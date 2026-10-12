#!/usr/bin/env python
"""LoRA SFT of Qwen3.5-0.8B on CPU (no quantization).

Why no 4-bit: on a CPU with no GPU, bitsandbytes' 4-bit path dequantizes to
bf16 before every matmul, so it is *slower* than native fp32 and the 0.8B base
already fits in RAM. We keep the base in fp32/bf16 and only train LoRA adapters,
which is where the real memory saving comes from (no grads/optimizer state for
the frozen 0.85B of weights).

CPU-specific landmines this script avoids on purpose:
  1. attn_implementation="eager"  -- CPU SDPA dispatches to
     _scaled_dot_product_flash_attention_for_cpu, which has NO backward kernel,
     so training with the default SDPA blows up.
  2. enable_input_require_grads() -- the LoRA-adapted layers are downstream of a
     frozen embedding; without this the graph has no leaf requiring grad and
     backward fails / silently produces no gradient.
  3. text-only forward -- the model is a VLM (100.6M-param vision tower); we
     never feed images so the tower is never executed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Prefer the flat local snapshot (fetch_model_mirror.sh) over the hub cache:
# huggingface.co is unreachable from this box and the cache layout needs symlinks.
LOCAL_MODEL = os.path.join(HERE, "model")
DEFAULT_MODEL = LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else "Qwen/Qwen3.5-0.8B"

DEFAULT_TARGETS = [
    # standard attention (6 layers)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # gated delta-net linear attention (18 layers)
    "in_proj_qkv", "in_proj_z", "out_proj",
    # MLP, all 24 layers
    "gate_proj", "up_proj", "down_proj",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_rss_gib() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    except Exception:
        return float("nan")


def system_ram_gib() -> dict:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total": vm.total / 1024 ** 3, "available": vm.available / 1024 ** 3,
                "used_pct": vm.percent}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_samples(ds, n: int, fmt: str) -> list[dict]:
    out = []
    for row in ds:
        if fmt == "alpaca":
            ins = row.get("instruction", "") or ""
            inp = row.get("input", "") or ""
            resp = row.get("output", "") or row.get("response", "") or ""
            if inp.strip():
                user = f"{ins.strip()}\n\n{inp.strip()}"
            else:
                user = ins.strip()
        else:  # dolly
            ins = row.get("instruction", "") or ""
            ctx = row.get("context", "") or ""
            resp = row.get("response", "") or ""
            user = f"{ins.strip()}\n\n{ctx.strip()}" if ctx.strip() else ins.strip()
        if not user or not resp.strip():
            continue
        out.append({"prompt": user, "response": resp.strip()})
        if len(out) >= n:
            break
    return out


def encode(tokenizer, sample: dict, seq_len: int, use_chat: bool):
    """Return (input_ids, labels) with the prompt masked out of the loss."""
    if use_chat:
        msgs_p = [{"role": "user", "content": sample["prompt"]}]
        msgs_f = msgs_p + [{"role": "assistant", "content": sample["response"]}]
        try:
            p_ids = tokenizer.apply_chat_template(
                msgs_p, tokenize=True, add_generation_prompt=True)
            f_ids = tokenizer.apply_chat_template(
                msgs_f, tokenize=True, add_generation_prompt=False)
        except Exception:
            p_ids = f_ids = None
        if p_ids and f_ids and len(f_ids) > len(p_ids) and f_ids[:len(p_ids)] == p_ids:
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            ids = f_ids
        else:
            use_chat = False  # fall through

    if not use_chat:
        text_p = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"
        ids = tokenizer(text_p + sample["response"] + tokenizer.eos_token)["input_ids"]
        p_len = len(tokenizer(text_p)["input_ids"])
        labels = [-100] * p_len + ids[p_len:]

    ids, labels = ids[:seq_len], labels[:seq_len]
    return ids, labels


def make_batch(tokenizer, encoded, max_len: int):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    L = min(max(len(e[0]) for e in encoded), max_len)
    input_ids, labels, attn = [], [], []
    for ids, lab in encoded:
        ids, lab = ids[:L], lab[:L]
        n = L - len(ids)
        input_ids.append(ids + [pad_id] * n)
        labels.append(lab + [-100] * n)
        attn.append([1] * len(ids) + [0] * n)
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long))


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default="yahma/alpaca-cleaned")
    p.add_argument("--dataset-fmt", default="alpaca", choices=["alpaca", "dolly"])
    p.add_argument("--n-samples", type=int, default=512, help="slice size to use")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--steps", type=int, default=20, help="optimizer steps to run")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default=",".join(DEFAULT_TARGETS))
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="load + one forward/backward, report cost, then exit")
    p.add_argument("--out-dir", default=os.path.join(HERE, "runs", "default"))
    p.add_argument("--save-adapter", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    log(f"torch {torch.__version__} | threads {args.threads} | dtype {args.dtype}")
    log(f"ram: {system_ram_gib()}")

    # ---- tokenizer -------------------------------------------------------- #
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    log(f"tokenizer: {type(tok).__name__} | vocab {tok.vocab_size} | "
        f"pad={tok.pad_token_id} eos={tok.eos_token_id}")

    # ---- model ------------------------------------------------------------ #
    from transformers import AutoModelForImageTextToText
    dt = getattr(torch, args.dtype)
    log(f"loading {args.model} as {args.dtype}, attn_implementation=eager ...")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dt,
        attn_implementation="eager",   # CPU SDPA has no backward kernel
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()   # grads must reach the frozen embed
    log(f"model loaded in {time.time() - t0:.1f}s | rss {peak_rss_gib():.2f} GiB")

    base_total = sum(p.numel() for p in model.parameters())
    log(f"base params: {base_total:,}")

    # ---- lora ------------------------------------------------------------- #
    from peft import LoraConfig, get_peft_model
    targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    present = {n.split(".")[-1] for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)}
    missing = [t for t in targets if t not in present]
    if missing:
        log(f"WARNING target modules not found in model: {missing}")
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[t for t in targets if t in present],
        task_type=None,   # keep VLM head intact
    )
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"lora: r={args.lora_r} alpha={args.lora_alpha} targets={lcfg.target_modules}")
    log(f"trainable params: {trainable:,} ({100 * trainable / base_total:.3f}% of base)")

    # ---- data ------------------------------------------------------------- #
    from datasets import load_dataset
    log(f"loading dataset {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")
    samples = build_samples(ds, args.n_samples, args.dataset_fmt)
    log(f"dataset rows={len(ds):,} -> using {len(samples)} samples")
    encoded = [encode(tok, s, args.seq_len, use_chat=True) for s in samples]
    lens = [len(e[0]) for e in encoded]
    log(f"token len: min {min(lens)} mean {sum(lens)/len(lens):.0f} max {max(lens)}")

    # ---- optimizer -------------------------------------------------------- #
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    model.train()

    # ---- run -------------------------------------------------------------- #
    metrics: dict = {
        "args": vars(args),
        "base_params": base_total,
        "trainable_params": trainable,
        "dataset_rows": len(ds),
        "used_samples": len(samples),
        "token_len": {"min": min(lens), "mean": sum(lens) / len(lens), "max": max(lens)},
        "steps": [],
        "peak_rss_gib": None,
        "wall_s": None,
    }

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    def forward_backward(encoded_batch, scale: float = 1.0):
        """One micro-batch forward + backward. Caller owns zero_grad/step."""
        ids, labels, attn = make_batch(tok, encoded_batch, args.seq_len)
        t0 = time.time()
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
        fwd = time.time() - t0
        loss = out.loss
        t0 = time.time()
        (loss * scale).backward()
        bwd = time.time() - t0
        return float(loss.detach()), fwd, bwd, int(attn.sum())

    if args.dry_run:
        log("=== DRY RUN: one forward+backward on a real batch ===")
        e = encoded[0]
        log(f"example tokens={len(e[0])} supervised={sum(1 for x in e[1] if x != -100)}")
        opt.zero_grad(set_to_none=True)
        loss, fwd, bwd, nt = forward_backward([e])
        gn = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        log(f"DRY RUN OK | loss={loss:.4f} grad_norm={float(gn):.4f} "
            f"fwd={fwd:.2f}s bwd={bwd:.2f}s tokens={nt} peak_rss={peak_rss_gib():.2f} GiB")
        metrics["dry_run"] = {"loss": loss, "grad_norm": float(gn), "fwd_s": fwd,
                              "bwd_s": bwd, "tokens": nt,
                              "peak_rss_gib": peak_rss_gib()}
        metrics["peak_rss_gib"] = peak_rss_gib()
        metrics["wall_s"] = time.time() - t_start
        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=1, ensure_ascii=False)
        log(f"wrote {os.path.join(args.out_dir, 'metrics.json')}")
        return 0

    log(f"=== TRAIN: {args.steps} steps, batch {args.batch_size} x accum {args.grad_accum} ===")
    cursor, t_train0 = 0, time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        losses, fwd_t, bwd_t, ntok = [], 0.0, 0.0, 0
        for _ in range(args.grad_accum):
            batch = [encoded[(cursor + i) % len(encoded)]
                     for i in range(args.batch_size)]
            cursor += args.batch_size
            loss, fwd, bwd, nt = forward_backward(batch, 1.0 / args.grad_accum)
            losses.append(loss); fwd_t += fwd; bwd_t += bwd; ntok += nt
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0))
        opt.step()
        rec = {"step": step, "loss": sum(losses) / len(losses), "grad_norm": gnorm,
               "fwd_s": round(fwd_t, 2), "bwd_s": round(bwd_t, 2), "tokens": ntok,
               "rss_gib": round(peak_rss_gib(), 3),
               "elapsed_s": round(time.time() - t_train0, 2)}
        metrics["steps"].append(rec)
        rate = ntok / max(rec["fwd_s"] + rec["bwd_s"], 1e-9)
        log(f"step {step:3d}/{args.steps}  loss {rec['loss']:.4f}  "
            f"gnorm {rec['grad_norm']:.3f}  {rec['fwd_s'] + rec['bwd_s']:.1f}s  "
            f"{ntok} tok  ({rate:.1f} tok/s)  rss {rec['rss_gib']:.2f} GiB")

    wall = time.time() - t_train0
    steps_done = len(metrics["steps"])
    mean_step = wall / max(steps_done, 1)
    mean_tok = sum(s["tokens"] for s in metrics["steps"]) / max(steps_done, 1)
    metrics.update({
        "wall_s": time.time() - t_start,
        "train_wall_s": wall,
        "mean_step_s": mean_step,
        "mean_tokens_per_step": mean_tok,
        "tokens_per_s": mean_tok / mean_step if mean_step else None,
        "peak_rss_gib": peak_rss_gib(),
        "final_loss": metrics["steps"][-1]["loss"] if steps_done else None,
        "first_loss": metrics["steps"][0]["loss"] if steps_done else None,
    })
    log(f"=== DONE === {steps_done} steps in {wall/60:.1f} min | "
        f"{mean_step:.1f} s/step | {metrics['tokens_per_s']:.1f} tok/s | "
        f"peak rss {metrics['peak_rss_gib']:.2f} GiB")

    if args.save_adapter:
        ap = os.path.join(args.out_dir, "adapter")
        model.save_pretrained(ap)
        tok.save_pretrained(ap)
        log(f"adapter saved -> {ap}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1, ensure_ascii=False)
    log(f"wrote {os.path.join(args.out_dir, 'metrics.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
