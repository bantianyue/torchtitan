#!/usr/bin/env python
"""Self-test for the data pipeline: chat-template prefix alignment + loss mask.

The failure mode this guards against is silent: if the prompt prefix used for
masking does not match the prefix of the full sequence, the mask shifts by a few
tokens and the model gets trained on its own prompt (or the response is dropped).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402

sys.path.insert(0, HERE)
from train_lora import build_samples, encode, make_batch  # noqa: E402

_LOCAL = os.path.join(HERE, "model")
MODEL = os.environ.get("QWEN_MODEL") or (_LOCAL if os.path.isdir(_LOCAL)
                                         else "Qwen/Qwen3.5-0.8B")
ok = True


def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    ok = ok and bool(cond)


print(f"tokenizer: {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"  class={type(tok).__name__} vocab={tok.vocab_size} "
      f"pad={tok.pad_token_id} eos={repr(tok.eos_token)}")

print("\nchat template present:", bool(getattr(tok, "chat_template", None)))

ds = load_dataset("yahma/alpaca-cleaned", split="train")
samples = build_samples(ds, 200, "alpaca")
print(f"\nsamples built: {len(samples)}")

bad_chat = 0
lens, sup = [], []
for i, s in enumerate(samples[:50]):
    ids, labels = encode(tok, s, seq_len=512, use_chat=True)
    lens.append(len(ids))
    sup.append(sum(1 for x in labels if x != -100))
    # supervised positions must be a contiguous suffix
    n_mask = sum(1 for x in labels if x == -100)
    contiguous = labels[:n_mask] == [-100] * n_mask and \
        all(x != -100 for x in labels[n_mask:])
    if not contiguous:
        bad_chat += 1
    if i == 0:
        print(f"\n--- example 0 ({len(ids)} tok, {sup[-1]} supervised) ---")
        print("PROMPT   :", repr(s["prompt"][:110]))
        print("RESPONSE :", repr(s["response"][:110]))
        print("decoded (supervised part):",
              repr(tok.decode([x for x in labels if x != -100])[:160]))
        tok_p = len(tok.decode(ids[:n_mask]))
        print(f"masked prefix decodes to {tok_p} chars")

print(f"\nlengths: min {min(lens)} mean {sum(lens)/len(lens):.0f} max {max(lens)}")
print(f"supervised: min {min(sup)} mean {sum(sup)/len(sup):.0f} max {max(sup)}")
check(bad_chat == 0, f"loss mask is a contiguous suffix on all 50 samples (bad={bad_chat})")
check(all(s > 0 for s in sup), "every sample has at least one supervised token")
check(max(lens) <= 512, "lengths within seq_len")

# batching / padding
enc = [encode(tok, s, 256, True) for s in samples[:8]]
ids, labels, attn = make_batch(tok, enc, 256)
print(f"\nbatch shapes: ids={tuple(ids.shape)} labels={tuple(labels.shape)} "
      f"attn={tuple(attn.shape)}")
check(ids.shape == labels.shape == attn.shape, "ids/labels/attn shapes match")
check(bool((attn.sum(1) <= 256).all()), "attention lengths <= seq_len")
check(int((labels != -100).sum()) > 0, "batch has supervised tokens")
# padded positions must be masked out of both attention and loss
padded = attn == 0
check(int(labels[padded].max().item() if padded.any() else 0) <= 0,
      "padded positions carry no supervised labels")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
