"""Convert the HuggingFace Qwen2.5-0.5B checkpoint into torchtitan's DCP layout.

Same shape as ``cpu_lab/convert_qwen35_hf.py``: build the torchtitan model,
convert its (empty) state dict to HF naming, let ``HuggingFaceStorageReader``
fill it from the safetensors, convert back, write DCP.

Two checks are worth their cost here, because both of Qwen2.5's architectural
deltas from the Qwen3 skeleton it reuses are *silent* if they are wrong:

- **q/k/v bias.** Qwen2.5 has it, Qwen3 does not. If the mapping dropped it,
  every coverage check would still pass -- the bias would simply keep its zero
  initialisation, and the loss would start at the random-init value and never
  recover. The coverage report counts the 72 bias keys explicitly.
- **q_norm / k_norm.** Qwen3 has them, Qwen2.5 does not. If the model were
  built with them on, they would be randomly initialised and the load would
  report full coverage. ``--dry-check`` compares shapes against the checkpoint
  so a spurious pair cannot slip through.

Run from the repository root::

    python cpu_lab/convert_qwen25_hf.py --dry-check   # names + shapes only
    python cpu_lab/convert_qwen25_hf.py               # the real conversion
    python cpu_lab/convert_qwen25_hf.py --untied      # for pipeline parallel
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_HF_DIR = os.path.join(_REPO_ROOT, "cpu_lab", "hf25")
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "cpu_lab", "checkpoints", "qwen25_0_5b_hf")


def _peak_rss_gib() -> float:
    """Peak working set of this process, via the lab's existing hostmem shim.

    Deliberately not hand-rolled ctypes: a mis-declared PROCESS_MEMORY_COUNTERS
    (or a truncated process handle) makes ``GetProcessMemoryInfo`` return a
    zeroed struct with no error, which reads as "0.00 GiB" rather than failing.
    """
    import hostmem

    return hostmem.process_memory()["peak_rss"] / 1024**3


def _hf_index_keys(hf_dir: str) -> set[str]:
    """Every tensor name the checkpoint on disk claims to contain."""
    index = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as fh:
            return set(json.load(fh)["weight_map"])
    from safetensors import safe_open

    single = os.path.join(hf_dir, "model.safetensors")
    with safe_open(single, framework="pt") as fh:
        return set(fh.keys())


def _report_coverage(hf_dir: str, hf_state_dict: dict, adapter) -> None:
    """Diff the adapter's HF view against the checkpoint's own key list."""
    on_disk = _hf_index_keys(hf_dir)
    mapped = set(hf_state_dict)

    unmapped = sorted(on_disk - mapped)
    invented = sorted(mapped - on_disk)

    print(f"[convert] HF keys on disk     : {len(on_disk)}")
    print(f"[convert] HF keys after to_hf : {len(mapped)}")
    if unmapped:
        print(f"[convert] !! {len(unmapped)} checkpoint keys the adapter drops "
              f"(these keep random init):")
        for key in unmapped[:20]:
            print(f"    - {key}")
    else:
        print("[convert] every checkpoint key is mapped")

    if invented:
        print(f"[convert] !! {len(invented)} keys the adapter invents "
              f"(not in the checkpoint):")
        for key in invented[:20]:
            print(f"    + {key}")
    else:
        print("[convert] no invented keys")

    # The Qwen2.5-specific assertion: 24 layers x (q, k, v) biases.
    biases = [k for k in mapped if k.endswith((".q_proj.bias", ".k_proj.bias",
                                               ".v_proj.bias"))]
    expected = 3 * len(adapter.model_config.layers)
    status = "OK" if len(biases) == expected else "MISMATCH"
    print(f"[convert] qkv biases: {len(biases)} (expected {expected}) [{status}]")

    # And the reverse: Qwen2.5 must NOT grow q_norm/k_norm.
    qk = [k for k in mapped if k.endswith((".q_norm.weight", ".k_norm.weight"))]
    print(f"[convert] q_norm/k_norm keys: {len(qk)} (expected 0) "
          f"[{'OK' if not qk else 'MISMATCH'}]")


def _shape_check(hf_dir: str, hf_state_dict: dict) -> int:
    """Compare each mapped tensor's shape with the checkpoint's own metadata."""
    from safetensors import safe_open

    single = os.path.join(hf_dir, "model.safetensors")
    index = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = sorted(set(weight_map.values()))
        actual = {}
        for shard in shards:
            with safe_open(os.path.join(hf_dir, shard), framework="pt") as fh:
                for key in fh.keys():
                    actual[key] = fh.get_slice(key).get_shape()
    else:
        with safe_open(single, framework="pt") as fh:
            actual = {k: fh.get_slice(k).get_shape() for k in fh.keys()}

    mismatches = 0
    for key, tensor in sorted(hf_state_dict.items()):
        want = actual.get(key)
        if want is None:
            continue
        got = tuple(tensor.shape)
        if got != tuple(want):
            if mismatches < 20:
                print(f"    ! {key}: model {got} vs checkpoint {tuple(want)}")
            mismatches += 1
    print(f"[convert] shape check: {mismatches} mismatch(es)")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_HF_DIR,
                        help="directory holding the HF safetensors checkpoint")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR,
                        help="destination directory for the DCP checkpoint")
    parser.add_argument("--flavor", default="0.5B")
    parser.add_argument("--dry-check", action="store_true",
                        help="build on the meta device and only verify the "
                             "HF<->torchtitan mapping (names and shapes)")
    parser.add_argument("--untied", action="store_true",
                        help="materialise a real lm_head.weight (a copy of the "
                             "embedding) instead of tying it. Required by "
                             "pipeline parallel; see qwen25_0_5b_pp_bf16.")
    args = parser.parse_args()

    # cpu_compat first: it degrades torch.compile before torchtitan's attention
    # module is imported. qwen25_cpu then supplies the flex/mask/checkpoint
    # patches (reused from qwen35_cpu -- see its docstring).
    import cpu_compat

    cpu_compat.apply()
    import qwen25_cpu

    qwen25_cpu.apply()

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageReader

    from torchtitan.components.checkpoint import ModelWrapper

    from qwen25 import model_registry

    t0 = time.time()
    spec = model_registry(
        args.flavor, attn_backend="flex", weight_tying=not args.untied
    )
    model_config = spec.model
    if args.dry_check:
        with torch.device("meta"):
            model = model_config.build()
    else:
        with torch.device("cpu"):
            model = model_config.build()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[convert] built {args.flavor}: {n_params:,} parameters "
          f"({n_params * 4 / 1024**3:.2f} GiB fp32) in {time.time() - t0:.1f}s")

    wrapped = ModelWrapper(model)
    # hf_assets_path is the HF dir so from_hf() can validate the rope config
    # against config.json -- a theta mismatch here is silent otherwise.
    adapter = spec.state_dict_adapter(model_config, args.input_dir)
    assert adapter is not None, "qwen25 must provide a state_dict_adapter"

    t0 = time.time()
    hf_state_dict = adapter.to_hf(wrapped._get_state_dict())
    print(f"[convert] to_hf: {len(hf_state_dict)} tensors in {time.time() - t0:.1f}s, "
          f"peak RSS {_peak_rss_gib():.2f} GiB")

    # With --untied the HF view now contains lm_head.weight, but the safetensors
    # on disk do not (HF ties it). Drop it before the coverage report and the
    # load so neither looks for a key that cannot exist, then restore it by
    # copying the embedding -- exactly what tying does at runtime, written out.
    head_key, emb_key = "lm_head.weight", "model.embed_tokens.weight"
    if args.untied:
        if head_key not in hf_state_dict:
            raise SystemExit(f"[convert] --untied but {head_key} absent")
        if emb_key not in hf_state_dict:
            raise SystemExit(f"[convert] --untied but {emb_key} absent")
        del hf_state_dict[head_key]
        print(f"[convert] --untied: {head_key} withheld from the load "
              f"(absent on disk), will be copied from {emb_key}")

    _report_coverage(args.input_dir, hf_state_dict, adapter)
    mismatches = _shape_check(args.input_dir, hf_state_dict)

    if args.dry_check:
        print(f"[convert] dry check done (mismatches={mismatches})")
        return 0 if mismatches == 0 else 1
    if mismatches:
        print("[convert] refusing to load: shape mismatches above")
        return 1

    t0 = time.time()
    dcp.load(hf_state_dict, storage_reader=HuggingFaceStorageReader(path=args.input_dir))
    print(f"[convert] loaded HF weights in {time.time() - t0:.1f}s, "
          f"peak RSS {_peak_rss_gib():.2f} GiB")

    if args.untied:
        hf_state_dict[head_key] = hf_state_dict[emb_key].clone()
        print(f"[convert] --untied: copied {emb_key} -> {head_key} "
              f"{tuple(hf_state_dict[head_key].shape)}")

    t0 = time.time()
    tt_state_dict = adapter.from_hf(hf_state_dict)
    print(f"[convert] from_hf: {len(tt_state_dict)} tensors in {time.time() - t0:.1f}s, "
          f"peak RSS {_peak_rss_gib():.2f} GiB")

    del hf_state_dict

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    dcp.save(tt_state_dict, checkpoint_id=args.output_dir)
    print(f"[convert] wrote {args.output_dir} in {time.time() - t0:.1f}s")

    with open(os.path.join(args.output_dir, "_conversion.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": args.input_dir,
                "flavor": args.flavor,
                "weight_tying": not args.untied,
                "n_params": n_params,
                "hf_tensors": len(tt_state_dict),
            },
            fh,
            indent=2,
        )
    print(f"[convert] peak RSS {_peak_rss_gib():.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
