"""Convert the HuggingFace Qwen3.5-0.8B checkpoint into torchtitan's DCP layout.

Same five steps as ``scripts/checkpoint_conversion/convert_from_hf.py`` -- build
the torchtitan model, convert the (empty) state dict to HF naming, let
``HuggingFaceStorageReader`` fill it from the safetensors shards, convert back --
plus two things that script does not do, both of which matter more here than they
would upstream:

1. **Key coverage report.** Both directions of the mapping are diffed against the
   checkpoint's own ``weight_map``. A name the adapter drops means that weight
   silently keeps its random initialisation and trains from noise; a name the
   adapter invents means a shape will not line up. Either failure is invisible in
   the loss curve for a long time, so it is checked up front.
2. **Per-stage timing and memory**, because at 1.1B parameters fp32 this is the
   one step of the pipeline whose peak memory (fused HF tensors and split
   torchtitan tensors alive at once) can exceed the host.

Run it from the repository root::

    python cpu_lab/convert_qwen35_hf.py
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

DEFAULT_HF_DIR = os.path.join(_REPO_ROOT, "qlora_lab", "model")
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "cpu_lab", "checkpoints", "qwen35_0_8b_hf")


def _peak_rss_gib() -> float:
    """Peak working set of this process, via the lab's existing hostmem shim.

    Deliberately not hand-rolled ctypes: a mis-declared PROCESS_MEMORY_COUNTERS
    (or a truncated process handle) makes ``GetProcessMemoryInfo`` return a
    zeroed struct with no error, which reads as "0.00 GiB" rather than failing.
    ``hostmem`` already carries that lesson in its docstring.
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
        print(f"[convert] !! {len(invented)} adapter keys absent from the "
              f"checkpoint (these load as zeros/unjinitialised):")
        for key in invented[:20]:
            print(f"    + {key}")
    else:
        print("[convert] adapter invents no keys")

    # lm_head is expected to be "invented" only when tying is off; turning it on
    # (which build_model_spec does) removes the key from the HF view entirely.
    if invented == ["lm_head.weight"]:
        print("[convert] (lm_head.weight only -- expected if weight tying is "
              "off: HF ties it to embed_tokens and from_hf copies it across)")
    del adapter


def _shape_check(hf_dir: str, hf_state_dict: dict) -> int:
    """Compare every mapped tensor's shape against the safetensors header.

    Runs on meta-device tensors, so it costs nothing and is worth doing before
    the real conversion: a transposed or fused-vs-split mistake shows up here as
    a shape mismatch instead of as a silently mis-trained model.

    Returns the number of mismatches.
    """
    import torch
    from safetensors import safe_open

    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    shards: dict[str, list[str]] = {}
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        for name, shard in weight_map.items():
            shards.setdefault(shard, []).append(name)
    else:
        shards["model.safetensors"] = list(hf_state_dict)

    mismatches = 0
    checked = 0
    for shard, names in sorted(shards.items()):
        path = os.path.join(hf_dir, shard)
        if not os.path.isfile(path):
            print(f"[shape] !! missing shard {shard}")
            mismatches += 1
            continue
        with safe_open(path, framework="pt") as fh:
            present = set(fh.keys())
            for name in names:
                if name not in hf_state_dict or name not in present:
                    continue
                expected = tuple(hf_state_dict[name].shape)
                actual = tuple(fh.get_slice(name).get_shape())
                checked += 1
                if expected != actual:
                    mismatches += 1
                    if mismatches <= 20:
                        print(f"[shape] !! {name}: adapter {expected} "
                              f"vs checkpoint {actual}")
    print(f"[shape] checked {checked} tensors, {mismatches} mismatches")
    del torch
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_HF_DIR,
                        help="directory holding the HF safetensors checkpoint")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR,
                        help="destination directory for the DCP checkpoint")
    parser.add_argument("--flavor", default="0.8B")
    parser.add_argument("--dry-check", action="store_true",
                        help="build on the meta device and only verify the "
                             "HF<->torchtitan mapping (names and shapes)")
    parser.add_argument("--untied", action="store_true",
                        help="materialise a real lm_head.weight (a copy of the "
                             "embedding) instead of tying it. Required by "
                             "pipeline parallel; see qwen35_0_8b_pp_bf16.")
    args = parser.parse_args()

    # cpu_compat first: it degrades torch.compile before torchtitan's attention
    # module is imported. qwen35_cpu then supplies the fla and attention shims.
    import cpu_compat

    cpu_compat.apply()
    import qwen35_cpu

    qwen35_cpu.apply()

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageReader

    from torchtitan.components.checkpoint import ModelWrapper

    t0 = time.time()
    # build_model_spec (not model_registry directly): it enables weight tying,
    # which the HF checkpoint requires -- see its docstring. --untied turns that
    # off, because PP cannot tie (see the flag's help and the PP config).
    spec = qwen35_cpu.build_model_spec(
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
    adapter = spec.state_dict_adapter(model_config, None)
    assert adapter is not None, "qwen3_5 must provide a state_dict_adapter"

    # Model state dict -> HF naming. to_hf only allocates fused tensors
    # (GDN in_proj_qkv / conv1d, vision qkv) on top of the parameter storage.
    t0 = time.time()
    hf_state_dict = adapter.to_hf(wrapped._get_state_dict())
    print(f"[convert] to_hf: {len(hf_state_dict)} tensors in {time.time() - t0:.1f}s, "
          f"peak RSS {_peak_rss_gib():.2f} GiB")

    # With --untied the HF view now contains lm_head.weight, but the safetensors
    # on disk do not (HF ties it). Drop it before the coverage report and the
    # load so neither looks for a key that cannot exist, then restore it by
    # copying the embedding -- which is exactly what tying would have done at
    # runtime, just written out instead.
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
                "flavor": args.flavor,
                "input_dir": os.path.abspath(args.input_dir),
                "hf_dir": args.input_dir,
                "parameters": n_params,
                "tensors": len(tt_state_dict),
                "peak_rss_gib": round(_peak_rss_gib(), 3),
            },
            fh,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
