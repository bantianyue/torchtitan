# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build the HuggingFace ``datasets`` arrow cache once, from a single process.

Why this exists
---------------
``run_cpu_train.py`` can pre-warm the cache by running one real rank first
(``_prewarm_dataset``), but that path constructs the whole model in that process.
For Qwen3.5-0.8B that is 852 M parameters: ~3.4 GB weights + ~3.4 GB grads +
~6.8 GB Adam moments, i.e. 13 GB+ against this host's 15.7 GiB of physical
memory. The warmup's only product is files under ``HF_DATASETS_CACHE``, so the
model is pure overhead -- and enough of it to be the difference between a
warmup that works and one that gets OOM-killed.

This script creates the same files without a model and without
``torch.distributed``: it builds the config's tokenizer and then its dataloader
once per ``dp_rank``, which are exactly the calls ``Trainer.__init__`` makes
(``trainer.py:521`` and ``trainer.py:529``), and throws them away.

Why the cache must pre-exist
----------------------------
``ChatDataset.__init__`` calls ``dataset.shuffle(seed=42)`` and
``split_dataset_by_node(...)``, and both make ``datasets`` write an Arrow
indices cache (``cache-<hash>.arrow``) next to the dataset cache. When N ranks
start together they race on that same destination path. Observed consequence: a
rank died with an obscure

    OSError: [Errno 22] Invalid argument:
      ...\\datasets\\json\\default-...\\0.0.0\\...\\cache-...arrow

raised from ``shutil.move`` -> ``open(dst, 'wb')``, while the other seven ranks
sat at ``Training starts at step 1`` with 0%% CPU until gloo reported
``Timed out waiting 300000ms for recv operation to complete`` in the first FSDP
all-gather. Two failures that look unrelated but are one: the survivors were
waiting on the rank that had already died.

With the files already on disk every rank only reads them, so the race cannot
happen. ``run_cpu_train.py --no-warmup`` skips its own (model-building) warmup
and relies on this script having been run first.

Usage::

    python cpu_lab/prewarm_dataset.py --hf-cache D:/02_source/torchtitan/cpu_lab/hf \\
        --config qwen35_0_8b_smoke --dp-world-size 8

The cache root must match what the training run uses (``--hf-cache``).
"""

import argparse
import importlib
import os
import shutil
import sys
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hf-cache",
        required=True,
        help="HuggingFace cache root; must equal the training run's --hf-cache.",
    )
    parser.add_argument(
        "--module",
        default="cpu_lab.qwen35",
        help="Config package holding <config_registry>.py",
    )
    parser.add_argument(
        "--config",
        default="qwen35_0_8b_smoke",
        help="Config function name inside <module>.config_registry",
    )
    parser.add_argument(
        "--dp-world-size",
        type=int,
        default=8,
        help="Number of dp ranks to warm index caches for (= batch degree).",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete the datasets cache first. Use after a crashed run: a "
        "half-written cache is worse than none.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    shared_cache = os.path.abspath(args.hf_cache)
    datasets_cache = os.path.join(shared_cache, "datasets")

    # Cross-drive path spellings would produce different cache hashes, so the
    # script resolves everything to absolute paths before handing them over.
    if args.wipe and os.path.isdir(datasets_cache):
        shutil.rmtree(datasets_cache, ignore_errors=True)
        print(f"[prewarm] wiped {datasets_cache}", flush=True)

    os.makedirs(datasets_cache, exist_ok=True)

    # These must be set before `datasets` is imported: it reads HF_DATASETS_CACHE
    # at import time into a module-level constant.
    os.environ["HF_HOME"] = shared_cache
    os.environ["HF_DATASETS_CACHE"] = datasets_cache
    # Everything is local (a vendored json file), so never touch the network --
    # same as the training launcher does.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    lab_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(lab_dir)
    for path in (lab_dir, repo_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    # Same install order as cpu_lab_child: cpu_compat first (it patches the
    # device snapshot torchtitan takes at import time), then the Qwen3.5 shim
    # (the model module imports `fla` at module scope, and building the config
    # imports that module).
    import cpu_compat

    cpu_compat.apply()
    import qwen35_cpu

    qwen35_cpu.apply()

    config_registry = importlib.import_module(f"{args.module}.config_registry")
    config_fn = getattr(config_registry, args.config)
    config = config_fn()

    print(
        f"[prewarm] HF_DATASETS_CACHE={datasets_cache}\n"
        f"[prewarm] config={args.module}.{args.config}\n"
        f"[prewarm] dp_world_size={args.dp_world_size}",
        flush=True,
    )

    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    seq_len = config.training.seq_len
    local_batch_size = config.training.local_batch_size

    # dp_rank only changes which slice of the shuffled dataset this process
    # keeps, and `select()` is what writes the per-rank indices cache -- so every
    # rank the launcher will start needs to be warmed, not just rank 0.
    for dp_rank in range(args.dp_world_size):
        started = time.time()
        config.dataloader.build(
            dp_world_size=args.dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
            snapshot_every_n_steps=1,
        )
        print(
            f"[prewarm] dp_rank {dp_rank}/{args.dp_world_size} ready "
            f"in {time.time() - started:.1f}s",
            flush=True,
        )

    total = 0
    for root, _dirs, files in os.walk(datasets_cache):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    print(
        f"[prewarm] done: {total / 1024:.0f} KiB in {datasets_cache}\n"
        "[prewarm] now launch training with --no-warmup and the same --hf-cache",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
