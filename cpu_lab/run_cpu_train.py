#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch TinyGPT on CPU as N ranks, one parallel strategy combination at a time.

Why this exists instead of ``torchrun``: official PyTorch *Windows CPU* wheels
are built without libuv, but ``TCPStore`` requests it by default, so any
``torchrun`` rendezvous fails with ``DistStoreError: use_libuv was requested but
PyTorch was built without libuv support``. This launcher spawns plain
subprocesses and installs the CPU compatibility patches (``cpu_compat``) in each
one before torchtitan is imported. See ``cpu_lab/README.md``.

Parallel strategies are expressed as ``world_size = dp_replicate * dp_shard *
cp * tp * pp`` and forwarded as ``--parallelism.*`` flags, which is exactly how
torchtitan itself selects them -- nothing about the strategy is baked into the
model or the training config.

Examples::

    python cpu_lab/run_cpu_train.py --combo single              # 1 rank, no parallelism
    python cpu_lab/run_cpu_train.py --combo fsdp    -n 8        # FSDP-8
    python cpu_lab/run_cpu_train.py --combo tp      -n 8        # TP-8
    python cpu_lab/run_cpu_train.py --combo tp_fsdp -n 8        # TP-2 x FSDP-4
    python cpu_lab/run_cpu_train.py --combo pp_tp   -n 8        # PP-4 x TP-2
    python cpu_lab/run_cpu_train.py --combo cp      -n 8        # CP-8
    python cpu_lab/run_cpu_train.py --list                      # show every preset
    python cpu_lab/run_cpu_train.py --combo fsdp -n 8 --dry-run # config check only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

import hostmem

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
LAB_DIR = HERE

DEFAULT_MODULE = "torchtitan.models.tinygpt"
DEFAULT_CONFIG = "tinygpt_debugmodel"

# Presets that cannot use the default config. Pipeline parallelism rejects
# weight tying and needs enough graph nodes to split, so PP presets switch to a
# dedicated untied 8-layer config unless the user passes --config.
_COMBO_DEFAULT_CONFIG = {
    "pp": "tinygpt_pp",
    "pp_tp": "tinygpt_pp",
    "pp_fsdp": "tinygpt_pp",
    "pp_tp_fsdp": "tinygpt_pp",
}

# Each preset maps a rank count to parallelism dims. They are chosen so the
# *combination* is the interesting part, not the arithmetic: every preset
# divides ``n`` evenly and exercises a distinct axis.
_COMBOS: dict[str, str] = {
    "single": "no parallelism (1-rank baseline)",
    "fsdp": "FSDP over all ranks (shard params + optimizer state)",
    "hsdp": "dp_replicate=2 x dp_shard=n/2 (hybrid shard + replicate)",
    "tp": "tensor parallel over all ranks (split heads + MLP)",
    "tp_fsdp": "TP=2 x FSDP=n/2 (the classic 2D combination)",
    "tp_sp": "TP=2 x FSDP=n/2 with sequence parallelism on",
    "pp": "pipeline parallel over all ranks (split layers)",
    "pp_tp": "PP=n/2 x TP=2 (2D: layers x heads)",
    "pp_fsdp": "PP=2 x FSDP=n/2",
    "pp_tp_fsdp": "PP=2 x TP=2 x FSDP=n/4 (3D: layers x heads x data)",
    "cp": "context parallel over all ranks (split the sequence)",
    "tp_cp": "TP=2 x CP=n/2 (heads x sequence)",
    # Same dimension mapping as "fsdp", but a distinct output directory: the
    # Qwen3.5 runs are 40x the parameter count of the TinyGPT presets and their
    # artifacts must not land on top of them.
    "qwen35": "FSDP over all ranks, for the Qwen3.5 CPU runs",
    # Qwen2.5-0.5B: the model that actually leaves room for several ranks on
    # this host (494M params, 3.68 GiB of bf16 state vs 6.36 GiB for Qwen3.5).
    "qwen25": "FSDP over all ranks, for the Qwen2.5 CPU runs",
}


# Presets that don't run on this CPU/gloo setup. They are kept because the
# configuration is correct and will work on a CUDA host; the launcher warns
# before starting them so a failure is not mistaken for a model bug.
_CPU_LIMITATIONS: dict[str, str] = {
    "cp": "context parallelism wraps K/V in DTensors, but CPU SDPA dispatches to "
    "aten._scaled_dot_product_flash_attention_for_cpu, which has no DTensor "
    "sharding strategy registered.",
    "tp_cp": "same CPU SDPA sharding gap as 'cp'.",
}

# Presets that run, but only under conditions worth stating up front. Printed as
# info, not as a failure warning.
_CPU_NOTES: dict[str, str] = {
    "hsdp": "verified at 4 ranks (dp_replicate=2 x dp_shard=2). At 8 ranks on "
    "this 8-core machine it hangs in a gloo recv timeout -- not thread "
    "oversubscription (--threads 1 hangs too) and not rank startup contention "
    "(--stagger 8 hangs too). dp_replicate alone (2 x 1) is fine.",
    "pp_tp_fsdp": "pass --sp false. With sequence parallelism on, this "
    "combination reaches 'Training starts at step 1' and then stops making "
    "progress; with SP off it completes. PP, PPxTP and PPxFSDP work with SP on.",
}


def _resolve_combo(combo: str, n: int) -> dict[str, int]:
    """Map a preset name + rank count to concrete parallelism degrees."""
    half = n // 2
    table = {
        "single": dict(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1),
        "fsdp": dict(dp_replicate=1, dp_shard=n, cp=1, tp=1, pp=1),
        "hsdp": dict(dp_replicate=2, dp_shard=half, cp=1, tp=1, pp=1),
        "tp": dict(dp_replicate=1, dp_shard=1, cp=1, tp=n, pp=1),
        "tp_fsdp": dict(dp_replicate=1, dp_shard=half, cp=1, tp=2, pp=1),
        "tp_sp": dict(dp_replicate=1, dp_shard=half, cp=1, tp=2, pp=1),
        "pp": dict(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=n),
        "pp_tp": dict(dp_replicate=1, dp_shard=1, cp=1, tp=2, pp=half),
        "pp_fsdp": dict(dp_replicate=1, dp_shard=half, cp=1, tp=1, pp=2),
        "pp_tp_fsdp": dict(dp_replicate=1, dp_shard=n // 4, cp=1, tp=2, pp=2),
        "cp": dict(dp_replicate=1, dp_shard=1, cp=n, tp=1, pp=1),
        "tp_cp": dict(dp_replicate=1, dp_shard=1, cp=half, tp=2, pp=1),
        # Same dimension mapping as "fsdp", but a distinct output directory: the
        # Qwen3.5 runs are ~40x the parameter count of the TinyGPT presets, and
        # _clear_outputs() wipes a combo's folder, so sharing "fsdp" would
        # destroy the scaling-ladder artifacts.
        "qwen35": dict(dp_replicate=1, dp_shard=n, cp=1, tp=1, pp=1),
        "qwen25": dict(dp_replicate=1, dp_shard=n, cp=1, tp=1, pp=1),
    }
    if combo not in table:
        raise SystemExit(f"unknown preset {combo!r}; known: {', '.join(sorted(table))}")
    return table[combo]


def _clear_outputs(combo: str) -> None:
    """Remove this strategy's output folder so the run starts from scratch.

    Needed because torchtitan's ``checkpoint.load_step`` defaults to ``-1``
    ("load the latest"): with checkpointing enabled, a second run against the
    same folder tries to resume, and because these configs save model-only
    (``last_save_model_only=True``) the load fails on the missing dataloader
    state -- ``Missing key in checkpoint state_dict: dataloader.dp_rank_0``.
    Starting clean also keeps the run reproducible, which matters more here than
    resumability.

    The warmup's own dump folder is cleared for the same reason (see
    ``_prewarm_dataset``): it is a run like any other, so a leftover checkpoint
    there made the second and every later warmup fail instantly -- silently,
    because the warmup is best-effort. The visible symptom appeared much later
    and looked unrelated: with the warmup failing, all N ranks built the dataset
    cache concurrently and intermittently wedged in it.

    Guarded so it can only ever touch ``cpu_lab/outputs/<combo>``.
    """
    outputs_root = os.path.realpath(os.path.join(HERE, "outputs"))
    target = os.path.realpath(_dump_dir(combo))
    if (
        os.path.commonpath([outputs_root, target]) != outputs_root
        or target == outputs_root
    ):
        raise RuntimeError(f"refusing to clear unexpected path: {target}")
    if not os.path.isdir(target):
        return
    import shutil

    shutil.rmtree(target, ignore_errors=True)
    print(f"[launcher] cleared previous outputs: {target}", flush=True)


def _prewarm_dataset(args, child_args, shared_cache, log_dir, ranks_to_spawn) -> None:
    """Prepare the dataset cache once, before any rank starts.

    Building the HuggingFace cache is the file-operation-heavy part of startup.
    Doing it concurrently from N ranks is what intermittently wedges (observed as
    ``state lock timeout`` inside 'Preparing c4_test dataset', after which the
    ranks that did start die on a gloo timeout). Running it once, single-process,
    removes the concurrency entirely; the ranks then only read.

    Skipped for single-process runs (they would warm it anyway) and when
    ``--no-warmup`` is passed.
    """
    if ranks_to_spawn <= 1 or args.no_warmup:
        return

    warmup_dump = os.path.join(HERE, "outputs", "_warmup")
    # Start the warmup from scratch, and turn its checkpointing off entirely --
    # the only thing it is for is building the dataset cache. A leftover
    # checkpoint in this folder makes the next warmup resume and die with
    # ``Missing key in checkpoint state_dict: dataloader.dp_rank_0``, and because
    # the launcher treats a failed warmup as non-fatal, every rank then builds
    # the cache concurrently instead.
    import shutil

    shutil.rmtree(warmup_dump, ignore_errors=True)

    warmup_cmd = [
        sys.executable, "-u", "-m", os.environ.get("CPU_LAB_ENTRY", "cpu_lab_child"),
        "--module", args.module, "--config", args.config,
        "--training.steps", "1",
        "--dump-folder", warmup_dump,
        "--checkpoint.no-enable",
    ]
    env = os.environ.copy()
    env.update(
        RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(args.master_port),
        OMP_NUM_THREADS=str(args.threads),
        HF_HOME=shared_cache,
        HF_DATASETS_CACHE=os.path.join(shared_cache, "datasets"),
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", USE_LIBUV="0",
        PYTHONPATH=os.pathsep.join([LAB_DIR, REPO_ROOT, env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        ),
        PYTHONUNBUFFERED="1",
    )
    print("[launcher] pre-warming dataset cache (single process) ...", flush=True)
    with open(os.path.join(log_dir, "_warmup.log"), "w", encoding="utf-8") as fh:
        rc = subprocess.call(warmup_cmd, env=env, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        print(
            f"[launcher] warmup exited with {rc}; continuing anyway "
            "(see cpu_lab/logs/_warmup.log)",
            file=sys.stderr,
        )
    else:
        print("[launcher] dataset cache ready", flush=True)


def _dump_dir(combo: str) -> str:
    """Per-strategy output directory, so artifacts never overwrite each other."""
    return os.path.join(HERE, "outputs", combo)


def _build_child_args(args, dims: dict[str, int]) -> list[str]:
    # CPU_LAB_ENTRY lets a debugging module replace the default child entry
    # (it must call cpu_compat.apply() itself before importing torchtitan).
    entry = os.environ.get("CPU_LAB_ENTRY", "cpu_lab_child")
    child = [sys.executable, "-u", "-m", entry]
    child += ["--module", args.module, "--config", args.config]
    child += [
        "--parallelism.data_parallel_replicate_degree", str(dims["dp_replicate"]),
        "--parallelism.data_parallel_shard_degree", str(dims["dp_shard"]),
        "--parallelism.context_parallel_degree", str(dims["cp"]),
        "--parallelism.tensor_parallel_degree", str(dims["tp"]),
        "--parallelism.pipeline_parallel_degree", str(dims["pp"]),
    ]
    # Give each strategy its own dump folder so checkpoints from different runs
    # do not overwrite each other and are easy to attribute afterwards.
    child += ["--dump-folder", _dump_dir(args.combo)]
    if args.seq_len is not None:
        child += ["--training.seq_len", str(args.seq_len)]
    if args.steps is not None:
        child += ["--training.steps", str(args.steps)]
    if args.local_batch_size is not None:
        child += ["--training.local_batch_size", str(args.local_batch_size)]
    if args.dataset_path is not None:
        # tyro exposes the dataloader's own dataset_path field, so a run can be
        # pointed at any local HF-style directory without touching torchtitan's
        # DATASETS registry. Used by --probe to train on the planted corpus.
        child += ["--dataloader.dataset-path", args.dataset_path]
    if args.sp is not None:
        # tyro renders booleans as a flag pair, not `--field true`.
        if args.sp:
            child += ["--parallelism.enable-sequence-parallel"]
        else:
            child += ["--parallelism.no-enable-sequence-parallel"]
    if args.dry_run:
        # fake_backend keeps world_size=N but performs no real communication, so
        # a single process can validate the config without gloo.
        child += ["--comm.mode=fake_backend"]
    return child


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--combo", default="fsdp", help="parallel strategy preset")
    p.add_argument(
        "-n", "--nproc", type=int, default=8, help="number of ranks (simulated GPUs)"
    )
    p.add_argument("--module", default=DEFAULT_MODULE, help="torchtitan module path")
    p.add_argument(
        "--config",
        default=None,
        help=f"config_registry function (default: per-combo; "
        f"{DEFAULT_CONFIG} for most presets)",
    )
    p.add_argument("--steps", type=int, default=None, help="override --training.steps")
    p.add_argument(
        "--seq-len", type=int, default=None, help="override --training.seq_len"
    )
    p.add_argument(
        "--local-batch-size",
        type=int,
        default=None,
        help="override --training.local_batch_size (per rank; global = x dp degree)",
    )
    p.add_argument(
        "--dataset-path",
        default=None,
        help="override --dataloader.dataset-path, e.g. cpu_lab/planted for the "
        "planted-secret corpus",
    )
    p.add_argument(
        "--collectives",
        action="store_true",
        help="count collective ops (call sites, bytes per rank, per step) into "
        "cpu_lab/logs/collectives_rank_*.json. Adds one Python call per aten op.",
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="after training, load the checkpoint and measure whether the model "
        "predicts the planted secrets (cpu_lab/logs/probe.json)",
    )
    p.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="render cpu_lab/reports/<combo>.html (default: on)",
    )
    p.add_argument(
        "--no-report",
        dest="report",
        action="store_false",
        help="skip HTML report generation",
    )
    p.add_argument(
        "--sp",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=None,
        help="force enable_sequence_parallel on/off (default: torchtitan's)",
    )
    p.add_argument("--threads", type=int, default=2, help="OMP threads per rank")
    p.add_argument(
        "--stagger",
        type=float,
        default=5.0,
        help="seconds between launching ranks. Dataset preparation touches "
        "shared lock files, and starting all ranks at once makes them contend "
        "(observed as a hang in 'Preparing c4_test dataset'). 2s proved too "
        "little for 8 ranks on this machine.",
    )
    p.add_argument(
        "--master-port", type=int, default=29540, help="TCPStore rendezvous port"
    )
    p.add_argument(
        "--hf-cache",
        default=None,
        help="HuggingFace cache root (default: cpu_lab/.hf_cache/shared). Point "
        "a fresh directory here when a run wedges in 'Preparing ... dataset': "
        "datasets rewrites a cache whose config hash changed, and the rewrite "
        "deletes files.",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="where to write logs and run.json (default: cpu_lab/logs). Point "
        "different experiments at different directories to keep their reports "
        "and artifacts side by side instead of overwriting each other.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="use fake_backend (no real collectives) to validate the config only",
    )
    p.add_argument("--list", action="store_true", help="list presets and exit")
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the single-process dataset-cache warmup (see _prewarm_dataset)",
    )
    p.add_argument(
        "--keep-outputs",
        action="store_true",
        help="do not clear cpu_lab/outputs/<combo> first. Note that torchtitan "
        "then tries to resume from the existing checkpoint.",
    )
    p.add_argument(
        "--show-logs", action="store_true", help="print rank 0's step lines after the run"
    )
    return p


# Signatures of a *native* crash, i.e. the process died in C++ rather than
# raising a Python exception. They matter because they are how a memory-exhausted
# Windows host reports a failed allocation: there is no catchable torch OOM on
# CPU, so the failure surfaces inside whatever native code touched memory last
# (an FSDP all-gather, a reduce-scatter, a oneDNN kernel). A rank that dies this
# way leaves its peers blocked in a collective with a peer that no longer exists,
# which is what makes the run look like a hang instead of a crash.
_NATIVE_CRASH_SIGNATURES = (
    "Windows fatal exception: access violation",
    "could not execute a primitive",
    "could not create a primitive",
    "Fatal Python error: Aborted",
    "Fatal Python error: Segmentation fault",
)


def _native_crash_scan(log_dir: str, ranks: int) -> list[dict]:
    """Find which rank died natively, and where.

    Scans every rank's log, not just the first failure: the launcher's watchdog
    kills the survivors of a dead rank, so the first non-zero exit code is often
    a rank that was killed *because* another one crashed. The rank whose log
    contains a native signature is the one that actually died first.
    """
    found: list[dict] = []
    for rank in range(ranks):
        name = "train.log" if ranks == 1 else f"rank_{rank}.log"
        path = os.path.join(log_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        for index, line in enumerate(lines):
            if any(sig in line for sig in _NATIVE_CRASH_SIGNATURES):
                # Keep a little context: the signature line plus the innermost
                # frames that follow it, which name the native call site.
                context = "".join(lines[index : index + 14])
                found.append(
                    {"rank": rank, "log": path, "signature": line.strip(), "context": context}
                )
                break
    return found


def _report_native_crash(log_dir: str, ranks: int, memory_peaks: dict) -> bool:
    """Explain a native crash as the memory wall it usually is. True if found."""
    crashes = _native_crash_scan(log_dir, ranks)
    if not crashes:
        return False
    first = crashes[0]
    print(
        f"\n[hint] rank {first['rank']} died in native code, not in Python:",
        file=sys.stderr,
    )
    print(f"       {first['signature']}", file=sys.stderr)
    # The innermost frames after the signature name the native call site.
    frames = [
        ln.strip()
        for ln in first["context"].splitlines()
        if ln.strip().startswith("File ") or "reduce_scatter" in ln or "all_gather" in ln
    ]
    for frame in frames[:6]:
        print(f"       {frame}", file=sys.stderr)
    per_rank = memory_peaks.get("per_rank") or {}
    if per_rank:
        total_rss = sum(e["peak_rss"] for e in per_rank.values())
        total_commit = sum(e["peak_commit"] for e in per_rank.values())
        low = memory_peaks.get("system_low") or {}
        print(
            f"\n       peak commit over {len(per_rank)} ranks: "
            f"{hostmem.gib(total_commit):.2f} GiB, peak RSS {hostmem.gib(total_rss):.2f} GiB.",
            file=sys.stderr,
        )
        if low.get("min_commit_free") is not None:
            print(
                f"       during the run: commit free fell to "
                f"{hostmem.gib(low['min_commit_free']):.2f} GiB and available "
                f"physical memory to {hostmem.gib(low['min_avail_phys']):.2f} GiB.",
                file=sys.stderr,
            )
        print(
            "\n       On a CPU-only Windows build a failed allocation has no catchable\n"
            "       error: it surfaces as a crash inside whatever allocated last --\n"
            "       here an FSDP collective or a oneDNN kernel. The other ranks then\n"
            "       block forever on a collective whose peer is already gone, which is\n"
            "       why the run reads as a hang and why a longer timeout cannot help.\n"
            "       Compare the numbers above with the host's free memory; if they do\n"
            "       not fit, this is a sizing problem, not a training bug.",
            file=sys.stderr,
        )
    return True


def _tail(path: str, n: int = 25) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return "".join(fh.readlines()[-n:])


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# The TinyGPT configs turn colour off (stdout goes to a file), but a run with
# colour on would otherwise break this parse -- hence the strip above.
_STEP_RE = re.compile(
    r"step:\s*(\d+)\s+loss:\s*(-?[\d.]+)\s+grad_norm:\s*(-?[\d.]+)"
)


def _collect_results(log_dir: str, ranks: int) -> dict:
    """Parse the per-rank logs into a per-rank result table.

    Only ranks that hold the last pipeline stage report a real loss; the others
    report a negative sentinel from ``pp_forward_backward_step``. Flagging that
    here is the whole point -- otherwise a PP run looks like it trained on
    negative loss.
    """
    rows = []
    for rank in range(ranks):
        name = "train.log" if ranks == 1 else f"rank_{rank}.log"
        path = os.path.join(log_dir, name)
        steps = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = _STEP_RE.search(_ANSI_RE.sub("", line))
                    if m:
                        steps.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        last = steps[-1] if steps else None
        rows.append(
            {
                "rank": rank,
                "log": path,
                "n_steps": len(steps),
                "last_step": last[0] if last else None,
                "last_loss": last[1] if last else None,
                "last_grad_norm": last[2] if last else None,
                "has_real_loss": bool(last and last[1] >= 0),
                # The whole series, not just the final value: the loss *shape*
                # is what shows whether a run learned anything, and under TP the
                # series must be bit-identical across ranks.
                "series": [[s, loss, grad] for s, loss, grad in steps],
            }
        )
    return {"rows": rows}


def _artifact_summary(dump_dir: str) -> list[str]:
    """Files the run left behind, grouped by kind."""
    out: list[str] = []
    if not os.path.isdir(dump_dir):
        return out
    for root, _dirs, files in os.walk(dump_dir):
        rel = os.path.relpath(root, dump_dir)
        for name in sorted(files):
            full = os.path.join(root, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            out.append(f"{size:>12,} B  {os.path.join(rel, name) if rel != '.' else name}")
    return out


def _write_summary(
    path: str,
    args,
    dims: dict[str, int],
    world_size: int,
    elapsed: float,
    results: dict,
    dump_dir: str,
    log_dir: str,
    memory_peaks: dict,
) -> str:
    """Human-readable record of one run, written next to the logs."""
    lines = []
    add = lines.append
    rule = "=" * 78
    add(rule)
    add("RUN SUMMARY")
    add(rule)
    add(f"  combo            {args.combo}")
    add(f"  module/config    {args.module} / {args.config}")
    add(
        f"  parallelism      dp_replicate={dims['dp_replicate']} "
        f"dp_shard={dims['dp_shard']} cp={dims['cp']} tp={dims['tp']} pp={dims['pp']}"
    )
    add(f"  world_size       {world_size}")
    add(f"  wall time        {elapsed:.1f} s")
    add(f"  command          {_launcher_command()}")
    add("")
    add("  per-rank results (real loss only on last-pipeline-stage ranks)")
    add(
        "  rank  steps  last_step  last_loss      last_grad_norm   note"
    )
    for row in results["rows"]:
        note = "" if row["has_real_loss"] else "sentinel, not a loss"
        loss = "n/a" if row["last_loss"] is None else f"{row['last_loss']:.6f}"
        grad = "n/a" if row["last_grad_norm"] is None else f"{row['last_grad_norm']:.4f}"
        add(
            f"  {row['rank']:<5} {row['n_steps']:<6} "
            f"{str(row['last_step']):<10} {loss:<13} {grad:<16} {note}"
        )

    real = [r for r in results["rows"] if r["has_real_loss"]]
    if real:
        losses = [r["last_loss"] for r in real]
        grads = {round(r["last_grad_norm"], 4) for r in real}
        add("")
        add(f"  last-stage ranks         {[r['rank'] for r in real]}")
        add(f"  their last losses        {[f'{v:.6f}' for v in losses]}")
        add(f"  distinct loss values     {len(set(round(v, 6) for v in losses))}"
            "   (1 = all last-stage ranks agree)")
        add(f"  distinct grad_norms      {sorted(grads)}"
            "   (1 value across all ranks = clipping synced)")

    per_rank_mem = memory_peaks.get("per_rank") or {}
    if per_rank_mem:
        add("")
        add("  peak memory per rank (sampled while the ranks ran)")
        add("  rank  peak RSS      peak commit")
        for rank in sorted(per_rank_mem, key=int):
            entry = per_rank_mem[rank]
            add(
                f"  {rank:<5} {hostmem.gib(entry['peak_rss']):>8.2f} GiB "
                f"{hostmem.gib(entry['peak_commit']):>8.2f} GiB"
            )
        total_rss = sum(e["peak_rss"] for e in per_rank_mem.values())
        total_commit = sum(e["peak_commit"] for e in per_rank_mem.values())
        add(
            f"  {'sum':<5} {hostmem.gib(total_rss):>8.2f} GiB "
            f"{hostmem.gib(total_commit):>8.2f} GiB"
            "   <- what the whole run needs, over 8 processes"
        )
        system = memory_peaks.get("system_after") or {}
        limit = system.get("commit_limit", -1)
        used = system.get("commit_used", -1)
        if limit > 0:
            add(
                f"  system commit after run  {hostmem.gib(used):.2f} / "
                f"{hostmem.gib(limit):.2f} GiB   "
                f"(headroom {hostmem.gib(limit - used):.2f} GiB)"
            )
        low = memory_peaks.get("system_low") or {}
        if low.get("min_commit_free") is not None:
            add(
                f"  system low-water mark    "
                f"commit free {hostmem.gib(low['min_commit_free']):.2f} GiB, "
                f"avail phys {hostmem.gib(low['min_avail_phys']):.2f} GiB"
                "   <- during the run; near zero means the job did not fit"
            )
        add("")

    add("")
    add("  artifacts")
    add(f"    logs                   {os.path.abspath(log_dir)}")
    arts = _artifact_summary(dump_dir)
    if arts:
        add(f"    dump folder            {dump_dir}")
        for line in arts[:20]:
            add(f"      {line}")
        if len(arts) > 20:
            add(f"      ... and {len(arts) - 20} more")
    else:
        add(f"    dump folder            {dump_dir}  <no files>")
    add(rule)

    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def _artifact_files(dump_dir: str) -> list[dict]:
    """Files the run left behind, as data (the summary keeps a text version)."""
    out: list[dict] = []
    if not os.path.isdir(dump_dir):
        return out
    for root, _dirs, files in os.walk(dump_dir):
        rel_root = os.path.relpath(root, dump_dir)
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.join(rel_root, name) if rel_root != "." else name
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            out.append({"rel": rel.replace("\\", "/"), "bytes": size})
    return out


def _cmdline(parts: list[str]) -> str:
    """Join a command for copy-paste, quoting only what needs it."""
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def _kill_tree(pid: int) -> None:
    """Kill a rank and everything it spawned.

    The venv's ``python.exe`` is a trampoline: the pid ``Popen`` returned is not
    the process holding the model, so killing only that pid leaves the real rank
    running -- and still attached to the TCPStore, which keeps the rendezvous
    (and the next run's port) alive.
    """
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            timeout=60,
        )
    except Exception:  # pragma: no cover - best effort
        pass


def _wait_for_ranks(procs: list) -> list[tuple[int, int]]:
    """Wait for every rank, but do not let a dead rank hold the others hostage.

    A rank that exits non-zero while its peers still run means the peers are
    blocked in a collective whose other half will never be posted -- the
    PP x FSDP failure, where rank 1 died inside a stage backward and ranks 0/2/3
    then sat in gloo until the 30-minute timeout. Waiting for the peers turns a
    three-second crash into a half-hour hang that reads like a training bug, so
    the survivors are killed and reported instead. A rank that exits *cleanly*
    while peers still run is normal (PP stages finish at slightly different
    times) and is not treated as a failure.
    """
    codes: list[tuple[int, int]] = []
    pending = {rank: (proc, log_file) for proc, log_file, rank in procs}
    while pending:
        for rank in sorted(pending):
            proc, log_file = pending[rank]
            rc = proc.poll()
            if rc is None:
                continue
            log_file.close()
            codes.append((rank, rc))
            del pending[rank]
        failed = [r for r, rc in codes if rc != 0]
        if failed and pending:
            print(
                f"[launcher] rank {failed} exited non-zero while "
                f"{sorted(pending)} still running. Killing the survivors: they "
                "are waiting on a collective with a peer that is already gone, "
                "so waiting for them only burns the gloo timeout.",
                file=sys.stderr,
                flush=True,
            )
            for rank in sorted(pending):
                _kill_tree(pending[rank][0].pid)
            for rank in sorted(pending):
                proc, log_file = pending[rank]
                rc = proc.wait()
                log_file.close()
                codes.append((rank, rc))
            pending.clear()
            break
        time.sleep(0.5)
    return codes


def _start_memory_sampler(procs) -> None:
    """Poll each rank's memory while it runs, keeping the per-rank peaks.

    "How many parameters fit" is a question about the *rank*, not about the
    launcher, so the figure has to come from outside. The per-step metrics line
    reports one rank's own view, but the peak also happens during checkpointing
    and during the first all-gather, which that line can miss.

    Physically resident bytes and *committed* bytes are both tracked: on Windows
    a process is refused memory against RAM + pagefile, so commit is what decides
    whether a size can start at all, while RSS is what decides whether it can run
    without thrashing.
    """
    import threading

    pids = [(rank, proc.pid) for proc, _log, rank in procs]
    state: dict[str, Any] = {"stop": False, "peaks": {}}

    def poll() -> None:
        while not state["stop"]:
            for rank, pid in pids:
                try:
                    # The whole subtree, not the pid Popen returned: the venv's
                    # python.exe is a trampoline whose *child* is the process
                    # holding the model.
                    info = hostmem.tree_memory(pid)
                except Exception:  # pragma: no cover - process may have exited
                    continue
                if info["rss"] < 0:
                    continue
                entry = state["peaks"].setdefault(
                    rank, {"peak_rss": 0, "peak_commit": 0, "samples": 0}
                )
                entry["peak_rss"] = max(entry["peak_rss"], info["rss"])
                entry["peak_commit"] = max(entry["peak_commit"], info["commit"])
                entry["samples"] += 1
            # Summed per-rank peaks answer "how big is the job"; they cannot
            # answer "did the job fit", because the host is shared with whatever
            # else is running. Tracking the *system* low-water mark during the
            # run is what separates the two: a job whose summed peak fits in the
            # nominal commit limit can still die if the baseline was already high
            # when it started.
            try:
                sysmem = hostmem.system_memory()
            except Exception:  # pragma: no cover
                sysmem = None
            if sysmem is not None:
                low = state.setdefault(
                    "system_low",
                    {
                        "min_avail_phys": sysmem["avail_phys"],
                        "min_commit_free": sysmem["commit_limit"]
                        - sysmem["commit_used"],
                        "max_commit_used": sysmem["commit_used"],
                    },
                )
                low["min_avail_phys"] = min(low["min_avail_phys"], sysmem["avail_phys"])
                low["min_commit_free"] = min(
                    low["min_commit_free"],
                    sysmem["commit_limit"] - sysmem["commit_used"],
                )
                low["max_commit_used"] = max(
                    low["max_commit_used"], sysmem["commit_used"]
                )
            state["stop"] = state["stop"] or False
            time.sleep(0.5)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    _MEMORY_SAMPLER["state"] = state


_MEMORY_SAMPLER: dict[str, Any] = {}


def _stop_memory_sampler() -> dict:
    state = _MEMORY_SAMPLER.get("state")
    if state is None:
        return {}
    state["stop"] = True
    # Give the thread one poll interval to notice; it is a daemon, so a slow
    # exit can never hold the run open.
    time.sleep(0.6)
    _MEMORY_SAMPLER.pop("state", None)
    system = hostmem.system_memory()
    low = state.get("system_low")
    return {
        "per_rank": {str(k): v for k, v in sorted(state["peaks"].items())},
        "system_low": (
            {
                "min_avail_phys": low["min_avail_phys"],
                "min_commit_free": low["min_commit_free"],
                "max_commit_used": low["max_commit_used"],
                "commit_limit": system["commit_limit"],
            }
            if low is not None
            else None
        ),
        "system_after": {
            "total_phys": system["total_phys"],
            "avail_phys": system["avail_phys"],
            "commit_limit": system["commit_limit"],
            "commit_used": system["commit_used"],
        },
        "source": hostmem.implementation(),
    }


def _launcher_command() -> str:
    return _cmdline(
        [sys.executable, os.path.join("cpu_lab", "run_cpu_train.py"), *sys.argv[1:]]
    )


def _write_run_json(
    path: str,
    args,
    dims: dict[str, int],
    world_size: int,
    elapsed: float,
    results: dict,
    dump_dir: str,
    log_dir: str,
    child_args: list[str],
    memory_peaks: dict | None = None,
) -> dict:
    """One machine-readable record of the run, which ``report.py`` renders.

    Written separately from ``summary.txt`` on purpose: the text summary is for
    a human reading the console, this is the contract the HTML report consumes.
    Re-rendering the report from the same directory therefore reproduces the same
    page, and nothing in it is recomputed from a different source.
    """
    real = [r for r in results["rows"] if r.get("has_real_loss")]
    payload = {
        "combo": args.combo,
        "module": args.module,
        "config": args.config,
        "python": sys.executable,
        "command": _launcher_command(),
        "child_command": _cmdline(child_args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_s": round(elapsed, 1),
        "world_size": world_size,
        "ranks_spawned": len(results["rows"]),
        "dims": dims,
        "overrides": {
            "seq_len": args.seq_len,
            "steps": args.steps,
            "local_batch_size": args.local_batch_size,
            "dataset_path": args.dataset_path,
            "sequence_parallel": args.sp,
        },
        "collectives": bool(args.collectives),
        "probe": bool(args.probe),
        "results": results,
        "last_step": max((r["last_step"] or 0) for r in real) if real else None,
        "dump_dir": os.path.abspath(dump_dir),
        "log_dir": os.path.abspath(log_dir),
        "artifacts": _artifact_files(dump_dir),
        "memory": memory_peaks or {},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


def _run_probe(args, dump_dir: str, log_dir: str, last_step: int | None) -> str | None:
    """Score the trained checkpoint against the planted secrets.

    Runs in a separate process because it must build the model and load the
    checkpoint *without* the distributed setup the training run had: DCP merges
    the per-rank shards, so one process can probe a run that used eight.
    """
    if not last_step:
        return None
    ckpt = os.path.join(dump_dir, "checkpoint", f"step-{last_step}")
    if not os.path.isdir(ckpt):
        print(f"[launcher] --probe: no checkpoint at {ckpt}", file=sys.stderr)
        return None
    out = os.path.join(log_dir, "probe.json")
    cmd = [
        sys.executable,
        os.path.join(LAB_DIR, "probe_inference.py"),
        "--module",
        args.module,
        "--config",
        args.config,
        "--checkpoint",
        ckpt,
        "--out",
        out,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([LAB_DIR, REPO_ROOT])
    print(f"\n[launcher] inference probe: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, env=env, cwd=REPO_ROOT)
    if rc != 0:
        print(f"[launcher] probe exited {rc}", file=sys.stderr)
        return None
    return out


def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        print("Parallel strategy presets:\n")
        print("  Works on this CPU/gloo setup:")
        for name, desc in _COMBOS.items():
            if name not in _CPU_LIMITATIONS and name not in _CPU_NOTES:
                print(f"    {name:<10} {desc}")
        print("\n  Works, with a caveat:")
        for name, note in _CPU_NOTES.items():
            print(f"    {name:<10} {_COMBOS.get(name, '')}")
            print(f"    {'':<10} {note}")
        print("\n  Configured correctly, but blocked by this environment:")
        for name, reason in _CPU_LIMITATIONS.items():
            print(f"    {name:<10} {reason}")
        print(
            "\n  Use: python cpu_lab/run_cpu_train.py --combo <name> -n <ranks>\n"
            "  Anything else (seq_len, steps, sequence parallelism, module,\n"
            "  config) is an override on top of the preset; see --help."
        )
        return 0

    n = args.nproc
    dims = _resolve_combo(args.combo, n)
    # Resolve the config only after the combo is known, so a preset can pick its
    # own default (see _COMBO_DEFAULT_CONFIG) while an explicit --config wins.
    if args.config is None:
        args.config = _COMBO_DEFAULT_CONFIG.get(args.combo, DEFAULT_CONFIG)

    world_size = (
        dims["dp_replicate"] * dims["dp_shard"] * dims["cp"] * dims["tp"] * dims["pp"]
    )
    if world_size != n:
        print(
            f"[launcher] preset {args.combo!r} needs {n} ranks to divide evenly as "
            f"{dims['dp_replicate']}x{dims['dp_shard']}x{dims['cp']}x{dims['tp']}x"
            f"{dims['pp']} = {world_size}. Pick a preset that divides {n}.",
            file=sys.stderr,
        )
        return 2

    child_args = _build_child_args(args, dims)

    print(f"[launcher] combo={args.combo}")
    print(
        f"[launcher] world_size={world_size} = "
        f"dp_replicate({dims['dp_replicate']}) x dp_shard({dims['dp_shard']}) x "
        f"cp({dims['cp']}) x tp({dims['tp']}) x pp({dims['pp']})"
    )
    print(f"[launcher] {' '.join(child_args)}\n", flush=True)

    if args.combo in _CPU_NOTES:
        print(
            f"[launcher] note for {args.combo!r}: {_CPU_NOTES[args.combo]}\n",
            flush=True,
        )

    if args.combo in _CPU_LIMITATIONS:
        print(
            f"[launcher] WARNING: preset {args.combo!r} is expected to fail on "
            f"this CPU/gloo setup:\n"
            f"            {_CPU_LIMITATIONS[args.combo]}\n"
            "            The config itself is correct and should work on CUDA. "
            "Continuing anyway.\n",
            file=sys.stderr,
            flush=True,
        )

    # --dry-run is single-process: fake_backend has no real collectives and is
    # only meaningful for rank 0.
    ranks_to_spawn = 1 if args.dry_run else world_size
    log_dir = (
        os.path.abspath(args.log_dir)
        if args.log_dir
        else os.path.join(HERE, "logs")
    )
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    t0 = time.time()

    # Preflight: state how much room the host has before spending minutes on a
    # run that may not fit. The per-combo footprint is measured, not guessed --
    # see the "Measured on this host" table in cpu_lab/README.md -- but printing
    # the free memory here makes the comparison possible without leaving the
    # console, and it is the number that decides whether a multi-rank job pages.
    try:
        sysmem = hostmem.system_memory()
        print(
            f"[launcher] host memory before launch: available physical "
            f"{hostmem.gib(sysmem['avail_phys']):.2f} GiB, commit free "
            f"{hostmem.gib(sysmem['commit_limit'] - sysmem['commit_used']):.2f} GiB"
            f" (measured footprints: see cpu_lab/README.md)",
            flush=True,
        )
    except Exception:  # pragma: no cover - diagnostics only
        pass

    shared_cache = (
        os.path.abspath(args.hf_cache)
        if args.hf_cache
        else os.path.join(HERE, ".hf_cache", "shared")
    )
    os.makedirs(shared_cache, exist_ok=True)
    if not args.keep_outputs:
        _clear_outputs(args.combo)
    _prewarm_dataset(args, child_args, shared_cache, log_dir, ranks_to_spawn)

    for rank in range(ranks_to_spawn):
        env = os.environ.copy()
        # HuggingFace datasets guards its processed cache with lock files, and
        # preparing that cache concurrently from N processes is what makes a run
        # hang inside dataset preparation (the training loop then dies with a
        # gloo collective timeout, which looks like a training bug). Two
        # mitigations: one shared cache, warmed once by _prewarm_dataset before
        # any rank starts, and staggered rank startups.
        env.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE=str(world_size),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(args.master_port),
            OMP_NUM_THREADS=str(args.threads),
            HF_HOME=shared_cache,
            HF_DATASETS_CACHE=os.path.join(shared_cache, "datasets"),
            # Everything here is local (vendored tokenizer + c4_test), so never
            # reach for the network.
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            # Belt-and-braces for the C++ TCPStore read at process start; the
            # real fix is no_libuv_patch (see cpu_compat).
            USE_LIBUV="0",
            # Makes run_logging write run_meta_rank_<r>.json next to the logs
            # and collectives.py dump its counters there. Absolute, because the
            # children run with the repository root as cwd.
            CPU_LAB_LOG_DIR=os.path.abspath(log_dir),
            # cpu_lab first so `cpu_compat` / `no_libuv_patch` resolve; repo root
            # next so `torchtitan` resolves.
            PYTHONPATH=os.pathsep.join(
                [LAB_DIR, REPO_ROOT, env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
            PYTHONUNBUFFERED="1",
        )
        if args.collectives:
            env["CPU_LAB_COLLECTIVES"] = "1"
        if args.dry_run:
            # fake_backend performs no real communication but still builds a
            # device mesh, and torchtitan reads the mesh's world size from NGPU
            # rather than from RANK/WORLD_SIZE. Without this, every --dry-run
            # with -n > 1 dies with "NGPU environment variable must be set
            # when using comm_mode=fake_backend" -- i.e. the cheapest way to
            # validate a multi-dimensional (PP/TP/FSDP) config was unusable.
            env["NGPU"] = str(world_size)
        log_name = "train.log" if ranks_to_spawn == 1 else f"rank_{rank}.log"
        log_file = open(
            os.path.join(log_dir, log_name), "w", encoding="utf-8", errors="replace"
        )
        procs.append(
            (
                subprocess.Popen(
                    child_args,
                    env=env,
                    cwd=REPO_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                ),
                log_file,
                rank,
            )
        )
        if rank + 1 < ranks_to_spawn and args.stagger > 0:
            # Space out startups so dataset preparation does not have N ranks
            # fighting over the same lock files at the same instant.
            time.sleep(args.stagger)

    _start_memory_sampler(procs)
    codes = _wait_for_ranks(procs)

    # Sampled while the ranks were alive; stopping here is what freezes the peaks.
    memory_peaks = _stop_memory_sampler()

    elapsed = time.time() - t0
    failed = [r for r, rc in codes if rc != 0]

    if failed:
        print(
            f"[launcher] FAILED ranks={failed} after {elapsed:.1f}s "
            "(see cpu_lab/logs/)",
            file=sys.stderr,
        )
        # Single-process runs write to train.log; multi-rank runs per-rank files.
        primary = "train.log" if ranks_to_spawn == 1 else f"rank_{failed[0]}.log"
        tail = _tail(os.path.join(log_dir, primary))
        print(f"\n--- tail of {primary} ---", file=sys.stderr)
        print(tail, file=sys.stderr)
        # A native crash is the memory wall wearing a training bug's clothes, and
        # the first non-zero exit code is usually a rank the watchdog killed, not
        # the one that died -- so scan all of them before drawing a conclusion.
        _report_native_crash(log_dir, ranks_to_spawn, memory_peaks)

        if "arg_mbs" in tail:
            print(
                "\n[hint] This is a torch/torchtitan API mismatch, not a model "
                "problem.\n"
                "  torchtitan drives the pipeline schedule with "
                "step(arg_mbs=..., kwarg_mbs=...),\n"
                "  but torch 2.12's schedule.step() signature is "
                "step(*args, target=, losses=, **kwargs),\n"
                "  so arg_mbs lands in **kwargs and is treated as model input. "
                "Pipeline\n"
                "  parallelism needs a torch version whose schedules expose the "
                "microbatch-level\n"
                "  step() API; every other preset (single/fsdp/hsdp/tp/tp_fsdp/"
                "cp/tp_cp) is unaffected.",
                file=sys.stderr,
            )
        # The per-rank memory peaks are the whole point of a memory-wall
        # investigation, and the wall is exactly the case where the run fails --
        # so record them on this path too instead of discarding them with the
        # run. A rank that died in a native allocation fault may produce no
        # parseable result row, hence the guard: a missing summary must never
        # mask the original failure.
        try:
            results = _collect_results(log_dir, ranks_to_spawn)
            summary_text = _write_summary(
                os.path.join(log_dir, "summary.txt"),
                args,
                dims,
                world_size,
                elapsed,
                results,
                _dump_dir(args.combo),
                log_dir,
                memory_peaks,
            )
            print()
            print(summary_text, end="")
            _write_run_json(
                os.path.join(log_dir, "run.json"),
                args,
                dims,
                world_size,
                elapsed,
                results,
                _dump_dir(args.combo),
                log_dir,
                child_args,
                memory_peaks,
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(
                f"[launcher] could not record memory peaks: {exc!r}",
                file=sys.stderr,
            )
        if "access violation" in tail:
            print(
                "\n[hint] Windows raised a native access violation. On a CPU-only\n"
                "  build this is how an exhausted host reports a failed allocation\n"
                "  -- there is no catchable torch OOM on CPU, so it surfaces as a\n"
                "  crash inside whatever allocated last (often an FSDP all-gather).\n"
                "  Compare this run's per-rank peak commit/RSS in cpu_lab/run.json\n"
                "  (or <log-dir>/run.json) against the host's physical memory\n"
                "  before reading it as a model bug.",
                file=sys.stderr,
            )
        return 1

    print(f"[launcher] all {ranks_to_spawn} process(es) finished in {elapsed:.1f}s")

    # Aggregate the per-rank logs into one readable result table. Parsing the
    # logs (rather than having the children report back) keeps the aggregation
    # honest: it reflects exactly what the run wrote, including which ranks
    # reported a sentinel instead of a loss.
    # ``ranks_to_spawn`` (not world_size) decides the log file names: a
    # --dry-run has world_size=N but only rank 0 actually ran.
    results = _collect_results(log_dir, ranks_to_spawn)
    summary_path = os.path.join(log_dir, "summary.txt")
    summary_text = _write_summary(
        summary_path, args, dims, world_size, elapsed, results,
        _dump_dir(args.combo), log_dir, memory_peaks,
    )
    print()
    print(summary_text, end="")

    run = _write_run_json(
        os.path.join(log_dir, "run.json"),
        args,
        dims,
        world_size,
        elapsed,
        results,
        _dump_dir(args.combo),
        log_dir,
        child_args,
        memory_peaks,
    )

    if args.probe:
        _run_probe(args, _dump_dir(args.combo), log_dir, run["last_step"])

    if args.report:
        try:
            import report

            out = report.render(log_dir, args.combo)
            print(f"[launcher] HTML report: {out}", flush=True)
        except Exception as exc:  # pragma: no cover - never fail a run for a report
            print(
                f"[launcher] report generation failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if args.show_logs:
        primary = "train.log" if ranks_to_spawn == 1 else "rank_0.log"
        path = os.path.join(log_dir, primary)
        print(f"\n--- {path} ---")
        for line in _tail(path, n=200).splitlines():
            if "step:" in line or "ERROR" in line:
                print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())