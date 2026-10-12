# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observability patches: make a run's parallelism, data, and outputs legible.

torchtitan logs the parallelism *axes* once, in one line:

    Building device mesh with parallelism: pp=2, dp_replicate=1, dp_shard=2, ...

Three things a reader still cannot answer from the logs:

1. **What those axes mean for this rank.** Which coordinate it occupies, which
   pipeline stages it holds, and whether it is a first/last stage. Without the
   last part, the sentinel that non-last PP stages log instead of a loss
   (``pp_forward_backward_step`` returns ``-1``) reads as a broken run.
2. **Which file is actually being trained on**, and how one step maps to tokens.
3. **What the run left on disk.**

None of this needs torchtitan to change: the three patches below wrap
``Trainer.__init__`` (banner), ``Trainer.train`` (closing summary) and
``MetricsProcessor.log`` (capture the last reported metrics).

Installed by ``cpu_lab_child.py``; independent of ``cpu_compat``, which is about
making the code *run* rather than making it legible.
"""

from __future__ import annotations

import os
import time
from typing import Any

_INSTALLED = False

# MetricsProcessor.log is the single place torchtitan reports a step's numbers,
# so it is also the cheapest place to remember the final ones.
_last_metrics: dict[str, Any] = {}
_started_at: dict[int, float] = {}

_RULE = "=" * 78
_AXIS_ORDER = [
    "pp",
    "dp_replicate",
    "fsdp",
    "cp",
    "tp",
    "ep",
    "batch",
    "loss",
]


def _fmt_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} GB"


def _dataset_files(path: str) -> str:
    """Human-readable listing of the files under a dataset directory."""
    if not os.path.exists(path):
        return "  <not found>"
    if os.path.isfile(path):
        return f"{_fmt_size(os.path.getsize(path))}"
    entries = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            entries.append(f"{name} ({_fmt_size(os.path.getsize(full))})")
    return ", ".join(entries) if entries else "<empty>"


def _parallelism_block(trainer: Any) -> list[str]:
    """The parallelism degrees as configured, then as resolved into meshes."""
    import torch.distributed as dist

    parallelism = trainer.config.parallelism
    world = dist.get_world_size() if dist.is_initialized() else int(
        os.environ.get("WORLD_SIZE", "1")
    )
    lines = ["--- parallelism (as configured) ---"]
    lines.append(f"  world_size                       {world}")
    rows = [
        ("data_parallel_replicate_degree", parallelism.data_parallel_replicate_degree),
        ("data_parallel_shard_degree", parallelism.data_parallel_shard_degree),
        ("context_parallel_degree", parallelism.context_parallel_degree),
        ("tensor_parallel_degree", parallelism.tensor_parallel_degree),
        ("pipeline_parallel_degree", parallelism.pipeline_parallel_degree),
        ("expert_parallel_degree", parallelism.expert_parallel_degree),
    ]
    notes = {
        "data_parallel_shard_degree": "  <- FSDP shard axis",
        "data_parallel_replicate_degree": "  <- HSDP replicate axis",
        "context_parallel_degree": "  <- splits the sequence",
        "tensor_parallel_degree": "  <- splits heads + MLP",
        "pipeline_parallel_degree": "  <- splits layers",
    }
    for name, value in rows:
        note = notes.get(name, "")
        if name == "data_parallel_shard_degree" and value in (-1,):
            note += "   (-1 = all remaining ranks)"
        lines.append(f"  {name:<32} {value}{note}")
    lines.append(
        f"  {'enable_sequence_parallel':<32} {parallelism.enable_sequence_parallel}"
    )
    lines.append(
        f"  {'pipeline_parallel_schedule':<32} {parallelism.pipeline_parallel_schedule}"
    )
    lines.append(
        f"  {'pipeline_parallel_microbatch_size':<32} "
        f"{parallelism.pipeline_parallel_microbatch_size}"
    )

    parallel_dims = trainer.parallel_dims
    try:
        meshes = parallel_dims.get_all_one_dimensional_meshes()
    except Exception:  # pragma: no cover - defensive
        meshes = {}
    if meshes:
        lines.append("")
        lines.append("--- meshes (resolved) ---")
        names = [n for n in _AXIS_ORDER if n in meshes]
        names += [n for n in meshes if n not in _AXIS_ORDER]
        for name in names:
            mesh = meshes[name]
            try:
                size = mesh.size()
            except Exception:  # pragma: no cover - defensive
                continue
            try:
                coord = mesh.get_coordinate()
            except Exception:  # pragma: no cover - defensive
                coord = "?"
            note = "  (disabled)" if size == 1 else ""
            lines.append(f"  {name:<14} size={size:<4} this rank at {coord}{note}")
    return lines


def _this_rank_block(trainer: Any) -> list[str]:
    """What this particular rank is responsible for."""
    import torch.distributed as dist

    lines = ["--- this rank ---"]
    lines.append(f"  global rank                      {dist.get_rank()}")

    schedule = getattr(trainer, "pp_schedule", None)
    stages = getattr(schedule, "_stages", None)
    if stages:
        idxs = sorted(getattr(s, "stage_index", -1) for s in stages)
        lines.append(f"  pipeline stages held             {idxs}")
        lines.append(
            f"  pipeline schedule                {type(schedule).__name__} "
            f"({getattr(schedule, '_n_microbatches', '?')} microbatches)"
        )
        holds_first = bool(getattr(trainer, "pp_has_first_stage", False))
        holds_last = bool(getattr(trainer, "pp_has_last_stage", False))
        lines.append(f"  holds first pipeline stage       {holds_first}")
        lines.append(f"  holds last pipeline stage        {holds_last}")
        if holds_last:
            lines.append(
                "    -> this rank computes the real loss; it appears in its per-step lines"
            )
        else:
            lines.append(
                "    -> this rank logs a NEGATIVE SENTINEL instead of a loss;"
            )
            lines.append(
                "       the sentinel is not a loss. Read the last-stage ranks' logs."
            )
    else:
        lines.append("  pipeline stages held             (pipeline parallel disabled)")
    return lines


def _data_block(trainer: Any) -> list[str]:
    """Where the tokens come from, and the arithmetic of one training step.

    Two dataloader shapes are handled: the dataset-registry one
    (``HuggingFaceTextDataLoader``, keyed by ``dataset``) and the chat/SFT one
    (``ChatDataLoader``, keyed by ``dataset_path`` + ``load_dataset_kwargs``).
    """
    config = trainer.config
    training = config.training
    lines = ["--- data ---"]

    dl_cfg = config.dataloader
    dataset_name = getattr(dl_cfg, "dataset", None)
    dataset_path = getattr(dl_cfg, "dataset_path", None)

    if dataset_name:
        lines.append(f"  dataset name                     {dataset_name}")
        try:
            from torchtitan.hf_datasets.text_datasets import DATASETS

            entry = DATASETS.get(dataset_name)
            # ``_validate_dataset`` resolves ``dataset_path or config.path``, so a
            # run that overrides the path (--dataloader.dataset-path) is reading
            # its own directory while still *named* after a registry entry. Print
            # the path actually used, and say when it is not the registry's.
            override = getattr(dl_cfg, "dataset_path", None)
            if entry is not None:
                resolved = os.path.abspath(override or entry.path)
                lines.append(f"  dataset path                     {resolved}")
                lines.append(f"  dataset contents                 {_dataset_files(resolved)}")
                if override and os.path.abspath(override) != os.path.abspath(entry.path):
                    lines.append(
                        f"  dataset source                   OVERRIDDEN from the "
                        f"'{dataset_name}' registry entry ({entry.path})"
                    )
            else:
                lines.append(
                    "  dataset path                     (not in DATASETS registry)"
                )
        except Exception as exc:  # pragma: no cover - defensive
            lines.append(f"  dataset path                     <unresolved: {exc}>")
    elif dataset_path:
        # ChatDataLoader: the real file lives in load_dataset_kwargs["data_files"].
        lines.append(
            f"  dataloader                       {type(dl_cfg).__qualname__} (SFT/chat)"
        )
        lines.append(f"  dataset path                     {dataset_path}")
        kwargs = getattr(dl_cfg, "load_dataset_kwargs", None) or {}
        data_files = kwargs.get("data_files")
        if data_files:
            resolved = os.path.abspath(data_files)
            lines.append(f"  dataset file                     {resolved}")
            lines.append(f"  dataset contents                 {_dataset_files(resolved)}")
        if kwargs.get("split"):
            lines.append(f"  split                            {kwargs['split']}")
        lines.append(
            "  label masking                    prompt tokens -> IGNORE_INDEX (-100)"
        )
    else:
        lines.append(f"  dataloader                       {type(dl_cfg).__name__}")

    assets = os.path.abspath(config.hf_assets_path)
    tok = os.path.join(assets, "tokenizer.json")
    lines.append(f"  tokenizer assets                 {assets}")
    lines.append(
        f"  tokenizer file                   {tok}"
        + ("" if os.path.exists(tok) else "   <missing>")
    )

    lines.append(f"  seq_len                          {training.seq_len}")
    lines.append(f"  local_batch_size                 {training.local_batch_size}")
    lines.append(
        f"  gradient_accumulation_steps      {trainer.gradient_accumulation_steps}"
    )
    lines.append(
        f"  pipeline microbatches per rank   {trainer.num_pipeline_parallel_microbatches}"
    )
    try:
        batch_degree = trainer.parallel_dims.get_mesh("batch").size()
    except Exception:  # pragma: no cover - defensive
        batch_degree = 1
    global_batch = (
        training.local_batch_size * batch_degree * trainer.gradient_accumulation_steps
    )
    lines.append(f"  global_batch_size                {global_batch}  (x batch degree {batch_degree})")
    lines.append(
        f"  tokens per step                  {global_batch} x {training.seq_len} = "
        f"{global_batch * training.seq_len}"
    )
    lines.append(f"  total steps requested            {training.steps}")
    return lines


def _output_block(trainer: Any) -> list[str]:
    """Where this run writes anything."""
    config = trainer.config
    dump = os.path.abspath(config.dump_folder)
    ckpt_cfg = config.checkpoint
    ckpt_dir = os.path.join(dump, ckpt_cfg.folder)
    lines = ["--- outputs ---"]
    lines.append(f"  dump folder                      {dump}")
    lines.append(
        f"  checkpoint folder                {ckpt_dir}"
        f"   ({'enabled' if ckpt_cfg.enable else 'DISABLED'}"
        f", folder='{ckpt_cfg.folder}', interval={ckpt_cfg.interval};"
        " the final step always saves)"
    )
    rank = os.environ.get("RANK", "0")
    world = os.environ.get("WORLD_SIZE", "1")
    log_name = "train.log" if world == "1" else f"rank_{rank}.log"
    lines.append(
        f"  this rank's log                  "
        f"{os.path.abspath(os.path.join('cpu_lab', 'logs', log_name))}"
    )
    return lines


def _emit(text: str) -> None:
    from torchtitan.tools.logging import logger

    logger.info("\n" + text)


def _local_param_count(trainer: Any) -> int:
    """Parameters materialised on this rank.

    Deliberately *not* summed across ranks to get a "global" figure: under TP
    and FSDP a DTensor reports its **global** shape from ``numel()``, so a naive
    cross-rank sum multiplies the count by the replication factor. torchtitan
    already logs the authoritative total ("size: N total parameters") from the
    unsharded model, before this banner; this is only here to show how much of
    it landed on this rank.
    """
    return sum(p.numel() for m in trainer.model_parts for p in m.parameters())


def _banner(trainer: Any) -> str:
    config = trainer.config
    spec = config.model_spec
    parts = [
        _RULE,
        "RUN CONFIG",
        _RULE,
        f"  model                            {spec.name} / {spec.flavor}",
        f"  params on this rank              {_local_param_count(trainer):,}"
        "   (see the 'size: ... total parameters' line above for the model total)",
        f"  module                           {type(spec.model).__module__}",
        "",
    ]
    parts += _parallelism_block(trainer)
    parts.append("")
    parts += _this_rank_block(trainer)
    parts.append("")
    parts += _data_block(trainer)
    parts.append("")
    parts += _output_block(trainer)
    parts.append(_RULE)
    return "\n".join(parts)


def _closing(trainer: Any, elapsed: float) -> str:
    config = trainer.config
    parts = [_RULE, "RUN SUMMARY", _RULE]
    parts.append(f"  steps completed                  {trainer.step} / {config.training.steps}")
    parts.append(f"  tokens seen                      {trainer.ntokens_seen:,}")

    if _last_metrics:
        parts.append(f"  last logged loss                 {_last_metrics.get('loss')}")
        parts.append(f"  last logged grad_norm            {_last_metrics.get('grad_norm')}")
    holds_last = getattr(trainer, "pp_has_last_stage", None)
    if holds_last is False:
        parts.append(
            "  (this rank is not the last pipeline stage, so it never saw a real loss)"
        )
    parts.append(f"  wall time                        {elapsed:.1f} s")

    parts.append("")
    parts.append("--- artifacts on disk ---")
    ckpt_cfg = config.checkpoint
    ckpt_dir = os.path.join(os.path.abspath(config.dump_folder), ckpt_cfg.folder)
    if ckpt_cfg.enable and os.path.isdir(ckpt_dir):
        entries = sorted(os.listdir(ckpt_dir))
        parts.append(f"  checkpoints                      {ckpt_dir}")
        for name in entries[:12]:
            parts.append(f"      {name}")
        if len(entries) > 12:
            parts.append(f"      ... and {len(entries) - 12} more")
    elif ckpt_cfg.enable:
        parts.append(f"  checkpoints                      {ckpt_dir}  <not created>")
    else:
        parts.append(
            "  checkpoints                      none"
            "   (checkpoint.enable = false; set it to true to save weights)"
        )

    log_dir = os.path.abspath(os.path.join("cpu_lab", "logs"))
    n_logs = len([f for f in os.listdir(log_dir) if f.endswith(".log")]) if os.path.isdir(log_dir) else 0
    parts.append(f"  per-rank logs                    {log_dir}  ({n_logs} .log files)")

    # Metrics are the only per-step record unless a checkpoint/metrics sink is on.
    parts.append(
        "  metrics                          one per-step line per rank in the logs above"
    )
    parts.append(_RULE)
    return "\n".join(parts)


def _patch_metrics_capture() -> None:
    try:
        from torchtitan.components.metrics import MetricsProcessor
    except ImportError:
        return
    if getattr(MetricsProcessor.log, "_cpu_logged", False):
        return

    real_log = MetricsProcessor.log

    def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        _last_metrics["step"] = step
        _last_metrics["loss"] = global_avg_loss
        _last_metrics["max_loss"] = global_max_loss
        _last_metrics["grad_norm"] = grad_norm
        return real_log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics)

    log._cpu_logged = True  # type: ignore[attr-defined]
    MetricsProcessor.log = log


def _meta(trainer: Any) -> dict:
    """The same facts as the banner, as data, so a report can consume them.

    Written per rank rather than only by rank 0: under TP/PP/FSDP each rank owns
    a different slice, and the per-rank coordinates are part of what the run *is*.
    """
    import torch.distributed as dist

    config = trainer.config
    spec = config.model_spec
    training = config.training
    parallelism = config.parallelism

    initialized = dist.is_initialized()
    rank = dist.get_rank() if initialized else int(os.environ.get("RANK", "0"))
    world = (
        dist.get_world_size() if initialized else int(os.environ.get("WORLD_SIZE", "1"))
    )

    meshes: dict[str, dict] = {}
    try:
        for name, mesh in trainer.parallel_dims.get_all_one_dimensional_meshes().items():
            meshes[name] = {
                "size": int(mesh.size()),
                "coord": str(mesh.get_coordinate()),
            }
    except Exception:  # pragma: no cover - defensive
        pass

    schedule = getattr(trainer, "pp_schedule", None)
    stages = getattr(schedule, "_stages", None)
    pp_info = {
        "enabled": bool(stages),
        "stage_indices": sorted(getattr(s, "stage_index", -1) for s in stages)
        if stages
        else [],
        "schedule": type(schedule).__name__ if schedule else None,
        "microbatches": getattr(schedule, "_n_microbatches", None) if schedule else None,
        "holds_first": bool(getattr(trainer, "pp_has_first_stage", False)),
        "holds_last": bool(getattr(trainer, "pp_has_last_stage", False)),
    }

    dl_cfg = config.dataloader
    dataset_name = getattr(dl_cfg, "dataset", None)
    dataset_path = getattr(dl_cfg, "dataset_path", None)
    kwargs = getattr(dl_cfg, "load_dataset_kwargs", None) or {}
    resolved_dataset = None
    dataset_override = getattr(dl_cfg, "dataset_path", None)
    dataset_registry_path = None
    if dataset_name:
        try:
            from torchtitan.hf_datasets.text_datasets import DATASETS

            entry = DATASETS.get(dataset_name)
            if entry is not None:
                dataset_registry_path = os.path.abspath(entry.path)
            # Mirrors ``_validate_dataset``: dataset_path wins over the registry.
            base = dataset_override or (entry.path if entry else None)
            resolved_dataset = os.path.abspath(base) if base else None
        except Exception:  # pragma: no cover - defensive
            resolved_dataset = dataset_override
    elif dataset_override:
        data_files = kwargs.get("data_files")
        resolved_dataset = os.path.abspath(data_files) if data_files else dataset_override

    try:
        batch_degree = trainer.parallel_dims.get_mesh("batch").size()
    except Exception:  # pragma: no cover - defensive
        batch_degree = 1
    global_batch = (
        training.local_batch_size * batch_degree * trainer.gradient_accumulation_steps
    )

    dump = os.path.abspath(config.dump_folder)
    ckpt_cfg = config.checkpoint

    return {
        "rank": rank,
        "world_size": world,
        "model": {
            "name": spec.name,
            "flavor": spec.flavor,
            "module": type(spec.model).__module__,
            "local_params": _local_param_count(trainer),
        },
        "parallelism": {
            "data_parallel_replicate_degree": parallelism.data_parallel_replicate_degree,
            "data_parallel_shard_degree": parallelism.data_parallel_shard_degree,
            "context_parallel_degree": parallelism.context_parallel_degree,
            "tensor_parallel_degree": parallelism.tensor_parallel_degree,
            "pipeline_parallel_degree": parallelism.pipeline_parallel_degree,
            "expert_parallel_degree": parallelism.expert_parallel_degree,
            "enable_sequence_parallel": bool(parallelism.enable_sequence_parallel),
            "pipeline_parallel_schedule": parallelism.pipeline_parallel_schedule,
            "pipeline_parallel_microbatch_size": parallelism.pipeline_parallel_microbatch_size,
        },
        "meshes": meshes,
        "pipeline": pp_info,
        "data": {
            "dataloader": type(dl_cfg).__qualname__,
            "dataset_name": dataset_name,
            "dataset_path": resolved_dataset,
            "dataset_registry_path": dataset_registry_path,
            "dataset_overridden": bool(
                dataset_override
                and dataset_registry_path
                and os.path.abspath(dataset_override) != dataset_registry_path
            ),
            "dataset_contents": _dataset_files(resolved_dataset)
            if resolved_dataset
            else None,
            "split": kwargs.get("split"),
            "tokenizer_dir": os.path.abspath(config.hf_assets_path),
            "tokenizer_file": os.path.join(
                os.path.abspath(config.hf_assets_path), "tokenizer.json"
            ),
        },
        "training": {
            "seq_len": training.seq_len,
            "local_batch_size": training.local_batch_size,
            "gradient_accumulation_steps": trainer.gradient_accumulation_steps,
            "pipeline_microbatches_per_rank": trainer.num_pipeline_parallel_microbatches,
            "batch_degree": int(batch_degree),
            "global_batch_size": int(global_batch),
            "tokens_per_step": int(global_batch * training.seq_len),
            "steps": training.steps,
            "lr": getattr(getattr(config.optimizer, "lr", None), "__float__", lambda: None)()
            if hasattr(config.optimizer, "lr")
            else None,
        },
        "outputs": {
            "dump_folder": dump,
            "checkpoint_folder": os.path.join(dump, ckpt_cfg.folder),
            "checkpoint_enabled": bool(ckpt_cfg.enable),
            "checkpoint_interval": ckpt_cfg.interval,
            "checkpoint_model_only": bool(ckpt_cfg.last_save_model_only),
        },
    }


def _meta_path(rank: int) -> str:
    log_dir = os.environ.get("CPU_LAB_LOG_DIR") or os.path.abspath(
        os.path.join("cpu_lab", "logs")
    )
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"run_meta_rank_{rank}.json")


def _patch_trainer() -> None:
    try:
        from torchtitan.trainer import Trainer
    except ImportError:
        return
    if getattr(Trainer.__init__, "_cpu_logged", False):
        return

    real_init = Trainer.__init__

    def __init__(self, config):
        real_init(self, config)
        _started_at[id(self)] = time.perf_counter()
        try:
            _emit(_banner(self))
        except Exception as exc:  # pragma: no cover - never break a run for a log
            _emit(f"[run-logging] banner unavailable: {type(exc).__name__}: {exc}")
        try:
            import json

            meta = _meta(self)
            with open(_meta_path(meta["rank"]), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - never break a run for a log
            _emit(f"[run-logging] run metadata unavailable: {type(exc).__name__}: {exc}")

    __init__._cpu_logged = True  # type: ignore[attr-defined]
    Trainer.__init__ = __init__

    real_train = Trainer.train

    def train(self):
        try:
            return real_train(self)
        finally:
            elapsed = time.perf_counter() - _started_at.pop(id(self), time.perf_counter())
            try:
                _emit(_closing(self, elapsed))
            except Exception as exc:  # pragma: no cover
                _emit(f"[run-logging] summary unavailable: {type(exc).__name__}: {exc}")

    train._cpu_logged = True  # type: ignore[attr-defined]
    Trainer.train = train


def install() -> None:
    """Install the observability patches. Safe to call more than once."""
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_metrics_capture()
    _patch_trainer()
    _INSTALLED = True