# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Count the collective communication a run actually performs, per step.

Every rule in this file was measured, not assumed. The findings that shaped it:

- **FSDP2** gathers parameters with ``dist.all_gather_into_tensor`` and
  reduce-scatters gradients with ``dist.reduce_scatter_tensor``. The latter
  dispatches into the aten op ``c10d._reduce_scatter_base_``, so the *same*
  collective is visible twice unless the nested aten op is suppressed.
- FSDP also calls ``fsdp.all_gather_copy_in`` the same number of times. That op
  is a **local** copy-in that stages shards into the gather buffer; it moves no
  data between ranks and is therefore not counted as communication.
- **TP** (DTensor) uses ``_c10d_functional.all_reduce`` for rowwise partial
  products, and ``_c10d_functional.all_gather_into_tensor`` /
  ``reduce_scatter_tensor`` once sequence parallelism is on.
- **PP** does *not* call ``dist.batch_isend_irecv``: torch's pipeline stages
  take a fast path through ``dist.isend`` / ``dist.irecv`` for homogeneous
  batches. Those cannot be wrapped -- ``P2POp.__new__`` validates its argument
  with ``op not in [isend, irecv]`` against the module-level originals, so a
  wrapper makes pipeline parallelism fail with ``Invalid op``. They are counted
  by hooking ``P2POp.__new__`` instead, which changes no identities.
- PP's gradient-norm sync calls plain ``torch.distributed.all_reduce``
  (``torchtitan/distributed/utils.py:646``) rather than the functional API, so a
  dispatcher-only probe would report a PP run as having no all-reduce at all.

Two layers are recorded, each keeping the raw op name: ``aten`` (dispatcher) and
``python`` (``torch.distributed.*`` wrappers plus the P2POp hook). Where a Python
call is known to dispatch into an aten op of the same kind -- reduce-scatter and
broadcast -- the nested aten op is suppressed so one collective is counted once.
Rows keep their layer and source op name, so any remaining overlap is visible
instead of averaged away.

What is measured
----------------
Per op: call count and the **payload bytes on one rank** -- the tensor the rank
contributes or reads (the shard going into an all-gather, the full gradient going
into a reduce-scatter), not the size of the output. That is what answers "how
much did this strategy put on the wire", and it is identical on every rank by
construction. Bytes are counted only for op names with an explicit rule;
anything unmatched reports 0 and is listed under ``unknown_payload`` so a gap is
visible rather than silently reported as zero traffic.

Installed only when ``CPU_LAB_COLLECTIVES=1``: one Python call per aten op is
fine for a 1M-parameter CPU model and pointless overhead otherwise.
"""

from __future__ import annotations

import collections
import json
import os
from typing import Any

# Bucket detection, longest pattern first so "reduce_scatter" is not shadowed and
# "all_gather_into_tensor" beats a bare "all_gather".
#
# ``fsdp.all_gather_copy_in`` is deliberately absent: it is a local staging copy
# with no inter-rank traffic. Counting it would inflate the FSDP all-gather
# traffic by 50% and describe a copy as communication.
_BUCKET_PATTERNS: list[tuple[str, str]] = [
    ("all_gather_into_tensor", "all_gather"),
    ("all_gather", "all_gather"),
    ("reduce_scatter", "reduce_scatter"),
    ("all_reduce", "all_reduce"),
    ("all_to_all", "all_to_all"),
    ("broadcast", "broadcast"),
    ("isend_irecv", "p2p"),
]

_FRIENDLY = {
    "all_gather": "all-gather",
    "reduce_scatter": "reduce-scatter",
    "all_reduce": "all-reduce",
    "broadcast": "broadcast",
    "all_to_all": "all-to-all",
    "p2p": "P2P send/recv",
    "p2p_send": "P2P send",
    "p2p_recv": "P2P recv",
}

# Op names that contain a collective-ish keyword but move nothing between ranks.
# ``fsdp.all_gather_copy_in`` stages the local shards into the gather buffer
# (a device-local copy); counting it would roughly double FSDP's reported
# all-gather traffic while describing a copy as communication.
_EXCLUDED = ("all_gather_copy_in",)

# Which argument carries the payload, per fully qualified op name (the trailing
# overload is stripped first). Per-op rather than per-bucket matters:
# reduce-scatter exists in two spellings with the tensor in different positions.
_PAYLOAD_ARG: dict[str, Any] = {
    "_c10d_functional.all_gather_into_tensor": 0,
    "_c10d_functional.reduce_scatter_tensor": 0,
    "_c10d_functional.all_reduce": 0,
    "_c10d_functional.all_to_all_single": 0,
    "_c10d_functional.broadcast": 0,
    "c10d._reduce_scatter_base_": 1,  # (output, input)
    "c10d.broadcast_": 0,  # (Tensor[] tensors, group, ...)
    "c10d.all_gather_into_tensor": 1,  # (output, input, ...)
    "c10d.reduce_scatter_tensor": 1,
    "dist.all_reduce": 0,
    "dist.all_gather_into_tensor": 1,
    "dist.reduce_scatter_tensor": 1,
    "dist.broadcast": 0,
    "dist.all_to_all_single": 1,
    "dist.send": 0,
    "dist.recv": 0,
}

# ops whose payload is a list of tensors rather than one tensor.
_PAYLOAD_LIST_ARG: dict[str, int] = {
    "dist.all_gather": 1,  # (tensor_list, tensor) -- the payload is the input
    "dist.reduce_scatter": 1,  # (output, input_list)
    "dist.all_to_all": 1,
}

# Fallback parameter names, for callers that pass by keyword instead of position.
# Measured need: FSDP's reduce-scatter calls ``dist.reduce_scatter_tensor`` with
# ``input=`` as a keyword, so a purely positional rule reads 0 bytes.
_PAYLOAD_KW: dict[str, str] = {
    "dist.reduce_scatter_tensor": "input",
    "dist.all_gather_into_tensor": "input",
    "dist.all_reduce": "tensor",
    "dist.broadcast": "tensor",
    "dist.all_to_all_single": "input",
}

# What each strategy should emit, for the report to check the observed table
# against. "missing" is as informative as "present".
EXPECTED_BY_STRATEGY: dict[str, list[tuple[str, str]]] = {
    "fsdp": [
        ("all-gather", "materialise each layer's full parameters from the shards"),
        ("reduce-scatter", "average gradients, then re-shard them"),
        ("all-reduce", "sync the gradient norm across data-parallel ranks"),
    ],
    "hsdp": [
        ("all-gather", "materialise each layer's full parameters from the shards"),
        ("reduce-scatter", "average gradients, then re-shard them"),
        ("all-reduce", "reduce across the *replicate* axis (the H in HSDP)"),
    ],
    "tp": [
        ("all-reduce", "sum the partial products of rowwise projections"),
        ("all-gather", "sequence parallelism: gather along the sequence axis"),
        ("reduce-scatter", "sequence parallelism: scatter partial sums back"),
    ],
    "pp": [
        ("P2P send", "hand activations forward to the next stage"),
        ("P2P recv", "take activations from the previous stage"),
        ("all-reduce", "sync the gradient norm across pipeline stages"),
    ],
}

_STATE: dict[str, Any] = {
    "installed": False,
    "rows": collections.Counter(),  # (raw, layer, bucket) -> calls
    "bytes": collections.Counter(),  # (raw, layer, bucket) -> bytes
    "schemas": {},
    "stacks": {},
    "unknown_payload": collections.Counter(),
    "zero_payload": collections.Counter(),
    "payload_debug": {},
    "per_step": [],
    "steps_seen": 0,
    "window": collections.Counter(),
    "window_bytes": collections.Counter(),
    "in_step": collections.Counter(),
    "in_step_bytes": collections.Counter(),
    "in_step_active": False,
    "phase": [],
    "py_stack": [],
    "mode": None,
}


def _strip_overload(op_name: str) -> str:
    """``c10d._reduce_scatter_base_.default`` -> ``c10d._reduce_scatter_base_``."""
    parts = op_name.split(".")
    if len(parts) > 2 and parts[-1] in {"default", "out", "Tensor", "Scalar"}:
        return ".".join(parts[:-1])
    return op_name


def _bucket_of(op_name: str) -> str | None:
    low = op_name.lower()
    if any(skip in low for skip in _EXCLUDED):
        return None
    for pattern, bucket in _BUCKET_PATTERNS:
        if pattern in low:
            return bucket
    return None


def _tensor_bytes(value: Any) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _payload_bytes(op_name: str, args: tuple, kwargs: dict) -> int | None:
    """Bytes of the tensor this rank sends/reads, or None if there is no rule."""
    if op_name.endswith("batch_isend_irecv"):
        return sum(_tensor_bytes(getattr(op, "tensor", None)) for op in (args[0] if args else []))
    if op_name in _PAYLOAD_LIST_ARG:
        index = _PAYLOAD_LIST_ARG[op_name]
        if index < len(args):
            return _tensor_bytes(args[index])
        return 0
    if op_name in _PAYLOAD_ARG:
        index = _PAYLOAD_ARG[op_name]
        if index < len(args):
            return _tensor_bytes(args[index])
        name = _PAYLOAD_KW.get(op_name)
        if name and name in kwargs:
            return _tensor_bytes(kwargs[name])
        return 0
    return None


def _brief(args: tuple, kwargs: dict) -> str:
    """A short, safe rendering of a call, for diagnosing a zero-byte payload."""
    def show(value: Any) -> str:
        import torch

        if isinstance(value, torch.Tensor):
            return f"Tensor{tuple(value.shape)}"
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(show(v) for v in value[:3]) + "]"
        text = repr(value)
        return text if len(text) <= 40 else text[:37] + "..."

    parts = [show(a) for a in args]
    parts += [f"{k}={show(v)}" for k, v in kwargs.items()]
    return "(" + ", ".join(parts) + ")"


def _measure(raw: str, args: tuple, kwargs: dict) -> int:
    """Payload bytes, recording anything that could silently under-report."""
    nbytes = _payload_bytes(raw, args, kwargs)
    if nbytes is None:
        _STATE["unknown_payload"][raw] += 1
        _STATE["payload_debug"].setdefault(raw, _brief(args, kwargs))
        return 0
    if nbytes == 0:
        # A rule exists but matched nothing -- most often a caller passing by
        # keyword. Recording the shape makes the miss fixable instead of silent.
        _STATE["zero_payload"][raw] += 1
        _STATE["payload_debug"].setdefault(raw, _brief(args, kwargs))
    return nbytes


def _capture_stack(raw: str) -> None:
    """Remember where an op was first issued, once per op name.

    Off unless ``CPU_LAB_COLLECTIVES_STACK=1``. It exists because "which line
    issued this collective" is the question that cannot be answered from counts:
    a per-step broadcast appears in the totals with no caller, and guessing at
    the caller is how a measurement stops being evidence.
    """
    if os.environ.get("CPU_LAB_COLLECTIVES_STACK") != "1":
        return
    if raw in _STATE["stacks"]:
        return
    import traceback

    frames = traceback.extract_stack()[:-1]
    _STATE["stacks"][raw] = [
        f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}" for f in frames[-14:]
    ]


def _record(raw: str, layer: str, bucket: str, nbytes: int, schema: str = "") -> None:
    key = (raw, layer, bucket, _phase())
    _STATE["rows"][key] += 1
    _STATE["bytes"][key] += nbytes
    if _STATE["in_step_active"]:
        _STATE["window"][f"{bucket}@{layer}"] += 1
        _STATE["window_bytes"][f"{bucket}@{layer}"] += nbytes
    if schema:
        _STATE["schemas"].setdefault(raw, schema)


def _phase() -> str:
    """Which part of the run issued this op: ``checkpoint``, ``step``, ``init``.

    The distinction is not cosmetic. A checkpoint save performs collectives of
    its own -- DCP's ``scatter_object_list`` is implemented as a broadcast from
    rank 0 -- and on a 1-rank run that is the *only* collective there is. Calling
    it "per-step communication" would report checkpoint I/O as training traffic.
    """
    if "checkpoint" in _STATE["phase"]:
        return "checkpoint"
    return "step" if _STATE["in_step_active"] else "init"


def _install_phase_hooks() -> None:
    """Mark the checkpoint-save window so its collectives can be told apart."""
    try:
        from torchtitan.components.checkpoint import CheckpointManager
    except ImportError:  # pragma: no cover - defensive
        return
    if getattr(CheckpointManager.dcp_save, "_cpu_lab_wrapped", False):
        return

    real = CheckpointManager.dcp_save

    def dcp_save(self, *args, **kwargs):
        _STATE["phase"].append("checkpoint")
        try:
            return real(self, *args, **kwargs)
        finally:
            _STATE["phase"].pop()

    dcp_save._cpu_lab_wrapped = True  # type: ignore[attr-defined]
    CheckpointManager.dcp_save = dcp_save


def _install_dispatch_mode() -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    keys = ("all_gather", "reduce_scatter", "all_reduce", "all_to_all", "broadcast")

    class CollectiveCounter(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            low = name.lower()
            if any(k in low for k in keys):
                raw = _strip_overload(name)
                bucket = _bucket_of(raw)
                if bucket is not None:
                    schema = "<unavailable>"
                    try:
                        schema = str(func._schema)
                    except Exception:  # pragma: no cover - schema is display only
                        pass
                    # A Python wrapper already counted this collective when the
                    # aten op is its implementation (reduce-scatter, broadcast).
                    # Counting both would double the table.
                    if bucket in _STATE["py_stack"]:
                        _STATE["schemas"].setdefault(raw, schema)
                        return func(*args, **(kwargs or {}))
                    _capture_stack(raw)
                    _record(
                        raw, "aten", bucket, _measure(raw, args, kwargs or {}), schema
                    )
            return func(*args, **(kwargs or {}))

    _STATE["mode"] = CollectiveCounter()
    _STATE["mode"].__enter__()


def _install_python_wrappers() -> None:
    """Wrap the ``torch.distributed.*`` collectives the dispatcher cannot see."""
    import torch.distributed as dist

    names = [
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "all_to_all_single",
        "broadcast",
        "send",
        "recv",
        "batch_isend_irecv",
    ]
    for attr in names:
        real = getattr(dist, attr, None)
        if real is None or getattr(real, "_cpu_lab_wrapped", False):
            continue

        def make(real=real, attr=attr):
            full = f"dist.{attr}"
            bucket = _bucket_of(full)

            def wrapper(*args, **kwargs):
                if bucket is None:
                    return real(*args, **kwargs)
                _capture_stack(full)
                _record(full, "python", bucket, _measure(full, args, kwargs))
                # The nested aten op of the same kind is this call's
                # implementation (reduce-scatter, broadcast); the stack marker
                # lets the dispatcher skip it so one collective counts once.
                _STATE["py_stack"].append(bucket)
                try:
                    return real(*args, **kwargs)
                finally:
                    _STATE["py_stack"].pop()

            wrapper._cpu_lab_wrapped = True  # type: ignore[attr-defined]
            return wrapper

        setattr(dist, attr, make())


def _install_p2p_hook() -> None:
    """Count point-to-point ops where they are *defined*: ``P2POp.__new__``.

    Wrapping ``dist.isend`` / ``dist.irecv`` is not an option -- ``P2POp``
    checks ``op not in [isend, irecv]`` against the module-level originals, so
    replacing the attributes makes pipeline parallelism raise
    ``ValueError: Invalid op``. ``P2POp`` is public API and is the single place
    a send or receive is described, so counting there catches both torch's
    fast path and ``batch_isend_irecv`` without touching any identity.
    """
    import torch.distributed as dist

    p2p_op = dist.P2POp
    if getattr(p2p_op, "_cpu_lab_wrapped", False):
        return

    real_new = p2p_op.__new__

    def __new__(cls, op, tensor, peer=None, group=None, tag=0, group_peer=None):
        bucket = "p2p_send" if op is dist.isend else "p2p_recv" if op is dist.irecv else "p2p"
        _capture_stack(f"P2POp[{bucket}]")
        _record(
            f"P2POp[{bucket.removeprefix('p2p_')}]",
            "python",
            bucket,
            _tensor_bytes(tensor),
            "dist.P2POp(op, tensor, peer, group, tag, group_peer)",
        )
        return real_new(cls, op, tensor, peer, group, tag, group_peer)

    p2p_op.__new__ = staticmethod(__new__)
    p2p_op._cpu_lab_wrapped = True  # type: ignore[attr-defined]


def _install_step_hook() -> None:
    """Snapshot counters once per step.

    Step 1 is not representative -- FSDP gathers all parameters on the first
    forward (nothing is prefetched yet) before settling into per-layer
    re-gathering -- so the per-step series is kept, not just a total, to tell a
    one-off setup cost apart from the steady state.
    """
    from torchtitan.trainer import Trainer

    if getattr(Trainer.train_step, "_cpu_lab_wrapped", False):
        return

    real = Trainer.train_step

    def train_step(self, *args, **kwargs):
        _STATE["window"].clear()
        _STATE["window_bytes"].clear()
        _STATE["in_step_active"] = True
        try:
            result = real(self, *args, **kwargs)
        finally:
            _STATE["in_step_active"] = False
        _STATE["steps_seen"] += 1
        # Accumulate as well as snapshot: ``window`` is cleared every step, so
        # reading it after the run only ever sees the last step. The running
        # total is what tells in-step traffic apart from one-off setup traffic.
        _STATE["in_step"].update(_STATE["window"])
        _STATE["in_step_bytes"].update(_STATE["window_bytes"])
        _STATE["per_step"].append(
            {
                "step": int(getattr(self, "step", _STATE["steps_seen"])),
                "calls": dict(_STATE["window"]),
                "bytes": dict(_STATE["window_bytes"]),
            }
        )
        return result

    train_step._cpu_lab_wrapped = True  # type: ignore[attr-defined]
    Trainer.train_step = train_step


def install() -> None:
    if _STATE["installed"]:
        return
    _STATE["installed"] = True
    _install_dispatch_mode()
    _install_python_wrappers()
    _install_p2p_hook()
    _install_step_hook()
    _install_phase_hooks()
    print("[collectives] counting aten + python collectives, per step", flush=True)


def summary() -> dict:
    steps = max(_STATE["steps_seen"], 1)
    ops = []
    for (raw, layer, bucket, phase), calls in _STATE["rows"].most_common():
        nbytes = _STATE["bytes"][(raw, layer, bucket, phase)]
        ops.append(
            {
                "op": _FRIENDLY[bucket],
                "bucket": bucket,
                "raw": raw,
                "layer": layer,
                "phase": phase,
                "calls": calls,
                "bytes_per_rank": nbytes,
                "calls_per_step": round(calls / steps, 2),
                "bytes_per_step": round(nbytes / steps, 1),
            }
        )

    by_bucket: dict[str, dict] = {}
    for row in ops:
        entry = by_bucket.setdefault(
            row["bucket"],
            {
                "op": row["op"],
                "bucket": row["bucket"],
                "calls": 0,
                "bytes_per_rank": 0,
                "calls_per_step": 0.0,
                "bytes_per_step": 0.0,
                "sources": [],
                "phase_calls": {},
                "phase_bytes": {},
            },
        )
        entry["calls"] += row["calls"]
        entry["bytes_per_rank"] += row["bytes_per_rank"]
        entry["calls_per_step"] = round(entry["calls"] / steps, 2)
        entry["bytes_per_step"] = round(entry["bytes_per_rank"] / steps, 1)
        entry["phase_calls"][row["phase"]] = (
            entry["phase_calls"].get(row["phase"], 0) + row["calls"]
        )
        entry["phase_bytes"][row["phase"]] = (
            entry["phase_bytes"].get(row["phase"], 0) + row["bytes_per_rank"]
        )
        entry["sources"].append(
            {
                "raw": row["raw"],
                "layer": row["layer"],
                "phase": row["phase"],
                "calls": row["calls"],
            }
        )

    for entry in by_bucket.values():
        bucket = entry["bucket"]
        entry["calls_during_steps"] = sum(
            _STATE["in_step"].get(f"{bucket}@{layer}", 0) for layer in ("aten", "python")
        )
        entry["calls_outside_steps"] = entry["calls"] - entry["calls_during_steps"]
        entry["bytes_during_steps"] = sum(
            _STATE["in_step_bytes"].get(f"{bucket}@{layer}", 0)
            for layer in ("aten", "python")
        )
        # Traffic that is not training traffic, called out explicitly so a
        # checkpoint save can never be read as per-step bandwidth.
        phase_calls = entry["phase_calls"]
        entry["calls_step"] = phase_calls.get("step", 0)
        entry["calls_checkpoint"] = phase_calls.get("checkpoint", 0)
        entry["calls_init"] = phase_calls.get("init", 0)
        entry["bytes_checkpoint"] = entry["phase_bytes"].get("checkpoint", 0)

    return {
        "steps": _STATE["steps_seen"],
        "per_step": _STATE["per_step"],
        "ops": ops,
        "by_bucket": sorted(by_bucket.values(), key=lambda e: -e["calls"]),
        "schemas": _STATE["schemas"],
        "stacks": _STATE["stacks"],
        "unknown_payload": dict(_STATE["unknown_payload"]),
        "zero_payload": dict(_STATE["zero_payload"]),
        "payload_debug": _STATE["payload_debug"],
        "total_calls": sum(_STATE["rows"].values()),
        "total_bytes_per_rank": sum(_STATE["bytes"].values()),
        "total_calls_checkpoint": sum(
            n for (_, _, _, phase), n in _STATE["rows"].items() if phase == "checkpoint"
        ),
    }


def dump(rank: int, log_dir: str) -> str:
    path = os.path.join(log_dir, f"collectives_rank_{rank}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary(), fh, indent=2)
    return path
