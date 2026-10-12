# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run torchtitan on a CPU-only torch build without editing torchtitan.

This module is the whole reason ``cpu_lab/`` exists as a sibling of
``torchtitan/``: every gap between "torchtitan expects a GPU cluster" and
"there is one CPU and gloo" is closed here, at *runtime*, by monkeypatching.
Nothing under ``torchtitan/`` is modified.

Must be imported **before** ``torchtitan`` (and before ``torch._utils`` is
consulted for device discovery), because several patches have to be in place
before the modules that read them are imported. The required order is:

1. triton stub                       -- ``torch._dynamo`` and
   ``torchtitan.distributed.minimal_async_ep.kernels`` import triton at module
   scope, and it has no CPU wheel.
2. device discovery -> "cpu"         -- ``torchtitan.tools.utils`` snapshots
   ``device_type``/``device_module`` at import time via
   ``torch._utils._get_available_device_type()``, which returns None on a
   CPU-only build and used to fall back to "cuda". Patching the probe (rather
   than the torchtitan global) is what makes the rest of the stack -- backend
   selection, ``DeviceMesh``, ``set_device`` -- resolve to CPU/gloo correctly.
3. TCPStore without libuv            -- see ``no_libuv_patch``.
4. barrier / set_timeout             -- CPU collectives take no device ids, and
   ``torch.distributed.set_timeout`` was removed from the public API.
5. CPU memory monitor                -- ``torch.cpu`` exposes no allocator stats.
6. FSDP SUM collectives              -- FSDP defaults to NCCL's fused
   ``PREMUL_SUM``, which gloo rejects.
7. Pipeline schedule ``step()``       -- torchtitan passes microbatches via
   ``arg_mbs=``; torch 2.12's ``step()`` takes a whole batch and swallows that
   keyword into ``**kwargs``.

``apply()`` is idempotent.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

_APPLIED = False


# --------------------------------------------------------------------------- #
# 1. triton stub
# --------------------------------------------------------------------------- #
def _install_triton_stub() -> None:
    """Make ``import triton`` (and ``triton.<anything>``) succeed.

    Only import-time evaluation matters: ``kernels.py`` uses ``@triton.jit``
    decorators and ``tl.<dtype>`` tables at module scope, and
    ``torch._inductor.runtime.hints`` does ``import triton.backends.compiler``.
    None of those kernels ever execute on CPU, so a placeholder that answers
    any attribute access keeps the import chain alive.

    Submodule imports (``triton.backends.compiler``) must go through the import
    machinery, not ``getattr``, so a meta path finder is registered rather than
    just seeding ``sys.modules``. The finder is appended last so a real triton
    installation always wins if one exists.
    """
    try:
        import triton  # noqa: F401  (real triton, if present)

        return False
    except ImportError:
        pass

    import importlib.abc
    import importlib.machinery

    class _StubModule(types.ModuleType):
        """Module whose missing attributes resolve to a nested placeholder."""

        def __getattr__(self, name: str) -> Any:
            if name == "__version__":
                # torch._inductor reads triton.__version__ for a compatibility
                # check; a string is all it needs.
                return "0.0.0+cpu-stub"
            if name.startswith("__") and name.endswith("__"):
                # Keep real dunders (__all__, __file__, copy/pickle probes)
                # raising AttributeError so they do not silently become stubs.
                raise AttributeError(name)
            child = _StubModule(f"{self.__name__}.{name}")
            child.__path__ = []
            setattr(self, name, child)
            return child

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            # ``@triton.jit`` with no parentheses must return the decorated
            # function unchanged, so the module still defines real callables.
            if len(args) == 1 and not kwargs and callable(args[0]):
                return args[0]
            return _StubModule(f"{self.__name__}()")

    class _StubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            module = _StubModule(spec.name)
            module.__path__ = []
            return module

        def exec_module(self, module) -> None:
            pass

    class _TritonFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "triton" and not fullname.startswith("triton."):
                return None
            if fullname in sys.modules:
                return None
            return importlib.machinery.ModuleSpec(fullname, _StubLoader())

    sys.meta_path.append(_TritonFinder())

    # Seed the two names that are imported most eagerly, so `import triton` and
    # `import triton.language as tl` resolve without going through the finder.
    triton_mod = _StubModule("triton")
    triton_mod.__path__ = []
    language_mod = _StubModule("triton.language")
    language_mod.__path__ = []
    triton_mod.language = language_mod
    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = language_mod
    print("[cpu-compat] installed triton stub (CPU-only build)", flush=True)
    return True


def _patch_torch_compile() -> None:
    """Degrade ``torch.compile`` to eager on a triton-less build.

    ``torchtitan.models.common.attention`` calls ``torch.compile`` at *module
    scope* (``_compiled_flex_attn = torch.compile(...)``), so merely importing
    the attention module drags in ``torch._dynamo`` -> ``torch._inductor`` ->
    real ``triton`` -- which has no CPU wheel and no usable stub (inductor
    subclasses triton internals). Returning the function unchanged keeps the
    import alive and the semantics correct, just unfused.

    Only used when triton is genuinely absent, i.e. when inductor could not have
    produced a kernel anyway. TinyGPT configs ship with ``compile`` disabled, so
    nothing silently loses an optimization it would otherwise have had.
    """
    import torch

    if getattr(torch.compile, "_cpu_patched", False):
        return

    def compile_stub(fn: Any = None, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        if fn is None:
            return decorator
        if callable(fn):
            return fn
        raise TypeError(
            f"torch.compile stub expected a callable or None, got {type(fn).__name__}"
        )

    compile_stub._cpu_patched = True  # type: ignore[attr-defined]
    torch.compile = compile_stub
    print(
        "[cpu-compat] torch.compile -> eager passthrough (no triton/inductor on CPU)",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# 2. device discovery
# --------------------------------------------------------------------------- #
def _patch_device_discovery() -> str:
    """Make ``torch._utils._get_available_device_type()`` report "cpu".

    ``torchtitan.tools.utils`` does ``device_type = _get_available_device_type()
    or "cuda"``, and the probe checks only accelerators -- cuda / mps / xpu /
    mtia -- so a CPU-only build has no path to ``"cpu"``. Patching the probe
    fixes every downstream consumer at once: the module-level ``device_type`` /
    ``device_module`` globals, ``DeviceMesh("cpu", ...)``, and gloo backend
    selection (``Backend.default_device_backend_map["cpu"] == "gloo"``).
    """
    import torch._utils as torch_utils

    if getattr(torch_utils._get_available_device_type, "_cpu_patched", False):
        return "cpu"

    real_probe = torch_utils._get_available_device_type

    def probe() -> str:
        found = real_probe()
        if found is None:
            return "cpu"
        return found

    probe._cpu_patched = True  # type: ignore[attr-defined]
    probe._real_probe = real_probe  # type: ignore[attr-defined]
    torch_utils._get_available_device_type = probe
    return "cpu"


# --------------------------------------------------------------------------- #
# 4. collectives
# --------------------------------------------------------------------------- #
def _patch_collectives() -> None:
    import datetime

    import torch
    import torch.distributed as dist

    # ``barrier(device_ids=[...])`` is an accelerator hint. On CPU,
    # ``torch.cpu.current_device()`` returns the *string* "cpu", which pybind
    # rejects ("Expected Sequence[SupportsInt]"). Strip the hint on CPU.
    if not getattr(dist.barrier, "_cpu_patched", False):
        real_barrier = dist.barrier

        def barrier(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("device_ids") is not None:
                kwargs["device_ids"] = None
            return real_barrier(*args, **kwargs)

        barrier._cpu_patched = True  # type: ignore[attr-defined]
        dist.barrier = barrier

    # ``torch.distributed.set_timeout`` was removed from the public API in
    # newer torch builds; torchtitan still calls it as ``(timeout, group)``.
    if not hasattr(dist, "set_timeout"):
        from torch.distributed.distributed_c10d import _set_pg_timeout

        def set_timeout(timeout: datetime.timedelta, group: Any = None) -> None:
            _set_pg_timeout(timeout, group)

        dist.set_timeout = set_timeout  # type: ignore[attr-defined]

    # ``torch.cpu`` lacks ``synchronize`` on some builds; it is a no-op anyway.
    if not hasattr(torch.cpu, "synchronize"):
        torch.cpu.synchronize = lambda *a, **kw: None  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 5. CPU memory monitor
# --------------------------------------------------------------------------- #
def _patch_metrics_monitor() -> None:
    """Replace the allocator-stats monitor with a host-RSS one.

    ``DeviceMemoryMonitor`` calls ``get_device_name`` / ``memory_stats`` /
    ``reset_peak_memory_stats`` / ``empty_cache``, none of which exist on
    ``torch.cpu``. The interface is only used for informational logging, so a
    small stand-in that reports process RSS keeps the metrics path happy.
    """
    try:
        from torchtitan.components import metrics as M
    except ImportError:
        return
    if getattr(M.build_device_memory_monitor, "_cpu_patched", False):
        return

    from torchtitan.tools.logging import logger, warn_once

    import hostmem

    class _HostMem:
        """Process memory via ``hostmem`` (psutil, else ctypes, else /proc).

        The first version of this stand-in only worked with psutil, which is not
        installed here, so every run logged ``memory: -1.00GiB`` -- the one number
        that says whether a model size actually fits was simply absent.
        """

        def __init__(self) -> None:
            self._proc = hostmem.process_memory()
            self._system = hostmem.system_memory()

        @property
        def available(self) -> bool:
            return self._proc["rss"] >= 0

        @property
        def rss(self) -> int:
            return self._proc["rss"]

        @property
        def commit(self) -> int:
            return self._proc["commit"]

        @property
        def total(self) -> int:
            return self._system["total_phys"]

    class CpuMemoryMonitor:
        def __init__(self) -> None:
            self._host = _HostMem()
            self._peak = 0
            self.device_name = "cpu"
            total = self._host.total
            self.device_capacity = total if total > 0 else max(self._host.rss, 1)
            self.device_capacity_gib = self.device_capacity / (1024**3)
            if not self._host.available:
                warn_once(
                    logger,
                    "Host memory stats unavailable (install psutil for CPU "
                    "memory metrics); reporting -1.",
                )

        def reset_peak_stats(self) -> None:
            self._peak = 0

        def get_peak_stats(self):
            rss = self._host.rss
            self._peak = max(self._peak, rss)
            gib = self._peak / (1024**3) if self._peak > 0 else -1
            pct = (
                100 * self._peak / self.device_capacity
                if self.device_capacity > 0 and self._peak > 0
                else -1
            )
            # num_alloc_retries / num_ooms have no CPU analogue.
            return M.DeviceMemStats(gib, pct, gib, pct, 0, 0)

    def build_device_memory_monitor():
        return CpuMemoryMonitor()

    build_device_memory_monitor._cpu_patched = True  # type: ignore[attr-defined]
    M.build_device_memory_monitor = build_device_memory_monitor


# --------------------------------------------------------------------------- #
# 6. FSDP collectives
# --------------------------------------------------------------------------- #
def _patch_fsdp_sum_reduction() -> None:
    """Force FSDP to reduce with SUM so gloo accepts the collective.

    FSDP's gradient-division path builds ``ReduceOp.PREMUL_SUM``, a fused NCCL
    op; gloo raises "Cannot use ReduceOp.PREMUL_SUM with Gloo". Setting
    ``force_sum_reduction_for_comms`` switches to plain SUM, which is
    numerically identical here because the caller still applies the 1/N
    scaling. Applied right after FSDP wraps the model.
    """
    try:
        from torchtitan.distributed import fsdp as F
    except ImportError:
        return
    if getattr(F.apply_fsdp_to_decoder, "_cpu_patched", False):
        return

    real_apply = F.apply_fsdp_to_decoder

    def apply_fsdp_to_decoder(*args: Any, **kwargs: Any) -> Any:
        # ``apply_fsdp_to_decoder`` mutates in place and returns None, so take
        # the model from the call rather than from the return value.
        result = real_apply(*args, **kwargs)
        model = args[0] if args else kwargs.get("model")
        if model is not None:
            for module in model.modules():
                setter = getattr(module, "set_force_sum_reduction_for_comms", None)
                if callable(setter):
                    setter(True)
        return result

    apply_fsdp_to_decoder._cpu_patched = True  # type: ignore[attr-defined]
    F.apply_fsdp_to_decoder = apply_fsdp_to_decoder


# --------------------------------------------------------------------------- #
# 7. Pipeline schedule step() API
# --------------------------------------------------------------------------- #
def _patch_pipeline_schedule_step() -> None:
    """Bridge torchtitan's ``step(arg_mbs=...)`` call onto torch's ``_step_microbatches``.

    torchtitan drives the pipeline with already-split microbatches::

        pp_schedule.step(arg_mbs=..., kwarg_mbs=..., target_mbs=...,
                         losses=..., loss_kwargs=..., return_outputs=False)

    In torch 2.12 every ``step`` signature is ``step(*args, target=None,
    losses=None, return_outputs=True, **kwargs)`` -- a *whole-batch* entry
    point that chunks its input itself. ``arg_mbs`` therefore lands in
    ``**kwargs``, gets treated as model input, and the run dies with the
    confusing ``Expecting N arg_mbs but got 1``.

    The microbatches-level entry point still exists and still takes exactly the
    arguments torchtitan passes: ``_step_microbatches(arg_mbs, kwarg_mbs,
    target_mbs, losses, return_outputs)``. So the adapter installs a per-instance
    ``step`` that accepts torchtitan's keywords and calls ``_step_microbatches``
    directly, reproducing the two pieces of bookkeeping that the whole-batch
    ``step`` would otherwise do (per-stage ``has_backward`` and runtime-state
    reset). ``_prepare_schedule_with_comms`` needs no replay -- the concrete
    schedule classes call it from ``__init__``.

    ``loss_kwargs`` has no slot in ``_step_microbatches`` either, so it is bound
    into the loss function instead: ``_build_pipeline_schedule`` is wrapped to
    insert a mutable holder that the adapter fills before each call.
    """
    try:
        from torchtitan.distributed import pipeline_parallel as PP
    except ImportError:
        return
    if getattr(PP._build_pipeline_schedule, "_cpu_patched", False):
        return

    real_build = PP._build_pipeline_schedule

    def _install_step_adapter(schedule: Any, loss_kwargs_holder: dict) -> None:
        real_step = schedule.step

        def step(*args: Any, **kwargs: Any) -> Any:
            # Everything torchtitan passes is a keyword; positional args would
            # mean a whole-batch call, which is not our case.
            arg_mbs = kwargs.pop("arg_mbs", None)
            kwarg_mbs = kwargs.pop("kwarg_mbs", None)
            target_mbs = kwargs.pop("target_mbs", None)
            loss_kwargs = kwargs.pop("loss_kwargs", None)
            explicit_target = kwargs.pop("target", None)
            losses = kwargs.pop("losses", None)
            return_outputs = kwargs.pop("return_outputs", False)

            # Unrecognized leftovers mean this is not the torchtitan convention
            # (or torch changed again) -- hand it to the real implementation
            # rather than silently dropping arguments.
            if kwargs or (arg_mbs is None and kwarg_mbs is None):
                return real_step(
                    *args,
                    target=explicit_target,
                    losses=losses,
                    return_outputs=return_outputs,
                    **kwargs,
                )

            loss_kwargs_holder.clear()
            if loss_kwargs:
                loss_kwargs_holder.update(loss_kwargs)

            stages = getattr(schedule, "_stages", None)
            if stages is None:
                single = getattr(schedule, "_stage", None)
                stages = [single] if single is not None else []
            for stage in stages:
                stage.has_backward = schedule._has_backward
                stage.clear_runtime_states()

            return schedule._step_microbatches(
                arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs
            )

        schedule.step = step

    def _build_pipeline_schedule(*args: Any, **kwargs: Any) -> Any:
        loss_fn = kwargs.get("loss_fn")
        holder: dict = {}
        if loss_fn is not None:
            def loss_fn_with_kwargs(*a: Any, **kw: Any) -> Any:
                # ``holder`` is repopulated per step; the schedule's own wrapper
                # only ever forwards (output, target), so the extra loss kwargs
                # (e.g. global_valid_tokens) arrive here.
                return loss_fn(*a, **kw, **holder)

            loss_fn_with_kwargs._cpu_pp_holder = holder  # type: ignore[attr-defined]
            kwargs["loss_fn"] = loss_fn_with_kwargs

        schedule = real_build(*args, **kwargs)
        _install_step_adapter(schedule, holder)
        return schedule

    _build_pipeline_schedule._cpu_patched = True  # type: ignore[attr-defined]
    PP._build_pipeline_schedule = _build_pipeline_schedule


# --------------------------------------------------------------------------- #
def apply() -> None:
    """Install every patch. Safe to call more than once."""
    global _APPLIED
    if _APPLIED:
        return

    _install_triton_stub()  # must precede torch.compile patching
    _patch_torch_compile()
    device_type = _patch_device_discovery()

    # no_libuv_patch must run before any process group is created.
    import no_libuv_patch  # noqa: F401

    _patch_collectives()
    _patch_metrics_monitor()
    _patch_fsdp_sum_reduction()
    _patch_pipeline_schedule_step()

    os.environ.setdefault("TORCHTITAN_DEVICE_TYPE", device_type)
    _APPLIED = True
    print(
        f"[cpu-compat] ready: device_type={device_type}, backend=gloo, "
        "triton=stub, fsdp=sum-reduction, pp=step-adapter",
        flush=True,
    )