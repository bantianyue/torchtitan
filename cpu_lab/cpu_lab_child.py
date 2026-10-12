# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-rank entry point: install CPU patches and run logging, then train.

Invoked by ``run_cpu_train.py`` as::

    python -m cpu_lab_child <args forwarded to torchtitan.train>

Installs happen before ``torchtitan.train`` is imported:

1. ``cpu_compat`` -- makes the stack *run* on a CPU-only torch build.
2. ``qwen35_cpu`` -- only for ``--module ...qwen3_5``: supplies the pure-torch
   GatedDeltaNet kernels (``fla`` has no CPU wheel) and a differentiable
   attention forward (flex attention has no CPU backward).
3. ``run_logging`` -- makes the run *legible* (per-rank parallelism, dataset
   path, outputs). It patches ``Trainer``, so it must be installed before the
   trainer class is instantiated.

The ordering matters for one reason only: ``torchtitan.tools.utils`` snapshots
the device type at import time, so ``cpu_compat`` has to be first.
"""

import os
import sys

import cpu_compat

cpu_compat.apply()


def _requested_module() -> str:
    """The ``--module`` value, read from argv so no launcher change is needed."""
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if arg.startswith("--module="):
            return arg.split("=", 1)[1]
        if arg == "--module" and index + 1 < len(argv):
            return argv[index + 1]
    return ""


if any(tag in _requested_module() for tag in ("qwen3_5", "qwen35")):
    # Only for Qwen3.5: the shim registers a synthetic `fla` package, which
    # would otherwise paper over a genuinely missing dependency elsewhere.
    # Both tags are matched because the module is reached through two names:
    # `torchtitan.models.qwen3_5` (upstream) and `cpu_lab.qwen35` (this lab's
    # configs). Missing a tag here is not a loud failure -- the config_registry
    # import raises ImportError on the absent `fla`, and ConfigManager swallows
    # that and reports "Config function ... not found" against the package.
    import qwen35_cpu

    qwen35_cpu.apply()

import run_logging

run_logging.install()

if os.environ.get("CPU_LAB_COLLECTIVES") == "1":
    import collectives

    collectives.install()


def main() -> None:
    from torchtitan.train import main as train_main

    train_main()

    # Written after training returns, so the counters include every step.
    # Only rank 0 and the launcher's aggregation read these, but every rank
    # writes its own file: a mismatch between ranks is itself a finding.
    if os.environ.get("CPU_LAB_COLLECTIVES") == "1":
        import collectives

        log_dir = os.environ.get("CPU_LAB_LOG_DIR", ".")
        os.makedirs(log_dir, exist_ok=True)
        path = collectives.dump(int(os.environ.get("RANK", "0")), log_dir)
        print(f"[collectives] wrote {path}", flush=True)


if __name__ == "__main__":
    main()