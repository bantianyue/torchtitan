# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Force ``TCPStore(use_libuv=False)`` so ``env://`` rendezvous works on CPU.

Official PyTorch **Windows CPU** wheels are compiled without libuv, but
``torch.distributed`` requests libuv by default when constructing a
``TCPStore``. That makes every rendezvous path (``torchrun``, ``env://``
``init_process_group``) fail before training starts::

    DistStoreError: use_libuv was requested but PyTorch was built without
    libuv support, run with USE_LIBUV=0 to disable it.

The ``USE_LIBUV=0`` environment variable is not honored by the C++ layer on
these builds, so the only reliable fix is to pass the constructor argument
explicitly. This module wraps ``TCPStore`` to do that; ``run_cpu_train.py``
activates it by importing this file first (``sitecustomize`` is not used so the
patch stays scoped to the ranks we spawn).
"""

import torch.distributed as dist

_ORIG_TCPSTORE = dist.TCPStore


class _NoLibuvTCPStore(_ORIG_TCPSTORE):
    """TCPStore that always disables the unavailable libuv backend."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_libuv", False)
        super().__init__(*args, **kwargs)


def apply() -> None:
    """Install the patched TCPStore into torch.distributed (idempotent)."""
    if getattr(dist.TCPStore, "_no_libuv_patched", False):
        return
    _NoLibuvTCPStore._no_libuv_patched = True
    dist.TCPStore = _NoLibuvTCPStore
    # distributed_c10d imported the symbol directly; keep both in sync so
    # whichever reference init_process_group resolves still gets the patch.
    try:
        import torch.distributed.distributed_c10d as c10d

        if getattr(c10d, "TCPStore", None) is _ORIG_TCPSTORE:
            c10d.TCPStore = _NoLibuvTCPStore
    except ImportError:
        pass


apply()