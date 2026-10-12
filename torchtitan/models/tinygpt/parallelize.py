# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Applies PT-D parallelisms and training techniques to TinyGPT.

The ordering below matters and is the same for every model in this repo:

1. **CP** -- wrap the inner attention *before* ``model.parallelize()``, so the
   CP wrapper ends up inside the ``local_map`` boundary that ``parallelize()``
   installs (the kernel must see local tensors).
2. **TP** -- ``model.parallelize(parallel_dims)`` walks the ``sharding_config``
   declarations written by ``update_from_config`` and turns them into DTensors
   plus input/output redistribution wrappers.
3. **AC** -- activation checkpointing trades compute for memory; apply it after
   TP so the wrappers land on the sharded modules.
4. **compile** -- applied after AC so ``torch.compile`` sees the final module
   structure.
5. **FSDP** -- last, so it wraps the already-TP-sharded modules and can shard
   whatever is left (DP/CP axes) on top.
"""

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, validate_config
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp

from .model import TinyGPTModel


def parallelize_tinygpt(
    model: TinyGPTModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply TP, CP, activation checkpointing, compile, and FSDP to TinyGPT.

    NOTE: the passed-in model is expected to be on the meta device, so nothing
    here materializes real weights.
    """
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    else:
        # CP first: the wrapper must be captured inside the local_map region
        # that model.parallelize() installs in step 2.
        if parallel_dims.cp_enabled:
            apply_cp_to_forward(
                # pyrefly: ignore [missing-attribute, not-callable]
                [block.attention.inner_attention for block in model.layers.values()],
                parallel_dims.get_mesh("cp"),
            )
        if parallel_dims.tp_enabled:
            model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if model_compile_enabled:
        apply_compile(model, compile_config)

    # Always run apply_fsdp_to_decoder. With shard_degree=1 the all-gather is a
    # no-op, but the MixedPrecisionPolicy and the final fully_shard(model) are
    # still installed, which keeps the FSDP-only vs TP-only paths comparable.
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        dp_mesh_dims = None

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model