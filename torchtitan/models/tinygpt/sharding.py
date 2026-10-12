# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-parallel / sequence-parallel declarations for TinyGPT.

This is the file that actually decides *where each tensor lives* under TP. It
is called once from ``TinyGPTModel.Config.update_from_config`` and only writes
``sharding_config`` fields -- nothing here runs at model-construction time.

Reading guide for the TP plan
-----------------------------
Two sharding styles appear below and they always come in pairs:

``colwise``
    The weight is split along its **output** dimension (``Shard(0)``), so every
    rank computes a different slice of the output. No communication. Used for
    the first matmul of a pair (``wq``/``wk``/``wv``, ``c_fc``, ``lm_head``,
    the token table).

``rowwise``
    The weight is split along its **input** dimension (``Shard(1)``), so every
    rank computes a *partial* sum that must be reduced. ``out_src=Partial``
    -> ``out_dst=Replicate`` performs that all-reduce. Used for the second
    matmul of the pair (``wo``, ``c_proj``).

A colwise followed by a rowwise is the standard way to shard an MLP or an
attention block with exactly one collective per block. Sequence parallelism
(SP) is the refinement that replaces that all-reduce with a reduce-scatter and
keeps the activation sequence-sharded in between, so the LayerNorms after the
attention and MLP do not need the full sequence.

Two TinyGPT-specific departures from ``set_llama3_sharding_config``:

1. The token embedding carries a second, *learned position* table. The position
   table is replicated (every rank needs all positions) and must never take the
   vocab-parallel path, so it gets its own explicit declaration instead of
   reusing ``set_decoder_sharding_config`` wholesale.
2. The LayerNorms have a ``bias`` in addition to ``weight`` (RMSNorm has only
   ``weight``), so the norm sharding has to declare both parameters --
   ``Module._distribute_states`` raises if any parameter is left undeclared.
"""

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    rowwise_config,
    set_gqa_inner_attention_local_map,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

__all__ = ["set_tinygpt_sharding_config"]


def _layernorm_sharding(*, enable_sp: bool, pre_lm_head: bool) -> ShardingConfig:
    """LayerNorm sharding, declaring both ``weight`` and ``bias``.

    LayerNorm is elementwise, so its parameters are replicated under TP; only
    the *activations* move between replicated and sequence-sharded. The
    pre-lm_head norm additionally gathers the sequence shard so ``lm_head`` sees
    full-length hidden states, matching the root-norm contract the chunked
    cross-entropy loss relies on.
    """
    activation = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    state = {"weight": dense_param_placement(tp=spmd.R)}
    if enable_sp:
        state["bias"] = dense_param_placement(tp=spmd.R)
    else:
        state["bias"] = dense_param_placement(tp=spmd.I)

    config = ShardingConfig(
        state_shardings=state,
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
    )
    if pre_lm_head:
        # All-gather the TP sequence shard back to replicated hidden states.
        config.out_dst_shardings = dense_activation_placement(tp=spmd.R)
    return config


def _set_token_embedding_sharding(config, *, activation_layout) -> None:
    """Vocabulary-parallel token table.

    ``weight`` is split by vocabulary rows; each rank computes the embeddings
    for the token ids it owns and zeros elsewhere, so the per-rank results are
    ``Partial`` and the all-reduce produces the exact same tensor a single rank
    would have computed.
    """
    config.tok.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=activation_layout,
        local_map=LocalMapConfig(in_grad_placements=None),
    )


def _set_position_embedding_sharding(config) -> None:
    """Learned position table: replicated, never vocab-parallel.

    ``positions`` is an index tensor every rank needs in full, so both the
    table and the positions input stay replicated and the output is replicated
    too. It then meets the (already reduced) token embedding, where DTensor's
    pointwise rule happily adds a Replicate operand to whatever layout the
    token embedding produced.
    """
    config.pos.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"positions": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"positions": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.I),
    )


def _set_attention_sharding(config, *, enable_sp: bool) -> None:
    """Q/K/V colwise + output projection rowwise.

    ``wq``/``wk``/``wv`` split the head dimension across ranks, so each rank
    owns ``n_heads / tp`` heads; ``wo`` consumes those head slices and reduces
    them back. ``inner_attention`` runs inside a ``local_map`` because SDPA is
    a kernel call, not a DTensor-aware op.
    """
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    config.sharding_config = ShardingConfig(
        in_src_shardings={"x_BLD": attn_x_layout},
        in_dst_shardings={"x_BLD": dense_activation_placement(tp=spmd.R)},
    )
    config.wq.sharding_config = colwise_config()
    config.wk.sharding_config = colwise_config()
    config.wv.sharding_config = colwise_config()
    config.wo.sharding_config = rowwise_config(output_sp=enable_sp)
    set_gqa_inner_attention_local_map(config.inner_attention)


def _set_mlp_sharding(config, *, attn_x_layout, enable_sp: bool) -> None:
    """``c_fc`` colwise + ``c_proj`` rowwise (the 4x expansion splits evenly)."""
    config.sharding_config = ShardingConfig(
        in_src_shardings={"x": attn_x_layout},
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R)},
    )
    config.c_fc.sharding_config = colwise_config()
    config.c_proj.sharding_config = rowwise_config(output_sp=enable_sp)


def set_tinygpt_sharding_config(config, *, enable_sp: bool) -> None:
    """Fill ``sharding_config`` on every TinyGPT sub-config.

    Declarations are written unconditionally: the mesh actually passed to
    ``Module.parallelize()`` decides which of them apply. Under FSDP-only (no
    TP axis) every declaration here resolves to a single-rank mesh and becomes
    a no-op, which is what makes one model definition work for all strategies.
    """
    activation_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )

    _set_token_embedding_sharding(config.tok_embeddings, activation_layout=activation_layout)
    _set_position_embedding_sharding(config.tok_embeddings)

    config.norm.sharding_config = _layernorm_sharding(
        enable_sp=enable_sp, pre_lm_head=True
    )

    config.lm_head.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.S(-1)),
        out_dst_shardings=dense_activation_placement(tp=spmd.S(-1)),
    )

    for layer_cfg in config.layers:
        layer_cfg.attention_norm.sharding_config = _layernorm_sharding(
            enable_sp=enable_sp, pre_lm_head=False
        )
        layer_cfg.ffn_norm.sharding_config = _layernorm_sharding(
            enable_sp=enable_sp, pre_lm_head=False
        )
        _set_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
        _set_mlp_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=activation_layout,
            enable_sp=enable_sp,
        )