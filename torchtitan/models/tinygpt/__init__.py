# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TinyGPT: a GPT-2 style model for learning torchtitan's training loop and
parallel strategies.

Registered under a fully-qualified module path rather than a short name, so it
can be used without editing ``torchtitan/models/__init__.py``::

    # Runs the model on the ``debugmodel`` flavor with the ``tinygpt_*`` configs.
    python -m torchtitan.train --module torchtitan.models.tinygpt --config <name>

The interesting part for a reader is ``sharding.py`` (where every tensor goes
under TP/SP) and ``parallelize.py`` (the order in which the strategies are
applied). ``config_registry.py`` wires up one training config per parallel
strategy combination.
"""

from collections.abc import Callable
from functools import partial
from typing import cast

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    PositionEmbedding,
    TinyGPTAttention,
    TinyGPTBlock,
    TinyGPTEmbedding,
    TinyGPTMLP,
    TinyGPTModel,
)
from .parallelize import parallelize_tinygpt

__all__ = [
    "TinyGPTModel",
    "model_registry",
    "parallelize_tinygpt",
    "tinygpt_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_TOKEN_EMB_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_POS_EMB_INIT = {"weight": partial(nn.init.normal_, std=0.01)}
_TOKEN_EMB_SKIP_INIT = {"weight": skip_param_init}
_NORM_INIT = {
    "weight": nn.init.ones_,
    "bias": nn.init.zeros_,
}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    """GPT-2's ``1/sqrt(2 * n_layer)`` scaling for residual-output projections."""
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _build_tinygpt_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    ffn_dim: int,
    head_dim: int | None = None,
) -> list[TransformerBlock.Config]:
    """Per-layer configs, with GPT-2's depth-scaled init on residual outputs.

    ``wo`` and ``c_proj`` are the two projections that write into the residual
    stream, so they get the depth-scaled init; everything else uses 0.02.

    All Linear layers use ``bias=False``, matching the convention elsewhere in
    torchtitan (``FeedForward``, ``GQAttention``). GPT-2 does put a bias on
    these projections, but a bias on the *rowwise* half of a TP pair needs a
    per-rank scale factor or it is counted once per rank -- a subtlety that adds
    nothing to what these runs are for. LayerNorm keeps its affine parameters.
    """
    layers: list[TinyGPTBlock.Config] = []
    for layer_id in range(n_layers):
        layers.append(
            TinyGPTBlock.Config(
                attention_norm=LayerNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                attention=TinyGPTAttention.Config(
                    dim=dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    wq=Linear.Config(
                        in_features=dim, out_features=dim, param_init=_LINEAR_INIT
                    ),
                    wk=Linear.Config(
                        in_features=dim, out_features=dim, param_init=_LINEAR_INIT
                    ),
                    wv=Linear.Config(
                        in_features=dim, out_features=dim, param_init=_LINEAR_INIT
                    ),
                    wo=Linear.Config(
                        in_features=dim,
                        out_features=dim,
                        param_init=_depth_init(layer_id),
                    ),
                    inner_attention=ScaledDotProductAttention.Config(),
                ),
                ffn_norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                feed_forward=TinyGPTMLP.Config(
                    c_fc=Linear.Config(
                        in_features=dim,
                        out_features=ffn_dim,
                        param_init=_LINEAR_INIT,
                    ),
                    act=GELU.Config(approximate="tanh"),
                    c_proj=Linear.Config(
                        in_features=ffn_dim,
                        out_features=dim,
                        param_init=_depth_init(layer_id),
                    ),
                ),
            )
        )
    return cast(list[TransformerBlock.Config], layers)


def _tinygpt(
    *,
    dim: int,
    n_heads: int,
    n_layers: int,
    vocab_size: int,
    max_position_embeddings: int,
    enable_weight_tying: bool,
    ffn_multiplier: int = 4,
    head_dim: int | None = None,
) -> TinyGPTModel.Config:
    """Assemble a TinyGPT config from its dimensions."""
    ffn_dim = ffn_multiplier * dim
    emb_init = _TOKEN_EMB_SKIP_INIT if enable_weight_tying else _TOKEN_EMB_INIT
    return TinyGPTModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        enable_weight_tying=enable_weight_tying,
        tok_embeddings=TinyGPTEmbedding.Config(
            tok=Embedding.Config(
                num_embeddings=vocab_size,
                embedding_dim=dim,
                param_init=emb_init,
            ),
            pos=PositionEmbedding.Config(
                num_embeddings=max_position_embeddings,
                embedding_dim=dim,
                param_init=_POS_EMB_INIT,
            ),
        ),
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_tinygpt_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            head_dim=head_dim,
        ),
    )


def _debugmodel() -> TinyGPTModel.Config:
    """Smallest flavor: enough to exercise every code path in a few seconds.

    ``n_heads=8`` and ``ffn_dim=512`` keep TP degrees 1/2/4/8 valid, and
    ``n_layers=4`` keeps PP degrees 1/2/4 valid, so one flavor covers every
    strategy combination the configs below demonstrate. ``vocab_size=2048``
    matches the vendored ``tests/assets/tokenizer`` rather than GPT-2's 50257,
    which keeps the embedding table (and the CPU step time) small.
    """
    return _tinygpt(
        dim=128,
        n_heads=8,
        n_layers=4,
        vocab_size=2048,
        max_position_embeddings=256,
        enable_weight_tying=True,
    )


def _debugmodel_untied() -> TinyGPTModel.Config:
    """``debugmodel`` with input/output embeddings left untied.

    Pipeline parallelism rejects weight tying (the tied parameter would have to
    live on two different stages), and torchtitan raises for that combination in
    ``Decoder.Config.update_from_config``. So PP runs use this flavor; it is
    identical to ``debugmodel`` apart from the tying, which keeps a PP run
    directly comparable to an FSDP/TP run of the same shape.
    """
    config = _tinygpt(
        dim=128,
        n_heads=8,
        n_layers=4,
        vocab_size=2048,
        max_position_embeddings=256,
        enable_weight_tying=False,
    )
    # The untied token table needs its own init instead of skipping it.
    config.tok_embeddings.tok.param_init = _TOKEN_EMB_INIT
    return config


def _8l512_untied() -> TinyGPTModel.Config:
    """``8L512`` with untied embeddings, for pipeline parallelism.

    PP needs enough graph nodes to split: the looped schedules require at least
    two virtual stages per rank, so a 4-layer model (6 effective nodes: 4 blocks
    + embedding + head) cannot fill ``pp_degree * 2`` stages once pp >= 3. Eight
    layers give 10 effective nodes, which covers pp up to 5 with stages to spare.
    """
    config = _tinygpt(
        dim=512,
        n_heads=8,
        n_layers=8,
        vocab_size=2048,
        max_position_embeddings=512,
        enable_weight_tying=False,
    )
    config.tok_embeddings.tok.param_init = _TOKEN_EMB_INIT
    return config


def _tinygpt_4l256() -> TinyGPTModel.Config:
    """The architecture of https://huggingface.co/coderian/TinyGPT.

    4 layers, 256 hidden, 8 heads, 50257-token vocab (the GPT-2 BPE table), a
    128-position learned table, and tied input/output embeddings -- ~29M
    parameters, of which ~26M live in the embedding/head and only ~3M in the
    transformer body.
    """
    return _tinygpt(
        dim=256,
        n_heads=8,
        n_layers=4,
        vocab_size=50257,
        max_position_embeddings=128,
        enable_weight_tying=True,
    )


def _tinygpt_8l512() -> TinyGPTModel.Config:
    """Mid-size flavor, still runnable on CPU.

    ``dim=512`` / ``n_heads=8`` / ``n_layers=8`` is large enough that TP and PP
    splits are visibly different from a single-device run, while staying at
    ~25M parameters and a 2048-token vocab.
    """
    return _tinygpt(
        dim=512,
        n_heads=8,
        n_layers=8,
        vocab_size=2048,
        max_position_embeddings=512,
        enable_weight_tying=True,
    )


def _scale(n_layers: int, dim: int) -> TinyGPTModel.Config:
    """A size on the scaling ladder: ``dim = n_layers * 64``, head_dim 64.

    Built by one function rather than hand-written flavors so the ladder is
    reproducible and the shape rules are stated once:

    - ``head_dim=64`` and ``dim`` a multiple of 128 keep ``n_heads = dim / 64``
      even, so a TP degree of 2 (and 4 for the larger rungs) divides the heads.
    - ``ffn_multiplier=4`` gives an ``ffn_dim`` that divides by the same degrees.
    - embeddings are **untied**, because pipeline parallelism rejects tying.
    - ``vocab_size=2048`` matches the vendored byte-level tokenizer, so the
      embedding table stays a small fraction of the body and the ladder measures
      the transformer rather than the vocabulary.

    The point of the ladder is to measure how a step's cost and a rank's memory
    grow with parameter count, on this machine, instead of extrapolating from
    arithmetic that ignores FSDP's all-gather buffers and gloo's staging copies.
    """
    config = _tinygpt(
        dim=dim,
        n_heads=dim // 64,
        n_layers=n_layers,
        vocab_size=2048,
        max_position_embeddings=512,
        enable_weight_tying=False,
    )
    config.tok_embeddings.tok.param_init = _TOKEN_EMB_INIT
    return config


def _scale_12l768() -> TinyGPTModel.Config:
    return _scale(12, 768)


def _scale_16l1024() -> TinyGPTModel.Config:
    return _scale(16, 1024)


def _scale_24l1536() -> TinyGPTModel.Config:
    return _scale(24, 1536)


def _scale_32l2048() -> TinyGPTModel.Config:
    return _scale(32, 2048)


tinygpt_configs: dict[str, Callable[[], TinyGPTModel.Config]] = {
    "debugmodel": _debugmodel,
    "debugmodel_untied": _debugmodel_untied,
    "4L256": _tinygpt_4l256,
    "8L512": _tinygpt_8l512,
    "8L512_untied": _8l512_untied,
    # Scaling ladder, ordered smallest first. Used to measure how step time and
    # per-rank memory grow with parameter count on this machine.
    "scale_12L768": _scale_12l768,
    "scale_16L1024": _scale_16l1024,
    "scale_24L1536": _scale_24l1536,
    "scale_32L2048": _scale_32l2048,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    """Build the ``ModelSpec`` for a TinyGPT flavor.

    ``attn_backend`` exists for signature parity with the other models, but
    TinyGPT only supports ``"sdpa"``: its attention has no positional encoding
    inside the block, so the flex/varlen backends (which expect per-document
    positions to build a mask) do not apply.
    """
    if attn_backend != "sdpa":
        raise ValueError(
            f"TinyGPT only supports attn_backend='sdpa' (got {attn_backend!r}). "
            "Its attention carries no in-block positional encoding, so the "
            "flex/varlen mask backends do not apply."
        )
    if flavor not in tinygpt_configs:
        raise ValueError(
            f"Unknown TinyGPT flavor {flavor!r}. "
            f"Available: {sorted(tinygpt_configs)}."
        )

    config = tinygpt_configs[flavor]()
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)

    return ModelSpec(
        name="tinygpt",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_tinygpt,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
