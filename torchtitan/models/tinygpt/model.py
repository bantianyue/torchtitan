# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A minimal GPT-2 style decoder, sized for learning the torchtitan stack.

This module intentionally keeps the classic GPT-2 recipe rather than the modern
Llama/DeepSeek recipe, so the differences from the other models in this repo are
easy to point at:

======================  =========================  ==========================
Component               TinyGPT (this file)        llama3 / deepseek_v3
======================  =========================  ==========================
Position encoding       learned absolute table     RoPE
Normalization           LayerNorm                  RMSNorm
MLP activation          GELU (two matmuls)         SwiGLU (three matmuls)
Attention               MHA (n_kv_heads == heads)  GQA / MLA
Weight tying            optional (GPT-2 default)   optional
======================  =========================  ==========================

Everything else -- the ``Decoder`` plumbing, the config/override system, and
the parallel-strategy entry points -- is identical, which is the point: the same
FSDP / TP / PP / CP combinations that train Llama train this too. See
``README.md`` in this directory for the parallel-strategy walkthrough.

Reference: https://huggingface.co/coderian/TinyGPT
(trained on TinyStories; config below matches its 4-layer / 256-dim / 8-head /
50257-vocab shape.)
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.attention import (
    AttentionMasksType,
    ScaledDotProductAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from torchtitan.config import ParallelismConfig

__all__ = [
    "PositionEmbedding",
    "TinyGPTAttention",
    "TinyGPTBlock",
    "TinyGPTEmbedding",
    "TinyGPTMLP",
    "TinyGPTModel",
]


# GPT-2 initializes every projection with N(0, 0.02), the token embedding with
# N(0, 0.02) and the position embedding with N(0, 0.01) (a smaller std because
# the position signal should not dominate the token signal at init).
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_TOKEN_EMB_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_POS_EMB_INIT = {"weight": partial(nn.init.normal_, std=0.01)}
_NORM_INIT = {
    "weight": nn.init.ones_,
    "bias": nn.init.zeros_,
}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    """GPT-2's scaled init for residual-output projections.

    GPT-2 scales the init of the projections writing into a residual stream
    (``c_proj`` and ``attn.c_proj``) by ``1/sqrt(2 * n_layer)`` to keep the
    residual variance from growing with depth. torchtitan expresses the same
    idea through :func:`depth_scaled_std`, applied per layer id.
    """
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


class PositionEmbedding(Module):
    """Learned absolute position embedding (GPT-2 ``wpe``).

    Deliberately not ``common.Embedding``: that class optionally shards its
    table across the tensor-parallel axis, which is right for the *vocabulary*
    table but wrong for the *position* table -- every rank needs all positions
    because every rank sees the whole (sequence-sharded or replicated)
    ``positions`` tensor. Keeping a separate, always-replicated module makes
    that requirement explicit.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        # Plain Parameter rather than nn.Embedding so the owning Module's
        # ``param_init`` machinery (keyed by parameter name) applies.
        self.weight = nn.Parameter(
            torch.empty(config.num_embeddings, config.embedding_dim)
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return F.embedding(positions, self.weight)


class TinyGPTEmbedding(Module):
    """Token embedding + learned position embedding.

    A single named child of the model (rather than two) so that pipeline
    parallel's stage splitting keeps both on the first stage: ``_split_module``
    only preserves FQNs it knows about (``tok_embeddings``, ``layers.*``,
    ``norm``, ``lm_head``) and would drop an unrelated sibling.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        tok: Embedding.Config
        pos: PositionEmbedding.Config

    def __init__(self, config: Config):
        super().__init__()
        self.tok = config.tok.build()
        self.pos = config.pos.build()

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.tok(tokens)
        if positions is not None:
            h = h + self.pos(positions)
        return h


class TinyGPTAttention(Module):
    """Multi-head causal self-attention with an SDPA inner attention.

    Not ``common.GQAttention`` because that module hard-requires a RoPE config
    and applies it unconditionally, whereas GPT-2 carries position information
    in the embedding. MHA (every query head has its own key/value head) is
    expressed as ``n_kv_heads == n_heads``; the projections stay separate so TP
    plan can shard each one colwise.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_heads: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        inner_attention: Module.Config
        head_dim: int | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim or config.dim // config.n_heads
        assert self.head_dim * self.n_heads == config.dim, (
            f"head_dim ({self.head_dim}) * n_heads ({self.n_heads}) must equal "
            f"dim ({config.dim})."
        )

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()
        self.inner_attention = config.inner_attention.build()
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape

        # (B, L, D) -> (B, L, N, H); the split keeps the head axis explicit so
        # TP can shard it (heads are evenly divisible by the TP degree).
        xq_BLNH = self.wq(x_BLD).view(B, L, self.n_heads, self.head_dim)
        xk_BLNH = self.wk(x_BLD).view(B, L, self.n_heads, self.head_dim)
        xv_BLNH = self.wv(x_BLD).view(B, L, self.n_heads, self.head_dim)

        # positions is ignored here on purpose: for GPT-2 the position signal
        # enters once, at the embedding, not per layer.
        out_BLNH = self.inner_attention(
            xq_BLNH,
            xk_BLNH,
            xv_BLNH,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=False,
        )
        out_BLD = out_BLNH.reshape(B, L, self.n_heads * self.head_dim)
        return self.wo(out_BLD)


class TinyGPTMLP(Module):
    """GPT-2 feed-forward: ``Linear(D -> 4D) -> GELU -> Linear(4D -> D)``.

    ``common.FeedForward`` is SwiGLU (three matmuls, gated), so it is not
    reusable here. Two matmuls makes this the cheapest block to TP-shard and a
    good place to observe colwise/rowwise pairing.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        c_fc: Linear.Config
        c_proj: Linear.Config
        act: GELU.Config

    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = config.c_fc.build()
        self.act = config.act.build()
        self.c_proj = config.c_proj.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class TinyGPTBlock(Module):
    """Pre-LayerNorm transformer block (GPT-2 block order).

    ``x + attn(ln_1(x))`` then ``x + mlp(ln_2(x))`` -- LayerNorm inside the
    residual branch, applied before each sublayer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention_norm: LayerNorm.Config
        attention: TinyGPTAttention.Config
        ffn_norm: LayerNorm.Config
        feed_forward: TinyGPTMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention_norm = config.attention_norm.build()
        self.attention = config.attention.build()
        self.ffn_norm = config.ffn_norm.build()
        self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), attention_masks, positions)
        return h + self.feed_forward(self.ffn_norm(h))


class TinyGPTModel(Decoder):
    """GPT-2 style decoder-only language model.

    Inherits ``Decoder`` for the pieces that are architecture-independent --
    the ``layers`` ModuleDict, weight tying, the ``None``-guarded forward used
    by pipeline parallelism, and recursive ``init_states`` -- and overrides the
    three members that assume RoPE:

    - ``Config.max_seq_len``: GPT-2's context length is the position table size.
    - ``Config.update_from_config``: the base implementation rewrites
      ``attention.rope.max_seq_len``, which does not exist here.
    - ``get_attention_masks``: SDPA takes an ``is_causal`` flag, not a mask
      object, so this returns ``None``. (The base implementation raises
      ``TypeError`` for any inner attention that is not Flex/Varlen.)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # Decoder.Config.max_seq_len reads the first attention layer's RoPE, so
        # the context length has to be carried explicitly instead.
        max_position_embeddings: int
        # Re-annotated (same field names) to the GPT-2 modules. The base
        # annotations say RMSNorm/Embedding; only the annotations change, the
        # dataclass fields keep their names and positions.
        tok_embeddings: TinyGPTEmbedding.Config
        norm: LayerNorm.Config

        @property
        def max_seq_len(self) -> int:
            return self.max_position_embeddings

        def update_from_config(self, *, config, **kwargs) -> None:
            """Resolve runtime knobs into the model config.

            Mirrors ``Decoder.Config.update_from_config`` minus the RoPE cache
            resize, and additionally fills in the TP sharding declarations.
            """
            from torchtitan.config import ParallelismConfig
            from torchtitan.trainer import Trainer

            assert hasattr(config, "parallelism"), (
                "config passed to update_from_config must provide "
                "a parallelism field."
            )
            parallelism: ParallelismConfig = config.parallelism
            assert isinstance(parallelism, ParallelismConfig), (
                "config.parallelism must be a ParallelismConfig, got "
                f"{type(parallelism).__name__}."
            )

            if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallel."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layers[0].attention.n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_heads ({n_heads})."
                    )
                # The MLP is sharded colwise on c_fc and rowwise on c_proj, so
                # the 4x expansion must split evenly across ranks.
                for i, layer_cfg in enumerate(self.layers):
                    tinygpt_layer = cast(TinyGPTBlock.Config, layer_cfg)
                    ffn_dim = tinygpt_layer.feed_forward.c_fc.out_features
                    if ffn_dim % tp != 0:
                        raise ValueError(
                            f"tensor_parallel_degree ({tp}) must divide the "
                            f"MLP hidden dim ({ffn_dim}) of layer {i}."
                        )

            if isinstance(config, Trainer.Config):
                seq_len = config.training.seq_len
                if seq_len > self.max_position_embeddings:
                    raise ValueError(
                        f"Training sequence length {seq_len} exceeds the "
                        f"position embedding table size "
                        f"{self.max_position_embeddings}."
                    )

            from .sharding import set_tinygpt_sharding_config

            set_tinygpt_sharding_config(
                self, enable_sp=parallelism.enable_sequence_parallel
            )

        def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, int]:
            attention = cast(TinyGPTAttention.Config, self.layers[0].attention)
            head_dim = attention.head_dim or (self.dim // attention.n_heads)
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=attention.n_heads,
                # Same convention as llama3: the qk+v head dims, doubled for
                # the two matmuls each attention performs.
                head_dims=2 * head_dim,
                seq_len=seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        if config.enable_weight_tying:
            self._retie()
            # Decoder.__init__ already tried to tie, by assigning
            # ``self.tok_embeddings.weight`` -- which on this model registers a
            # stray Parameter on the *wrapper* module (the real table lives at
            # ``tok_embeddings.tok.weight``). Drop it, or it would show up as a
            # separate untied parameter under FSDP.
            self.tok_embeddings._parameters.pop("weight", None)

    def _retie(self) -> None:
        """Point the token table at the output projection's weight."""
        self.tok_embeddings.tok.weight = self.lm_head.weight

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if self.enable_weight_tying:
            # Re-tie before init (on meta device the init-time tying is a no-op)
            # and skip Decoder.init_states, whose re-tie targets a ``.weight``
            # attribute this model does not have at the top level.
            self._retie()
            self.tok_embeddings._parameters.pop("weight", None)
        Module.init_states(self, buffer_device=buffer_device)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        # ``tok_embeddings`` is None on non-first pipeline stages.
        h = (
            self.tok_embeddings(tokens, positions)
            if self.tok_embeddings is not None
            else tokens
        )

        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h

        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h

    def get_attention_masks(self, positions: torch.Tensor) -> None:
        """SDPA applies causality internally; GPT-2 uses no other mask."""
        return None
