"""Qwen2.5-0.5B, built on torchtitan's shipped Qwen3 dense skeleton.

torchtitan has no ``models/qwen2_5``, but Qwen2.5 is a Qwen3 dense model minus
two details, and both are already expressible in the shipped config objects:

- **Qwen3 applies RMSNorm to q and k** after the projection (``qk_norm``);
  Qwen2.5 does not. ``GQAttention.Config.qk_norm`` is optional, so pass
  ``None``. Getting this wrong is silent: the norms would be randomly
  initialised and every conversion check would still pass.
- **Qwen2.5's q/k/v projections carry a bias**; Qwen3's do not.
  ``Linear.Config`` has a ``bias`` flag but ``make_gqa_config`` does not expose
  it, so the attention config is rebuilt here with ``bias=True`` on the fused
  ``wqkv`` only -- ``o_proj`` and the three MLP matrices have no bias, matching
  the HF checkpoint.

Reusing the skeleton matters: ``parallelize_qwen3``,
``set_qwen3_sharding_config`` (called from ``Qwen3Model.Config.
update_from_config``) and ``pipeline_llm`` all stay in play, and none of them
inspects either difference.

Architecture, from ``cpu_lab/hf25/config.json``::

    dim=896  n_layers=24  n_heads=14  n_kv_heads=2  head_dim=64
    hidden_dim=4864  vocab_size=151936  eps=1e-6  rope_theta=1e6
    max_position_embeddings=32768  tie_word_embeddings=true

which is 493,962,112 parameters with the head tied -- 42% fewer than the
Qwen3.5-0.8B run, and the reason this model is the one that fits several ranks
on this host.
"""

import dataclasses
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import CosSinRoPE, Embedding, Linear, RoPE
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.qwen3 import Qwen3TransformerBlock
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = ["model_registry", "Qwen25StateDictAdapter", "qwen25_configs"]

_EPS = 1e-6

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


def _norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _output_linear_init(dim: int) -> dict:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _gqa_config(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    layer_id: int,
    inner_attention,
    rope: RoPE.Config,
    qkv_bias: bool,
) -> GQAttention.Config:
    """``make_gqa_config`` with ``qk_norm`` off and, optionally, qkv bias on."""
    cfg = make_gqa_config(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        wqkv_param_init=_LINEAR_INIT,
        wo_param_init=_depth_init(layer_id),
        inner_attention=inner_attention,
        rope=rope,
        fuse_qkv=True,
        # Qwen2.5 has no q_norm/k_norm. Qwen3 does; that is the whole delta.
        qk_norm=None,
    )
    if qkv_bias:
        wqkv = cfg.qkv_linear.wqkv
        cfg = dataclasses.replace(
            cfg,
            qkv_linear=dataclasses.replace(
                cfg.qkv_linear,
                wqkv=dataclasses.replace(wqkv, bias=True),
            ),
        )
    return cfg


def _build_qwen25_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    attn_backend: str,
    rope: RoPE.Config,
    qkv_bias: bool = True,
) -> list:
    inner_attention = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_norm(dim),
                ffn_norm=_norm(dim),
                attention=_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    layer_id=layer_id,
                    inner_attention=inner_attention,
                    rope=rope,
                    qkv_bias=qkv_bias,
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _0_5b(attn_backend: str, qkv_bias: bool = True) -> Qwen3Model.Config:
    dim = 896
    head_dim = 64
    n_layers = 24
    n_heads = 14
    n_kv_heads = 2
    hidden_dim = 4864
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen25_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_seq_len=32768,
                theta=1000000.0,
            ),
            qkv_bias=qkv_bias,
        ),
    )


qwen25_configs = {"0.5B": _0_5b}


class Qwen25StateDictAdapter(Qwen3StateDictAdapter):
    """``Qwen3StateDictAdapter`` plus the q/k/v bias keys of Qwen2.5.

    The base map covers weights only. Qwen2.5 keeps ``q_proj.bias``,
    ``k_proj.bias`` and ``v_proj.bias``, and dropping them would load a model
    whose attention is subtly wrong while still reporting full key coverage --
    the failure mode where loss starts at the random-init value and never
    recovers. ``to_hf`` needs no change: it inverts ``from_hf_map``, so the bias
    entries convert in both directions once they are listed here.
    """

    def __init__(self, model_config: Qwen3Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map.update(
            {
                "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.qkv_linear.wq.bias",
                "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.qkv_linear.wk.bias",
                "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.qkv_linear.wv.bias",
            }
        )


def model_registry(
    flavor: str = "0.5B",
    attn_backend: str = "flex",
    weight_tying: bool = True,
    qkv_bias: bool = True,
) -> ModelSpec:
    """Build the Qwen2.5-0.5B ``ModelSpec``.

    ``weight_tying`` defaults to on because ``tie_word_embeddings=true``: the
    safetensors on disk have no ``lm_head.weight``, and ``dcp.load`` refuses a
    checkpoint that is missing a key the model declares. Pass ``False`` only
    alongside the untied checkpoint from ``convert_qwen25_hf.py --untied``,
    which pipeline parallel requires (a tied tensor cannot live on both the
    first and the last stage).
    """
    config = qwen25_configs[flavor](attn_backend, qkv_bias)
    config.enable_weight_tying = weight_tying
    return ModelSpec(
        name="qwen25",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        # None for dense models, same as llama3: this hook only exists to
        # register MoE load-balancing, and Qwen2.5-0.5B has no experts.
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen25StateDictAdapter,
    )
