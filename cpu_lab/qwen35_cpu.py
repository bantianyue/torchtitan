"""Make ``torchtitan.models.qwen3_5`` run on a CPU-only torch build.

Six gaps separate the shipped Qwen3.5 implementation from a machine with no
CUDA, no triton, and a CPU that cannot differentiate through flex attention.
Each is closed here, at runtime, so nothing under ``torchtitan/`` is touched --
the same contract ``cpu_compat`` follows.

1. **``fla`` is imported at module scope.** ``qwen3_5/model.py`` does
   ``from fla.ops.gated_delta_rule import chunk_gated_delta_rule, ...``, and
   ``flash-linear-attention`` ships Triton kernels. There is no CPU wheel for
   triton, so ``import torchtitan.models.qwen3_5`` fails outright.
   ``_install_fla_shim`` registers a synthetic ``fla`` package whose kernels are
   the chunkwise gated-delta-rule algorithm in plain PyTorch -- parallel within
   chunks, sequential only across chunks -- so it is differentiable and usable
   for training, unlike the sequential recurrence in
   ``tests/unit_tests/test_qwen3_5_deltanet.py``. The math follows the
   Apache-2.0 reference in ``transformers/models/qwen3_5/modeling_qwen3_5.py``.

2. **flex attention has no CPU backward.** Its forward works, so the failure is
   late (first ``loss.backward()``) and looks like a model bug:
   ``NotImplementedError: FlexAttention does not support backward on CPU``.
   6 of Qwen3.5's 24 blocks are full attention; all 6 would be unrunnable.
   ``_patch_eager_flex_attention`` replaces ``FlexAttention.forward`` with an
   equivalent matmul/softmax implementation.

3. **The varlen conv is Triton-only.** Only reached under the ``varlen``
   attention backend (unreachable here, since ``torch.nn.attention.varlen`` is
   itself CUDA-only), but patched anyway so the model cannot half-work.

4. **The forward requires a ``special_tokens`` batch key.** Only ``MMDataLoader``
   emits it, so text-only SFT through ``ChatDataLoader`` raises
   ``KeyError: 'image_id'`` on step 1. ``_patch_text_only_special_tokens``
   supplies sentinels for that case.

5. **Flex BlockMask construction is incompatible with the pinned torch.** The
   torch path builds a document mask every step via ``create_block_mask(..., separate_full_blocks=...)``;
   torch 2.12.1's ``create_block_mask`` has no such parameter, so *every* rank
   dies on step 1 with a ``TypeError`` -- see
   ``_patch_attention_mask_creation``, which skips mask construction entirely.

6. **``initial_load_path`` is validated with a POSIX-only test.** torchtitan rejects
   the path unless it ``startswith("/")`` or looks remote, so on Windows an
   absolute ``D:\\...`` path is refused before training starts.
   ``_patch_checkpoint_path_validation`` lets the platform decide what absolute
   means.

Consequences the caller must know about (see ``cpu_lab/README.md``):

- The eager attention honours **causality only**. Under the ``flex`` backend the
  GatedDeltaNet layers already run the packed sequence as one continuous stream
  with no per-document state reset (``qwen3_5/model.py`` documents this), so
  packed documents are *not* isolated from each other regardless of what the
  attention layers do. Matching that is a deliberate choice, not an oversight.
- Everything is compute-bound differently than on GPU: the 18 GDN layers cost
  ~``L/chunk_size`` sequential chunk steps, and the 6 attention layers materialise
  a dense ``L x L`` score matrix. Both are fine at ``seq_len`` in the low
  hundreds and neither scales to 4096.
"""

import sys
import types

import torch
import torch.nn.functional as F

_APPLIED = False
_CHUNK_SIZE = 64
_FLA_MODULES = (
    "fla",
    "fla.ops",
    "fla.ops.gated_delta_rule",
    "fla.modules",
    "fla.modules.conv",
    "fla.modules.conv.causal_conv1d",
)


# --------------------------------------------------------------------------- #
# 1. Gated DeltaNet: pure-torch replacements for the FLA Triton kernels
# --------------------------------------------------------------------------- #
def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """``x / sqrt(sum(x^2) + eps)`` -- matches the FLA kernel, not F.normalize."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _as_state(x: torch.Tensor) -> torch.Tensor:
    """Move to the contiguous fp32 layout the recurrence accumulates in."""
    return x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = _CHUNK_SIZE,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gated delta rule, chunked along the sequence dimension (drop-in for FLA).

    Args (FLA layout, which is also the layout ``GatedDeltaKernel`` passes):
        query, key: ``[B, L, N_k, K]``
        value: ``[B, L, N_v, V]``
        g: ``[B, L, N_v]`` -- decay in **log** space, so entries must be <= 0
        beta: ``[B, L, N_v]`` -- the delta-rule write gate, in ``(0, 1)``

    Returns:
        ``(output [B, L, N_v, V], final_state | None)``.
        ``cu_seqlens`` (accepted through ``**kwargs``) is ignored: per-document
        state resets exist only on the CUDA varlen path, and the flex backend
        this module targets passes no such metadata. See the module docstring.
    """
    del kwargs  # cu_seqlens / cu_seqlens_cpu / output_final_state synonyms
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]

    query, key, value, beta, decay = (
        _as_state(x) for x in (query, key, value, beta, g)
    )

    # Grouped linear attention: N_k may divide N_v. The caller normally expands
    # already; doing it here keeps the shim honest if it does not.
    if key.shape[1] != value.shape[1]:
        repeat = value.shape[1] // key.shape[1]
        query = query.repeat_interleave(repeat, dim=1)
        key = key.repeat_interleave(repeat, dim=1)

    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query = query * (query.shape[-1] ** -0.5)

    pad = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad:
        query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
        beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    total_len = sequence_length + pad
    num_chunks = total_len // chunk_size

    # beta turns K/V into the "write" the delta rule applies to the state.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    )
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    # Cumulative log-decay inside a chunk, so pairwise decay is a difference.
    cum_decay = decay.cumsum(dim=3)
    strictly_upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)
    pairwise_decay = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(
        strictly_upper, float("-inf")
    ).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # UT transform: solve (I - strictly_lower)^-1 once, then the per-position
    # delta updates become two matmuls instead of a per-token loop.
    new_values = torch.linalg.solve_triangular(
        ut_system, v_beta, upper=False, unitriangular=True
    )
    k_cumdecay = torch.linalg.solve_triangular(
        ut_system, decayed_k_beta, upper=False, unitriangular=True
    )

    state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    if initial_state is None:
        state = torch.zeros(state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        state = initial_state.to(new_values.dtype)

    out = torch.zeros_like(new_values)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Sequential only across chunks. This is the whole point of the chunked
    # formulation: intra-chunk work is batched, so the loop is L/chunk_size
    # steps rather than L.
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra_chunk_attn[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    out = out.reshape(batch_size, num_v_heads, -1, v_head_dim)[:, :, :sequence_length]
    out = out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return out, (state if output_final_state else None)


def fused_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FLA's inference-only kernel. Same math, so reuse the chunked one."""
    assert beta is not None, "gated delta rule requires beta"
    return chunk_gated_delta_rule(query, key, value, g, beta, **kwargs)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    backend: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Depthwise causal conv, optionally reset at document boundaries.

    ``weight`` arrives as ``[C, K]`` (the caller squeezes the ``groups`` axis).
    """
    del bias, backend, kwargs
    kernel_size = weight.shape[-1]
    apply_activation = activation in ("silu", "swish")

    def _one_segment(x_BTD: torch.Tensor) -> torch.Tensor:
        x_BDT = F.pad(x_BTD.transpose(1, 2), [kernel_size - 1, 0])
        return F.conv1d(x_BDT, weight, None, groups=weight.size(0)).transpose(1, 2)

    offsets = None
    if cu_seqlens_cpu is not None:
        offsets = cu_seqlens_cpu.tolist()
    elif cu_seqlens is not None:
        offsets = cu_seqlens.detach().cpu().tolist()

    if offsets is None:
        out = _one_segment(x)
    else:
        # Flattened (1, T, D) layout: conv each document on its own so state
        # cannot leak across the boundary.
        segments = [
            _one_segment(x[:, start:end])
            for start, end in zip(offsets[:-1], offsets[1:], strict=False)
        ]
        out = torch.cat(segments, dim=1) if segments else x

    return (F.silu(out) if apply_activation else out), None


def _new_module(name: str, *, package: bool = False, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _install_fla_shim() -> str:
    """Register a synthetic ``fla`` package covering what qwen3_5 imports."""
    existing = sys.modules.get("fla")
    if existing is not None and getattr(existing, "_cpu_shim", False):
        return "already installed"

    _new_module("fla", package=True, _cpu_shim=True)
    _new_module("fla.ops", package=True)
    _new_module(
        "fla.ops.gated_delta_rule",
        chunk_gated_delta_rule=chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule,
    )
    _new_module("fla.modules", package=True)
    _new_module("fla.modules.conv", package=True)
    _new_module("fla.modules.conv.causal_conv1d", causal_conv1d=causal_conv1d)
    return "registered pure-torch kernels"


# --------------------------------------------------------------------------- #
# 2. Attention: eager replacement for the compiled flex attention
# --------------------------------------------------------------------------- #
def _patch_eager_flex_attention() -> str:
    """Give ``FlexAttention.forward`` a CPU-differentiable implementation."""
    from torchtitan.models.common.attention import FlexAttention

    current = FlexAttention.forward
    if getattr(current, "_cpu_eager", False):
        return "already patched"

    def eager_forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks=None,
        score_mod=None,
        scale: float | None = None,
        enable_gqa: bool = False,
        out_transform=None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Causal attention as matmul + softmax, in fp32.

        Same contract as the flex path -- ``(B, L, N, H)`` in and out -- minus
        the sparse block structure. ``attention_masks`` is accepted and ignored;
        the flex path would use it to also block cross-document attention, but
        the GatedDeltaNet layers cannot honour document boundaries on CPU
        anyway, so pretending the attention layers do would be worse than
        being explicit (see this module's docstring).
        """
        del kwargs
        if score_mod is not None:
            raise NotImplementedError(
                "cpu_lab's eager attention ignores score_mod; only the plain "
                "causal path used by Qwen3.5 is supported."
            )
        del attention_masks

        q = q_BLNH.transpose(1, 2).float()
        k = k_BLNH.transpose(1, 2).float()
        v = v_BLNH.transpose(1, 2).float()
        if enable_gqa and k.shape[1] != q.shape[1]:
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        head_dim = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) * (
            scale if scale is not None else head_dim**-0.5
        )
        q_len, kv_len = scores.shape[-2:]
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))

        lse = torch.logsumexp(scores, dim=-1)
        out_BNLH = torch.matmul(torch.softmax(scores, dim=-1), v)
        out_BLNH = out_BNLH.to(q_BLNH.dtype).transpose(1, 2)

        if out_transform is None:
            return out_BLNH
        return out_transform(out_BLNH, lse.transpose(1, 2))

    eager_forward._cpu_eager = True  # type: ignore[attr-defined]
    FlexAttention.forward = eager_forward
    return "matmul+softmax, fp32, causal"


def _patch_causal_conv1d_varlen() -> str:
    """Replace the Triton varlen conv with the per-document torch reference."""
    import torchtitan.models.qwen3_5.model as qwen35_model

    if getattr(qwen35_model._causal_conv1d_varlen, "_cpu_eager", False):
        return "already patched"

    def causal_conv1d_varlen(
        x_BTD: torch.Tensor,
        weight: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: torch.Tensor | None,
    ) -> torch.Tensor:
        out, _ = causal_conv1d(
            x=x_BTD,
            weight=weight.squeeze(1),
            activation="silu",
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        return out

    causal_conv1d_varlen._cpu_eager = True  # type: ignore[attr-defined]
    qwen35_model._causal_conv1d_varlen = causal_conv1d_varlen
    return "per-document torch conv"


def _patch_text_only_special_tokens() -> str:
    """Let Qwen35Model accept a text-only batch.

    ``_prepare_multimodal_embeds`` reads ``special_tokens["image_id"]`` before it
    checks whether any image is present, so a batch with no ``special_tokens``
    key (which is every batch ``ChatDataLoader`` produces -- only
    ``MMDataLoader``'s collator emits that key) raises ``KeyError: 'image_id'``
    on the first step.

    Substituting sentinel ids is a no-op rather than a workaround: the ids are
    only used to locate vision placeholder tokens, and they are only read when
    ``pixel_values``/``pixel_values_videos`` are not None. Sentinel ``-1`` never
    equals a real token id, so even a stray vision token would not match.
    """
    from torchtitan.models.qwen3_5.model import Qwen35Model

    if getattr(Qwen35Model._prepare_multimodal_embeds, "_cpu_text_only", False):
        return "already patched"

    original = Qwen35Model._prepare_multimodal_embeds

    def prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> torch.Tensor:
        if special_tokens is None:
            special_tokens = {"image_id": -1, "video_id": -1}
        return original(
            self,
            tokens,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            grid_thw=grid_thw,
            grid_thw_videos=grid_thw_videos,
            special_tokens=special_tokens,
        )

    prepare_multimodal_embeds._cpu_text_only = True  # type: ignore[attr-defined]
    Qwen35Model._prepare_multimodal_embeds = prepare_multimodal_embeds
    return "text-only batches accepted"


# --------------------------------------------------------------------------- #
# 4. Model spec: one flag the shipped config factory leaves off
# --------------------------------------------------------------------------- #
def build_model_spec(
    flavor: str = "0.8B", *, attn_backend: str = "flex", weight_tying: bool = True
):
    """``model_registry`` for Qwen3.5, with weight tying turned on.

    ``Qwen/Qwen3.5-0.8B/config.json`` sets ``tie_word_embeddings=true``, which is
    why the checkpoint on disk has **no** ``lm_head.weight`` at all. The shipped
    ``_0_8b`` factory leaves ``enable_weight_tying`` off with the comment that
    torchtitan does not support tied embeddings -- stale, since
    ``Decoder.Config`` implements it (``tok_embeddings.weight = lm_head.weight``,
    with FSDP grouping the two into one unit to avoid a duplicate all-gather).

    Leaving it off has two costs, one fatal and one merely large:

    - ``dcp.load`` refuses to proceed with a key the checkpoint does not contain
      (``Missing key in checkpoint state_dict: lm_head.weight``), so the HF
      checkpoint cannot be converted at all.
    - The untied head is 248320 x 1024 = 254M parameters, i.e. ~1 GiB of fp32
      weights plus a same-sized gradient and 2 GiB of Adam state on a 16 GiB
      host. Tying removes all four.

    Nothing else about the model changes; ``state_dict_adapter`` already has the
    matching branch that omits ``lm_head.weight`` when tying is on.

    ``weight_tying=False`` exists to A/B one specific hazard rather than to be
    used in production: ``apply_fsdp_to_decoder`` puts the tied
    ``tok_embeddings + norm + lm_head`` in a single FSDP unit and passes
    ``reshard_after_forward=(policy == "always")``, which for the default policy
    is **False** -- so that unit's ~1 GiB all-gather is held for the whole step
    on every rank instead of being released after the forward
    (``torchtitan/distributed/fsdp.py``). Turning tying off is the cheapest way
    to tell "the tied group is too big to keep resident" apart from "the whole
    0.8B model is too big to train here", because the untied branch reshard the
    embedding after the forward instead.
    """
    from torchtitan.models.qwen3_5 import model_registry

    spec = model_registry(flavor, attn_backend=attn_backend)
    spec.model.enable_weight_tying = weight_tying
    return spec


# --------------------------------------------------------------------------- #
# 5. Attention masks: stop torchtitan from building flex BlockMasks here
# --------------------------------------------------------------------------- #
def _patch_attention_mask_creation() -> str:
    """Make ``create_attention_mask`` a no-op that returns ``None``.

    Two independent reasons, either of which is sufficient on this host.

    **It crashes.** ``decoder._create_flex_attention_mask`` calls
    ``create_block_mask(..., separate_full_blocks=...)``, but the pinned torch
    2.12.1 signature is::

        create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN,
                          device=None, BLOCK_SIZE=128, _compile=False)

    so every rank dies identically with ``TypeError: create_block_mask() got an
    unexpected keyword argument 'separate_full_blocks'``. This is a torchtitan
    CI-tracks-PyTorch-nightly artifact, not a bug reachable on a release torch,
    which is why it has no upstream workaround to copy.

    **It would be discarded.** ``_patch_eager_flex_attention`` replaces the
    forward with matmul + softmax, which has no sparse-block kernel to consume
    the BlockMask. Building one per step would be pure cost.

    ``None`` is already a first-class value for every consumer:
    ``Decoder.get_attention_masks`` returns it whenever there are no
    full-attention layers, ``BaseAttention`` accepts it, and the Qwen3.5 layers
    read it only as "document boundaries are unknown", which is exactly the
    BlockMask's effective meaning on this path too. So this replaces an object
    nobody reads with the canonical empty object, and the compute is unchanged.

    The cost is the limitation documented at the top of this module: attention
    stays purely causal, so tokens can attend across packed-document boundaries.
    Removing that needs ``positions`` plumbed into the attention forward (the
    qwen3_5 full-attention layer calls ``inner_attention`` without it) plus a
    per-document reset of the GatedDeltaNet recurrence, which the ``cu_seqlens``
    argument of :func:`causal_conv1d` already supports on the varlen path only.
    """
    from torchtitan.models.common import attention, decoder

    def no_mask(*args: object, **kwargs: object) -> None:
        del args, kwargs
        return None

    # decoder.py line 22 does `from ...attention import create_attention_mask`,
    # so it holds its own reference. Patching only `attention` would leave
    # decoder calling the original; both are rebound here.
    attention.create_attention_mask = no_mask
    decoder.create_attention_mask = no_mask
    return "flex BlockMasks disabled (unusable on CPU)"


# --------------------------------------------------------------------------- #
# 6. Checkpoint paths: let the platform decide what "absolute" means
# --------------------------------------------------------------------------- #
def _patch_checkpoint_path_validation() -> str:
    """Accept a Windows drive-letter ``initial_load_path``.

    torchtitan's ``CheckpointManager.Config.__post_init__`` gates the field on::

        self.initial_load_path.startswith("/") or filesystem.is_remote(...)

    which is a POSIX assumption -- ``"D:\\\\..."`` and ``"D:/..."`` both fail it, so
    any Windows run that tries to resume from a converted checkpoint dies during
    config parsing, before a single parameter is built::

        ValueError: initial_load_path must be an absolute path or a remote
        URI (e.g. gs://...): D:\\02_source\\torchtitan\\cpu_lab\\checkpoints\\qwen35_0_8b_hf

    The check is inline in ``__post_init__``, so it cannot be bypassed by
    patching a helper. Instead the original is invoked with a *temporary*
    substitute: ``"/" + path`` satisfies ``startswith("/")``, is not remote
    (no scheme), and is restored in a ``finally`` so every later consumer --
    the DCP reader, the state-dict adapter, the logger -- still sees the real
    path. Nothing observes the substitute except the predicate under test.

    Gated on ``os.path.isabs`` rather than applied unconditionally: on POSIX the
    original predicate is already correct, and rewriting paths there would only
    add a difference between the two platforms.
    """
    import os.path as _ospath

    from torchtitan.components import checkpoint as _checkpoint

    target = _checkpoint.CheckpointManager.Config
    original_post_init = target.__post_init__
    if getattr(original_post_init, "_qwen35_cpu_patched", False):
        return "initial_load_path validation already patched"

    def _is_windows_absolute(path: object) -> bool:
        # "C:\x" or "C:/x": drive letter, colon, then a separator. Plain
        # "" or "\x" is left to the original logic.
        return (
            isinstance(path, str)
            and len(path) >= 3
            and path[0].isalpha()
            and path[1] == ":"
            and path[2] in "\\/"
        )

    def patched_post_init(self) -> None:
        real = self.initial_load_path
        if _is_windows_absolute(real):
            # Only the predicate sees this; restored before returning.
            self.initial_load_path = "/" + real
            try:
                original_post_init(self)
            finally:
                self.initial_load_path = real
        else:
            original_post_init(self)

    patched_post_init._qwen35_cpu_patched = True  # type: ignore[attr-defined]
    target.__post_init__ = patched_post_init  # type: ignore[method-assign]
    _drive_sample = "D:/"
    return (
        "initial_load_path accepts Windows absolute paths "
        f"(isabs={_ospath.isabs(_drive_sample)})"
    )


# --------------------------------------------------------------------------- #
def apply() -> None:
    """Install every shim. Safe to call more than once."""
    global _APPLIED
    if _APPLIED:
        return

    fla = _install_fla_shim()
    attn = _patch_eager_flex_attention()
    masks = _patch_attention_mask_creation()
    conv = _patch_causal_conv1d_varlen()
    text = _patch_text_only_special_tokens()
    ckpt = _patch_checkpoint_path_validation()

    _APPLIED = True
    print(
        f"[qwen35-cpu] ready: fla={fla}, attention={attn}, masks={masks}, "
        f"varlen-conv={conv}, batch={text}, checkpoint={ckpt}",
        flush=True,
    )
