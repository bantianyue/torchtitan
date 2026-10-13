"""Make torchtitan's Qwen3 skeleton train Qwen2.5 on a CPU-only torch build.

Only three of the seven ``cpu_compat``-family patches are needed, and none of
the Qwen3.5-specific ones:

- No ``fla`` shim. Qwen2.5 is a plain dense transformer; the Triton-based
  GatedDeltaNet kernels that made ``import torchtitan.models.qwen3_5`` fail
  outright are not on this path.
- No varlen conv or ``special_tokens`` patch. Those exist for the linear
  attention and the multimodal batch key respectively.

The three that remain are properties of *this host*, not of Qwen3.5, so they
are reused verbatim from ``cpu_lab/qwen35_cpu.py`` rather than duplicated:

1. **flex attention has no CPU backward.** Its forward works, so the failure is
   late -- the first ``loss.backward()`` -- and looks like a model bug:
   ``NotImplementedError: FlexAttention does not support backward on CPU``.
   Replaced with matmul + softmax.
2. **BlockMask construction is incompatible with the pinned torch.** The torch
   path calls ``create_block_mask(..., separate_full_blocks=...)``, a parameter
   torch 2.12.1 does not have, so every rank dies on step 1 with a
   ``TypeError``.
3. **``initial_load_path`` is validated with a POSIX-only test**, so a Windows
   ``D:\\...`` path is refused during config parsing, before anything is built.

Why flex at all rather than sdpa: torchtitan removed ``sdpa`` for language
models (``get_attention_config`` raises), and ``varlen`` goes through
``torch.nn.attention.varlen``, which is CUDA-only. Flex plus patch 1 is the only
backend that runs here.

Known limitation, inherited and harmless for what these runs measure: the eager
attention honours causality but ignores ``attention_masks``, so tokens can
attend across packed-document boundaries. On Qwen3.5 that was unavoidable
because the GatedDeltaNet layers cannot honour document boundaries on CPU
either. Here it is avoidable in principle -- the mask could be applied -- but
the runs below are for parallelism and memory behaviour, where cross-document
attention changes the loss value and not the sharding or the collectives.
"""

_APPLIED = False


def apply() -> None:
    """Install the three host-level patches. Safe to call more than once."""
    global _APPLIED
    if _APPLIED:
        return

    # Private by convention, not by contract: these three patch
    # torchtitan.models.common (attention + decoder) and the checkpoint config,
    # none of which is Qwen3.5-specific. Importing the module is also what makes
    # the reuse visible -- a second copy would drift.
    import qwen35_cpu

    attn = qwen35_cpu._patch_eager_flex_attention()
    masks = qwen35_cpu._patch_attention_mask_creation()
    ckpt = qwen35_cpu._patch_checkpoint_path_validation()

    _APPLIED = True
    print(
        f"[qwen25-cpu] ready: attention={attn}, masks={masks}, checkpoint={ckpt}",
        flush=True,
    )
