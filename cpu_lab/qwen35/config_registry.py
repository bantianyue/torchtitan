"""Qwen3.5-0.8B training configs that run on this CPU-only machine.

Why this module exists rather than a flavor inside ``torchtitan/models/qwen3_5``:
the shipped ``qwen35_0_8b`` config trains the *multimodal* recipe -- ``cc12m``
image-text WebDataset, ``seq_len=4096``, ``local_batch_size=4``, SelectiveAC --
none of which fits a 4-core/16 GiB CPU host. The model itself needs no changes,
so the only thing that varies is the config, and keeping it under ``cpu_lab/``
leaves ``torchtitan/`` untouched (the project constraint).

Launch it through the launcher, which installs the CPU patches first::

    python cpu_lab/run_cpu_train.py --combo qwen35 -n 8 \
        --module cpu_lab.qwen35 --config qwen35_0_8b_smoke -n 8 \
        --hf-cache D:/02_source/torchtitan/cpu_lab/hf

Choices that are about this machine, not about Qwen3.5:

- ``CrossEntropyLoss``, not ``ChunkedLossWrapper``. Chunking bounds logit memory
  but its FSDP "keep lm_head unsharded" path faults on CPU-only torch (the same
  reason ``tinygpt`` avoids it). A 248320-token vocabulary at ``seq_len=256``
  makes ~250 MB of logits, which is affordable here.
- ``mixed_precision_param="float32"`` by default: bf16 matmuls get no CPU
  speedup, and fp32 keeps the loss curve interpretable. Override with
  ``--training.mixed_precision_param bfloat16`` when memory is the binding
  constraint, which on this host it will be.
- ``activation_checkpoint=None``: it trades compute for activation memory. At
  the ``seq_len`` values used here the activations are not the peak, so AC would
  only make each step slower.
- The tokenizer is Qwen3.5's own (``MultiModalTokenizer`` + the shipped special
  tokens) pointed at the local converted model directory, so no vocabulary is
  silently substituted.
"""

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import CommConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import ChatDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import QWEN3_5_SPECIAL_TOKENS
from torchtitan.trainer import Trainer

from qwen35_cpu import build_model_spec

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

# Tokenizer + chat template, downloaded in the previous session. Absolute so the
# config does not depend on the process's working directory.
HF_ASSETS = os.path.join(REPO_ROOT, "qlora_lab", "model")

# Vendored single-turn SFT pairs (question/answer). Ten rows, no network, so a
# smoke run measures the model and nothing else.
SFT_JSON = os.path.join(REPO_ROOT, "tests", "assets", "sft_test", "data.json")

# Written by cpu_lab/convert_qwen35_hf.py: the HF weights in torchtitan's DCP
# layout, one directory, loadable via checkpoint.initial_load_path.
HF_CHECKPOINT_DCP = os.path.join(REPO_ROOT, "cpu_lab", "checkpoints", "qwen35_0_8b_hf")


def _process_sample(sample: dict) -> list[dict[str, str]]:
    """The alpaca-style pair ChatDataset expects, one user/assistant turn."""
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _chat_dataloader(*, sft_json: str = SFT_JSON) -> ChatDataLoader.Config:
    return ChatDataLoader.Config(
        dataset_path="json",
        load_dataset_kwargs={"data_files": sft_json, "split": "train"},
        sample_processor=_process_sample,
        # One rank per core already: worker processes would oversubscribe the
        # 4 physical cores 8 ways and thrash instead of feeding the model.
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def _config(
    *,
    seq_len: int,
    local_batch_size: int,
    steps: int,
    lr: float,
    save: bool,
    initial_load_path: str | None = None,
    dataloader: ChatDataLoader.Config | None = None,
    weight_tying: bool = True,
    dtype: str = "float32",
    mixed_precision_param: str = "float32",
    mixed_precision_reduce: str = "float32",
    fsdp_reshard_after_forward: str | None = None,
    comm: CommConfig | None = None,
) -> Trainer.Config:
    """Shared Trainer.Config skeleton for the Qwen3.5 CPU runs."""
    # attn_backend="flex" is the only language-model backend that survives on
    # CPU at all: "varlen" goes through torch.nn.attention.varlen (CUDA-only)
    # and flex_flash is Hopper/Blackwell-only. cpu_lab/qwen35_cpu.py replaces
    # FlexAttention's forward with an eager causal implementation, and
    # build_model_spec turns on weight tying (see its docstring).
    model_spec = build_model_spec(
        "0.8B", attn_backend="flex", weight_tying=weight_tying
    )
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_spec)),
        model_spec=model_spec,
        hf_assets_path=HF_ASSETS,
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        dataloader=dataloader if dataloader is not None else _chat_dataloader(),
        optimizer=default_adamw(lr=lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1, disable_color_printing=True),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
            dtype=dtype,
            mixed_precision_param=mixed_precision_param,
            mixed_precision_reduce=mixed_precision_reduce,
        ),
        # expert_parallel_degree=1 because Qwen3.5-0.8B is dense. Leaving
        # data_parallel_shard_degree at -1 ("all remaining ranks") is what lets
        # one config cover the single-rank and 8-rank FSDP cases.
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            **(
                {"fsdp_reshard_after_forward": fsdp_reshard_after_forward}
                if fsdp_reshard_after_forward is not None
                else {}
            ),
        ),
        # Enabled so a run leaves weights behind; disabled for smoke runs, where
        # writing 1.1B parameters would cost more time than the two steps.
        checkpoint=CheckpointManager.Config(
            enable=save,
            interval=steps,
            last_save_model_only=True,
            initial_load_path=initial_load_path,
            initial_load_model_only=True,
        ),
        activation_checkpoint=None,
        **({"comm": comm} if comm is not None else {}),
    )


def qwen35_0_8b_bf16_multi() -> Trainer.Config:
    """``qwen35_0_8b_bf16`` with collective timeouts this host can actually meet.

    Why a whole config for one field: at 2 and 4 ranks the run does not run out of
    memory, it runs out of *time*. Its measured low-water marks are 6.08 GiB of
    free commit (2 ranks) and a system snapshot showing 19.97 GiB still unused
    (4 ranks) -- comfortably more than the 1-rank run that succeeds, which
    bottoms out at 4.61 GiB. What the multi-rank runs do have is
    ``avail_phys`` pinned at 0.00 GiB: they thrash. A rank stuck waiting on a
    page-in cannot post the matching receive, gloo's 300 s
    ``unbound_buffer`` timeout fires, and the abort takes down every other rank.

    So the distinguishing question is "is this rank progressing, or wedged", and
    ``init_timeout_seconds`` is how long it gets to answer. The defaults are
    sized for accelerators: 300 s for init and the first step, then tightened to
    100 s afterwards. A paging CPU rank needs neither number -- it needs minutes
    per collective and no post-step tightening at all.

    This is a diagnostic, not a production setting. If it gets the run through,
    the     conclusion is "the job fits but is paging"; if it still times out at, say,
    1800 s, the ranks are genuinely deadlocked and more patience will not help.
    """
    return _config(
        seq_len=128,
        local_batch_size=1,
        steps=2,
        lr=1e-5,
        save=False,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        fsdp_reshard_after_forward="always",
        comm=CommConfig(
            init_timeout_seconds=1800,
            train_timeout_seconds=1800,
        ),
    )


def qwen35_0_8b_pp_bf16() -> Trainer.Config:
    """Same budget as ``qwen35_0_8b_bf16_multi``, but shaped for pipeline parallel.

    **``weight_tying=False`` is forced, not chosen.** torchtitan raises
    ``NotImplementedError: Weight tying is not supported with Pipeline Parallel``
    as soon as ``enable_weight_tying and pipeline_parallel_degree > 1``
    (``torchtitan/models/common/decoder.py``). The reason is structural: tying
    makes ``tok_embeddings.weight`` and ``lm_head.weight`` the *same* tensor,
    but PP puts the embedding on the first stage and the head on the last, and
    one tensor cannot live on two stages.

    That collides with a hard requirement of this checkpoint -- HF's
    ``tie_word_embeddings=true`` means the safetensors on disk contain no
    ``lm_head.weight`` at all, which is why every other config here ties. So a
    PP run additionally needs the *untied* DCP checkpoint produced by
    ``cpu_lab/untie_checkpoint.py``, which materialises ``lm_head.weight`` as a
    copy of the embedding. Budget for it: the model goes from 852,985,920 to
    ~1.107B parameters, i.e. bf16 state from 6.83 GiB to ~8.85 GiB.

    The second change is ``local_batch_size=4``. PP slices the
    batch into ``local_batch_size // pipeline_parallel_microbatch_size``
    microbatches, and with the default microbatch of 1 that number *is* the
    batch size. At ``local_batch_size=1`` a 4-stage pipeline therefore runs one
    microbatch: every stage idles while one works, so the pipeline is fully
    serialized -- it still saves memory, but it cannot be any faster than a
    single rank, and the P2P overhead makes it slower. Four microbatches is the
    smallest batch that lets a 2- or 4-stage pipeline actually overlap.

    The memory arithmetic that makes this the affordable way to grow the batch:
    PP cuts the *resident* state by the stage count before the batch multiplies
    the activations. At ``pp=4`` each rank owns 6 of the 24 layers, so
    4 microbatches cost roughly what 1 microbatch costs on the full model.

    Trainer enforces ``local_batch_size % pp_microbatch_size == 0``, which is
    satisfied for any microbatch <= 4 here. Raise the microbatch with
    ``--parallelism.pipeline_parallel_microbatch_size`` only if the batch also
    goes up.

    Also note the reporting caveat that comes with PP: only the last stage has
    the loss, so every other rank logs a ``-1`` sentinel. Judge a PP run by the
    last-stage rank's loss alone; ``grad_norm`` is the only value all ranks
    share.
    """
    return _config(
        seq_len=128,
        local_batch_size=4,
        steps=2,
        lr=1e-5,
        save=False,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        fsdp_reshard_after_forward="always",
        # Forced by PP; see the docstring. Pair this config with the untied
        # checkpoint from cpu_lab/untie_checkpoint.py.
        weight_tying=False,
        comm=CommConfig(
            init_timeout_seconds=1800,
            train_timeout_seconds=1800,
        ),
    )


def qwen35_0_8b_smoke() -> Trainer.Config:
    """Two steps from random init: does the stack run on CPU at all?

    Random init on purpose. The question this answers is whether torchtitan's
    Trainer, FSDP, the dataloader and the patched model survive a full
    forward/backward/optimizer step -- not whether the weights are any good. Not
    loading 1.1B parameters over gloo keeps the answer to that question cheap.
    """
    return _config(seq_len=128, local_batch_size=1, steps=2, lr=1e-5, save=False)


def qwen35_0_8b_smoke_bf16() -> Trainer.Config:
    """``qwen35_0_8b_smoke`` with a bf16 FSDP all-gather, optimiser still fp32.

    ``mixed_precision_param`` sizes only FSDP's all-gather output, so this is the
    narrowest of the memory levers: it halves the largest single transient (the
    248320 x 1024 ``tok_embeddings`` all-gather, 1.02 GB fp32 per rank) while
    leaving fp32 master weights, fp32 gradients and fp32 Adam moments untouched.

    ``mixed_precision_reduce`` is left at ``float32`` because that is the only
    value the field's annotation accepts -- tyro warns and proceeds otherwise,
    which produces a run that does not mean what its config says.

    Measured result: still dies at the same ``_fsdp_collectives.allocate`` call,
    so the all-gather buffer is not what sets the ceiling.
    """
    return _config(
        seq_len=128,
        local_batch_size=1,
        steps=2,
        lr=1e-5,
        save=False,
        mixed_precision_param="bfloat16",
    )


def qwen35_0_8b_bf16() -> Trainer.Config:
    """Full-bf16 training: parameters, gradients *and* optimiser state in bf16.

    ``training.dtype`` is the lever that matters here, and it is not the same
    knob as ``mixed_precision_param``. It sets torch's default dtype for the
    whole build (``torch.set_default_dtype`` around
    ``model_config.build()``, ``trainer.py:317``), so the parameters are created
    bf16 on the meta device, the gradients follow, and Adam's ``zeros_like``
    state is bf16 too -- with **no** fp32 master copy anywhere.

    That halves the entire fixed cost of the run, which on this host is the
    binding constraint:

                    fp32       bf16
      params        3.41 GB    1.71 GB
      gradients     3.41 GB    1.71 GB
      Adam m + v    6.83 GB    3.41 GB
      --------------------------------
      total        13.65 GB    6.83 GB   (whole run, all ranks)

    Combined with weight tying (drops a second 248320 x 1024 head) and
    ``fsdp_reshard_after_forward="always"`` (releases the tied
    ``tok_embeddings + norm + lm_head`` unit after the forward instead of
    holding ~0.5 GiB per rank for the whole step), this is the cheapest
    configuration that still trains every parameter of the 0.8B model.

    It is a memory recipe, not a speed one: CPU bf16 matmuls get no hardware
    acceleration here, so steps take longer than in fp32. Loss values are also
    noisier; the point is that the run completes at all.
    """
    return _config(
        seq_len=128,
        local_batch_size=1,
        steps=2,
        lr=1e-5,
        save=False,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        fsdp_reshard_after_forward="always",
    )


def qwen35_0_8b_untied() -> Trainer.Config:
    """``qwen35_0_8b_smoke`` with ``enable_weight_tying`` off.

    One-variable A/B for the memory wall: tying saves the 254M-parameter head
    but makes FSDP keep the ``tok_embeddings + norm + lm_head`` unit
    unresharded (~1 GiB per rank, resident for the whole step). Random init, so
    nothing has to be loaded and the only difference is the tying branch in
    ``apply_fsdp_to_decoder``.

    Measured result: dies at the same ``_fsdp_collectives.allocate`` call, so
    the tying branch is not what sets the ceiling either.
    """
    return _config(
        seq_len=128,
        local_batch_size=1,
        steps=2,
        lr=1e-5,
        save=False,
        weight_tying=False,
    )


def qwen35_0_8b_step() -> Trainer.Config:
    """One step, no checkpointing: the cheapest thing that proves CPU training."""
    return _config(seq_len=128, local_batch_size=1, steps=1, lr=1e-5, save=False)


def qwen35_0_8b_sft() -> Trainer.Config:
    """Continue training from the real Qwen3.5-0.8B weights.

    Requires ``cpu_lab/convert_qwen35_hf.py`` to have run first. ``initial_load_path``
    points at the converted DCP directory and ``initial_load_model_only=True``
    means only parameters are restored (there is no optimizer state to restore).

    ``seq_len=256`` rather than the model's 262144: with 18 of 24 layers being
    GatedDeltaNet, per-token work is small but the eager attention in
    ``qwen35_cpu`` materialises an ``L x L`` score matrix per full-attention
    layer, so sequence length is the term to watch.
    """
    return _config(
        seq_len=256,
        local_batch_size=1,
        steps=20,
        lr=5e-5,
        save=True,
        initial_load_path=HF_CHECKPOINT_DCP,
    )


def qwen35_0_8b_sft_bf16() -> Trainer.Config:
    """The SFT recipe that actually fits this host: bf16 + seq 128.

    ``qwen35_0_8b_sft`` above is the natural recipe and it does **not** fit.
    It leaves ``dtype`` at fp32, i.e. 13.65 GB of parameters + gradients +
    Adam state before any activation, against a host whose *entire* commit
    allowance is 45.63 GiB and whose desktop baseline already consumes ~30 GiB.
    Measured on the bf16 twin (seq 128, 1 rank), the low-water mark during a run
    is 4.61 GiB of free commit -- so 13.65 GB of optimizer state cannot be added
    on top of it.

    What this config changes relative to ``qwen35_0_8b_sft``, and why:

    - ``dtype="bfloat16"`` -- halves params, gradients and Adam to 6.83 GB total.
      Same lever as ``qwen35_0_8b_bf16``; see that docstring for the arithmetic.
    - ``seq_len=128`` rather than 256. Not a memory argument: the fixed state
      dominates and my eager CPU attention's score matrix is 4 MB at 256. It is
      a *time* argument -- 128 is the length the single-rank bf16 smoke actually
      measured at 64 s/step, and 20 steps at 256 is a 40-minute run on 8 cores.
    - ``steps=10``. Enough to see loss move on a 5-sample dataset and to exercise
      the checkpoint writer once, without committing to a long run on a host that
      is one allocation away from a native fault.
    - ``fsdp_reshard_after_forward="always"`` -- see ``qwen35_0_8b_bf16``.

    Everything else -- loading ``cpu_lab/checkpoints/qwen35_0_8b_hf`` produced by
    ``convert_qwen35_hf.py``, ``last_save_model_only=True`` -- is inherited from
    ``qwen35_0_8b_sft``, so a run here proves the whole chain: HF weights ->
    torchtitan DCP -> real optimiser steps -> torchtitan checkpoint on disk.
    """
    return _config(
        seq_len=128,
        local_batch_size=1,
        steps=10,
        lr=5e-5,
        save=True,
        initial_load_path=HF_CHECKPOINT_DCP,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        fsdp_reshard_after_forward="always",
    )
