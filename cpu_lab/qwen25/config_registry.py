"""Qwen2.5-0.5B training configs for this CPU-only host.

Same contract as ``cpu_lab/qwen35/config_registry.py`` -- the configs live here
so nothing under ``torchtitan/`` changes -- but the budgets are recomputed for
494M parameters instead of 853M, which is the point of the switch::

                     Qwen3.5-0.8B   Qwen2.5-0.5B
    parameters       852,985,920    493,962,112
    bf16 params          1.59 GiB       0.92 GiB
    bf16 grads           1.59           0.92
    bf16 Adam m+v        3.18           1.84
    --------------------------------------------
    fixed state          6.36 GiB       3.68 GiB

The per-rank overhead that does *not* shrink with rank count scales with the
model too (FSDP's all-gather buffer is one full copy of the parameters), so it
drops from ~4.8 GiB to ~3.2 GiB. Put together, four ranks move from
"~26 GiB, does not fit" to "~13 GiB, fits", which is what buying a smaller
model actually buys here.

Launch through the launcher so the CPU patches are installed first::

    python cpu_lab/run_cpu_train.py --combo qwen25 -n 4 --threads 2 \\
        --module cpu_lab.qwen25 --config qwen25_0_5b_bf16_multi --stagger 5

Choices that are about this host:

- ``CrossEntropyLoss``, not ``ChunkedLossWrapper``: chunking's "keep lm_head
  unsharded" FSDP path faults on CPU-only torch.
- ``activation_checkpoint=None``: at these sequence lengths the activations are
  not the peak, so AC would only cost time.
- ``attn_backend="flex"`` is not a preference -- torchtitan removed ``sdpa`` for
  language models and ``varlen`` is CUDA-only, so flex plus the eager replacement
  in ``cpu_lab/qwen25_cpu.py`` is the only path that runs.
"""

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CommConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import ChatDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from qwen25 import model_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

# Tokenizer + config.json, downloaded to the repo so the run needs no network.
# Absolute: the config must not depend on the process's working directory.
HF_ASSETS = os.path.join(REPO_ROOT, "cpu_lab", "hf25")

# Vendored single-turn SFT pairs (question/answer). No network.
SFT_JSON = os.path.join(REPO_ROOT, "tests", "assets", "sft_test", "data.json")

# Written by cpu_lab/convert_qwen25_hf.py.
HF_CHECKPOINT_DCP = os.path.join(REPO_ROOT, "cpu_lab", "checkpoints", "qwen25_0_5b_hf")

# Written by cpu_lab/convert_qwen25_hf.py --untied. Required by PP, which
# cannot tie a tensor that lives on two stages at once.
HF_CHECKPOINT_DCP_UNTIED = HF_CHECKPOINT_DCP + "_untied"


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
    weight_tying: bool = True,
    dtype: str = "float32",
    mixed_precision_param: str = "float32",
    mixed_precision_reduce: str = "float32",
    fsdp_reshard_after_forward: str | None = None,
    comm: CommConfig | None = None,
    enable_checkpoint: bool = False,
) -> Trainer.Config:
    """Shared Trainer.Config skeleton for the Qwen2.5 CPU runs.

    ``enable_checkpoint`` is deliberately separate from ``save``, because
    ``CheckpointManager`` gates *loading* on ``enable`` too: with
    ``enable=False`` it returns before ever reading ``initial_load_path``
    (``torchtitan/components/checkpoint.py``), so a config that sets
    ``initial_load_path`` but leaves ``enable`` off silently trains from random
    initialisation. That is invisible in the logs -- no error, no warning --
    and only shows up as a loss that starts at ~12 instead of ~2.
    """
    model_spec = model_registry("0.5B", attn_backend="flex", weight_tying=weight_tying)
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_spec)),
        model_spec=model_spec,
        hf_assets_path=HF_ASSETS,
        dataloader=_chat_dataloader(),
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
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            **(
                {"fsdp_reshard_after_forward": fsdp_reshard_after_forward}
                if fsdp_reshard_after_forward is not None
                else {}
            ),
        ),
        checkpoint=CheckpointManager.Config(
            enable=save or enable_checkpoint,
            interval=steps,
            last_save_model_only=True,
            initial_load_path=initial_load_path,
            initial_load_model_only=True,
        ),
        activation_checkpoint=None,
        **({"comm": comm} if comm is not None else {}),
    )


def qwen25_0_5b_smoke() -> Trainer.Config:
    """Two steps from random init: does the stack run on CPU at all?

    Random init on purpose -- this asks whether the Trainer, the patched
    attention, the dataloader and the optimiser survive a full step, not whether
    the weights are any good. Skipping the 494M-parameter load keeps it cheap.
    """
    return _config(seq_len=128, local_batch_size=1, steps=2, lr=1e-5, save=False)


def qwen25_0_5b_bf16() -> Trainer.Config:
    """Full bf16: parameters, gradients and Adam state all bf16, no fp32 master.

    ``training.dtype`` is the lever that matters and it is not
    ``mixed_precision_param``. The former sets torch's default dtype for the
    whole build, so parameters are created bf16, gradients follow, and Adam's
    ``zeros_like`` state is bf16 too. The latter only changes the dtype of
    FSDP's all-gather, leaving fp32 master weights behind.
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


def qwen25_0_5b_bf16_multi() -> Trainer.Config:
    """``qwen25_0_5b_bf16`` with collective timeouts this host can meet.

    Multi-rank failures here are timeouts, not OOMs: gloo defaults to 300 s for
    init and then tightens to 100 s after step 1, while a CPU rank that is
    paging can legally spend minutes inside one all-gather. A rank that cannot
    post its matching receive in time aborts, and the abort takes every other
    rank with it.

    This is a diagnostic, not a production setting. If it gets the run through,
    the conclusion is "it fits but is paging"; if it still times out at 1800 s,
    the ranks are genuinely deadlocked and more patience will not help.
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
        comm=CommConfig(init_timeout_seconds=1800, train_timeout_seconds=1800),
    )


def qwen25_0_5b_sft_bf16() -> Trainer.Config:
    """Continue training from the real Qwen2.5-0.5B weights.

    Requires ``cpu_lab/convert_qwen25_hf.py`` to have run first.

    **The step-1 loss is the correctness check.** Random init gives roughly 12.8
    on this vocabulary; a correct load of the HF weights puts step 1 near 1.3.
    Anything in between means the conversion mapped keys onto the wrong tensors
    -- most likely the q/k/v bias, which is the one place Qwen2.5 and the Qwen3
    skeleton disagree.
    """
    return _config(
        seq_len=256,
        local_batch_size=1,
        steps=10,
        lr=5e-5,
        save=True,
        initial_load_path=HF_CHECKPOINT_DCP,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        fsdp_reshard_after_forward="always",
    )


def qwen25_0_5b_pp_bf16() -> Trainer.Config:
    """Pipeline parallel: the one dimension whose cost falls as ranks grow.

    ``weight_tying=False`` is **forced**, not chosen: torchtitan raises
    ``NotImplementedError: Weight tying is not supported with Pipeline Parallel``
    because a tied tensor would have to live on both the first and the last
    stage. Pair this with ``cpu_lab/convert_qwen25_hf.py --untied``, which
    materialises ``lm_head.weight`` as a copy of the embedding (151936 x 896,
    +0.28 GiB of bf16 state).

    ``local_batch_size=4`` so a 2- or 4-stage pipeline has more than one
    microbatch. With one microbatch every stage idles while one works, so the
    pipeline still saves memory but cannot be faster than a single rank.

    Reporting caveat: only the last stage computes the loss, so other ranks log
    a ``-1`` sentinel. ``grad_norm`` is the only value every rank shares.
    """
    return _config(
        seq_len=128,
        local_batch_size=4,
        steps=2,
        lr=1e-5,
        save=False,
        initial_load_path=HF_CHECKPOINT_DCP_UNTIED,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        fsdp_reshard_after_forward="always",
        weight_tying=False,
        comm=CommConfig(init_timeout_seconds=1800, train_timeout_seconds=1800),
        # Not "save this run" -- "turn checkpointing on at all". Without it
        # CheckpointManager never reads initial_load_path and this trains from
        # random init (see _config). Cost: one 1.2 GB write at the last step.
        enable_checkpoint=True,
    )
