# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training configs for TinyGPT, one per *purpose* rather than per strategy.

Parallel strategy is deliberately **not** baked into these configs. It is
selected on the command line, because that is how torchtitan works and because
it makes the mapping from "strategy" to "flag" visible::

    # single device
    --module torchtitan.models.tinygpt --config tinygpt_debugmodel

    # FSDP over every rank (data_parallel_shard_degree defaults to -1 = "all")
    ... --config tinygpt_debugmodel

    # TP over every rank (dp_shard then defaults to the remaining 8/8 = 1)
    ... --config tinygpt_debugmodel --parallelism.tensor_parallel_degree 8

    # TP x FSDP on 8 ranks
    ... --config tinygpt_debugmodel --parallelism.tensor_parallel_degree 2

    # PP x TP on 8 ranks
    ... --config tinygpt_debugmodel --parallelism.pipeline_parallel_degree 4 \
        --parallelism.tensor_parallel_degree 2

``cpu_lab/run_cpu_train.py`` wraps these flags in named presets; see
``cpu_lab/README.md``.

Pretraining and SFT are the same program here -- same trainer, same launch
command, same parallelism flags. ``tinygpt_sft`` differs from
``tinygpt_debugmodel`` only in the dataloader (chat pairs with the prompt masked
to ``IGNORE_INDEX``) and, for a real run, in pointing
``checkpoint.initial_load_path`` at a pretrained checkpoint.

What *is* fixed here, and why:

- ``CrossEntropyLoss`` rather than ``ChunkedLossWrapper``: chunking caps peak
  logit memory at scale, which this model does not need, and its FSDP
  "keep lm_head unsharded" path faults on CPU-only torch.
- ``activation_checkpoint=None``: it trades compute for activation memory, and
  a CPU model this small has no memory pressure to relieve -- it would just
  slow the step down.
- ``mixed_precision_param="float32"``: bfloat16 matmuls are not accelerated on
  CPU, and fp32 keeps the numbers easy to inspect by hand.
- Checkpointing **is** enabled (torchtitan's default is off) so a run leaves
  concrete weights behind, and colour printing is off because stdout goes to a
  per-rank log file where ANSI escapes are just noise.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import ChatDataLoader, HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry


def _config(
    flavor: str,
    *,
    seq_len: int,
    local_batch_size: int,
    steps: int,
    lr: float,
    pipeline_parallel_schedule: str | None = None,
    dataloader=None,
) -> Trainer.Config:
    """Shared Trainer.Config skeleton for every TinyGPT run."""
    model_spec = model_registry(flavor)
    parallelism = ParallelismConfig(expert_parallel_degree=1)
    if pipeline_parallel_schedule is not None:
        # torchtitan drives the schedule with the ``arg_mbs=``/``kwarg_mbs=``
        # API, which only the PipelineScheduleMulti family implements. The
        # default "1F1B" resolves to a single-stage schedule with a different
        # ``step()`` signature, so PP runs must name a looped schedule.
        parallelism.pipeline_parallel_schedule = pipeline_parallel_schedule
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_spec),
        ),
        model_spec=model_spec,
        # Vendored tokenizer + vendored c4_test split: this whole config runs
        # offline, which is also what makes it a fair baseline across parallel
        # strategies (identical data and identical step count).
        hf_assets_path="./tests/assets/tokenizer",
        dataloader=(
            dataloader
            if dataloader is not None
            else HuggingFaceTextDataLoader.Config(dataset="c4_test")
        ),
        optimizer=default_adamw(lr=lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        # log_freq=1 so every step is visible: these runs are short and the
        # per-step numbers are the point. Colour is off because stdout is
        # redirected to a per-rank log file, and ANSI escapes make those files
        # awkward to read and to grep.
        metrics=MetricsProcessor.Config(log_freq=1, disable_color_printing=True),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
        ),
        # data_parallel_shard_degree=-1 means "take whatever ranks are left
        # after tp/cp/pp". Leaving it at -1 is what lets a single config work
        # for the single-device, FSDP, and TP cases alike.
        parallelism=parallelism,
        checkpoint=CheckpointManager.Config(
            # Enabled so a run leaves something concrete on disk: the weights
            # land in {dump_folder}/{checkpoint.folder}. interval == steps means
            # only the final save fires, and last_save_model_only keeps it to a
            # single model file rather than a full optimizer/resume state.
            enable=True,
            interval=steps,
            last_save_model_only=True,
        ),
        activation_checkpoint=None,
    )


def tinygpt_debugmodel() -> Trainer.Config:
    """Default learning config: 4 layers / 128 hidden / 8 heads / 2048 vocab.

    ``n_heads=8`` and ``ffn_dim=512`` make TP degrees 1/2/4/8 valid, and
    ``n_layers=4`` makes PP degrees 1/2/4 valid, so a single config covers
    every strategy combination the launcher offers. ``max_position_embeddings``
    is 256, so ``seq_len`` can be raised (e.g. for context parallelism) without
    touching the model.
    """
    return _config(
        "debugmodel",
        seq_len=128,
        local_batch_size=1,
        steps=20,
        lr=8e-4,
    )


def tinygpt_pp() -> Trainer.Config:
    """Config for pipeline parallelism (8 layers, untied embeddings).

    Two constraints force a dedicated config here:

    1. Weight tying and PP are mutually exclusive in torchtitan -- a tied
       parameter would have to exist on both the first stage (embedding) and the
       last stage (lm_head). Hence the ``8L512_untied`` flavor.
    2. Looped schedules (``Interleaved1F1B``) require at least two virtual stages
       per rank, so the graph needs enough nodes: with ``pp_degree=4`` that is 8
       stages, and a 4-layer model only has 6 effective nodes (4 blocks +
       embedding + head). Eight layers give 10, which fits.

    ``local_batch_size`` is 8 rather than 1 because torch's pipeline schedule
    wants at least one microbatch per stage (microbatch size defaults to 1).
    ``Interleaved1F1B`` (rather than the default ``1F1B``) is required because
    torchtitan drives the schedule through the ``arg_mbs=`` API, which only the
    multi-stage schedule family implements.
    """
    return _config(
        "8L512_untied",
        seq_len=128,
        local_batch_size=8,
        steps=20,
        lr=6e-4,
        pipeline_parallel_schedule="Interleaved1F1B",
    )


def tinygpt_sft() -> Trainer.Config:
    """Supervised fine-tuning on chat data, to show how little differs.

    Compare with :func:`tinygpt_debugmodel`: same ``Trainer.Config``, same
    ``ModelSpec``, same optimiser, same loss class, same parallelism fields, and
    the same launch command (``torchtitan.train`` with a different ``--config``).
    Only the dataloader changes -- and that is where the entire SFT difference
    lives.

    ``ChatDataset`` tokenises ``[user, assistant]`` pairs, **masks the prompt
    tokens with IGNORE_INDEX**, and packs documents greedily. The masking needs
    no loss-side change because ``cross_entropy_loss`` hardcodes
    ``ignore_index=IGNORE_INDEX`` (``-100``), so those positions contribute no
    gradient.

    One caveat specific to TinyGPT: packing puts several documents in one
    sequence, and correctness then depends on a *document* mask so tokens cannot
    attend across document boundaries. TinyGPT's SDPA backend applies causality
    only (``is_causal=True``) and has no way to express that mask -- the
    flex/varlen backends that torchtitan's ``get_attention_masks`` supports are
    what a production SFT setup would use, and they are the reason this model
    cannot simply switch backends. Worth knowing before reading the numbers here
    as a real SFT result: the flow is right, the masking across packed documents
    is not.

    To actually fine-tune from a pretrained model rather than from random init,
    point ``checkpoint.initial_load_path`` at a saved checkpoint and leave
    ``initial_load_model_only=True``.
    """
    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    return _config(
        "debugmodel",
        # seq_len is capped by the flavour's learned position table (256 for
        # debugmodel), unlike RoPE models where the context can be extended.
        seq_len=128,
        local_batch_size=4,
        steps=20,
        lr=3e-4,
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": "tests/assets/sft_test/data.json",
                "split": "train",
            },
            sample_processor=process_sample,
        ),
    )


def tinygpt_smoke() -> Trainer.Config:
    """Three steps, tiny sequence -- for checking that a strategy *runs*.

    Use this first when trying a new parallelism combination: it fails fast and
    says nothing about throughput.
    """
    return _config(
        "debugmodel",
        seq_len=64,
        local_batch_size=1,
        steps=3,
        lr=8e-4,
    )


def _scale_config(flavor: str) -> Trainer.Config:
    """Same recipe for every rung of the scaling ladder.

    Identical to :func:`tinygpt_pp` except for the flavor, so a measurement
    series only varies the model size. ``local_batch_size=8`` and
    ``seq_len=128`` are held constant too: a sweep that changes several things
    at once measures their product, not the scaling.
    """
    return _config(
        flavor,
        seq_len=128,
        local_batch_size=8,
        steps=20,
        lr=6e-4,
        pipeline_parallel_schedule="Interleaved1F1B",
    )


def scale_12L768() -> Trainer.Config:
    """~88M parameters: the first rung above the 8L512 default."""
    return _scale_config("scale_12L768")


def scale_16L1024() -> Trainer.Config:
    """~206M parameters."""
    return _scale_config("scale_16L1024")


def scale_24L1536() -> Trainer.Config:
    """~687M parameters."""
    return _scale_config("scale_24L1536")


def scale_32L2048() -> Trainer.Config:
    """~1.6B parameters. Expected to be past this machine's practical limit."""
    return _scale_config("scale_32L2048")


def tinygpt_reference_4l256() -> Trainer.Config:
    """The shape of https://huggingface.co/coderian/TinyGPT.

    256 hidden / 4 layers / 8 heads / 50257-token GPT-2 vocab / 128 positions
    and tied embeddings. The 50257-way output projection makes the step ~10x
    slower on CPU than ``tinygpt_debugmodel``, so this is for inspecting the
    architecture rather than for looping over strategies.
    """
    return _config(
        "4L256",
        seq_len=128,
        local_batch_size=1,
        steps=10,
        lr=3e-4,
    )


def tinygpt_8l512() -> Trainer.Config:
    """Mid-size config, for strategies whose effect is invisible at 4 layers.

    Pipeline parallelism in particular only becomes interesting once there are
    enough layers to split, and tensor parallelism needs width to show a
    per-rank memory drop.
    """
    return _config(
        "8L512",
        seq_len=128,
        local_batch_size=1,
        steps=10,
        lr=6e-4,
    )