# TinyGPT

A GPT-2 style decoder sized for learning how torchtitan trains a model and how
its parallel strategies compose. The architecture follows
[`coderian/TinyGPT`](https://huggingface.co/coderian/TinyGPT) (a 4-layer,
256-hidden, 50257-vocab model trained on TinyStories on a CPU).

Run it by fully-qualified module path, so `torchtitan/models/__init__.py` does
not need to list it:

```bash
python -m torchtitan.train \
  --module torchtitan.models.tinygpt --config tinygpt_debugmodel
```

On this machine (CPU-only torch) use the harness instead, which applies the
required compatibility patches and selects a strategy:

```bash
python cpu_lab/run_cpu_train.py --combo fsdp -n 8
python cpu_lab/run_cpu_train.py --list
```

## What this model is, and where it deliberately differs

TinyGPT keeps the classic GPT-2 recipe, so it is easy to diff against the
Llama/DeepSeek implementations in this repo:

| Component | TinyGPT | llama3 / deepseek_v3 |
|---|---|---|
| Position encoding | learned absolute table (`wpe`) | RoPE |
| Normalization | `LayerNorm` (pre-LN) | `RMSNorm` |
| MLP | `Linear -> GELU(tanh) -> Linear` | SwiGLU (3 matmuls) |
| Attention | MHA, `n_kv_heads == n_heads` | GQA / MLA |
| Weight tying | optional, on by default | optional |

Two deliberate simplifications, both documented where they appear:

- **Linear biases are off.** GPT-2 puts a bias on every projection. A bias on
  the *rowwise* half of a TP pair has to be scaled by the TP degree or it is
  counted once per rank; `Linear` with `bias=False` matches what `FeedForward`
  and `GQAttention` already do in this repo, and keeps the colwise/rowwise
  pairing — the actual teaching point — free of that subtlety. LayerNorm keeps
  its affine parameters.
- **Attention uses `ScaledDotProductAttention`, not FlexAttention.** FlexAttention
  has no CPU backward kernel and needs per-document positions to build a mask;
  GPT-2 carries position information in the embedding, not in the attention.

## Files

| File | Responsibility |
|---|---|
| `model.py` | The model: `TinyGPTModel`, `TinyGPTBlock`, `TinyGPTAttention`, `TinyGPTMLP`, `TinyGPTEmbedding`, `PositionEmbedding` |
| `sharding.py` | Where every tensor lives under TP/SP — `set_tinygpt_sharding_config` |
| `parallelize.py` | The order strategies are applied in — `parallelize_tinygpt` |
| `__init__.py` | Model flavors (`tinygpt_configs`) and `model_registry` |
| `config_registry.py` | Training configs (`--config <name>`) |

### Why `TinyGPTModel` overrides three methods

`TinyGPTModel` inherits from `Decoder` to get the things that are not
architecture-specific — the `layers` ModuleDict, weight tying, the
`None`-guarded forward that pipeline parallelism relies on, and recursive
`init_states`. Three members assume RoPE and are overridden:

- `Config.max_seq_len` — the base reads `attention.rope.max_seq_len`; GPT-2's
  context length is the position table size.
- `Config.update_from_config` — the base rewrites `attention.rope.max_seq_len`
  (which does not exist here) and would raise; this version validates TP
  divisibility, rejects tying+PP, checks `seq_len` against the position table,
  and then fills in the sharding declarations.
- `get_attention_masks` — returns `None`. SDPA takes an `is_causal` flag, and
  the base would raise `TypeError` for any inner attention that is not
  Flex/Varlen.

## The TP plan

`sharding.py` is the file worth reading. Every projection is one half of a
colwise/rowwise pair, and the pairing is what makes a block cost exactly one
collective:

```
attention:   wq,wk,wv  colwise (Shard(0))  ->  wo        rowwise (Shard(1) -> all-reduce)
MLP:         c_fc      colwise              ->  c_proj    rowwise
embedding:   token table colwise (vocab split, Partial -> all-reduce)
             position table Replicate (every rank needs all positions)
lm_head:     colwise
norms:       Replicate weights, activations move between Replicate and
             sequence-sharded
```

Two details specific to this model:

1. The token embedding has a second, learned position table. The position table
   must be replicated and must **not** take the vocab-parallel path, so it gets
   its own `PositionEmbedding` module and its own declaration.
2. `LayerNorm` has a `bias` as well as a `weight`. `Module._distribute_states`
   raises if any parameter is left undeclared, so the norm config declares both
   (the RMSNorm helpers in `decoder_sharding.py` only declare `weight`).

Facts to check after a TP run, and how:

```bash
# Every rank must agree on the loss -- TP does not shard the batch.
for i in 0 1 2 3 4 5 6 7; do
  grep "step:  1" cpu_lab/logs/rank_$i.log
done
```

## Strategies: measured on this machine

`local_batch_size` is 1 for the non-PP configs, so the *global* batch depends on
the DP degree. That is why losses are not comparable across strategies — a
useful thing to notice in itself:

| Preset | Parallelism | Global batch | Result |
|---|---|---|---|
| `single` | none (1 rank) | 1 | 8.91 -> 7.92 -> 6.30 (3 steps) |
| `fsdp` | dp_shard=8 | 8 | 10.87 -> 8.30 -> 6.75 (3 steps) |
| `tp` | tp=8 | 1 | step 1 = 8.00394, step 2 = 7.54181, **identical on all 8 ranks** |
| `tp_sp` | tp=2 x dp_shard=4, SP on | 4 | 10.40 -> 8.20 |
| `tp_fsdp` | tp=2 x dp_shard=4 | 4 | 9.997 (1 step) |
| `pp` | pp=4 | 8 | last stage 7.83547 |
| `pp_tp` | pp=4 x tp=2 | 8 | last stage 8.09366 |
| `pp_fsdp` | pp=2 x dp_shard=4 | 16 | last stage 8.08705 |
| `pp_tp_fsdp` | pp=2 x tp=2 x dp_shard=2, `--sp false` | 16 | last stage 8.25261 |
| `hsdp` | dp_replicate=2 x dp_shard=2 (4 ranks) | 4 | 9.92072 |

Signals worth checking after a run:

- Under `tp`, every rank must report a **bit-identical loss** -- TP does not
  shard the batch, so a disagreement means a collective is missing.
- Under PP, only the last stage logs a real loss; other stages log the `-1`
  sentinel from `pp_forward_backward_step`. But `grad_norm` is identical across
  all ranks, because clipping is synchronised over the PP mesh.
- Losses are **not** comparable across strategies: with `local_batch_size=1`,
  FSDP-8 trains on a global batch of 8 while TP-8 trains on 1. Match the global
  batch before comparing.

### Caveats in this environment

| Preset | What happens |
|---|---|
| `pp_tp_fsdp` | Runs, but only with `--sp false`. With sequence parallelism on it reaches `Training starts at step 1` and then stops making progress. |
| `hsdp` | Runs at 4 ranks. At 8 ranks on this 8-core machine it hangs in a gloo recv timeout; not caused by thread oversubscription (`--threads 1` hangs too) or by rank startup contention (`--stagger 8` hangs too). |
| `cp`, `tp_cp` | Cannot run: context parallelism wraps K/V in DTensors, but CPU SDPA dispatches to `aten._scaled_dot_product_flash_attention_for_cpu`, which has no DTensor sharding strategy. On CUDA the SDPA path reaches flash attention, which does have one, so this is a CPU-specific gap rather than a model limitation. |

So the 4-way PP x TP x FSDP x CP combination asked for is not reachable here,
and CP is the single blocker: PP x TP x FSDP is verified working. Note that
TinyGPT has no RoPE, so CP would also have to go through the SDPA path -- there
is no FlexAttention fallback, since FlexAttention needs per-document positions
to build a mask.

Pipeline parallelism imposes two config requirements that hold regardless of
hardware: tying and PP are mutually exclusive (`Decoder.Config` raises), and
looped schedules need at least two virtual stages per rank -- which is why the PP
config uses the 8-layer flavor (a 4-layer model only has 6 effective nodes:
4 blocks + embedding + head).

## Configs

| `--config` | Model | Purpose |
|---|---|---|
| `tinygpt_debugmodel` | 4L / 128 hidden / 8 heads / 2048 vocab | default; head count 8 and ffn 512 make TP 1/2/4/8 valid |
| `tinygpt_smoke` | same, 64-token sequence, 3 steps | fail fast when trying a new combination |
| `tinygpt_pp` | 8L / 512 hidden, untied, `Interleaved1F1B` | pipeline runs |
| `tinygpt_sft` | `debugmodel` + `ChatDataLoader` | supervised fine-tuning on chat data |
| `tinygpt_reference_4l256` | 256 hidden / 50257 vocab | the HF model's shape (~29M params, ~10x slower per step on CPU) |
| `tinygpt_8l512` | 8L / 512 hidden | mid-size, for when 4 layers is too shallow to show the effect |

## What a run prints, and what it leaves on disk

Every rank log carries two blocks that make the run self-explanatory. The
harness installs them (`cpu_lab/run_logging.py`); torchtitan itself only logs
the parallelism axes as a single line of numbers.

`RUN CONFIG` — at the top, per rank:

```
--- parallelism (as configured) ---
  data_parallel_shard_degree       2  <- FSDP shard axis
  tensor_parallel_degree           2  <- splits heads + MLP
  pipeline_parallel_degree         2  <- splits layers
  enable_sequence_parallel         False
  pipeline_parallel_schedule       Interleaved1F1B
--- meshes (resolved) ---
  pp             size=2    this rank at (1,)
  tp             size=2    this rank at (0,)
--- this rank ---
  pipeline stages held             [1, 3]
  holds last pipeline stage        True
    -> this rank computes the real loss; it appears in its per-step lines
--- data ---
  dataset path                     ...\tests\assets\c4_test
  dataset contents                 data.json (4.53 MB)
  tokenizer file                   ...\tests\assets\tokenizer\tokenizer.json
  global_batch_size                16  (x batch degree 2)
  tokens per step                  16 x 128 = 2048
--- outputs ---
  checkpoint folder                ...\cpu_lab\outputs\pp_tp_fsdp\checkpoint
```

`RUN SUMMARY` — at the bottom, per rank: steps completed, tokens seen, last loss
and grad norm, wall time, and the files produced.

Plus, once, in `cpu_lab/logs/summary.txt`:

```
  rank  steps  last_step  last_loss      last_grad_norm   note
  0     3      3          -2.000000     11.0815          sentinel, not a loss
  ...
  4     3      3          7.184210      11.0815

  distinct loss values     1   (1 = all last-stage ranks agree)
  distinct grad_norms      [11.0815]   (1 value across all ranks = clipping synced)
```

Training data is the vendored `tests/assets/c4_test/data.json` (4.53 MB) read
through the tokenizer at `tests/assets/tokenizer/tokenizer.json`; both paths are
printed in the banner. No network access is used.

## Where the initial weights come from

There is **no weights directory and no pretrained checkpoint**. Weights are
drawn in-process by `init_states()` from the `param_init` specs declared in
`__init__.py`, which is why the first step's loss is high and random:

| Module | Distribution | Spec |
|---|---|---|
| `wq`/`wk`/`wv`, `c_fc` | `trunc_normal_(std=0.02)` | `_LINEAR_INIT` |
| `wo`, `c_proj` | `trunc_normal_(std=depth_scaled_std(0.02, layer_id))` | `_depth_init` — GPT-2's `1/sqrt(2*n_layer)` residual scaling |
| token table | `normal_(std=0.02)` | `_TOKEN_EMB_INIT` |
| position table | `normal_(std=0.01)` | `_POS_EMB_INIT` |
| `LayerNorm` | `ones_` / `zeros_` | `_NORM_INIT` |
| `lm_head` | `trunc_normal_(std=1/sqrt(dim), ±3σ)` | `_output_linear_init` |

Verified after `init_states()`:

```
layers.0.attention.wq.weight  std=0.02004   (declared 0.02)
tok_embeddings.pos.weight     std=0.00998   (declared 0.01)
norm.weight                   mean=1.0000   (declared ones_)
```

`ModelSpec.state_dict_adapter` is `None`, so **nothing can load the HF
`coderian/TinyGPT` weights as-is**: the two also differ (GPT-2's linear biases
are omitted here). Loading a checkpoint requires a `state_dict_adapter`, or a
checkpoint saved by this model itself via `checkpoint.initial_load_path`.

## From-scratch pretraining vs SFT

The flow is the same: same `Trainer`, same `ModelSpec`, same loss class, same
optimiser and scheduler, same parallelism flags, and **the same launch script**
(`python -m torchtitan.train --module ... --config <name>`). Everything that
differs lives in the dataloader and the checkpoint settings:

| | pretraining (`tinygpt_debugmodel`) | SFT (`tinygpt_sft`) |
|---|---|---|
| dataloader | `HuggingFaceTextDataLoader` (`c4_test`) | `ChatDataLoader` (`sft_test/data.json`, user/assistant pairs) |
| labels | every token is a target | **prompt tokens set to `IGNORE_INDEX` (-100)** |
| initial weights | random init | normally `checkpoint.initial_load_path` -> a pretrained checkpoint |
| typical lr / steps | 8e-4, many steps | 3e-4, few steps |

The masking needs no loss-side change: `cross_entropy_loss` hardcodes
`ignore_index=IGNORE_INDEX`, so prompt positions contribute no gradient. So
"train from scratch" and "SFT" are the same program with a different
`--config` — the difference is *which labels carry signal*, not *which code
runs*.

One caveat when reading `tinygpt_sft` numbers: `ChatDataset` greedily packs
documents into one sequence, and correctness then needs a *document* mask so
tokens cannot attend across document boundaries. TinyGPT's SDPA backend applies
causality only, and the flex/varlen backends that could express that mask are
exactly the ones this model cannot use (no RoPE, and no CPU backward for flex).
The flow is representative; the packed-document masking is not.

Checkpointing is **enabled** in these configs (torchtitan's default is off), so a
run leaves real weights in `{dump_folder}/{checkpoint.folder}/step-N/` — one
`.distcp` shard per rank plus a `.metadata`. The harness passes a per-strategy
`--dump-folder` so different combinations do not overwrite each other:

```
cpu_lab/outputs/<combo>/checkpoint/step-N/       weights
cpu_lab/outputs/<combo>/structured_logs/*.jsonl  per-rank build/step timing trace
```
