# cpu_lab

> **Moving to a new machine?** Start with
> [`../README_NEW_MACHINE.md`](../README_NEW_MACHINE.md) (task goals, current
> progress, verification ladder, next steps), then run
> [`bootstrap.py`](bootstrap.py). The full rebuild recipe is in
> [`MIGRATION.md`](MIGRATION.md) — the repo is committed without venvs,
> weights or checkpoints, and that script puts them back.

A harness for running [TinyGPT](../torchtitan/models/tinygpt) with real
distributed collectives on a **CPU-only** torch build, without modifying
anything under `torchtitan/`.

It also trains **Qwen3.5-0.8B** on CPU — see
[Training Qwen3.5-0.8B on CPU](#training-qwen35-08b-on-cpu).

```bash
python cpu_lab/run_cpu_train.py --list
python cpu_lab/run_cpu_train.py --combo single              # 1 rank baseline
python cpu_lab/run_cpu_train.py --combo fsdp    -n 8        # FSDP-8
python cpu_lab/run_cpu_train.py --combo tp      -n 8        # TP-8
python cpu_lab/run_cpu_train.py --combo tp_fsdp -n 8        # TP-2 x FSDP-4
python cpu_lab/run_cpu_train.py --combo tp_sp   -n 8        # TP-2 x FSDP-4 + SP

python cpu_lab/run_cpu_train.py --combo single -n 1 --config tinygpt_smoke --steps 3 --show-logs
python cpu_lab/run_cpu_train.py --combo fsdp -n 8 --dry-run   # config check only
```

The full pipeline — train, count the collectives, prove the checkpoint learned
something, and write an HTML report — is one command:

```bash
python cpu_lab/planted_corpus.py                                  # once
python cpu_lab/run_cpu_train.py --combo pp_tp_fsdp -n 8 --sp false \
    --collectives --probe --dataset-path cpu_lab/planted --steps 60
# -> cpu_lab/reports/pp_tp_fsdp.html
```

`--report` is on by default, so every run leaves a report. See
[The HTML report](#the-html-report), [Measuring the
communication](#measuring-the-communication-collectives) and [Proving the
training did something](#proving-the-training-did-something-probe).

Ranks are real OS processes, the process group is real gloo, and FSDP really
shards — this is not a simulation. `--dry-run` switches to torchtitan's
`fake_backend`, which keeps `world_size=N` but performs no communication.

## Files

| File | Responsibility |
|---|---|
| `run_cpu_train.py` | Launcher: resolves a strategy preset into `--parallelism.*` flags, spawns N ranks, aggregates the results |
| `cpu_lab_child.py` | Per-rank entry point; installs the patches, then runs `torchtitan.train` |
| `cpu_compat.py` | Patches that make the stack **run** on CPU, with a comment per patch explaining the failure it works around |
| `run_logging.py` | Patches that make a run **legible**: the per-rank `RUN CONFIG` banner, the `RUN SUMMARY` block, and `run_meta_rank_<r>.json` |
| `no_libuv_patch.py` | Forces `TCPStore(use_libuv=False)` |
| `collectives.py` | Counts the collectives a run actually performs (calls + bytes per rank, per step), for `--collectives` |
| `planted_corpus.py` | Writes a corpus with two made-up secrets in it, plus the manifest of what to ask for, for `--probe` |
| `probe_inference.py` | Loads a checkpoint, asks it for the planted secrets, and compares against random-init weights |
| `report.py` | Renders one run's artifacts into a single self-contained HTML page |
| `scaling.py` | Runs the scaling ladder (below): fits measured rungs to get the largest model this machine can train, and renders `reports/scaling.html` |
| `hostmem.py` | Host RSS/commit sampling without psutil, for the launcher's memory column and for `scaling.py` |
| `qwen35_cpu.py` | The six patches that make **upstream** `torchtitan/models/qwen3_5` trainable on CPU (see [below](#training-qwen35-08b-on-cpu)) |
| `qwen35/config_registry.py` | Qwen3.5 CPU training recipes; pick one with `--module cpu_lab.qwen35 --config <name>` |
| `convert_qwen35_hf.py` | One-off `Qwen/Qwen3.5-0.8B` HF checkpoint → torchtitan DCP converter, with key-coverage and shape reports |
| `prewarm_dataset.py` | Builds the HF `datasets` cache in **one** process before the ranks launch, for a race the launcher cannot avoid (see below) |
| `qwen35_report.py` | Renders every Qwen3.5 run under a log root into one HTML page |

## What a run leaves behind

```
cpu_lab/logs/
  rank_0.log … rank_7.log        per-rank stdout: RUN CONFIG banner, per-step metrics, RUN SUMMARY
  summary.txt                    one aggregated result table for the run
  run.json                       the same run as data, for report.py
  run_meta_rank_<r>.json         per-rank mesh coordinates, pipeline stages, data, outputs
  collectives_rank_<r>.json      only with --collectives
  probe.json                     only with --probe
  _warmup.log                    output of the dataset-cache warmup step
cpu_lab/outputs/<combo>/
  checkpoint/step-N/             model weights (one .distcp shard per rank) + .metadata
  structured_logs/               per-rank JSONL trace of build/step timings
cpu_lab/reports/<combo>.html     the report for the last run of that preset
cpu_lab/planted/data.json        the planted corpus, for --dataset-path cpu_lab/planted
cpu_lab/probe_manifest.json      prompts/expected secrets, recorded as real tokenizer ids
```

**`RUN CONFIG`** (top of every rank log) answers "what strategy am I running":
the configured degrees with a note on what each one splits, the resolved meshes
and *this rank's coordinate* in each, which pipeline stages this rank holds and
whether it is a first/last stage, the dataset file being read, the batch/seq
arithmetic, and the output folders.

**`RUN SUMMARY`** (end of every rank log) answers "what did it produce": steps
completed, tokens seen, last loss and grad norm, wall time, and a listing of the
files on disk.

**`summary.txt`** aggregates all ranks into one table, flagging the ranks that
reported a sentinel rather than a loss, and reduces the run to two checkable
facts: the number of distinct loss values among last-stage ranks, and the number
of distinct grad norms across all ranks. Both should be 1.

Colour is disabled in the logs (a redirected stdout would otherwise be full of
ANSI escapes), and the parser tolerates colour in case it is turned back on.

## Why this exists

`torchtitan` targets a CUDA cluster. A CPU-only torch build fails in six
distinct places before the training loop starts. All six are fixed at runtime in
`cpu_compat.py`; nothing under `torchtitan/` is touched.

| # | Failure | Patch |
|---|---|---|
| 1 | `torch._dynamo` / `torchtitan...minimal_async_ep.kernels` import triton at module scope; triton has no CPU wheel. Separately, `models/common/attention.py` calls `torch.compile` at import, dragging in inductor, which subclasses triton internals. | A meta-path finder answers any `triton.*` import with a permissive placeholder, and `torch.compile` degrades to eager. |
| 2 | `torchtitan.tools.utils` does `device_type = _get_available_device_type() or "cuda"`. The probe only checks accelerators, so a CPU build has no path to `"cpu"` and later dies in `device_module.set_device()`. | Wrap the probe (`torch._utils._get_available_device_type`) so it reports `"cpu"`. This one patch also fixes `DeviceMesh`, and backend selection — `Backend.default_device_backend_map["cpu"] == "gloo"`. |
| 3 | Official Windows CPU wheels are built without libuv, but `TCPStore` requests it by default, so *any* rendezvous — including `torchrun` — fails with `DistStoreError: use_libuv was requested but PyTorch was built without libuv support`. `USE_LIBUV=0` is not honoured by the C++ layer on these builds. | `no_libuv_patch` subclasses `TCPStore` to pass `use_libuv=False` explicitly. |
| 4 | `torch.distributed.barrier(device_ids=[device_module.current_device()])` — on CPU `current_device()` returns the string `"cpu"`, which pybind rejects. `torch.distributed.set_timeout` also no longer exists. | Strip `device_ids` on CPU; provide `set_timeout` from `_set_pg_timeout`. |
| 5 | `DeviceMemoryMonitor` calls `get_device_name` / `memory_stats` / `reset_peak_memory_stats` / `empty_cache`, none of which exist on `torch.cpu`. | Substitute a host-RSS monitor with the same interface. |
| 6 | FSDP's gradient division builds `ReduceOp.PREMUL_SUM`, a fused NCCL op; gloo raises `Cannot use ReduceOp.PREMUL_SUM with Gloo`. | After FSDP wraps the model, call `set_force_sum_reduction_for_comms(True)`. Numerically identical — the 1/N scaling still happens. |
| 7 | torchtitan drives the pipeline with `pp_schedule.step(arg_mbs=..., kwarg_mbs=..., target_mbs=..., losses=..., loss_kwargs=..., return_outputs=False)`. In torch 2.12 every `step` signature is `step(*args, target=None, losses=None, return_outputs=True, **kwargs)` — a *whole-batch* entry point. `arg_mbs` lands in `**kwargs`, is treated as model input, and the run dies with `Expecting N arg_mbs but got 1`. | The microbatches-level entry point still exists and still takes exactly those arguments: `_step_microbatches(arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs)`. A per-instance `step` adapter calls it directly, replaying the two bits of bookkeeping the whole-batch `step` would do (`has_backward` per stage, runtime-state reset). `loss_kwargs` has no slot there either, so it is bound into the loss function instead. |

Two further issues belong to the launcher rather than the patch layer:

- **`torchrun` is unusable** (patch 3), so `run_cpu_train.py` spawns plain
  subprocesses with `RANK` / `LOCAL_RANK` / `WORLD_SIZE` set by hand.
- **Concurrent dataset-cache locking.** HuggingFace `datasets` guards its
  processed cache with lock files, and on this machine removing them can stall.
  Three mitigations are applied: the cache is built **once, single-process,
  before any rank starts** (`_prewarm_dataset`, skippable with `--no-warmup`),
  all ranks share that one cache, and rank startups are staggered
  (`--stagger`, default 5s). Without them a run intermittently hangs in
  `Preparing c4_test dataset` and the ranks that *did* start then die with a
  gloo collective timeout — which looks like a training bug but is not.
  Offline mode is forced, since everything used here is vendored.

  Symptom to look for when diagnosing this: some ranks never reach
  `Training starts at step 1`. Check that before concluding anything about the
  model or the strategy. (On a machine without a file-deletion guard this does
  not happen at all; it is the guard's global lock that wedges, not `datasets`.)

- **The warmup is a run, so it needs the same clean start.** It used to be
  pointed at `cpu_lab/outputs/_warmup` without clearing it, so from the *second*
  run onwards it resumed and died with `Missing key in checkpoint state_dict:
  dataloader.dp_rank_0`. The launcher treats a failed warmup as non-fatal (it is
  an optimisation, not a requirement), so the failure was silent — and the
  visible symptom arrived much later and looked unrelated: with no warm cache,
  every rank built it concurrently and intermittently wedged, which is exactly
  the failure the warmup exists to prevent. The warmup now clears its own dump
  folder and runs with `--checkpoint.no-enable`. A one-line check after any
  change here: run the same command twice and confirm **both** print
  `dataset cache ready`.

## Strategy presets

`world_size = dp_replicate x dp_shard x cp x tp x pp`, forwarded as
`--parallelism.*` flags. The preset only chooses the numbers — the model and the
training config are strategy-agnostic, which is the point.

Working here: `single`, `fsdp`, `tp`, `tp_sp`, `tp_fsdp`, `pp`, `pp_tp`,
`pp_fsdp`, `pp_tp_fsdp` (with `--sp false`), `hsdp` (up to 4 ranks).

Blocked by this environment: `cp`, `tp_cp` (CPU SDPA has no DTensor sharding
strategy). Caveats on `pp_tp_fsdp` and `hsdp` are printed by `--list` and are
explained in [TinyGPT's README](../torchtitan/models/tinygpt/README.md).

## Hybrid parallelism on this host (measured 2026-10-13)

The presets above were validated with **TinyGPT**. Running them against
**Qwen3.5-0.8B** turns up three constraints that are about the *model*, not the
machine, and they cut the hybrid space down a lot:

| Constraint | Where it bites | Consequence |
|---|---|---|
| **PP cannot tie embeddings** | `torchtitan/models/common/decoder.py` raises `NotImplementedError` as soon as `enable_weight_tying and pp > 1` | Structural, not a bug: tying makes `tok_embeddings.weight` and `lm_head.weight` the same tensor, but PP puts the embedding on stage 0 and the head on the last stage |
| **TP is capped at 2** | `tensor_parallel_degree (4) must divide n_kv_heads (2)` | Qwen3.5-0.8B has 2 KV heads, so `tp>2` is impossible at any rank count |
| **TP x FSDP hits a stride bug** | `view size is not compatible with input tensor's size and stride` | Reproduced at `tp=2 x dp_shard=2`; still open |

### Ordering the dimensions by what they cost

On CPU the goal is *fitting*, not speed — 2 ranks measured **slower** than 1
(140 s vs 64 s per step) because gloo ships 1.71 GiB per all-gather over TCP on
4 cores. So rank the dimensions by traffic, cheapest first:

| Dimension | What moves per step | Shards params/grads/Adam? |
|---|---|---|
| **PP** | activations across stage boundaries (P2P) | yes, 1/pp — and no full-sized all-gather buffer |
| **TP** | activations, one all-reduce per layer | yes, 1/tp |
| **FSDP** | **full params all-gather + grads reduce-scatter** | yes, but each rank carries a ~4.8 GiB transient buffer regardless of rank count |

Which is why the memory models diverge so sharply at `bf16`:

```
pure FSDP, N ranks :  6.83 GiB + 4.8 GiB x N      -> N=4: 26 GiB   (too big here)
PP=4, dp_shard=1   :  6.83/4 GiB per rank + ~0.5  ->       ~8.8 GiB
```

### Recipes

```bash
# 0. Fit check: single rank, bf16. This is the baseline everything else is
#    compared against (64 s/step, peak RSS 7.69 GiB).
python cpu_lab/run_cpu_train.py --combo qwen35 -n 1 --threads 4 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_bf16 \
    --hf-cache cpu_lab/hf --no-warmup

# 1. Shard the data, 2 ranks. Verified working with the relaxed timeouts.
python cpu_lab/run_cpu_train.py --combo qwen35 -n 2 --threads 3 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_bf16_multi \
    --hf-cache cpu_lab/hf --no-warmup --stagger 5

# 2. Hybrid shard + replicate, 2x2. Cheaper than dp_shard=4 because the
#    all-reduce only crosses the two replica groups.
python cpu_lab/run_cpu_train.py --combo hsdp -n 4 --threads 2 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_bf16_multi \
    --hf-cache cpu_lab/hf --no-warmup --stagger 5

# 3. Tensor parallel (capped at 2 here), optionally over FSDP.
python cpu_lab/run_cpu_train.py --combo tp      -n 2 --threads 4 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_bf16_multi \
    --hf-cache cpu_lab/hf --no-warmup
python cpu_lab/run_cpu_train.py --combo tp_fsdp -n 4 --threads 2 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_bf16_multi \
    --hf-cache cpu_lab/hf --no-warmup --stagger 5

# 4. Pipeline parallel -- needs the UNTIED checkpoint first, see below.
python cpu_lab/convert_qwen35_hf.py --untied \
    --output-dir cpu_lab/checkpoints/qwen35_0_8b_hf_untied
python cpu_lab/run_cpu_train.py --combo pp -n 2 --threads 4 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_pp_bf16 \
    --checkpoint.initial-load-path cpu_lab/checkpoints/qwen35_0_8b_hf_untied \
    --hf-cache cpu_lab/hf --no-warmup --stagger 5
```

`--untied` exists only because of the PP/tying collision: it materialises
`lm_head.weight` as a copy of the embedding (that is exactly what tying does at
runtime, just written out). It costs 852,985,920 -> ~1.107B parameters, i.e.
bf16 state 6.83 -> ~8.85 GiB. `qwen35_0_8b_pp_bf16` sets `weight_tying=False`
to match, and `local_batch_size=4` so a 2- or 4-stage pipeline has more than one
microbatch to overlap (at batch 1 a pipeline is fully serialised).

### `--dry-run` cannot validate multi-rank runs

Worth knowing before trusting one: `fake_backend` performs **no** communication.
Consequences measured here:

- **PP always dies under `--dry-run`.** Stage initialisation exchanges shape
  metadata over P2P (`_recv_meta`), so with no real P2P the received shape is
  garbage — observed as `tried to allocate 4517329193108106637 bytes` or
  `_pickle.UnpicklingError: invalid load key`.
- **TP / FSDP / HSDP "succeed" with `nan`.** The all-reduce and all-gather are
  no-ops, so the run is numerically meaningless: `Loss is not finite
  (global_avg_loss=nan) at step 2`.

Use `--dry-run` to confirm that a config parses and a mesh builds. Anything
numerical, and anything under PP, needs a real run. (Fixing this needed
`NGPU` to be exported for the child: torchtitan reads the mesh's world size from
`NGPU`, not from `WORLD_SIZE`, so every `--dry-run` with `-n > 1` used to fail
with `NGPU environment variable must be set`.)

## Reading the results

```bash
# per-rank logs
ls cpu_lab/logs/
grep "step:" cpu_lab/logs/rank_0.log

# the aggregated result table for the last run
type cpu_lab\logs\summary.txt        # cmd
cat cpu_lab/logs/summary.txt         # bash

# what strategy ran, per rank
grep -A30 "^RUN CONFIG" cpu_lab/logs/rank_4.log

# TP does not shard the batch, so every rank must report the same loss
for i in 0 1 2 3 4 5 6 7; do grep "step:  1" cpu_lab/logs/rank_$i.log; done

# the whole run as one HTML page
start cpu_lab\reports\pp_tp_fsdp.html      # cmd
open cpu_lab/reports/pp_tp_fsdp.html       # macOS
```

Losses are not comparable across strategies: with `local_batch_size=1`, FSDP-8
trains on a global batch of 8 while TP-8 trains on 1. Compare a strategy against
the `single` baseline only after matching the global batch.

## The HTML report

`--report` is **on by default**, so every run writes `cpu_lab/reports/<combo>.html`
— one self-contained file (inline CSS, plain-JS canvas chart, no CDN, opens from
`file://`) covering:

1. the exact command, the interpreter, and the `torchtitan` flags the preset expanded into;
2. the parallelism degrees with what each axis splits, the resolved meshes, the dataset
   path and contents, and the batch/seq arithmetic;
3. the collectives the run really issued, per step and in bytes (see below);
4. the loss curve, the per-rank table (with the PP sentinel flagged), the correctness
   checks, and where the weights landed;
5. the inference probe, when `--probe` was used.

Re-render it any time without re-running training:

```bash
python cpu_lab/report.py --log-dir cpu_lab/logs --combo pp_tp_fsdp
```

The report reads only what the run wrote (`run.json`, `run_meta_rank_*.json`,
`collectives_rank_*.json`, `probe.json`), so it never becomes a second source of
truth that can disagree with the logs.

## Measuring the communication (`--collectives`)

The interesting question for a parallel strategy is not "how fast" but "what does
it actually send". `--collectives` answers it with measured numbers, not theory,
by installing two probes before torchtitan is imported:

- a `TorchDispatchMode` over the `_c10d_functional.*` / `c10d.*` collectives;
- Python wrappers over `torch.distributed.*` plus a `P2POp.__new__` hook.

Two probes are needed because no single hook sees everything, and each finding
here changed the code:

| Finding | Consequence |
|---|---|
| FSDP2 gathers with `dist.all_gather_into_tensor` and scatters with `dist.reduce_scatter_tensor` | the caller is a **Python** entry point, not an aten op |
| `dist.reduce_scatter_tensor` then dispatches into aten `c10d._reduce_scatter_base_` | without suppressing the nested op, one collective is counted twice |
| FSDP also calls `fsdp.all_gather_copy_in` the same number of times | it is a **local** staging copy; counting it would inflate FSDP all-gather bytes by ~50% |
| PP does **not** call `dist.batch_isend_irecv` — torch's stages use `dist.isend`/`dist.irecv` directly | `isend`/`irecv` cannot be wrapped (`P2POp` checks `op not in [isend, irecv]` by identity, so a wrapper breaks PP with `Invalid op`); counted via `P2POp.__new__` instead |
| PP's gradient-norm sync calls plain `torch.distributed.all_reduce` | a dispatcher-only probe reports PP as having **no** all-reduce at all |
| `dist.all_gather_into_tensor` may take its payload as a keyword argument | a positional-only rule reads 0 bytes; there is a kwarg fallback, and anything still unmatched lands in `unknown_payload` rather than being reported as zero traffic |
| a 1-rank run still reports a per-step `broadcast` | it is **checkpoint I/O**, not training: DCP's `scatter_object_list` is implemented as a broadcast from rank 0. Ops are therefore classified by the phase they were issued in — `step`, `init` or `checkpoint` — and the report says so when a row is entirely save-time. `CPU_LAB_COLLECTIVES_STACK=1` additionally records the first call stack per op, which is how this one was identified rather than guessed |

Measured on 2 ranks (so the numbers are readable; scale by the degree), **per
rank per step**, counting only what happens inside the training steps:

| Strategy | all-gather | reduce-scatter | all-reduce | P2P |
|---|---|---|---|---|
| `fsdp` (dp_shard=2) | 9 calls, 3.74 MB | 5 calls, 4.33 MB | 4 calls, 24 B | — |
| `tp` (tp=2, SP off) | — | — | 12 calls, 591 KB | — |
| `tp` (tp=2, **SP on**) | 18 calls, 590 KB | 18 calls, 1.18 MB | 22 calls, 142 KB | — |
| `pp` (pp=2) | — | — | 1 call, 4 B | 26 send + 26 recv, 6.29 MB each way |
| `pp_tp_fsdp` (pp2×tp2×shard2, 8 ranks) | 5 calls, 13.7 MB | 5 calls, 27.0 MB | 77 calls, 18.0 MB | 24 send + 24 recv, 6.0 MB each way |

Reading them: FSDP all-gathers roughly **twice** the sharded parameter size per
step, because parameters are re-gathered for the backward pass after being freed;
the all-reduce is 12 bytes — a 3-element fp32 scalar, the gradient norm — which is
how you can tell "sync" traffic from "real" traffic at a glance. With sequence
parallelism on, TP trades all-reduce for all-gather + reduce-scatter; the
all-reduce count does not drop to zero, because the sharded norms still need one.
PP moves the largest volume by far, and the send/recv byte counts match to 0.7 B
in 6.29 MB, which is the sanity check that both directions are measured.

In the 3D combination all three families appear at once, and the totals are
per-rank: 4.15 GB over 60 steps on ranks 0–3 vs 3.95 GB on ranks 4–7. The
difference is real and expected — the two halves hold different pipeline stages
(rank 4+ owns the last stage, so it has the `lm_head` and one extra branch) — and
it is visible only because every rank dumps its own counters. Step 1 is also not
step 2: 85 all-reduce and 10 all-gather calls versus 76 and 5, the one-off cost of
the first forward.

Ops seen **outside** the training steps are reported separately, split into
`init` (module init, FSDP's parameter broadcast, process-group setup) and
`checkpoint` (save-time collectives). Both are one-off or per-interval costs, so
folding them into a per-step average would misread a startup spike as
steady-state bandwidth — and on a 1-rank run the *only* collective in the whole
process is the checkpoint save's broadcast.

## Proving the training did something (`--probe`)

A falling loss is weak evidence — it is equally consistent with learning byte
frequencies. `planted_corpus.py` writes a corpus containing two invented launch
codes, each anchored to its own prefix, and `--probe` then asks the trained
weights (and a random-init model of the same architecture) for them:

```
python cpu_lab/planted_corpus.py                      # writes data.json + probe_manifest.json
python cpu_lab/run_cpu_train.py --combo single -n 1 --config tinygpt_smoke \
    --seq-len 64 --local-batch-size 8 --steps 300 \
    --dataset-path cpu_lab/planted --probe --collectives
```

Measured after 300 single-rank steps (`single`, `tinygpt_smoke`, seq_len 64):

| | random init | trained |
|---|---|---|
| `p("ZORBAXQUILL7741" \| "the secret launch code is")` | 1.5e-56 | **0.914** |
| per-token accuracy on the secret | 0.13 | **1.00** |
| greedy output | `sssssssssssssss` | **`ZORBAXQUILL7741`** |

The same probe on the **8-rank PP×TP×FSDP** checkpoint (60 steps, 8L512_untied):

| | random init | trained |
|---|---|---|
| p(secret) A | 8.6e-51 | **0.608** |
| p(secret) B | 2.8e-49 | **0.845** |
| per-token accuracy A / B | 0.00 / 0.00 | **1.00 / 1.00** |
| greedy A | `ope he lo room upon lo…` | **`ZORBAXQUILL7741`** (exact) |
| greedy B | `ope he lo tea lo he lo…` | **`MIRVANTHOS9932`** (exact) |

Two secrets, both reproduced exactly, and the corpus is the only place those
strings exist. It also doubles as a checkpoint test: the probe runs in a
**separate process with no distributed setup**, which works because DCP merges
the eight per-rank shards — so a wrong load, a shape mismatch or a broken shard
would show up here rather than being taken on faith.

### Two ways to get a false negative from this probe

Both were hit while building it; both look like a weak checkpoint rather than a
flawed probe.

1. **Do not re-point `lm_head` at the embedding.** For a *tied* model the two
   are the same tensor, so re-tying after a load is a harmless no-op. Every PP
   config uses an **untied** flavor (a tied parameter cannot live on two pipeline
   stages), and there the same line replaces the trained output head with the
   embedding matrix: the model then emits noise (`\xda\xda\xda…`) while the load
   reports success. The probe checks `enable_weight_tying` instead of assuming.
2. **Include BOS in the prompt.** The corpus feeds each document as
   `[BOS] + text` with positions from 0, so a probe that starts the prompt at
   position 0 asks the question one position earlier than anything the model saw.
   On the 8-rank checkpoint that alone was the difference between 0.87/0.93 and
   1.00/1.00 accuracy — i.e. between "learned the shape of the code" and "knows
   the code".

`planted_corpus.py` keeps its manifest *outside* the dataset directory on purpose:
`load_dataset(<dir>)` globs everything under the directory and infers a single
schema, so a `manifest.json` sitting next to `data.json` makes the load fail with
`column names don't match`.

## How big a model fits (`scaling.py`)

The flavours in `torchtitan/models/tinygpt/config_registry.py` (`scale_12L768`,
`scale_16L1024`, `scale_24L1536`, `scale_32L2048`; all built by `_scale(n_layers,
dim)` with `dim = n_layers * 64`, untied) are the ladder's *independent
variable*. The recipe is to run one strategy at several sizes with everything
else pinned (`seq_len=128`, `local_batch_size=8`) and sample each rank's peak
memory, then fit and extrapolate:

```bash
python cpu_lab/run_cpu_train.py --combo pp_tp_fsdp -n 8 --sp false --steps 3 \
    --config scale_12L768 --no-report \
    --log-dir cpu_lab/logs_scale/scale_12L768 \
    --hf-cache "$TEMP/cpu_lab_hf"
python cpu_lab/scaling.py          # -> cpu_lab/reports/scaling.html
```

`--log-dir` is what makes a rung a rung: one directory per point, and
`scaling.py` reads them all. Add `--no-warmup` above ~200M parameters — the
warmup instantiates the *same* model single-process for one step, which at 687M
costs more than the measurement it precedes.

Measured here (16 GB RAM, 8 cores, CPU-only torch 2.12.1 + gloo):

| flavor | params | s/step | peak RSS/rank | peak commit/rank | commit, all 8 ranks |
|---|---|---|---|---|---|
| `8L512_untied` | 27.5M | 6.5 | 0.55 GiB | 0.66 GiB | 5.16 GiB |
| `scale_12L768` | 88.5M | 15.0 | 0.77 GiB | 0.94 GiB | 7.44 GiB |
| `scale_16L1024` | 206.1M | 36.5 | 1.00 GiB | 1.50 GiB | 12.01 GiB |
| `scale_24L1536` | 687M | — | — | — | **crashed in FSDP's all-gather allocation** |

Three fits come out, and the interesting one is not "bytes per parameter":

- **commit** `41.2 B/param x N + 4.08 GiB` — summed over the 8 ranks. The fixed
  term is *per rank* (python + torch + dataloader ~0.5 GiB, times 8), and at
  these sizes it dominates the parameter term. That is the whole finding: the
  ceiling is moved much further by using fewer ranks than by shrinking the model.
- **resident** `20.3 B/param x N + 3.35 GiB` — physical RAM, the limit that
  cannot be stretched.
- **time** `0.170 s per M param + 1.1 s` per step. Whatever fits, this decides
  whether it is *usable* — 2048 tokens/step makes 206M parameters about
  56 tokens/s.

Three traps, each of which changed the code:

1. `local_params` in `run_meta_<rank>.json` is `numel()` on a DTensor, and a
   DTensor answers with its **global** shape. Summing it over
   pp2 x tp2 x dp_shard2 therefore gives `4N`, not `N`. `scaling.py` parses
   torchtitan's own `size: N total parameters` line instead; that line is printed
   before any sharding. (The sum is kept only as a fallback, and the report says
   which source it used.)
2. The commit budget must not be sampled live — whatever is running at report
   time inflates `commit_used`, and a rung in flight can drive the budget to
   zero. Use the `system_after` snapshot each run already records, and take the
   *smallest* across rungs: that is "this machine with training stopped".
3. A rung that fails leaves no `run.json`, so it would drop silently out of the
   study — and it is usually the most informative point. `collect_failed()`
   picks up directories that have rank logs but no `run.json` and reports the
   signature that stopped them. Note that a large failed allocation on Windows
   does not always raise `MemoryError`: at 687M parameters the last-stage ranks
   died with `Windows fatal exception: access violation` inside
   `fsdp_collectives.py allocate`, while the first-stage ranks sat at "Training
   starts" waiting for partners that would never arrive.

## Training Qwen3.5-0.8B on CPU

Upstream torchtitan already ships a complete, production-grade Qwen3.5
implementation — `torchtitan/models/qwen3_5/` (865-line `model.py`, vision tower,
`state_dict_adapter`, and a `0.8B` flavor). So "train Qwen3.5 with torchtitan"
needs **no new model code**. What it needs is one compatibility layer, because the
implementation is written against CUDA:

| # | Gap | Patch |
|---|---|---|
| 1 | `model.py` imports `fla.ops.gated_delta_rule` at module scope; `flash-linear-attention` has no CPU wheel | synthetic `fla` module in `sys.modules`, with pure-torch chunked gated-delta-rule and causal-conv1d kernels |
| 2 | `flex_attention` has **no CPU backward** (`NotImplementedError`) | `FlexAttention.forward` → fp32 matmul + softmax causal (elementwise `score_mod` unsupported: raises) |
| 3 | varlen conv is Triton-only | per-document torch depthwise conv |
| 4 | the forward reads `batch["special_tokens"]`, which only `MMDataLoader` emits | substitute `{"image_id": -1, "video_id": -1}` for text-only batches |
| 5 | `create_block_mask(..., separate_full_blocks=...)` is a nightly kwarg absent from pinned torch 2.12.1 | skip BlockMask construction entirely; `get_attention_masks` returns `{}` |
| 6 | `initial_load_path` validation is POSIX-only (`startswith("/")`), so `D:\...` is rejected | re-validate with `os.path.isabs` |

None of these touch `torchtitan/`; all are installed by `qwen35_cpu.apply()`
before `torchtitan.train` runs.

### The one setting that decides whether it fits

`TrainingConfig.dtype` (not `mixed_precision_*`) is the lever. Per the field's own
docstring, `training_dtype=bfloat16` puts **parameters, gradients and optimizer
state** in bf16 with no fp32 master copy. For 852,985,920 parameters:

```
fp32: 3.41 GB params + 3.41 GB grads + 6.83 GB Adam = 13.65 GB
bf16: 1.71 GB params + 1.71 GB grads + 3.41 GB Adam =  6.83 GB
```

`mixed_precision_param` alone is **not** enough — it only sizes the all-gather
transient, so params/grads/Adam stay fp32. Measured: an 8-rank run with bf16
mixed precision but fp32 dtype crashed identically to the fp32 one.

### What runs today

```
# random init, 2 steps -- does the stack work at all?
./.venv/Scripts/python.exe cpu_lab/run_cpu_train.py --combo qwen35 -n 1 --threads 4 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_bf16 \
    --hf-cache D:/02_source/torchtitan/cpu_lab/hf \
    --no-warmup --log-dir cpu_lab/logs_qwen35/smoke --show-logs

# real HF weights -> 10 SFT steps -> torchtitan checkpoint
./.venv/Scripts/python.exe cpu_lab/run_cpu_train.py --combo qwen35 -n 1 --threads 4 \
    --module cpu_lab.qwen35 --config qwen35_0_8b_sft_bf16 \
    --hf-cache D:/02_source/torchtitan/cpu_lab/hf \
    --no-warmup --log-dir cpu_lab/logs_qwen35/sft --show-logs
```

The second run proves the whole chain and is the reference result:

```
step:  1  loss: 1.33863  grad_norm: 145.0000  memory: 0.38GiB  tps: 1
step: 10  loss: 0.24605  grad_norm:  41.2500  memory: 0.38GiB  tps: 2
```

Loss near 1.3 on step 1 (versus 12.8 from random init) is the tell that the
converted weights are the real pretrained ones.

### Making multi-rank FSDP actually run

It does run — at **2 ranks**, with the collective timeouts relaxed. Verified:

```
step:  1  loss: 12.94714  grad_norm: 69.5000  memory: 0.38GiB(2.41%)
step:  2  loss: 12.84149  grad_norm: 77.5000
distinct loss values 1   (last-stage ranks agree)
distinct grad_norms  1   (clipping synced)
peak  RSS 7.37 GiB / commit 16.32 GiB total
```

Three changes, in order of effect.

**1. Relax the collective timeouts — this is the one that mattered.**

```bash
--comm.init_timeout_seconds 1800 --comm.train_timeout_seconds 1800
```

torchtitan defaults to **300 s** for init and the first step, then *tightens* to
**100 s** afterwards. Both are sized for accelerators, where a collective that
has not completed in 100 s is a hung rank. On a paging CPU rank a single
all-gather legitimately takes minutes, and the post-step tightening guarantees a
failure on every subsequent step.

This is what was killing the multi-rank runs. At 2 ranks with the default 300 s,
rank 1 timed out waiting to send; with 1800 s the same run completed both steps
with matching losses and grad norms. The memory was never the immediate cause.

**2. Use 2 ranks, not 8.**

FSDP shards parameters, gradients and optimizer state — and nothing else. Each
rank additionally carries roughly **1.1 GiB it cannot shard**: the Python + torch
runtime (~0.5 GiB) plus the all-gather output and the flat gradient buffer
(~0.6 GiB at bf16). That term is multiplied by the rank count, so on this
machine more ranks is strictly worse. Measured total commit:

| ranks | total peak commit | system low-water (commit free) | result |
|---|---|---|---|
| 1 | 11.68 GiB | 4.61 GiB | completed |
| 2 | 16.32 GiB | 2.08 GiB | completed (with 1800 s timeouts) |
| 4 | 18.88 GiB | 0.47 GiB | `access violation` |

Two ranks is the sweet spot. Four is at the ceiling and dies there; eight is not
reachable.

**3. Keep `training.dtype="bfloat16"`.** See the section above. `mixed_precision_param`
alone leaves params/grads/Adam in fp32 and changes nothing.

**A note on `expandable_segments`.** `PYTORCH_CPU_ALLOC_CONF=expandable_segments:True`
is accepted silently by torch 2.12.1 but has **no effect** on CPU — the option is
CUDA-only. An A/B allocation-churn probe measured identical commit with it on and off.
Do not expect it to help here.

### What more RAM would and would not fix

Keep the two budgets separate:

- **Physical memory decides whether it is *usable*.** This host has 15.73 GiB with
  ~4.5 GiB available, against a ~7.4 GiB resident need at 4 ranks — so it thrashes,
  the ranks miss their collective deadlines, and everything dies. At 32 GiB the 8-rank
  resident need (~15 GiB) fits outright, the thrashing stops, and both the timeouts
  and the native faults go away. **This is the actual fix.**
- **Commit memory decides whether it *hard-faults*.** Commit limit = physical + pagefile,
  currently 45.63 GiB. The pagefile is on **C:, which has only 11.13 GiB free**. On a
  32 GiB machine with a system-managed pagefile on a nearly full C: the limit lands
  around 62 GiB — sufficient, but **a pagefile is not a substitute for RAM**: it adds
  no physical memory, it just lets the thrashing continue for longer.

Recommendation if upgrading: put a **fixed-size 32 GiB pagefile on D:** (510 GiB free)
instead of leaving it system-managed on C:.

Note that even at 8 ranks this stays a demonstration, not a training setup: measured
throughput is ~2 tokens/s at 1 rank, so an 0.8B model on 8 CPU cores is orders of
magnitude away from anything practical.

### Two traps worth knowing before extending this

**The dataset cache race.** `shuffle(seed=42)` derives the `cache-*.arrow`
filename from the dataset content, not the rank, so all 8 ranks target the *same*
destination and one loses with `OSError: [Errno 22] Invalid argument`. Symptom:
all ranks print "Training starts at step 1" and then nothing. Fix:
`prewarm_dataset.py` builds the cache in one process first.

**A patch that silently fails to install.** If the module gate in
`cpu_lab_child.py` misses the `--module` name, `config_registry` fails to import
`fla`, and `ConfigManager._load_config`'s `except ImportError: continue` swallows
it — the only symptom is `Config function '...' not found ... Available config
functions: []`. A missing patch does not announce itself.

### Reading a run: the system low-water mark

`run.json`'s `memory.system_low` records the *minimum* free commit and available
physical memory observed while the ranks ran. Per-rank peaks only answer "how big
is this job"; they cannot answer "does it fit", because the host is shared and the
baseline moves. The same job dies on a high baseline and survives on a low one,
and only the low-water mark distinguishes the two. It is what turned this
investigation around: three runs had been read as "a memory wall in the forward",
when the forward was innocent.

```bash
CPU_LAB_ENTRY=<module>   # replace the child entry point (for debugging)
--threads N              # OMP threads per rank (default 2; lower it if the
                         # machine has few cores and 8 ranks contend)
--stagger N              # seconds between rank launches (default 5). Raise it
                         # if a run hangs in 'Preparing c4_test dataset'.
--no-warmup              # skip the single-process dataset-cache warmup
--keep-outputs           # do NOT clear cpu_lab/outputs/<combo> first
--master-port N          # if a previous run left the rendezvous port busy
--seq-len N              # override --training.seq_len
--steps N                # override --training.steps
--local-batch-size N     # override --training.local_batch_size (per rank)
--dataset-path DIR       # override --dataloader.dataset-path
--collectives            # count collectives into collectives_rank_*.json
--probe                  # score the checkpoint against the planted secrets
--no-report              # skip the HTML report
```

`cpu_lab/logs/`, `cpu_lab/outputs/` and `cpu_lab/.hf_cache/` are run artifacts
and can be deleted; the launcher recreates them. Delete `cpu_lab/logs` between
comparable runs: the launcher overwrites the logs for the ranks it uses but does
not remove leftovers from a run with more ranks.

The launcher clears `cpu_lab/outputs/<combo>` before each run. That is not
tidiness — torchtitan's `checkpoint.load_step` defaults to `-1` ("load the
latest"), so a second run against the same folder *resumes*, and because these
configs save model-only it would fail with `Missing key in checkpoint
state_dict: dataloader.dp_rank_0`. Starting clean also keeps runs reproducible.
Use `--keep-outputs` if you want to keep a checkpoint and resume deliberately.
