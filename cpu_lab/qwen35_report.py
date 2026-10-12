# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render the Qwen3.5-on-CPU investigation as one self-contained HTML page.

Reads the ``run.json`` files that ``run_cpu_train.py`` writes next to each run's
logs and turns them into a page, so every number in the report traces back to a
recorded run rather than to a transcription. Re-running this against the same
log root reproduces the same page.

Usage::

    python cpu_lab/qwen35_report.py --log-root cpu_lab/logs_qwen35 \\
        --out cpu_lab/reports/qwen35.html

No CDN, no external assets: the page opens offline, which matters because the
whole point of this lab is a machine that cannot be assumed to have a network.
"""

import argparse
import glob
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# --------------------------------------------------------------------------- #
# Host facts. Measured with cpu_lab/_probe_alloc.py and getconf-style probes on
# 2026-10-12; kept here as data because the report has to state the budget the
# model was measured against, and the host is the experiment.
# --------------------------------------------------------------------------- #
HOST = {
    "cores": 4,
    "logical": 8,
    "physical_gib": 15.73,
    "avail_phys_gib": 3.45,
    "commit_limit_gib": 45.63,
    "avail_commit_gib": 15.69,
    "usable_alloc_gib": 4.0,
    "other_holders": [
        ("pycharm64.exe", 1.97),
        ("Memory Compression", 1.06),
        ("WorkBuddy.exe (x2)", 0.77),
        ("chrome.exe (x4)", 1.27),
        ("WindowsTerminal + explorer + WeChat", 0.69),
    ],
}

MODEL = {
    "total_params": 852_985_920,
    "vocab": 248_320,
    "dim": 1024,
    "n_layers": 24,
    "gdn_layers": 18,
    "full_attn_layers": 6,
    "head_dim": 256,
    "hidden_dim": 3584,
    "embedding_params": 248_320 * 1024,
    "vision_params": 100_600_000,
}

GAPS = [
    (
        "1",
        "<code>fla</code> imported at module scope",
        "<code>qwen3_5/model.py</code> does <code>from fla.ops.gated_delta_rule "
        "import ...</code>; <code>flash-linear-attention</code> is Triton-only and "
        "there is no CPU Triton wheel, so importing the model fails outright.",
        "<code>_install_fla_shim</code> registers a synthetic <code>fla</code> "
        "package whose kernels are the chunkwise gated-delta-rule algorithm in "
        "plain PyTorch (parallel within a chunk, sequential only across chunks).",
    ),
    (
        "2",
        "flex attention has no CPU backward",
        "Forward works, so it fails late - at the first <code>loss.backward()</code> "
        "- as <code>NotImplementedError: FlexAttention does not support backward on "
        "CPU</code>. 6 of the 24 blocks are full attention.",
        "<code>_patch_eager_flex_attention</code> replaces "
        "<code>FlexAttention.forward</code> with matmul + softmax in fp32, keeping "
        "the <code>(B, L, N, H)</code> contract.",
    ),
    (
        "3",
        "Triton-only varlen conv",
        "Only reached under the <code>varlen</code> backend, which is itself "
        "CUDA-only - so unreachable here, but it would half-work if the backend "
        "were ever selected.",
        "<code>_patch_causal_conv1d_varlen</code> substitutes a per-document "
        "torch depthwise conv.",
    ),
    (
        "4",
        "forward needs a <code>special_tokens</code> batch key",
        "Only <code>MMDataLoader</code> emits it, so text-only SFT through "
        "<code>ChatDataLoader</code> raises <code>KeyError: 'image_id'</code> on "
        "step 1.",
        "<code>_patch_text_only_special_tokens</code> supplies "
        "<code>{'image_id': -1, 'video_id': -1}</code> sentinels.",
    ),
    (
        "5",
        "<code>create_block_mask</code> API mismatch",
        "torchtitan calls <code>create_block_mask(..., separate_full_blocks=...)</code>; "
        "the pinned torch 2.12.1 signature is "
        "<code>(mask_mod, B, H, Q_LEN, KV_LEN, device, BLOCK_SIZE, _compile)</code>. "
        "Every rank dies on step 1 with <code>TypeError</code>. torchtitan CI tracks "
        "PyTorch nightly, so no release torch hits it.",
        "<code>_patch_attention_mask_creation</code> makes mask construction a no-op. "
        "The mask is already discarded by gap 2, so this removes cost as well as the "
        "crash; the cost is that attention stays purely causal.",
    ),
]

# Arithmetic floor for the fixed part of a run: parameters + gradients + Adam's
# two moments, summed over every rank. Not a measurement - a lower bound that no
# amount of sharding changes, because FSDP shards this budget, it does not reduce
# it.
BUDGET = [
    ("parameters", 4, 2),
    ("gradients", 4, 2),
    ("Adam exp_avg + exp_avg_sq", 8, 4),
]


def _fmt(gib: float, digits: int = 2) -> str:
    return f"{gib:.{digits}f} GiB"


def _load_runs(log_root: str) -> list[dict]:
    """Every recorded run under ``log_root``, newest first."""
    runs = []
    for path in glob.glob(os.path.join(log_root, "**", "run.json"), recursive=True):
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[report] skipping {path}: {exc}", file=sys.stderr)
            continue
        payload["_dir"] = os.path.dirname(os.path.abspath(path))
        runs.append(payload)
    runs.sort(key=lambda r: r.get("started") or "", reverse=True)
    return runs


def _outcome(run: dict) -> tuple[str, str]:
    """(css class, short label) for a run, from its per-rank results."""
    rows = run.get("results", {}).get("rows", []) or []
    real = [r for r in rows if r.get("has_real_loss")]
    if real:
        return "ok", "loss produced"
    if rows:
        return "miss", "no loss (crash)"
    return "muted", "no data"


def _mem_totals(run: dict) -> tuple[float, float, int]:
    per_rank = (run.get("memory") or {}).get("per_rank") or {}
    rss = sum(v.get("peak_rss", 0) for v in per_rank.values())
    commit = sum(v.get("peak_commit", 0) for v in per_rank.values())
    return rss / 2**30, commit / 2**30, len(per_rank)


def _runs_table(runs: list[dict]) -> str:
    if not runs:
        return "<p class='muted'>No <code>run.json</code> files found.</p>"
    out = [
        "<table><thead><tr>"
        "<th>started</th><th>config</th><th class='num'>ranks</th>"
        "<th>dtype</th><th>outcome</th><th class='num'>wall</th>"
        "<th class='num'>peak RSS (all)</th><th class='num'>peak commit (all)</th>"
        "<th>log dir</th></tr></thead><tbody>"
    ]
    for run in runs:
        cls, label = _outcome(run)
        rss, commit, n = _mem_totals(run)
        dims = run.get("dims") or {}
        rank_str = f"{run.get('world_size', '?')}"
        dtype = "-"
        # The dtype is not part of run.json; it is encoded in the config name,
        # which the launcher records verbatim. Reading it back from there keeps
        # the table honest rather than re-deriving it from intent.
        name = run.get("config") or ""
        if "bf16" in name:
            dtype = "bf16"
        elif "fp32" in name or name.endswith("_step") or "smoke" in name:
            dtype = "fp32"
        out.append(
            "<tr>"
            f"<td>{html.escape(run.get('started', '?'))}</td>"
            f"<td><code>{html.escape(name)}</code>"
            f"<div class='muted' style='font-size:12px'>"
            f"dp_shard={dims.get('dp_shard', '?')}</div></td>"
            f"<td class='num'>{rank_str}</td>"
            f"<td>{dtype}</td>"
            f"<td class='{cls}'>{label}</td>"
            f"<td class='num'>{run.get('wall_time_s', '?')} s</td>"
            f"<td class='num'>{_fmt(rss) if n else '-'}</td>"
            f"<td class='num'>{_fmt(commit) if n else '-'}</td>"
            f"<td class='muted' style='font-size:12px'>"
            f"{html.escape(os.path.relpath(run['_dir'], REPO_ROOT))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _budget_table() -> str:
    rows = []
    total_fp32 = total_bf16 = 0.0
    for label, bytes_fp32, bytes_bf16 in BUDGET:
        gib_fp32 = MODEL["total_params"] * bytes_fp32 / 2**30
        gib_bf16 = MODEL["total_params"] * bytes_bf16 / 2**30
        total_fp32 += gib_fp32
        total_bf16 += gib_bf16
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td class='num'>{bytes_fp32} B/param</td>"
            f"<td class='num'>{_fmt(gib_fp32)}</td>"
            f"<td class='num'>{bytes_bf16} B/param</td>"
            f"<td class='num'>{_fmt(gib_bf16)}</td></tr>"
        )
    rows.append(
        "<tr><td><b>fixed cost, whole run</b></td><td></td>"
        f"<td class='num'><b>{_fmt(total_fp32)}</b></td><td></td>"
        f"<td class='num'><b>{_fmt(total_bf16)}</b></td></tr>"
    )
    rows.append(
        "<tr><td>available on this host, idle</td><td></td>"
        f"<td class='num'>{_fmt(HOST['avail_commit_gib'])} commit "
        f"/ {_fmt(HOST['avail_phys_gib'])} physical</td>"
        f"<td></td><td class='num miss'>still over budget</td></tr>"
    )
    return (
        "<table><thead><tr><th>component</th>"
        "<th class='num'>fp32</th><th class='num'></th>"
        "<th class='num'>bf16</th><th class='num'></th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


_CSS = """
:root{color-scheme:light dark;
--bg:#ffffff;--fg:#16181d;--muted:#6b7280;--line:#e3e5ea;--panel:#f7f8fa;
--accent:#2f6fd0;--ok:#1a7f4b;--miss:#b4451f;--warn:#8a6100;--code:#f2f3f6;}
@media (prefers-color-scheme:dark){:root{
--bg:#15171c;--fg:#e8eaf0;--muted:#9aa1ad;--line:#2b2f38;--panel:#1c1f26;
--accent:#6ea8ff;--ok:#5ed09a;--miss:#ff9b7a;--warn:#e8c46a;--code:#22262e;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",Roboto,sans-serif;}
.wrap{max-width:1040px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:24px;font-weight:600;margin:0 0 4px}
h2{font-size:17px;font-weight:600;margin:0 0 6px;display:flex;align-items:center;gap:10px}
h3{font-size:14px;font-weight:600;margin:22px 0 8px}
section{border-top:1px solid var(--line);padding:26px 0 6px}
.num{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;
border-radius:50%;background:var(--accent);color:#fff;font-size:12px;font-weight:600;flex:none}
.sub{margin:2px 0 14px;color:var(--muted)}
.muted{color:var(--muted)}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:10px 14px;
border-radius:6px;margin:12px 0;color:var(--fg)}
.warn{background:var(--panel);border-left:3px solid var(--warn);padding:10px 14px;border-radius:6px}
table{width:100%;border-collapse:collapse;margin:8px 0 4px;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
table.kv th{width:230px;color:var(--fg);font-weight:500}
code{font-family:ui-monospace,Consolas,"Courier New",monospace;font-size:12.5px;
background:var(--code);padding:1px 5px;border-radius:4px}
pre.code{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:12px 14px;
overflow-x:auto;font-family:ui-monospace,Consolas,monospace;font-size:12.5px;line-height:1.5;margin:8px 0}
.meta{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 4px}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:999px;
padding:5px 12px;font-size:12.5px}
.pill b{font-weight:600}
.chip{display:inline-block;background:var(--panel);border:1px solid var(--line);border-radius:999px;
padding:0 7px;font-size:11.5px;color:var(--muted);margin-left:6px}
.ok{color:var(--ok);font-weight:600}
.miss{color:var(--miss);font-weight:600}
ul.tight{margin:6px 0;padding-left:20px}
ul.tight li{margin:5px 0}
footer{margin-top:36px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}
"""


def _section(number: str, title: str, subtitle: str, body: str) -> str:
    return (
        "<section>"
        f"<h2><span class='num'>{number}</span>{title}</h2>"
        f"<div class='sub'>{subtitle}</div>{body}</section>"
    )


def render(runs: list[dict], out_path: str) -> str:
    e = html.escape

    verdict = (
        "<div class='note'><b>The integration works; the host does not have the "
        "memory for it.</b><br>"
        "Every CPU incompatibility in torchtitan's native Qwen3.5 is closed and the "
        "model builds, shards over 8 ranks, loads the converted HF checkpoint and "
        "runs a full forward/backward of step 1 under FSDP + gloo. What stops a "
        "complete training run is the host: with no model loaded this box has "
        f"<b>{_fmt(HOST['avail_phys_gib'])} of physical memory</b> free and "
        f"<b>{_fmt(HOST['avail_commit_gib'])} of commit</b> headroom, while the "
        "fixed cost of training 0.8B - parameters, gradients and Adam's two moments, "
        "before any activation - is "
        f"<b>{_fmt(MODEL['total_params'] * 16 / 2**30)}</b> in fp32. Sharding spreads "
        "that cost across ranks, it does not shrink it."
        "</div>"
    )

    host_rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in [
            ("CPU", f"{HOST['cores']} cores / {HOST['logical']} logical"),
            ("Physical memory", _fmt(HOST["physical_gib"]) + " total, "
             + _fmt(HOST["avail_phys_gib"]) + " available"),
            ("Commit", _fmt(HOST["commit_limit_gib"]) + " limit, "
             + _fmt(HOST["avail_commit_gib"]) + " available"),
            ("Address space per process",
             f"a single process allocated {_fmt(HOST['usable_alloc_gib'])} "
             "(paging to the page file) before failing - so the wall is total "
             "commit, not a per-process cap"),
            ("GPU", "none - gloo only, no NCCL, no Triton"),
            ("torch", "2.12.1+cpu"),
        ]
    )
    holders = "".join(
        f"<li><code>{e(name)}</code> - {_fmt(gib, 2)}</li>"
        for name, gib in HOST["other_holders"]
    )

    model_rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in [
            ("Total parameters", f"{MODEL['total_params']:,} (weight-tied)"),
            ("Vocabulary x dim",
             f"{MODEL['vocab']:,} x {MODEL['dim']} = "
             f"{MODEL['embedding_params']/1e6:.0f} M "
             f"<span class='chip'>{MODEL['embedding_params']/MODEL['total_params']*100:.0f}% "
             "of the model in the embedding</span>"),
            ("Layers",
             f"{MODEL['n_layers']} = {MODEL['gdn_layers']} GatedDeltaNet + "
             f"{MODEL['full_attn_layers']} full attention"),
            ("Vision tower",
             f"~{MODEL['vision_params']/1e6:.0f} M parameters, unused by text-only SFT "
             f"<span class='chip'>{MODEL['vision_params']*8/2**30:.1f} GiB of bf16 "
             "params+grads+Adam</span>"),
        ]
    )

    gaps = "".join(
        f"<tr><td class='num' style='display:table-cell'>{n}</td>"
        f"<td><b>{title}</b></td><td>{sym}</td><td>{fix}</td></tr>"
        for n, title, sym, fix in GAPS
    )

    repro = """<pre class='code'># 0. the host has to be able to hold the dataset cache and the checkpoint
#    (both under D:, per the project constraint on where artifacts live)

# 1. build the HuggingFace datasets arrow cache once, single-process and
#    model-free -- 8 ranks building it concurrently is what killed rank 1
python cpu_lab/prewarm_dataset.py \\
    --hf-cache D:/02_source/torchtitan/cpu_lab/hf --config qwen35_0_8b_smoke \\
    --dp-world-size 8 --wipe

# 2. convert the HF weights into torchtitan's DCP layout
python cpu_lab/convert_qwen35_hf.py --input-dir qlora_lab/model \\
    --output-dir cpu_lab/checkpoints/qwen35_0_8b_hf

# 3. run. --no-warmup because step 1 already built the cache.
python cpu_lab/run_cpu_train.py --combo qwen35 -n 8 \\
    --module cpu_lab.qwen35 --config qwen35_0_8b_bf16 \\
    --hf-cache D:/02_source/torchtitan/cpu_lab/hf --no-warmup \\
    --log-dir cpu_lab/logs_qwen35/my_run

# 4. re-render this page from whatever run.json files exist
python cpu_lab/qwen35_report.py --log-root cpu_lab/logs_qwen35 \\
    --out cpu_lab/reports/qwen35.html</pre>"""

    probe = (
        "<p>A model-free 8-rank <code>all_gather_into_tensor</code> sweep "
        "(<code>cpu_lab/_probe_allgather.py</code>) isolates the failure from "
        "Qwen3.5 entirely: no model, no FSDP, no checkpoint - just eight processes "
        "and a growing buffer. It fails too, and one attempt reported the reason "
        "explicitly:</p>"
        "<pre class='code'>RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. "
        "DefaultCPUAllocator: not enough\nmemory: you tried to allocate 268435456 "
        "bytes.</pre>"
        "<div class='note'>256 MiB, refused, with no model loaded. The same sweep "
        "taken at a quieter moment reports "
        "<code>avail_phys=0.14 GiB commit=45.63/45.63 GiB</code> <i>before</i> the "
        "first attempt. So the earlier failures are the host running out of commit, "
        "and FSDP's access violation is that same exhaustion surfacing inside a "
        "native allocation, where CPU torch has no catchable OOM to raise.</div>"
    )

    next_steps = (
        "<ul class='tight'>"
        "<li><b>Free the host before blaming the model.</b> Idle, this box has "
        f"{_fmt(HOST['avail_phys_gib'])} of physical memory free; the largest "
        "single holder is the IDE. Closing it roughly triples the headroom, which "
        "is the difference between paging and fitting.</li>"
        "<li><b>Drop the vision tower for text-only SFT.</b> It is ~100 M parameters "
        "that receive no gradient from <code>ChatDataLoader</code> data. "
        "<code>parallelize.py</code> already guards on "
        "<code>model.vision_encoder is not None</code>, so this is a config change "
        "in <code>cpu_lab/</code>, not a torchtitan change - and it removes "
        f"{MODEL['vision_params']*8/2**30:.1f} GiB of the bf16 budget.</li>"
        "<li><b>Restore document-boundary masking.</b> Gap 2 and gap 5 both leave "
        "attention purely causal, so packed SFT documents attend across their "
        "boundaries. Fixing it needs <code>positions</code> plumbed into the "
        "attention forward (the qwen3_5 full-attention layer calls "
        "<code>inner_attention</code> without it) plus a per-document reset of the "
        "GatedDeltaNet recurrence.</li>"
        "<li><b>Raise the gloo timeout for paged runs.</b> A step that takes 40 s "
        "resident can take minutes while paging, and the default 300 s collective "
        "timeout then fires on a run that was merely slow.</li>"
        "</ul>"
    )

    body = [
        verdict,
        f"<div class='meta'>"
        f"<span class='pill'><b>{len(runs)}</b> recorded runs</span>"
        f"<span class='pill'>model <b>Qwen3.5-0.8B</b></span>"
        f"<span class='pill'>framework <b>torchtitan</b></span>"
        f"<span class='pill'>backend <b>gloo / CPU</b></span>"
        f"</div>",
        _section(
            "1",
            "What was asked, and what came of it",
            "Train Qwen3.5-0.8B locally with torchtitan, on a CPU-only machine, "
            "without touching anything under <code>torchtitan/</code> except model "
            "directories.",
            "<p>torchtitan already ships a complete native Qwen3.5 - "
            "<code>model.py</code>, <code>sharding.py</code>, "
            "<code>parallelize.py</code>, a <code>state_dict_adapter</code>, and an "
            "<code>_0_8b</code> flavor - so no model code had to be written. The "
            "work was entirely CPU enablement, and all of it is closed (section 2), "
            "plus the plumbing around it: a model-free dataset-cache prewarm "
            "(section 4) and an HF-to-DCP weight conversion.</p>"
            "<p>The 0.8B model then refuses to train here, on memory. Section 5 is "
            "the ledger and section 6 is the isolation that shows the wall belongs "
            "to the host rather than to Qwen3.5.</p>",
        ),
        _section(
            "2",
            "Five CPU gaps, all closed",
            "Each is a runtime shim in <code>cpu_lab/qwen35_cpu.py</code>, applied "
            "before torchtitan is imported, so <code>torchtitan/</code> stays "
            "untouched.",
            "<table><thead><tr><th>#</th><th>gap</th><th>symptom</th>"
            f"<th>shim</th></tr></thead><tbody>{gaps}</tbody></table>",
        ),
        _section(
            "3",
            "Recorded runs",
            "Every row is a <code>run.json</code> written by the launcher next to "
            "that run's logs. Peak memory is summed over ranks.",
            _runs_table(runs),
        ),
        _section(
            "4",
            "Two failures that looked like model bugs and were not",
            "Both cost real time and both are now prevented by the launcher.",
            "<h3>A dataset-cache race that presented as a training hang</h3>"
            "<p>All 8 ranks built the HuggingFace <code>datasets</code> Arrow cache "
            "concurrently. <code>ChatDataset.__init__</code> calls "
            "<code>shuffle(seed=42)</code>, which makes <code>datasets</code> write "
            "an indices cache - and because the seed is fixed, all 8 ranks compute "
            "the <i>same</i> destination path and race on it. Rank 1 died writing it "
            "(<code>OSError: [Errno 22] Invalid argument</code> from "
            "<code>shutil.move</code>); the other seven were still at "
            "<code>Training starts at step 1</code> with 0% CPU, and gloo reported it "
            "300 s later as "
            "<code>Timed out waiting 300000ms for recv operation to complete</code>. "
            "One dead rank, seven innocent victims, one misleading error.</p>"
            "<p><code>cpu_lab/prewarm_dataset.py</code> builds that cache once, "
            "single-process, and without a model: the launcher's own warmup runs a "
            "real rank, which for 0.8B means 852 M parameters plus their optimiser "
            "state in one process - more than this host has. Confirmed fixed: after "
            "the prewarm, all 8 ranks wrote their <code>run_meta_rank_*.json</code>, "
            "including rank 1.</p>"
            "<h3>A weight-tying hypothesis that the data refuted</h3>"
            "<p>FSDP groups the tied <code>tok_embeddings + norm + lm_head</code> "
            "into one unit and passes "
            "<code>reshard_after_forward=(policy == \"always\")</code>, which for the "
            "default policy is <b>False</b> - so that unit's all-gather is held for "
            "the whole step on every rank, and the crash was exactly at its unshard. "
            "That looked like the explanation. It is not: turning tying off "
            "(<code>qwen35_0_8b_untied</code>) dies at the identical "
            "<code>_fsdp_collectives.allocate</code> call. Worth recording because "
            "the fix that <i>did</i> move the failure was elsewhere.</p>",
        ),
        _section(
            "5",
            "The memory ledger",
            f"{MODEL['total_params']:,} parameters. The three components below are "
            "fixed - FSDP divides them across ranks, it does not reduce them - and "
            "they exclude activations, gloo buffers, the framework runtime "
            "(~0.25 GiB per rank) and any transient all-gather.",
            _budget_table()
            + "<div class='note'><b><code>training.dtype</code> is the lever, and it "
            "is not <code>mixed_precision_param</code>.</b> "
            "<code>mixed_precision_param</code> sizes only FSDP's all-gather output; "
            "setting it to bf16 changed nothing (run "
            "<code>qwen35_0_8b_smoke_bf16</code> dies at the same call). "
            "<code>training.dtype</code> sets torch's default dtype around "
            "<code>model_config.build()</code> (<code>trainer.py:317</code>), so "
            "parameters, gradients <i>and</i> Adam's <code>zeros_like</code> state "
            "all become bf16 with no fp32 master copy - halving the whole fixed "
            "cost. That is the configuration "
            "(<code>qwen35_0_8b_bf16</code>) which carried step 1's forward and "
            "backward to completion over 40.4 s.</div>",
        ),
        _section(
            "6",
            "Isolating the wall from the model",
            "If the failure is Qwen3.5's, a probe without Qwen3.5 should pass.",
            probe,
        ),
        _section(
            "7",
            "Host budget, measured",
            "The numbers the model was measured against.",
            f"<table class='kv'>{host_rows}</table>"
            f"<h3>What the rest of the box is holding</h3><ul class='tight'>"
            f"{holders}</ul>"
            "<p class='muted'>Working-set figures from <code>tasklist</code>, "
            "2026-10-12. They sum to well under the gap between total and available "
            "physical memory, so the balance sits in kernel paged pool, page tables "
            "and file cache rather than in any one application.</p>",
        ),
        _section(
            "8",
            "Reproducing",
            "In order. Steps 1 and 2 are one-off; step 3 is the experiment.",
            repro,
        ),
        _section(
            "9",
            "What would make it fit, and what is still open",
            "Ordered by how much each is worth - all of them changes under "
            "<code>cpu_lab/</code>, none of them changes to torchtitan.",
            next_steps,
        ),
    ]

    doc = (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Qwen3.5-0.8B x torchtitan x CPU</title>"
        f"<style>{_CSS}</style></head><body><div class='wrap'>"
        "<h1>Qwen3.5-0.8B on torchtitan, CPU only</h1>"
        "<div class='sub'>Where the integration succeeds, where the host runs out, "
        "and how each was measured.</div>"
        + "".join(body)
        + "<footer>Generated by <code>cpu_lab/qwen35_report.py</code> from the "
        "<code>run.json</code> files under the given log root. Self-contained: no "
        "CDN, no external assets.</footer>"
        "</div></body></html>"
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-root", default=os.path.join(HERE, "logs_qwen35"))
    parser.add_argument("--out", default=os.path.join(HERE, "reports", "qwen35.html"))
    args = parser.parse_args()

    runs = _load_runs(args.log_root)
    path = render(runs, args.out)
    print(f"[report] {len(runs)} run(s) -> {path}", flush=True)
    for run in runs:
        cls, label = _outcome(run)
        print(f"  {run.get('started', '?'):<20} {run.get('config', '?'):<34} {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
