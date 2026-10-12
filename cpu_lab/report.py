# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render one run's artifacts into a self-contained HTML report.

Everything the report needs already exists on disk after a run:

    cpu_lab/logs/run.json                    command, config, data, results, artifacts
    cpu_lab/logs/run_meta_rank_<r>.json      per-rank meshes, pipeline stages, data
    cpu_lab/logs/collectives_rank_<r>.json   collective ops per rank (with --collectives)
    cpu_lab/logs/probe.json                  inference probe (with --probe)
    cpu_lab/logs/rank_<r>.log                raw per-rank logs

So the report is a *view*, not a second source of truth: re-running it on the
same log directory reproduces the same page, and running a different strategy
just writes a different page. No numbers are recomputed here that were not
already measured.

The output is a single file with inline CSS and plain-JS canvas charts -- no
CDN, no build step, works offline from ``file://``.
"""

from __future__ import annotations

import glob
import html
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.path.join(HERE, "logs")
DEFAULT_OUT_DIR = os.path.join(HERE, "reports")

# What each collective buys, in one line. Kept next to the expected-op table so
# a reader can tell "this strategy needs it" from "this is what it does".
WHY: dict[str, str] = {
    "all-gather": "every rank reconstructs the full tensor from the shards "
    "(FSDP: full parameters; sequence parallelism: full activations)",
    "reduce-scatter": "sum across ranks, then keep only this rank's slice "
    "(FSDP: gradient averaging + re-sharding)",
    "all-reduce": "every rank ends up with the same summed value "
    "(TP: partial products of rowwise projections; all: gradient-norm sync)",
    "broadcast": "one rank's tensor is copied to the others "
    "(FSDP broadcasts module state at init; process group setup)",
    "P2P send": "hand a tensor to one specific rank "
    "(pipeline parallelism: activations forward)",
    "P2P recv": "take a tensor from one specific rank "
    "(pipeline parallelism: activations forward, gradients backward)",
}

_AXIS_MEANING = {
    "data_parallel_replicate_degree": "replicate the model, average gradients (HSDP's H)",
    "data_parallel_shard_degree": "shard parameters, gradients and optimiser state (FSDP)",
    "context_parallel_degree": "split the sequence across ranks",
    "tensor_parallel_degree": "split weight matrices / attention heads",
    "pipeline_parallel_degree": "split the layer stack into stages",
    "expert_parallel_degree": "split MoE experts (unused here, single expert)",
}


def _human_bytes(n: float) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.2f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024
    return f"{n:,.2f} GB"


def _esc(value) -> str:
    return html.escape(str(value))


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def collect(log_dir: str = DEFAULT_LOG_DIR) -> dict:
    """Gather every artifact a run left in ``log_dir``."""
    run = _read_json(os.path.join(log_dir, "run.json")) or {}
    metas = []
    for path in sorted(glob.glob(os.path.join(log_dir, "run_meta_rank_*.json"))):
        meta = _read_json(path)
        if meta:
            metas.append(meta)
    colls = []
    for path in sorted(glob.glob(os.path.join(log_dir, "collectives_rank_*.json"))):
        data = _read_json(path)
        if data:
            colls.append((os.path.basename(path), data))
    return {
        "run": run,
        "metas": metas,
        "collectives": colls,
        "probe": _read_json(os.path.join(log_dir, "probe.json")),
        "log_dir": os.path.abspath(log_dir),
    }


# ---------------------------------------------------------------- building blocks


def _kv_table(rows: list[tuple[str, str, str]]) -> str:
    body = "".join(
        f"<tr><th>{_esc(k)}</th><td>{v}</td><td class='muted'>{_esc(note)}</td></tr>"
        for k, v, note in rows
    )
    return f"<table class='kv'>{body}</table>"


def _section(number: str, title: str, body: str, subtitle: str = "") -> str:
    sub = f"<p class='sub'>{_esc(subtitle)}</p>" if subtitle else ""
    return (
        f"<section><h2><span class='num'>{_esc(number)}</span>{_esc(title)}</h2>"
        f"{sub}{body}</section>"
    )


def _code(text: str) -> str:
    return f"<pre class='code'>{_esc(text)}</pre>"


# ------------------------------------------------------------------ 1. commands


def _section_command(data: dict) -> str:
    run = data["run"]
    rows = [
        ("interpreter", _esc(run.get("python", "")), "the project venv; other pythons on this box lack torch"),
        ("launcher", _esc(run.get("command", "")), "reproduces this exact run"),
        ("torchtitan args", _esc(run.get("child_command", "")), "what the launcher expands the preset into"),
    ]
    body = _kv_table(rows)
    body += "<h3>Copy-paste (cmd.exe)</h3>"
    body += _code(run.get("command", ""))
    body += "<h3>Expanded per-rank command</h3>"
    body += _code(run.get("child_command", ""))
    return _section("1", "执行命令", body, "这个预设展开成了哪些 torchtitan 参数")


# ------------------------------------------------------------- 2. config & data


def _dataset_files(path: str) -> str:
    """One-line listing of a dataset directory (display only)."""
    if not path or not os.path.exists(path):
        return "<not found>"
    if os.path.isfile(path):
        return _human_bytes(os.path.getsize(path))
    entries = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            entries.append(f"{name} ({_human_bytes(os.path.getsize(full))})")
    return ", ".join(entries) if entries else "<empty>"


def _effective_dataset(dl: dict, run: dict) -> tuple[str | None, bool, str | None, str]:
    """Which directory the run actually read, and whether it was overridden.

    ``--dataloader.dataset-path`` wins over the registry entry inside
    torchtitan (``_validate_dataset`` does ``dataset_path or config.path``),
    while the dataset *name* stays the registry key. Reporting the registry path
    for an overridden run would name the wrong directory.

    Run metadata written before this was tracked only carries the registry path,
    so the launcher's recorded CLI override is used as the fallback -- it is the
    same fact, just from where it was recorded.
    """
    override = (run.get("overrides") or {}).get("dataset_path")
    meta_path = dl.get("dataset_path")
    if override:
        effective = os.path.abspath(override)
        if not meta_path or os.path.abspath(meta_path) != effective:
            registry = dl.get("dataset_registry_path") or meta_path
            return effective, True, registry, _dataset_files(effective)
    return (
        meta_path,
        bool(dl.get("dataset_overridden")),
        dl.get("dataset_registry_path"),
        dl.get("dataset_contents") or _dataset_files(meta_path or ""),
    )


def _section_config(data: dict) -> str:
    run = data["run"]
    meta = data["metas"][0] if data["metas"] else {}
    dims = run.get("dims", {})
    world = run.get("world_size", "?")
    training = meta.get("training", {})
    model = meta.get("model", {})
    dl = meta.get("data", {})
    outputs = meta.get("outputs", {})
    par = meta.get("parallelism", {})

    rows = [
        ("world_size", f"<b>{_esc(world)}</b>",
         f"= dp_replicate {dims.get('dp_replicate')} x dp_shard {dims.get('dp_shard')} "
         f"x cp {dims.get('cp')} x tp {dims.get('tp')} x pp {dims.get('pp')}"),
        ("model", f"{_esc(model.get('name'))} / <b>{_esc(model.get('flavor'))}</b>",
         f"{model.get('local_params', 0):,} params materialised on this rank"),
        ("module", _esc(model.get("module")), "the model package"),
    ]
    for axis, meaning in _AXIS_MEANING.items():
        value = par.get(axis)
        if value is not None:
            rows.append((axis, f"<b>{_esc(value)}</b>", meaning))
    rows += [
        ("enable_sequence_parallel", _esc(par.get("enable_sequence_parallel")),
         "shard the sequence across TP ranks instead of replicating activations"),
        ("pipeline_parallel_schedule", _esc(par.get("pipeline_parallel_schedule")),
         "looped schedules are required: only the multi-stage family implements torchtitan's step() calling convention"),
    ]
    body = "<h3>并行策略</h3>" + _kv_table(rows)

    meshes = meta.get("meshes", {})
    if meshes:
        mesh_rows = "".join(
            f"<tr><td>{_esc(name)}</td><td>{_esc(info.get('size'))}</td>"
            f"<td class='muted'>{_esc(info.get('coord'))}</td></tr>"
            for name, info in meshes.items()
        )
        body += (
            "<h3>解析后的进程网格 (rank 0 的坐标)</h3>"
            f"<table><tr><th>mesh</th><th>size</th><th>this rank at</th></tr>{mesh_rows}</table>"
        )

    pipe = meta.get("pipeline", {})
    if pipe.get("enabled"):
        body += (
            f"<p class='note'>本 rank 持有 pipeline stage "
            f"<b>{_esc(pipe.get('stage_indices'))}</b>，schedule "
            f"<code>{_esc(pipe.get('schedule'))}</code>，"
            f"{_esc(pipe.get('microbatches'))} 个 microbatch。"
            f"是否持有最后一级：<b>{_esc(pipe.get('holds_last'))}</b>"
            + ("" if pipe.get("holds_last") else "（不含最后一级 → 这个 rank 记录的是负哨兵，不是 loss）")
            + "</p>"
        )

    dataset_path_eff, overridden, registry_path, contents = _effective_dataset(dl, run)
    data_rows = [
        ("dataloader", _esc(dl.get("dataloader")), "which loader class"),
        ("dataset name", _esc(dl.get("dataset_name")),
         "key in torchtitan's DATASETS registry (kept even when the path is overridden)"),
        ("dataset path", f"<code>{_esc(dataset_path_eff)}</code>",
         "the directory actually read"
         + (
             f" — <b>overridden</b> from the '{dl.get('dataset_name')}' registry entry "
             f"({_esc(registry_path)}) by <code>--dataloader.dataset-path</code>"
             if overridden
             else ""
         )),
        ("dataset contents", _esc(contents), "vendored and offline"),
        ("tokenizer", f"<code>{_esc(dl.get('tokenizer_file'))}</code>",
         "byte-level, vocab 2020 -- one id per character"),
        ("seq_len", _esc(training.get("seq_len")), "tokens per sequence"),
        ("local_batch_size", _esc(training.get("local_batch_size")), "sequences per rank per step"),
        ("global_batch_size", f"<b>{_esc(training.get('global_batch_size'))}</b>",
         f"local {training.get('local_batch_size')} x batch degree {training.get('batch_degree')}"),
        ("tokens per step", f"<b>{_esc(training.get('tokens_per_step'))}</b>",
         f"= {training.get('global_batch_size')} x {training.get('seq_len')}"),
        ("gradient accumulation", _esc(training.get("gradient_accumulation_steps")), "extra micro-steps per update"),
        ("steps", _esc(training.get("steps")), "requested"),
    ]
    body += "<h3>输入数据与一步的规模</h3>" + _kv_table(data_rows)

    out_rows = [
        ("dump folder", f"<code>{_esc(outputs.get('dump_folder'))}</code>", "everything this run wrote"),
        ("checkpoint folder", f"<code>{_esc(outputs.get('checkpoint_folder'))}</code>",
         f"enabled={outputs.get('checkpoint_enabled')}, interval={outputs.get('checkpoint_interval')}, "
         f"model_only={outputs.get('checkpoint_model_only')}"),
    ]
    body += "<h3>输出位置</h3>" + _kv_table(out_rows)
    return _section("2", "关键配置、输入的数据、输出内容", body,
                    "并行策略怎么配的、数据从哪来、写到哪里去")


# ---------------------------------------------------------------- 3. collectives


def _section_collectives(data: dict, combo: str) -> str:
    colls = data["collectives"]
    if not colls:
        return _section(
            "3", "执行过程中的关键通信",
            "<p class='note'>本次运行没有采集通信（加 <code>--collectives</code> 开启）。</p>",
            "FSDP 的 All-Gather、TP 的 All-Reduce、PP 的 P2P",
        )

    name, summary = colls[0]
    steps = max(summary.get("steps", 1), 1)
    body = (
        f"<p class='note'>数据来自 <code>{_esc(name)}</code>，覆盖 {steps} 个训练步。"
        "字节数是 <b>单个 rank 的 payload</b>（该 rank 贡献或读入的张量），"
        "不是输出张量的大小——所有 rank 的这个数按构造相同。</p>"
    )

    rows = ""
    for entry in summary.get("by_bucket", []):
        sources = ", ".join(
            f"<code>{_esc(s['raw'])}</code>" for s in entry.get("sources", [])
        )
        phase_bits = []
        for label, key in (
            ("训练步内", "calls_step"),
            ("初始化", "calls_init"),
            ("存 checkpoint", "calls_checkpoint"),
        ):
            count = entry.get(key, 0)
            if count:
                phase_bits.append(f"{label} {count}")
        # A row whose traffic is entirely checkpoint I/O would otherwise read as
        # per-step bandwidth: DCP's object scatter is implemented as a broadcast,
        # and on a 1-rank run it is the only collective in the whole process.
        all_checkpoint = entry.get("calls_step", 0) == 0 and entry.get(
            "calls_checkpoint", 0
        ) > 0
        badge = ""
        if len(phase_bits) > 1:
            badge = f" <span class='chip'>{' · '.join(phase_bits)}</span>"
        if all_checkpoint:
            badge = (
                " <span class='miss'>全部是存盘 I/O，不是训练通信</span>"
                f" <span class='chip'>{' · '.join(phase_bits)}</span>"
            )
        rows += (
            f"<tr><td><b>{_esc(entry['op'])}</b></td>"
            f"<td class='num'>{entry['calls_per_step']:g}</td>"
            f"<td class='num'>{_human_bytes(entry['bytes_per_step'])}</td>"
            f"<td class='muted'>{sources}{badge}</td></tr>"
        )
    body += (
        "<h3>实测流量（每步）</h3>"
        "<table><tr><th>集合通信</th><th>call / step</th><th>payload / step (每 rank)</th>"
        f"<th>来源算子</th></tr>{rows}</table>"
        "<p class='note'>call / step 是「总次数 ÷ 训练步数」，所以初始化与存盘那些"
        "一次性调用会被摊到每步里——下面的 phase 标注是看这一点的依据："
        "<b>训练步内</b>才是每步真实开销，<b>初始化</b>（FSDP 广播模块状态、进程组建立）"
        "与<b>存 checkpoint</b>（DCP 内部用 broadcast 做 object scatter）都只发生一次或每"
        "interval 次。单卡运行时唯一出现的集合通信就全部属于存盘 I/O。</p>"
    )

    expected = _expected_rows(combo, summary)
    body += "<h3>应该出现 vs 实际出现</h3>" + expected

    steps_rows = ""
    previous = None
    repeats = 0
    for per in summary.get("per_step", []):
        calls = ", ".join(f"{k}={v}" for k, v in sorted(per.get("calls", {}).items()))
        if calls == previous:
            # The steady state is the same every step, so a 60-row table of one
            # repeated line hides the only thing worth seeing: how step 1 differs.
            repeats += 1
            continue
        if repeats:
            steps_rows += (
                f"<tr><td class='num'>…</td><td class='muted'>上一步的计数重复了 "
                f"{repeats} 步（稳态）</td></tr>"
            )
            repeats = 0
        steps_rows += (
            f"<tr><td class='num'>{_esc(per.get('step'))}</td>"
            f"<td class='muted'>{_esc(calls)}</td></tr>"
        )
        previous = calls
    if repeats:
        steps_rows += (
            f"<tr><td class='num'>…</td><td class='muted'>末尾 {repeats} 步与上一步相同</td></tr>"
        )
    if steps_rows:
        body += (
            "<h3>逐步调用数（首步与稳态的区别）</h3>"
            "<p class='note'>只在上一步的计数发生变化时另起一行：稳态每步相同，"
            "首步的差额才是要看的（FSDP 第一次 forward 要 gather 全部参数）。</p>"
            f"<table><tr><th>step</th><th>calls (bucket@layer)</th></tr>{steps_rows}</table>"
        )

    if len(colls) > 1:
        rows = "".join(
            f"<tr><td>{_esc(os.path.basename(path))}</td>"
            f"<td class='num'>{_esc(summary_.get('total_calls'))}</td>"
            f"<td class='num'>{_human_bytes(summary_.get('total_bytes_per_rank'))}</td></tr>"
            for path, summary_ in colls
        )
        body += (
            "<h3>各 rank 是否一致</h3>"
            "<p class='note'>同一策略下每个 rank 的集合通信次数应当一致；"
            "PP 的 P2P 数可能因 stage 角色不同而有差异。</p>"
            f"<table><tr><th>rank 文件</th><th>total calls</th><th>total bytes / rank</th></tr>{rows}</table>"
        )

    glossary = "".join(
        f"<li><b>{_esc(op)}</b> — {_esc(text)}</li>" for op, text in WHY.items()
    )
    body += f"<h3>这些集合通信各自在干什么</h3><ul class='tight'>{glossary}</ul>"

    schemas = summary.get("schemas", {})
    if schemas:
        items = "".join(
            f"<li><code>{_esc(raw)}</code><br><span class='muted'>{_esc(schema)}</span></li>"
            for raw, schema in sorted(schemas.items())
        )
        body += (
            "<details><summary>原始算子名与 schema（为什么字节数这么算）</summary>"
            f"<ul class='tight'>{items}</ul></details>"
        )

    stacks = summary.get("stacks") or {}
    if stacks:
        # Only present when the run was started with CPU_LAB_COLLECTIVES_STACK=1.
        items = ""
        for raw, frames in sorted(stacks.items()):
            listing = "".join(f"<br>&nbsp;&nbsp;{_esc(f)}" for f in frames)
            items += f"<li><code>{_esc(raw)}</code>{listing}</li>"
        body += (
            "<details><summary>首次调用栈（谁发的这个集合通信）</summary>"
            f"<ul class='tight muted'>{items}</ul></details>"
        )

    unknown = summary.get("unknown_payload") or {}
    zero = summary.get("zero_payload") or {}
    if unknown or zero:
        body += (
            "<p class='warn'>有算子没有取到 payload 字节："
            f"无规则={_esc(unknown)}，规则命中为 0={_esc(zero)}。"
            "这些算子的流量计为 0，属于已知缺口而不是“零流量”。</p>"
        )
    return _section("3", "执行过程中的关键通信", body,
                    "FSDP 的 All-Gather / Reduce-Scatter、TP 的 All-Reduce、PP 的 P2P")


def _expected_rows(combo: str, summary: dict) -> str:
    """Which collectives this strategy should emit, vs which were observed."""
    import collectives as coll_mod

    observed = {entry["op"] for entry in summary.get("by_bucket", [])}
    keys = []
    if "fsdp" in combo or combo.startswith("hsdp"):
        keys.append("fsdp")
    if combo.startswith("hsdp"):
        keys.append("hsdp")
    if "tp" in combo:
        keys.append("tp")
    if combo.startswith("pp") or combo == "pp":
        keys.append("pp")
    if not keys:
        keys.append("fsdp" if combo == "fsdp" else "")

    seen: set[str] = set()
    rows = ""
    for key in keys:
        for op, why in coll_mod.EXPECTED_BY_STRATEGY.get(key, []):
            if (key, op) in seen:
                continue
            seen.add((key, op))
            hit = op in observed
            mark = (
                "<span class='ok'>observed</span>"
                if hit
                else "<span class='miss'>not seen</span>"
            )
            rows += (
                f"<tr><td>{_esc(key)}</td><td><b>{_esc(op)}</b></td>"
                f"<td>{mark}</td><td class='muted'>{_esc(why)}</td></tr>"
            )
    if not rows:
        return "<p class='note'>没有为该预设登记期望值。</p>"
    return (
        "<table><tr><th>策略</th><th>集合通信</th><th>结果</th><th>它解决什么</th></tr>"
        f"{rows}</table>"
    )


# --------------------------------------------------------------------- 4. results


def _section_results(data: dict) -> str:
    run = data["run"]
    rows_data = run.get("results", {}).get("rows", [])
    body = ""

    series = next(
        (r for r in rows_data if r.get("has_real_loss") and r.get("series")),
        None,
    )
    if series:
        body += (
            "<div class='chartbox'><canvas id='lossChart' width='900' height='320'>"
            "loss curve</canvas></div>"
            "<p class='note'>loss 与 grad_norm 取自 "
            f"rank {series['rank']}（最后一个 pipeline stage）。"
            "虚线为 grad_norm，右轴。</p>"
        )

    rows = ""
    for row in rows_data:
        note = (
            "<span class='muted'>—</span>"
            if row.get("has_real_loss")
            else "<span class='miss'>sentinel, not a loss</span>"
        )
        loss = row.get("last_loss")
        grad = row.get("last_grad_norm")
        rows += (
            f"<tr><td class='num'>{row.get('rank')}</td>"
            f"<td class='num'>{row.get('n_steps')}</td>"
            f"<td class='num'>{'n/a' if loss is None else f'{loss:.6f}'}</td>"
            f"<td class='num'>{'n/a' if grad is None else f'{grad:.4f}'}</td>"
            f"<td>{note}</td></tr>"
        )
    body += (
        "<h3>各 rank 的最终结果</h3>"
        "<table><tr><th>rank</th><th>steps</th><th>last loss</th><th>last grad_norm</th>"
        f"<th>note</th></tr>{rows}</table>"
    )

    real = [r for r in rows_data if r.get("has_real_loss")]
    if real:
        losses = sorted({round(r["last_loss"], 6) for r in real})
        grads = sorted({round(r["last_grad_norm"], 4) for r in real})
        checks = [
            (len(losses) == 1,
             f"末 stage 组内 loss 一致：{len(losses)} 个不同值 "
             f"{losses if len(losses) <= 4 else ''}",
             "TP 不对 batch 分片，同一 stage 内各 rank 的 loss 必须逐位相同；"
             "不一致说明有集合通信漏了"),
            (len(grads) == 1,
             f"grad_norm 全 rank 一致：{grads}",
             "梯度裁剪在并行网格上做了同步"),
        ]
        items = "".join(
            f"<li><span class='{'ok' if ok else 'miss'}'>{'PASS' if ok else 'CHECK'}</span> "
            f"{_esc(text)}<br><span class='muted'>{_esc(why)}</span></li>"
            for ok, text, why in checks
        )
        body += f"<h3>正确性判据</h3><ul class='tight checks'>{items}</ul>"

    memory = run.get("memory") or {}
    per_rank_mem = memory.get("per_rank") or {}
    if per_rank_mem:
        rows = ""
        for rank in sorted(per_rank_mem, key=int):
            entry = per_rank_mem[rank]
            rows += (
                f"<tr><td class='num'>{_esc(rank)}</td>"
                f"<td class='num'>{_human_bytes(entry['peak_rss'])}</td>"
                f"<td class='num'>{_human_bytes(entry['peak_commit'])}</td></tr>"
            )
        total_rss = sum(e["peak_rss"] for e in per_rank_mem.values())
        total_commit = sum(e["peak_commit"] for e in per_rank_mem.values())
        rows += (
            f"<tr><td><b>sum</b></td><td class='num'><b>{_human_bytes(total_rss)}</b></td>"
            f"<td class='num'><b>{_human_bytes(total_commit)}</b></td></tr>"
        )
        system = memory.get("system_after") or {}
        limit = system.get("commit_limit", -1)
        used = system.get("commit_used", -1)
        note = (
            "采样来自启动器在运行期间对每个 rank 进程的轮询（每 0.5s），"
            "不是各 rank 自报——峰值也出现在首步 all-gather 和存 checkpoint 时。"
            "<b>RSS</b> 是实际驻留内存，<b>commit</b> 是进程向系统预留的字节；"
            "Windows 拒绝分配是按 commit 对 RAM+页面文件计账的，所以能否启动看 commit，"
            "能否不抖动地跑看 RSS。"
        )
        if limit > 0:
            note += (
                f" 运行结束后系统 commit 使用 {_human_bytes(used)} / "
                f"{_human_bytes(limit)}，余量 {_human_bytes(limit - used)}。"
            )
        body += (
            "<h3>每个 rank 的内存峰值</h3>"
            "<table><tr><th>rank</th><th>peak RSS</th><th>peak commit</th></tr>"
            f"{rows}</table><p class='note'>{note}</p>"
        )

    artifacts = run.get("artifacts", [])
    if artifacts:
        weights = [a for a in artifacts if "checkpoint" in a["rel"]]
        other = [a for a in artifacts if "checkpoint" not in a["rel"]]
        total = sum(a["bytes"] for a in weights)
        rows = "".join(
            f"<tr><td><code>{_esc(a['rel'])}</code></td><td class='num'>{_human_bytes(a['bytes'])}</td></tr>"
            for a in weights
        )
        shards = len([a for a in artifacts if a["rel"].endswith(".distcp")])
        saved_steps = sorted(
            {
                int(a["rel"].split("step-")[1].split("/")[0])
                for a in weights
                if "step-" in a["rel"] and a["rel"].split("step-")[1].split("/")[0].isdigit()
            }
        )
        note = (
            f"<p class='note'>{_esc(run.get('dump_dir'))} 下，{shards} 个分片共 "
            f"{_human_bytes(total)}。格式是 DCP（分布式 checkpoint）："
            "<code>.metadata</code> 是索引，每个 <code>.distcp</code> 是一个 rank 的分片。"
            "不是 safetensors，也不是 <code>.pt</code>。</p>"
        )
        if len(saved_steps) > 1:
            final = max(saved_steps)
            intermediates = [s for s in saved_steps if s != final]
            note += (
                "<p class='note'>存了 "
                f"{len(saved_steps)} 个 step：{saved_steps}。中间那些（{intermediates}）"
                "是<b>完整</b> checkpoint（含优化器状态，所以每个分片大 3 倍左右），"
                f"最后一步 step-{final} 是 <b>model-only</b>（"
                "<code>last_save_model_only=True</code>），分片明显更小。"
                "保存间隔来自配置里自己的 <code>steps</code> 值，"
                "<code>--steps</code> 只改训练步数、不改这个间隔——"
                "所以覆盖步数时会多出中间 checkpoint。</p>"
            )
        body += (
            "<h3>权重落在哪</h3>"
            + note
            + f"<table><tr><th>文件</th><th>大小</th></tr>{rows}</table>"
        )
        readme = _readme_snippet(run, data)
        if readme:
            body += readme
        if other:
            rows = "".join(
                f"<tr><td><code>{_esc(a['rel'])}</code></td><td class='num'>{_human_bytes(a['bytes'])}</td></tr>"
                for a in other[:10]
            )
            body += (
                "<details><summary>其他产出</summary>"
                f"<table><tr><th>文件</th><th>大小</th></tr>{rows}</table></details>"
            )
    return _section("4", "结果", body, "loss 曲线、各 rank 数值、正确性判据、权重位置")


def _readme_snippet(run: dict, data: dict) -> str:
    """How to load these weights back, using the real flavor name.

    The flavor matters: loading a checkpoint into the wrong flavor fails with
    ``Size mismatch ... for layers.0.attention.wk.weight``, which is the shape
    of a 512-hidden run being read into a 128-hidden model.
    """
    flavor = None
    for meta in data.get("metas", []):
        flavor = meta.get("model", {}).get("flavor") or flavor
    step = run.get("last_step")
    ckpt = os.path.join(run.get("dump_dir", ""), "checkpoint", f"step-{step}")
    return _code(
        "# 把权重读回模型（单进程也可以：DCP 会把各 rank 的分片合起来）\n"
        "import torch.distributed.checkpoint as dcp\n"
        f"spec = model_registry({flavor!r})      # 必须用训练时的 flavor\n"
        "model = spec.model.build(); model.init_states()\n"
        "state = model.state_dict()\n"
        f"dcp.load(state, checkpoint_id=r'{ckpt}')\n"
        "model.load_state_dict(state)"
    )


# ---------------------------------------------------------------------- 5. probe


def _section_probe(data: dict) -> str:
    probe = data["probe"]
    if not probe:
        return _section(
            "5", "用训练好的模型推理",
            "<p class='note'>本次运行没有跑推理探针（加 <code>--probe</code> 开启，"
            "并配合 <code>--dataset-path cpu_lab/planted</code> 训练在植入语料上）。</p>",
            "证明训练真的改变了模型",
        )

    body = (
        f"<p class='note'>语料 <code>{_esc(probe.get('manifest'))}</code> 共 "
        f"{_esc(probe.get('corpus_rows'))} 行，其中两个虚构暗号被反复重复。"
        "下面用同一套架构、同一个词表，对比 <b>随机初始化</b> 与 "
        f"<b>本次训练出来的权重</b>（<code>{_esc(probe.get('checkpoint'))}</code>，"
        f"载入 {_esc(probe.get('checkpoint_loaded'))}）。"
        "loss 下降只能说明学到了字节频率；能不能说出暗号，才是训练生效的证据。</p>"
    )

    rows = ""
    for entry in probe.get("results", {}).get("probes", []):
        before = entry["random_init"]
        after = entry["trained"]
        badge = (
            "<span class='ok'>EXACT MATCH</span>"
            if after.get("exact_match")
            else "<span class='miss'>未逐字命中</span>"
        )
        rows += (
            f"<tr><td><b>{_esc(entry['key'])}</b><br>"
            f"<code>{_esc(entry['prompt_decoded'])}</code> → "
            f"<code>{_esc(entry['secret_text'])}</code></td>"
            f"<td class='num'>{before['p_secret']:.2e}</td>"
            f"<td class='num'><b>{after['p_secret']:.2e}</b></td>"
            f"<td class='num'>{before['mean_nll_on_secret']:.3f} → "
            f"<b>{after['mean_nll_on_secret']:.3f}</b></td>"
            f"<td class='num'>{before['accuracy']:.2f} → <b>{after['accuracy']:.2f}</b></td>"
            f"<td><code>{_esc(after['greedy_text'])}</code> {badge}</td></tr>"
        )
    body += (
        "<h3>暗号的可信度对比</h3>"
        "<table><tr><th>prompt → 期望的暗号</th><th>p(暗号)<br>随机权重</th>"
        "<th>p(暗号)<br>训练后</th><th>每 token NLL<br>随机 → 训练后</th>"
        "<th>逐 token 准确率<br>随机 → 训练后</th><th>贪心解码（训练后）</th></tr>"
        f"{rows}</table>"
        "<p class='note'>p(暗号) 是整串暗号在 teacher forcing 下的联合概率："
        "把每个位置预测对下一个字符的概率连乘。随机权重下是 1/2020 量级的 15 次方，"
        "所以肉眼看到的是 10 的负几十次方；训练后接近 1。</p>"
    )

    doc_rows = ""
    for entry in probe.get("results", {}).get("documents", []):
        before = entry["random_init"]
        after = entry["trained"]
        doc_rows += (
            f"<tr><td>{_esc(entry['key'])}</td>"
            f"<td class='num'>{before['mean_nll']:.4f}</td>"
            f"<td class='num'><b>{after['mean_nll']:.4f}</b></td>"
            f"<td class='num'>{before['accuracy']:.3f} → <b>{after['accuracy']:.3f}</b></td></tr>"
        )
    body += (
        "<h3>整篇植入文档的 teacher-forced NLL</h3>"
        "<table><tr><th>文档</th><th>NLL 随机权重</th><th>NLL 训练后</th>"
        f"<th>逐 token 准确率</th></tr>{doc_rows}</table>"
    )

    detail = ""
    for entry in probe.get("results", {}).get("probes", []):
        before, after = entry["random_init"], entry["trained"]
        detail += (
            f"<li>{_esc(entry['key'])}：随机权重贪心输出 "
            f"<code>{_esc(before['greedy_text'])}</code>；训练后 "
            f"<code>{_esc(after['greedy_text'])}</code>；"
            f"第一个 token 的排名 随机={_esc(before['rank_first_token'])} → "
            f"训练后={_esc(after['rank_first_token'])}</li>"
        )
    body += (
        f"<details><summary>逐条明细（含未命中时的中间形态）</summary>"
        f"<ul class='tight'>{detail}</ul></details>"
    )
    return _section("5", "用训练好的模型推理：训练生效了吗", body,
                    "在训练数据里埋了一个暗号，看模型能不能说出来")


# ------------------------------------------------------------------------ render

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
table.kv th{width:210px;color:var(--fg);font-weight:500}
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
ul.checks{list-style:none;padding-left:0}
ul.checks li{padding-left:0}
details{margin:12px 0}
summary{cursor:pointer;color:var(--accent)}
.chartbox{border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0;background:var(--bg)}
canvas{width:100%;height:320px;display:block}
footer{margin-top:36px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}
"""

_JS = """
function drawLoss(el, data){
  const dpr = window.devicePixelRatio || 1;
  const cssW = el.clientWidth, cssH = el.clientHeight;
  el.width = cssW * dpr; el.height = cssH * dpr;
  const c = el.getContext('2d'); c.scale(dpr, dpr);
  const cs = getComputedStyle(document.body);
  const M = {l:56,r:58,t:14,b:34};
  const W = cssW - M.l - M.r, H = cssH - M.t - M.b;
  const steps = data.steps, loss = data.loss, grad = data.grad;
  const lmin = Math.min(...loss), lmax = Math.max(...loss);
  const gmin = 0, gmax = Math.max(...grad) * 1.1;
  const pad = (lmax - lmin) * 0.15 || 1;
  const y0 = lmin - pad, y1 = lmax + pad;
  const X = i => M.l + (steps.length === 1 ? W / 2 : (i / (steps.length - 1)) * W);
  const Y = v => M.t + H - ((v - y0) / (y1 - y0)) * H;
  const Y2 = v => M.t + H - ((v - gmin) / (gmax - gmin)) * H;

  c.strokeStyle = cs.getPropertyValue('--line') || '#ddd';
  c.fillStyle = cs.getPropertyValue('--muted') || '#888';
  c.font = '11px sans-serif'; c.lineWidth = 1;
  for (let k = 0; k <= 4; k++){
    const y = M.t + (H * k) / 4;
    c.beginPath(); c.moveTo(M.l, y); c.lineTo(M.l + W, y); c.stroke();
    const lv = y1 - ((y1 - y0) * k) / 4;
    c.textAlign = 'right'; c.fillText(lv.toFixed(2), M.l - 8, y + 4);
    const gv = gmax - ((gmax - gmin) * k) / 4;
    c.textAlign = 'left'; c.fillText(gv.toFixed(1), M.l + W + 8, y + 4);
  }
  c.textAlign = 'center';
  steps.forEach((s, i) => { if (steps.length <= 20 || i % Math.ceil(steps.length/12) === 0) {
    c.fillText(s, X(i), M.t + H + 18); } });

  c.strokeStyle = '#c8871f'; c.lineWidth = 1.5; c.setLineDash([5,4]); c.beginPath();
  grad.forEach((v, i) => { const x = X(i), y = Y2(v); i ? c.lineTo(x,y) : c.moveTo(x,y); });
  c.stroke(); c.setLineDash([]);

  c.strokeStyle = '#2f6fd0'; c.lineWidth = 2.2; c.beginPath();
  loss.forEach((v, i) => { const x = X(i), y = Y(v); i ? c.lineTo(x,y) : c.moveTo(x,y); });
  c.stroke();
  c.fillStyle = '#2f6fd0';
  loss.forEach((v, i) => { c.beginPath(); c.arc(X(i), Y(v), 3, 0, 7); c.fill(); });
}
function legend(){
  const cs = getComputedStyle(document.body);
  return "<span style='display:inline-block;width:14px;height:2px;background:#2f6fd0;vertical-align:middle'></span> "
    + "<span style='color:" + cs.getPropertyValue('--muted') + "'>loss</span> &nbsp; "
    + "<span style='display:inline-block;width:14px;border-top:2px dashed #c8871f;vertical-align:middle'></span> "
    + "<span style='color:" + cs.getPropertyValue('--muted') + "'>grad_norm (右轴)</span>";
}
document.addEventListener('DOMContentLoaded', function(){
  const holder = document.getElementById('chart-data');
  const el = document.getElementById('lossChart');
  if (holder && el){
    drawLoss(el, JSON.parse(holder.textContent));
    const lg = document.createElement('div');
    lg.style.cssText = 'font-size:12px;margin-top:6px';
    lg.innerHTML = legend();
    el.parentNode.appendChild(lg);
  }
});
"""


def render_html(data: dict, combo: str) -> str:
    run = data["run"]
    dims = run.get("dims", {})
    training = (data["metas"][0].get("training", {}) if data["metas"] else {})
    rows_data = run.get("results", {}).get("rows", [])
    real = [r for r in rows_data if r.get("has_real_loss")]

    pills = [
        ("combo", combo),
        ("world_size", run.get("world_size")),
        ("parallelism", " x ".join(
            f"{k}={dims.get(k)}" for k in ("dp_replicate", "dp_shard", "cp", "tp", "pp")
            if dims.get(k) not in (None, 1)
        ) or "none"),
        ("steps", training.get("steps")),
        ("wall time", f"{run.get('wall_time_s', 0):.1f} s"),
        ("last loss", f"{real[0]['last_loss']:.6f}" if real else "n/a"),
    ]
    strip = "".join(
        f"<span class='pill'>{_esc(k)} <b>{_esc(v)}</b></span>" for k, v in pills
    )

    series = next((r for r in rows_data if r.get("has_real_loss") and r.get("series")), None)
    chart_json = ""
    if series:
        chart_json = (
            "<script id='chart-data' type='application/json'>"
            + json.dumps(
                {
                    "steps": [s for s, _, _ in series["series"]],
                    "loss": [loss for _, loss, _ in series["series"]],
                    "grad": [grad for _, _, grad in series["series"]],
                }
            )
            + "</script>"
        )

    body = (
        _section_command(data)
        + _section_config(data)
        + _section_collectives(data, combo)
        + _section_results(data)
        + _section_probe(data)
    )
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>TinyGPT on CPU — {_esc(combo)}</title>"
        f"<style>{_CSS}</style></head><body><div class='wrap'>"
        f"<h1>TinyGPT on CPU — {_esc(combo)}</h1>"
        f"<p class='sub'>一次 8 卡并行策略实验的完整记录：命令、配置、通信、结果、推理验证。"
        f"由 cpu_lab/report.py 从本次运行的日志生成。</p>"
        f"<div class='meta'>{strip}</div>"
        f"{body}"
        f"<footer>生成于 {generated} · 日志目录 <code>{_esc(data['log_dir'])}</code> · "
        "所有数字都来自磁盘上的运行产物，报告本身不做计算。</footer>"
        f"</div>{chart_json}<script>{_JS}</script></body></html>"
    )


def render(
    log_dir: str = DEFAULT_LOG_DIR,
    combo: str | None = None,
    out_path: str | None = None,
) -> str:
    data = collect(log_dir)
    run = data["run"]
    combo = combo or run.get("combo") or "run"
    if out_path is None:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        out_path = os.path.join(DEFAULT_OUT_DIR, f"{combo}.html")
    html_text = render_html(data, combo)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--combo", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(render(args.log_dir, args.combo, args.out))
