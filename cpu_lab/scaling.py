# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Estimate the largest model this machine can train, from measured rungs.

The question "how big a model fits" is usually answered with arithmetic: bytes
per parameter times parameter count. That arithmetic is wrong here in both
directions, and the error is not small:

- it ignores a large *fixed* cost per rank (python + torch + dataloader), which
  at these sizes dominates the parameter term by several times;
- it ignores FSDP's all-gather buffers, gloo's staging copies and activation
  memory, all of which scale with the model but with coefficients that depend on
  the parallel layout, the batch size and the sequence length;
- it says nothing about *commit*, which is what Windows actually refuses to hand
  out (RAM + pagefile), as opposed to resident bytes.

So instead: run the same parallel strategy at several sizes, sample each rank's
peak memory and time each step, then fit and extrapolate. The fit is linear in
parameter count, which is what the mechanism predicts -- and if the measured
points are not on a line, that is itself the finding.

Two limits come out, and they are different limits:

- **memory** -- how many parameters can be resident / committed at once;
- **time** -- how long a step takes at that size. On 8 CPU cores with gloo over
  loopback this is usually the binding one, and it is worth stating in the same
  breath, because "it fits" and "it is usable" are not the same claim.
"""

from __future__ import annotations

import glob
import html
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE_ROOT = os.path.join(HERE, "logs_scale")
OUT_DIR = os.path.join(HERE, "reports")

import hostmem  # noqa: E402  (local module, after path setup)

_STEP_TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_TOTAL_PARAMS = re.compile(r"size:\s*([\d,]+)\s+total parameters")


def _total_params(log_dir: str, ranks: int) -> int | None:
    """Global parameter count from torchtitan's own init line.

    Printed before any sharding, so unlike ``numel()`` on a DTensor it is not
    replicated per rank. Returns ``None`` if no log carries it (older runs).
    """
    for rank in range(ranks):
        path = os.path.join(
            log_dir, "train.log" if ranks == 1 else f"rank_{rank}.log"
        )
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                match = _TOTAL_PARAMS.search(line)
                if match:
                    return int(match.group(1).replace(",", ""))
    return None


def _step_time(log_dir: str, ranks: int) -> tuple[float | None, float | None]:
    """(seconds per step, startup seconds) from the rank logs' timestamps."""
    for rank in range(ranks):
        path = os.path.join(
            log_dir, "train.log" if ranks == 1 else f"rank_{rank}.log"
        )
        if not os.path.exists(path):
            continue
        stamps: list[tuple[int, float]] = []
        first_log = last_log = None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                match = _STEP_TS.search(line)
                if not match:
                    continue
                stamp = time.mktime(
                    time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                )
                if first_log is None:
                    first_log = stamp
                last_log = stamp
                step = re.search(r"step:\s*(\d+)", line)
                if step:
                    stamps.append((int(step.group(1)), stamp))
        if len(stamps) >= 2:
            (s0, t0), (s1, t1) = stamps[0], stamps[-1]
            per_step = (t1 - t0) / max(s1 - s0, 1)
            startup = t0 - (first_log or t0)
            return per_step, max(startup, 0.0)
        if last_log and first_log:
            return None, last_log - first_log
    return None, None


_FAIL_SIGNATURES = (
    (re.compile(r"MemoryError|can't allocate memory|DefaultCPUAllocator|not enough memory|"
                r"out of memory|failed to allocate", re.I), "内存分配失败"),
    # Windows does not always raise MemoryError for a large failed allocation:
    # the process dies with an access violation inside the allocator instead.
    # Seen at 687M parameters, in FSDP's all-gather buffer allocation.
    (re.compile(r"access violation|fatal exception", re.I), "分配失败导致进程崩溃"),
    (re.compile(r"Timed out|Socket Timeout|watchdog", re.I), "等待其它 rank 超时"),
    (re.compile(r"Killed|Terminated|KeyboardInterrupt", re.I), "被中断 / 超时杀掉"),
)


def _failure_reason(log_paths: list[str]) -> tuple[str, str]:
    """``(short reason, last meaningful line)`` for a rung that left no run.json."""
    last = ""
    for pattern, reason in _FAIL_SIGNATURES:
        for path in log_paths:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            hits = pattern.findall(text)
            if hits:
                for line in text.splitlines():
                    if pattern.search(line):
                        last = line.strip()
                return reason, last
    for path in log_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
        except OSError:
            continue
        if lines:
            last = lines[-1]
    return "未产出 run.json（没跑到训练结束）", last


def collect_failed(root: str = SCALE_ROOT) -> list[dict]:
    """Rungs that started but never wrote a ``run.json``.

    A scaling study's most interesting rung is often the one that does not
    finish, so those directories are reported with the signature that stopped
    them rather than silently skipped.
    """
    failed = []
    for log_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(log_dir):
            continue
        if os.path.exists(os.path.join(log_dir, "run.json")):
            continue
        logs = sorted(glob.glob(os.path.join(log_dir, "rank_*.log")))
        if not logs:
            logs = sorted(glob.glob(os.path.join(log_dir, "train.log")))
        if not logs:
            continue
        reason, evidence = _failure_reason(logs)
        failed.append(
            {
                "label": os.path.basename(log_dir),
                "log_dir": os.path.abspath(log_dir),
                "flavor": _flavor_from_logs(logs),
                "reason": reason,
                "evidence": evidence,
            }
        )
    return failed


_FLAVOR = re.compile(r"Model\s+\S+\s+(\S+)\s+size:")


def _flavor_from_logs(log_paths: list[str]) -> str:
    for path in log_paths[:1]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    match = _FLAVOR.search(line)
                    if match:
                        return match.group(1)
        except OSError:
            continue
    return "n/a"


def _pagefile_drive_free() -> int:
    """Free bytes on the system drive -- the ceiling on pagefile growth.

    Windows raises the commit limit by growing ``pagefile.sys`` on demand, so the
    commit figure is not a constant: it moved from 40.75 to 43.88 GiB across this
    set of runs. What it can grow to is bounded by free space on the system drive,
    which is why that number belongs next to it.
    """
    drive = os.environ.get("SystemDrive", "C:") + os.sep
    try:
        import shutil

        return shutil.disk_usage(drive).free
    except OSError:
        return -1


def collect(root: str = SCALE_ROOT) -> list[dict]:
    rungs = []
    for run_json in sorted(glob.glob(os.path.join(root, "*", "run.json"))):
        log_dir = os.path.dirname(run_json)
        run = _read_json(run_json) or {}
        metas = [
            _read_json(p)
            for p in sorted(glob.glob(os.path.join(log_dir, "run_meta_rank_*.json")))
        ]
        metas = [m for m in metas if m]
        if not metas:
            continue
        ranks = len(metas)
        # ``local_params`` is NOT additive across ranks. A DTensor answers
        # ``numel()`` with its *global* shape, so under pp2 x tp2 x dp_shard2 each
        # rank reports N/pp and the sum over 8 ranks is 4N. The model's own line
        # ("Model <name> <flavor> size: N total parameters") is printed before
        # sharding and is the number to fit against; keep the sum only as a
        # fallback, and say which one was used so a wrong x-axis is visible.
        total_params, source = _total_params(log_dir, ranks), "log"
        if total_params is None:
            total_params = sum(m["model"]["local_params"] for m in metas)
            source = "sum(local_params)"
        rows = run.get("results", {}).get("rows", [])
        real = [r for r in rows if r.get("has_real_loss")]
        per_step, startup = _step_time(log_dir, ranks)
        mem = (run.get("memory") or {}).get("per_rank") or {}
        if not mem:
            continue
        commit = [e["peak_commit"] for e in mem.values()]
        rss = [e["peak_rss"] for e in mem.values()]
        training = metas[0].get("training", {})
        rungs.append(
            {
                "label": os.path.basename(log_dir),
                "log_dir": os.path.abspath(log_dir),
                "params": total_params,
                "params_source": source,
                "local_params_per_rank": [m["model"]["local_params"] for m in metas],
                "ranks": ranks,
                "flavor": metas[0]["model"]["flavor"],
                "seq_len": training.get("seq_len"),
                "local_batch_size": training.get("local_batch_size"),
                "global_batch_size": training.get("global_batch_size"),
                "tokens_per_step": training.get("tokens_per_step"),
                "steps_run": len(real[0]["series"]) if real and real[0].get("series") else None,
                "wall_time_s": run.get("wall_time_s"),
                "per_step_s": per_step,
                "startup_s": startup,
                "peak_commit_per_rank": max(commit),
                "peak_rss_per_rank": max(rss),
                "total_commit": sum(commit),
                "total_rss": sum(rss),
                "system_after": (run.get("memory") or {}).get("system_after") or {},
            }
        )
    rungs.sort(key=lambda r: r["params"])
    return rungs


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares ``y = slope * x + intercept``; falls back to a ratio."""
    n = len(xs)
    if n < 2:
        return (ys[0] / xs[0] if xs and xs[0] else 0.0), 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    return slope, mean_y - slope * mean_x


def _human_bytes(n: float) -> str:
    if n is None or n < 0:
        return "n/a"
    for unit in ("B", "KB", "MB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:,.2f} {unit}"
        n /= 1024
    return f"{n:,.2f} GiB"


def extrapolate(rungs: list[dict], budget_commit: float, budget_rss: float) -> dict:
    """Solve both memory limits, plus a time projection at each size."""
    params = [float(r["params"]) for r in rungs]
    commit = [float(r["total_commit"]) for r in rungs]
    rss = [float(r["total_rss"]) for r in rungs]

    commit_slope, commit_fixed = _fit(params, commit)
    rss_slope, rss_fixed = _fit(params, rss)

    def solve(slope: float, fixed: float, budget: float) -> float | None:
        if budget <= fixed:
            return None
        if slope <= 0:
            return None
        return (budget - fixed) / slope

    max_by_commit = solve(commit_slope, commit_fixed, budget_commit)
    max_by_rss = solve(rss_slope, rss_fixed, budget_rss)

    per_step = [
        (float(r["params"]), r["per_step_s"])
        for r in rungs
        if r.get("per_step_s")
    ]
    time_model = None
    if len(per_step) >= 2:
        slope, fixed = _fit([p for p, _ in per_step], [t for _, t in per_step])
        time_model = {"slope_s_per_param": slope, "fixed_s": fixed}

        def step_time_at(n_params: float) -> float | None:
            return slope * n_params + fixed
    else:
        def step_time_at(n_params: float) -> float | None:
            return None

    return {
        "commit_slope_bytes_per_param": commit_slope,
        "commit_fixed_bytes": commit_fixed,
        "rss_slope_bytes_per_param": rss_slope,
        "rss_fixed_bytes": rss_fixed,
        "budget_commit_bytes": budget_commit,
        "budget_rss_bytes": budget_rss,
        "max_params_by_commit": max_by_commit,
        "max_params_by_rss": max_by_rss,
        "time_model": time_model,
        "step_time_at": {
            "max_by_commit": step_time_at(max_by_commit) if max_by_commit else None,
            "max_by_rss": step_time_at(max_by_rss) if max_by_rss else None,
        },
        "step_time_at_current": step_time_at(params[-1]) if params else None,
    }


# ------------------------------------------------------------------ rendering

_CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#16181d;--muted:#6b7280;--line:#e3e5ea;
--panel:#f7f8fa;--accent:#2f6fd0;--ok:#1a7f4b;--miss:#b4451f;--code:#f2f3f6;}
@media (prefers-color-scheme:dark){:root{--bg:#15171c;--fg:#e8eaf0;--muted:#9aa1ad;
--line:#2b2f38;--panel:#1c1f26;--accent:#6ea8ff;--ok:#5ed09a;--miss:#ff9b7a;--code:#22262e;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:32px 24px 72px}
h1{font-size:23px;font-weight:600;margin:0 0 4px}h2{font-size:16px;font-weight:600;margin:26px 0 8px}
h3{font-size:14px;font-weight:600;margin:20px 0 6px}
.sub{color:var(--muted);margin:2px 0 14px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
code{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;background:var(--code);
padding:1px 5px;border-radius:4px}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:10px 14px;
border-radius:6px;margin:12px 0}.muted{color:var(--muted)}.ok{color:var(--ok);font-weight:600}
.miss{color:var(--miss);font-weight:600}
.chartbox{border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}
canvas{width:100%;height:300px;display:block}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:5px 12px;
font-size:12.5px;display:inline-block;margin:0 8px 8px 0}.pill b{font-weight:600}
ul.tight{margin:6px 0;padding-left:20px}
"""

_JS = """
function draw(canvas, series, yLabel){
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w*dpr; canvas.height = h*dpr;
  const c = canvas.getContext('2d'); c.scale(dpr,dpr);
  const cs = getComputedStyle(document.body);
  const M = {l:62,r:16,t:14,b:36}, W = w-M.l-M.r, H = h-M.t-M.b;
  const xs = series[0].x.concat(series.flatMap(s=>s.x));
  const ys = series.flatMap(s=>s.y);
  const x0 = 0, x1 = Math.max(...xs)*1.05;
  const y0 = 0, y1 = Math.max(...ys)*1.15;
  const X = v => M.l + (v-x0)/(x1-x0)*W;
  const Y = v => M.t + H - (v-y0)/(y1-y0)*H;
  c.strokeStyle = cs.getPropertyValue('--line')||'#ddd';
  c.fillStyle = cs.getPropertyValue('--muted')||'#888';
  c.font = '11px sans-serif'; c.lineWidth = 1;
  for (let k=0;k<=4;k++){
    const y = M.t + H*k/4;
    c.beginPath(); c.moveTo(M.l,y); c.lineTo(M.l+W,y); c.stroke();
    c.textAlign='right'; c.fillText(((y1-(y1-y0)*k/4)/1073741824).toFixed(1)+' GiB', M.l-8, y+4);
  }
  c.textAlign='center';
  for (let k=0;k<=4;k++){
    const v = x0+(x1-x0)*k/4;
    c.fillText((v/1e6).toFixed(0)+'M', X(v), M.t+H+18);
  }
  series.forEach((s,idx)=>{
    c.strokeStyle = s.color; c.lineWidth = 2; c.setLineDash(s.dash||[]);
    c.beginPath();
    s.x.forEach((xv,i)=>{ const px=X(xv), py=Y(s.y[i]); i?c.lineTo(px,py):c.moveTo(px,py); });
    c.stroke(); c.setLineDash([]);
    c.fillStyle = s.color;
    s.x.forEach((xv,i)=>{ c.beginPath(); c.arc(X(xv),Y(s.y[i]),3,0,7); c.fill(); });
  });
}
document.addEventListener('DOMContentLoaded', function(){
  const data = JSON.parse(document.getElementById('scaling-data').textContent);
  const c1 = document.getElementById('memChart');
  if (c1) draw(c1, data.memory, 'bytes');
  const c2 = document.getElementById('timeChart');
  if (c2){
    const dpr = window.devicePixelRatio||1, w=c2.clientWidth, h=c2.clientHeight;
    c2.width=w*dpr; c2.height=h*dpr;
    const c=c2.getContext('2d'); c.scale(dpr,dpr);
    const cs=getComputedStyle(document.body);
    const M={l:62,r:16,t:14,b:36}, W=w-M.l-M.r, H=h-M.t-M.b;
    const xs=data.time.map(p=>p[0]), ys=data.time.map(p=>p[1]);
    const x1=Math.max(...xs)*1.05, y1=Math.max(...ys)*1.15;
    const X=v=>M.l+v/x1*W, Y=v=>M.t+H-v/y1*H;
    c.strokeStyle=cs.getPropertyValue('--line')||'#ddd'; c.fillStyle=cs.getPropertyValue('--muted')||'#888';
    c.font='11px sans-serif';
    for(let k=0;k<=4;k++){const y=M.t+H*k/4;c.beginPath();c.moveTo(M.l,y);c.lineTo(M.l+W,y);c.stroke();
      c.textAlign='right';c.fillText((y1-(y1)*k/4).toFixed(0)+' s',M.l-8,y+4);}
    c.textAlign='center';
    for(let k=0;k<=4;k++){const v=x1*k/4;c.fillText((v/1e6).toFixed(0)+'M',X(v),M.t+H+18);}
    c.strokeStyle='#2f6fd0';c.lineWidth=2;c.beginPath();
    xs.forEach((xv,i)=>{const px=X(xv),py=Y(ys[i]);i?c.lineTo(px,py):c.moveTo(px,py);});
    c.stroke();c.fillStyle='#2f6fd0';
    xs.forEach((xv,i)=>{c.beginPath();c.arc(X(xv),Y(ys[i]),3,0,7);c.fill();});
  }
});
"""


def render_html(
    rungs: list[dict],
    est: dict,
    out_path: str,
    failed: list[dict] | None = None,
    system: dict | None = None,
) -> str:
    def esc(v):
        return html.escape(str(v))

    failed = failed or []
    system = system or {}

    rows = ""
    for r in rungs:
        per_step = f"{r['per_step_s']:.1f}" if r["per_step_s"] else "n/a"
        rows += (
            f"<tr><td><b>{esc(r['flavor'])}</b></td>"
            f"<td class='num'>{r['params']/1e6:,.1f}M</td>"
            f"<td class='num'>{esc(r['tokens_per_step'])}</td>"
            f"<td class='num'>{per_step}</td>"
            f"<td class='num'>{_human_bytes(r['peak_rss_per_rank'])}</td>"
            f"<td class='num'>{_human_bytes(r['peak_commit_per_rank'])}</td>"
            f"<td class='num'>{_human_bytes(r['total_commit'])}</td></tr>"
        )
    table = (
        "<table><tr><th>flavor</th><th>params (total)</th><th>tokens/step</th>"
        "<th>s/step</th><th>peak RSS/rank</th><th>peak commit/rank</th>"
        f"<th>commit, all ranks</th></tr>{rows}</table>"
    )

    max_commit = est["max_params_by_commit"]
    max_rss = est["max_params_by_rss"]
    budget = est["budget_commit_bytes"]

    def fmt_params(value):
        return "n/a" if not value else f"{value/1e6:,.0f}M ({value/1e9:.2f}B)"

    pills = [
        ("物理内存", _human_bytes(system.get("total_phys", -1))),
        ("commit 上限", _human_bytes(system.get("commit_limit", -1))),
        ("系统盘剩余", _human_bytes(system.get("pagefile_drive_free", -1))),
        ("per-param commit", f"{est['commit_slope_bytes_per_param']:.2f} B/param (all ranks)"),
        ("固定 commit", _human_bytes(est["commit_fixed_bytes"])),
        ("commit 口径上限", fmt_params(max_commit)),
        ("驻留口径上限", fmt_params(max_rss)),
    ]
    strip = "".join(f"<span class='pill'>{esc(k)} <b>{esc(v)}</b></span>" for k, v in pills)

    if failed:
        failed_rows = "".join(
            f"<tr><td><b>{esc(f['flavor'])}</b></td>"
            f"<td class='miss'>{esc(f['reason'])}</td>"
            f"<td><code>{esc((f['evidence'] or '')[:200])}</code></td></tr>"
            for f in failed
        )
        failed_block = (
            "<h2>4. 跑不起来的那一档</h2>"
            "<table><tr><th>flavor</th><th>结果</th><th>日志里的最后一条关键行</th></tr>"
            f"{failed_rows}</table>"
            "<p class='note'>这一档是<b>实测的边界</b>，不是外推：它按同一套命令启动，"
            "却没能跑完。它的意义在于把「线性外推说能到哪」压到「实际上到哪」。</p>"
        )
    else:
        failed_block = ""

    time_rows = ""
    for r in rungs:
        per_step = f"{r['per_step_s']:.1f} s" if r["per_step_s"] else "n/a"
        rate = (
            f"{r['tokens_per_step']/r['per_step_s']:,.0f}"
            if r["per_step_s"] and r["tokens_per_step"]
            else "n/a"
        )
        time_rows += (
            f"<tr><td>{esc(r['flavor'])}</td>"
            f"<td class='num'>{r['params']/1e6:,.1f}M</td>"
            f"<td class='num'>{per_step}</td>"
            f"<td class='num'>{rate}</td></tr>"
        )

    def time_note():
        model = est.get("time_model")
        if not model:
            return ""
        lines = [
            f"每步耗时拟合 <code>t ≈ {model['slope_s_per_param']*1e6:.3f} s 每 M 参数 "
            f"+ {model['fixed_s']:.1f} s 固定开销</code>。"
        ]
        for label, value in (
            ("commit 上限处", est["step_time_at"]["max_by_commit"]),
            ("RSS 上限处", est["step_time_at"]["max_by_rss"]),
        ):
            if value:
                lines.append(f"{label}约 <b>{value/60:.1f} 分钟/步</b>。")
        return "<p class='note'>" + "".join(lines) + "</p>"

    chart_data = {
        "memory": [
            {"x": [r["params"] for r in rungs],
             "y": [r["peak_commit_per_rank"] for r in rungs],
             "color": "#2f6fd0"},
            {"x": [r["params"] for r in rungs],
             "y": [r["peak_rss_per_rank"] for r in rungs],
             "color": "#1d9e75", "dash": [5, 4]},
        ],
        "time": [[r["params"], r["per_step_s"]] for r in rungs if r["per_step_s"]],
    }

    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    candidates = [v for v in (max_commit, max_rss) if v]
    limiting = ""
    if candidates:
        limiting = (
            f"<li>两者取小：<b>{fmt_params(min(candidates))}</b> 是这套配置下能启动的上限；"
            "再大就要么分配不到内存，要么开始疯狂换页。</li>"
        )
    body = f"""
<div class="wrap">
<h1>这台机器能跑多大的模型</h1>
<p class="sub">实测估算：同一并行策略（pp2 × tp2 × dp_shard2，8 rank，gloo/CPU，fp32）
在若干尺寸上各跑几步，采样每 rank 峰值内存与每步耗时，再线性外推。
先量化目标，再看架构，最后给规划——数字全部来自 <code>cpu_lab/logs_scale/</code> 下的实测。</p>
<div>{strip}</div>
<h2>1. 实测各档</h2>
{table}
<div class="chartbox"><canvas id="memChart">memory vs parameters</canvas></div>
<p class="note">实线为每 rank 峰值 <b>commit</b>（进程向系统预留的字节），虚线为峰值 <b>RSS</b>（实际驻留）。
两者都随参数量线性增长；直线在 0 参数处的截距就是每 rank 的固定开销
（python + torch + dataloader），在本文的尺寸区间里它由截距主导，
所以「能跑多大」首先取决于开了几个 rank，而不是模型本身多大。</p>
<h2>2. 每步耗时</h2>
<table><tr><th>flavor</th><th>params</th><th>s/step</th><th>tokens/s</th></tr>{time_rows}</table>
<div class="chartbox"><canvas id="timeChart">step time vs parameters</canvas></div>
{time_note()}
<h2>3. 结论</h2>
<ul class="tight">
<li>内存上限（commit 口径）：约 <b>{fmt_params(max_commit)}</b> —— 系统 commit 预算决定。</li>
<li>内存上限（驻留口径）：约 <b>{fmt_params(max_rss)}</b> —— 物理 RAM 决定。</li>
{limiting}
</ul>
<p class="note">commit 那一栏不是常数：Windows 需要时会自动把 <code>pagefile.sys</code> 撑大，
本次实验期间 commit 上限就从 {_human_bytes(40.75 * 1024**3)} 长到
{_human_bytes(system.get('commit_limit', -1))}，能长多大取决于系统盘剩余空间
（实测 {_human_bytes(system.get('pagefile_drive_free', -1))}）。
所以「commit 口径上限」是个会浮动的数；真正硬的是<b>物理内存</b>那一栏——
超出它的部分只能在页面文件里来回倒，每步耗时会被换页吞掉。</p>
{failed_block}
<p class="note">估算前提：固定 <code>seq_len=128</code>、<code>local_batch_size=8</code>
（global batch 16）、<code>pp2×tp2×dp_shard2</code>、fp32 参数与优化器状态。
换更大的 batch/seq 会抬高激活内存的系数，因此上面的数字是这套配置下的上限。</p>
<footer class="muted">生成于 {generated} · 数据来自 {esc(SCALE_ROOT)} 下各档的 run.json</footer>
</div>
<script id="scaling-data" type="application/json">{json.dumps(chart_data)}</script>
<script>{_JS}</script>
"""
    page = (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>能跑多大的模型</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=SCALE_ROOT)
    parser.add_argument(
        "--out", default=os.path.join(OUT_DIR, "scaling.html")
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=0.0,
        help="commit to leave for the rest of the machine, on top of what is "
        "already committed (default: 0 -- use the full headroom)",
    )
    parser.add_argument(
        "--resident-reserve-gib",
        type=float,
        default=5.0,
        help="physical RAM to assume the desktop keeps. The live 'available' "
        "figure is NOT the budget: the OS reclaims pageable pages from other "
        "processes for the run, and summing RSS over 8 ranks double-counts the "
        "read-only pages (torch/python images) they share. Measured: a rung "
        "whose ranks sum to 7.5 GiB RSS ran at full speed while 'available' "
        "read 4.6--6.7 GiB.",
    )
    args = parser.parse_args()

    rungs = collect(args.root)
    if not rungs:
        print(f"no rungs found under {args.root}", file=os.sys.stderr)
        return 2
    failed = collect_failed(args.root)

    system = hostmem.system_memory()
    system["pagefile_drive_free"] = _pagefile_drive_free()
    # The budget must not be read live: whatever else is running at report time
    # (including a rung in flight) inflates ``commit_used`` and can drive the
    # budget to zero. Each rung already recorded the system state right after its
    # ranks exited, so the smallest of those is "this machine with training
    # stopped" -- that is the baseline to subtract.
    baselines = [
        r["system_after"]["commit_used"]
        for r in rungs
        if (r.get("system_after") or {}).get("commit_used", -1) > 0
    ]
    limit = max(
        [system["commit_limit"]]
        + [
            r["system_after"]["commit_limit"]
            for r in rungs
            if (r.get("system_after") or {}).get("commit_limit", -1) > 0
        ]
    )
    baseline_used = min(baselines) if baselines else system["commit_used"]
    budget_commit = max(limit - baseline_used - args.reserve_gib * 1024**3, 0)
    budget_rss = max(
        system["total_phys"] - args.resident_reserve_gib * 1024**3, 0
    )
    est = extrapolate(rungs, budget_commit, budget_rss)

    print(f"rungs: {len(rungs)}")
    for r in rungs:
        per_step = f"{r['per_step_s']:.1f}s" if r["per_step_s"] else "n/a"
        print(
            f"  {r['flavor']:<14} {r['params']/1e6:8,.1f}M  {per_step:>8}  "
            f"commit/rank {hostmem.gib(r['peak_commit_per_rank']):.2f} GiB  "
            f"all ranks {hostmem.gib(r['total_commit']):.2f} GiB"
        )
    for f in failed:
        print(f"  FAILED {f['flavor']:<14} {f['reason']}")
    print(f"system commit: baseline {hostmem.gib(baseline_used):.2f} / limit "
          f"{hostmem.gib(limit):.2f} GiB -> budget {hostmem.gib(budget_commit):.2f} GiB "
          f"(live used {hostmem.gib(system['commit_used']):.2f} GiB ignored)")
    print(f"physical RAM {hostmem.gib(system['total_phys']):.2f} GiB -> "
          f"resident budget {hostmem.gib(budget_rss):.2f} GiB")
    print(f"fit commit = {est['commit_slope_bytes_per_param']:.2f} B/param x N "
          f"+ {hostmem.gib(est['commit_fixed_bytes']):.2f} GiB")
    print(f"max params by commit: "
          f"{(est['max_params_by_commit'] or 0)/1e6:,.0f}M")
    print(f"max params by RSS:    {(est['max_params_by_rss'] or 0)/1e6:,.0f}M")
    if est.get("time_model"):
        print(f"fit step time = {est['time_model']['slope_s_per_param']*1e6:.3f} s/M "
              f"+ {est['time_model']['fixed_s']:.1f} s")
    out = render_html(rungs, est, args.out, failed=failed, system=system)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
