#!/usr/bin/env python
"""Render one qlora_lab run (metrics.json) into a self-contained HTML report.

No CDN, no external assets -- opens offline, follows the OS light/dark scheme,
same visual language as cpu_lab/reports/*.html.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUTREPORTS = os.path.join(HERE, "reports")

CSS = """
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
canvas{width:100%;height:280px;display:block}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:5px 12px;
font-size:12.5px;display:inline-block;margin:0 8px 8px 0}.pill b{font-weight:600}
ul.tight{margin:6px 0;padding-left:20px}
.kv{display:flex;flex-wrap:wrap;gap:6px 28px;margin:6px 0}
.kv div{font-size:13px}.kv span{color:var(--muted)}
"""

JS = r"""
function drawLine(canvas, series, yLabel, opts){
  opts = opts || {};
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w*dpr; canvas.height = h*dpr;
  const c = canvas.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0);
  const cs = getComputedStyle(document.body);
  const fg = cs.getPropertyValue('--fg').trim(), muted = cs.getPropertyValue('--muted').trim();
  const line = cs.getPropertyValue('--line').trim();
  const M = {l:66,r:18,t:16,b:38}, W = w-M.l-M.r, H = h-M.t-M.b;
  const allY = series.flatMap(s=>s.y);
  let y0 = Math.min(...allY), y1 = Math.max(...allY);
  const padY = (y1-y0)*0.12 || Math.abs(y0)*0.05 || 1;
  y0 -= padY; y1 += padY;
  if (opts.y0 !== undefined) y0 = opts.y0;
  const xs = series[0].x;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const X = v => M.l + (x1===x0?W/2:((v-x0)/(x1-x0))*W);
  const Y = v => M.t + H - ((v-y0)/(y1-y0))*H;
  c.font = '11px -apple-system,"Segoe UI",sans-serif';
  c.strokeStyle = line; c.fillStyle = muted; c.lineWidth = 1;
  for (let i=0;i<=4;i++){
    const v = y0 + (y1-y0)*i/4, y = Y(v);
    c.beginPath(); c.moveTo(M.l,y); c.lineTo(M.l+W,y); c.stroke();
    const lbl = Math.abs(v)>=1000? v.toFixed(0) : v.toFixed(3);
    c.textAlign='right'; c.textBaseline='middle'; c.fillText(lbl, M.l-8, y);
  }
  for (let i=0;i<=4;i++){
    const v = x0 + (x1-x0)*i/4, x = X(v);
    c.textAlign='center'; c.textBaseline='top'; c.fillText(String(Math.round(v)), x, M.t+H+8);
  }
  c.save(); c.translate(14, M.t+H/2); c.rotate(-Math.PI/2);
  c.textAlign='center'; c.textBaseline='middle'; c.fillText(yLabel,0,0); c.restore();
  series.forEach(s=>{
    c.strokeStyle = s.color; c.lineWidth = 2; c.beginPath();
    s.y.forEach((v,i)=>{ const x=X(s.x[i]), y=Y(v); i?c.lineTo(x,y):c.moveTo(x,y); });
    c.stroke();
    if (s.dots!==false) s.y.forEach((v,i)=>{
      c.fillStyle = s.color; c.beginPath();
      c.arc(X(s.x[i]), Y(v), 2.6, 0, Math.PI*2); c.fill();
    });
  });
}
"""


def _fmt_int(v):
    return f"{int(v):,}"


def _fmt_params(v):
    if v >= 1e9:
        return f"{v/1e9:.3f}B"
    return f"{v/1e6:.1f}M"


def _human_min(sec):
    if sec < 90:
        return f"{sec:.1f} 秒"
    if sec < 5400:
        return f"{sec/60:.1f} 分钟"
    if sec < 86400 * 2:
        return f"{sec/3600:.1f} 小时"
    return f"{sec/86400:.1f} 天"


def render(m: dict, out_path: str) -> str:
    esc = lambda v: html.escape(str(v))
    a = m["args"]
    steps = m.get("steps", [])
    dry = m.get("dry_run")

    pills = [
        ("模型", a["model"]),
        ("主干参数", _fmt_params(m["base_params"])),
        ("可训练", f"{_fmt_int(m['trainable_params'])} ({100*m['trainable_params']/m['base_params']:.3f}%)"),
        ("LoRA", f"r={a['lora_r']} α={a['lora_alpha']}"),
        ("数据集", f"{a['dataset']} ({_fmt_int(m['dataset_rows'])} 行 → 用 {m['used_samples']})"),
        ("seq_len", str(a["seq_len"])),
        ("dtype", a["dtype"]),
        ("threads", str(a["threads"])),
    ]
    if dry:
        pills.append(("峰值 RSS", f"{dry['peak_rss_gib']:.2f} GiB"))
    else:
        pills.append(("s/step", f"{m['mean_step_s']:.1f}"))
        pills.append(("tokens/s", f"{m['tokens_per_s']:.1f}"))
        pills.append(("峰值 RSS", f"{m['peak_rss_gib']:.2f} GiB"))
    strip = "".join(f"<span class='pill'>{esc(k)} <b>{esc(v)}</b></span>" for k, v in pills)

    # ---- per-step table
    rows = "".join(
        f"<tr><td class='num'>{s['step']}</td>"
        f"<td class='num'>{s['loss']:.4f}</td>"
        f"<td class='num'>{s['grad_norm']:.3f}</td>"
        f"<td class='num'>{s['fwd_s']:.1f}</td>"
        f"<td class='num'>{s['bwd_s']:.1f}</td>"
        f"<td class='num'>{s['fwd_s']+s['bwd_s']:.1f}</td>"
        f"<td class='num'>{s['tokens']}</td>"
        f"<td class='num'>{s['tokens']/max(s['fwd_s']+s['bwd_s'],1e-9):.1f}</td>"
        f"<td class='num'>{s['rss_gib']:.2f}</td></tr>"
        for s in steps
    )
    step_table = (
        "<table><tr><th>step</th><th class='num'>loss</th><th class='num'>grad_norm</th>"
        "<th class='num'>fwd s</th><th class='num'>bwd s</th><th class='num'>合计 s</th>"
        "<th class='num'>tokens</th><th class='num'>tok/s</th><th class='num'>RSS GiB</th></tr>"
        f"{rows}</table>"
    )

    chart_data = {
        "loss": [{"x": [s["step"] for s in steps], "y": [s["loss"] for s in steps],
                  "color": "#2f6fd0"}],
        "time": [{"x": [s["step"] for s in steps],
                  "y": [round(s["fwd_s"] + s["bwd_s"], 2) for s in steps],
                  "color": "#1d9e75"}],
        "split": [
            {"x": [s["step"] for s in steps], "y": [s["fwd_s"] for s in steps],
             "color": "#2f6fd0", "dots": False},
            {"x": [s["step"] for s in steps], "y": [s["bwd_s"] for s in steps],
             "color": "#c98a1b", "dots": False},
        ],
    }

    # ---- projections
    proj = ""
    if steps:
        n_full = m["dataset_rows"]
        per_step = m["mean_step_s"]
        # one optimizer step consumes batch_size * grad_accum samples
        consumed = a.get("batch_size", 1) * a.get("grad_accum", 1)
        steps_per_epoch = n_full / consumed
        epoch_s = steps_per_epoch * per_step
        proj = (
            f"<li>按当前配置（每步消耗 <b>{consumed}</b> 条样本、{per_step:.1f} s/步），"
            f"在完整 {esc(a['dataset'])}（{_fmt_int(n_full)} 行）上跑满 1 个 epoch "
            f"约需 <b>{_human_min(epoch_s)}</b>。</li>"
        )

    if dry:
        verdict = f"""<h2>1. dry-run 结果</h2>
<table><tr><th>项</th><th class='num'>值</th></tr>
<tr><td>loss</td><td class='num'>{dry['loss']:.4f}</td></tr>
<tr><td>grad_norm</td><td class='num'>{dry['grad_norm']:.4f}</td></tr>
<tr><td>forward</td><td class='num'>{dry['fwd_s']:.2f} s</td></tr>
<tr><td>backward</td><td class='num'>{dry['bwd_s']:.2f} s</td></tr>
<tr><td>tokens（含被 mask 的 prompt）</td><td class='num'>{dry['tokens']}</td></tr>
<tr><td>峰值 RSS</td><td class='num'>{dry['peak_rss_gib']:.2f} GiB</td></tr></table>
<p class="note">这一步只验证<b>一次真实的前向+反向能不能在本机跑通</b>：重点是 GDN
线性注意力层在 CPU 上有没有可用的 autograd 路径，以及 6 层标准注意力在
<code>eager</code> 下能否反传。能出 loss 且 grad_norm &gt; 0，说明梯度确实到达了
LoRA 参数。</p>"""
        body_mid = ""
    else:
        first, last = m.get("first_loss"), m.get("final_loss")
        drop = (first - last) if (first is not None and last is not None) else None
        verdict = f"""<h2>1. 训练曲线</h2>
<div class="chartbox"><canvas id="lossChart">loss</canvas></div>
<p class="note">起始 loss <b>{first:.4f}</b> → 结束 loss <b>{last:.4f}</b>
（下降 <b>{drop:+.4f}</b>）。{esc(a['steps'])} 步、每条样本平均
<b>{m['mean_tokens_per_step']:.0f}</b> token。</p>
<h2>2. 每步耗时</h2>
<div class="chartbox"><canvas id="timeChart">s/step</canvas></div>
<div class="chartbox"><canvas id="splitChart">fwd vs bwd</canvas></div>
<p class="note">蓝线为 forward，黄线为 backward。backward 明显更长是正常的：
LoRA 虽不为主干算权重梯度，但仍要对<b>全部 24 层</b>求输入梯度，
且要经过冻结的 GDN 与注意力层。</p>"""
        body_mid = f"""<h2>3. 逐 step 明细</h2>
{step_table}
<h2>4. 结论</h2>
<ul class="tight">
<li>这台机器上 0.8B 级模型的 LoRA SFT 实测 <b>{m['mean_step_s']:.1f} s/step</b>、
<b>{m['tokens_per_s']:.1f} token/s</b>（seq {esc(a['seq_len'])}、batch {esc(a.get('batch_size'))}、
fp32、{esc(a['threads'])} 线程）。</li>
<li>峰值 RSS <b>{m['peak_rss_gib']:.2f} GiB</b>，距离 16 GB 物理内存还有余量——
<b>内存不是这台机器训 0.8B 的瓶颈，时间才是</b>。</li>
{proj}
</ul>
<p class="note">为什么没有用 4bit：CPU 上 bitsandbytes 的 4bit 是「反量化成 bf16 再算」，
比原生 fp32 多一遍 dequant，<b>更慢</b>；而 0.8B 主干 fp32 也只有 ~3.4 GB，本来就装得下。
QLoRA 的价值是省显存，在 16 GB 内存的 CPU 上省不出有意义的东西。
真正的省内存手段是 LoRA 本身：主干冻结，不带梯度也不带优化器状态。</p>"""

    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    sub = ("实测：Qwen3.5-0.8B 在纯 CPU 上做 LoRA SFT。"
           "本机为 i7-1165G7（4 核 / 8 线程，AVX512，无 GPU），16 GB 内存。"
           "数字全部来自本目录一次真实运行。")

    return f"""<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Qwen3.5-0.8B CPU LoRA 实测</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Qwen3.5-0.8B 在 CPU 上做 LoRA 微调</h1>
<p class="sub">{sub}</p>
<div>{strip}</div>
{verdict}
{body_mid}
<footer class="muted">生成于 {generated} · 数据来自 <code>{esc(os.path.relpath(out_path))}</code> 对应的 metrics.json</footer>
</div>
<script id="run-data" type="application/json">{json.dumps(chart_data)}</script>
<script>{JS}
(function(){{
  const d = JSON.parse(document.getElementById('run-data').textContent);
  const el = id => document.getElementById(id);
  try {{
    if (el('lossChart')) drawLine(el('lossChart'), d.loss, 'loss');
    if (el('timeChart')) drawLine(el('timeChart'), d.time, 's / step');
    if (el('splitChart')) drawLine(el('splitChart'), d.split, 's');
  }} catch (e) {{ console.error(e); }}
}})();
</script></body></html>"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="dir containing metrics.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    mp = os.path.join(args.run_dir, "metrics.json")
    with open(mp, encoding="utf-8") as fh:
        m = json.load(fh)
    tag = os.path.basename(os.path.normpath(args.run_dir))
    out = args.out or os.path.join(OUTREPORTS, f"{tag}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render(m, out))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
