#!/usr/bin/env python3
"""Generate the self-contained campaign report HTML with inline SVG charts."""
import json

import numpy as np

S = json.load(open("/mnt/nvme0/campaign/plan/report_stats.json"))
OUT = "/home/ubuntu/TensorDex/eval/results/full_hub_campaign_report.html"

W, H, ML, MR, MT, MB = 860, 300, 52, 16, 14, 40
PW, PH = W - ML - MR, H - MT - MB


def poly(xs, ys, xmax, ymax):
    pts = []
    for x, y in zip(xs, ys):
        px = ML + x / xmax * PW
        py = MT + PH - y / ymax * PH
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


# ---- CDF chart (reduction distribution) ----
xs = S["cdf_x"]
p_count = poly(xs, S["cdf_count"], 1.0, 1.0)
p_bytes = poly(xs, S["cdf_bytes"], 1.0, 1.0)
gridx = "".join(
    f'<line x1="{ML + f*PW:.0f}" y1="{MT}" x2="{ML + f*PW:.0f}" y2="{MT+PH}" class="grid"/>'
    f'<text x="{ML + f*PW:.0f}" y="{MT+PH+18}" class="axis" text-anchor="middle">{int(f*100)}%</text>'
    for f in (0, .25, .5, .75, 1))
gridy = "".join(
    f'<line x1="{ML}" y1="{MT + PH - f*PH:.0f}" x2="{ML+PW}" y2="{MT + PH - f*PH:.0f}" class="grid"/>'
    f'<text x="{ML-8}" y="{MT + PH - f*PH + 4:.0f}" class="axis" text-anchor="end">{int(f*100)}%</text>'
    for f in (0, .25, .5, .75, 1))
med = S["red_pcts"]["50"]
cdf_svg = f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="CDF of per-tensor reduction ratio" id="cdfsvg">
{gridx}{gridy}
<line x1="{ML + med*PW:.1f}" y1="{MT}" x2="{ML + med*PW:.1f}" y2="{MT+PH}" class="refline"/>
<text x="{ML + med*PW + 6:.1f}" y="{MT+14}" class="note">median {med*100:.0f}%</text>
<polyline points="{p_count}" fill="none" class="s1" stroke-width="2"/>
<polyline points="{p_bytes}" fill="none" class="s2" stroke-width="2"/>
<text x="{ML+PW-4}" y="{MT+PH+34}" class="axis" text-anchor="end">per-tensor reduction ratio (1 − compressed/original)</text>
<line id="cdf-cross" x1="0" x2="0" y1="{MT}" y2="{MT+PH}" class="cross" visibility="hidden"/>
<rect id="cdf-hit" x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="transparent"/>
</svg>'''

# ---- stage core-hours bar ----
stages = [("S3 fetch", S["stage_hours"]["fetch"]),
          ("upload", S["stage_hours"]["upload"]),
          ("verify (decode+hash)", S["stage_hours"]["verify"]),
          ("TensorX compress", S["stage_hours"]["compress"])]
tot_h = sum(v for _, v in stages)
bw, bh, bml = 860, 190, 190
rows = []
for i, (name, v) in enumerate(stages):
    y = 18 + i * 42
    w = v / stages[0][1] * (bw - bml - 120)
    rows.append(
        f'<text x="{bml-10}" y="{y+15}" class="lbl" text-anchor="end">{name}</text>'
        f'<rect x="{bml}" y="{y}" width="{max(w,3):.0f}" height="22" rx="4" class="bar s1f" data-tip="{name}: {v:.0f} core-hours ({v/tot_h*100:.0f}%)"/>'
        f'<text x="{bml + max(w,3) + 8:.0f}" y="{y+15}" class="val">{v:.0f} core-h · {v/tot_h*100:.0f}%</text>')
stage_svg = f'<svg viewBox="0 0 {bw} {bh}" role="img" aria-label="Core-hours per pipeline stage">{"".join(rows)}</svg>'

# ---- timeline: pairs + bytes (two stacked mini charts, shared time axis) ----
ev = [(0,0),(8,31),(28,1481),(48,3025),(68,12576),(88,28396),(108,44219),
      (128,60095),(148,75170),(168,89180),(188,103481),(208,119564),
      (228,135665),(248,157386),(268,199526),(288,261756),(308,305304),
      (328,354536),(348,453729),(354,581274)]
by = [(0,0),(65,3.29),(83,5.08),(128,8.14),(158,14.23),(180,16.41),
      (241,21.88),(270,24.20),(322,30.22),(354,32.12)]
tmax = 354
tw, th, tml, tmt, tmb = 860, 200, 52, 12, 34
tph = th - tmt - tmb
def tpoly(data, ymax, ph, mt):
    return " ".join(f"{tml + t/tmax*(tw-tml-16):.1f},{mt + ph - v/ymax*ph:.1f}" for t, v in data)
timegrid = "".join(
    f'<line x1="{tml + m/tmax*(tw-tml-16):.0f}" y1="{tmt}" x2="{tml + m/tmax*(tw-tml-16):.0f}" y2="{tmt+tph}" class="grid"/>'
    f'<text x="{tml + m/tmax*(tw-tml-16):.0f}" y="{th-12}" class="axis" text-anchor="middle">{m//60}h</text>'
    for m in (0, 60, 120, 180, 240, 300, 354))
tl1 = f'''<svg viewBox="0 0 {tw} {th}" role="img" aria-label="Pairs completed over wall time">
{timegrid}
<text x="{tml-8}" y="{tmt+10}" class="axis" text-anchor="end">581k</text>
<text x="{tml-8}" y="{tmt+tph}" class="axis" text-anchor="end">0</text>
<polyline points="{tpoly(ev, 581274, tph, tmt)}" fill="none" class="s1" stroke-width="2"/>
<text x="{tml+6}" y="{tmt+14}" class="note">pairs completed</text>
</svg>'''
tl2 = f'''<svg viewBox="0 0 {tw} {th}" role="img" aria-label="Target TB completed over wall time">
{timegrid}
<text x="{tml-8}" y="{tmt+10}" class="axis" text-anchor="end">32.1</text>
<text x="{tml-8}" y="{tmt+tph}" class="axis" text-anchor="end">0</text>
<polyline points="{tpoly(by, 32.12, tph, tmt)}" fill="none" class="s2" stroke-width="2"/>
{"".join(f'<circle cx="{tml + t/tmax*(tw-tml-16):.1f}" cy="{tmt + tph - v/32.12*tph:.1f}" r="4" class="dot2" data-tip="{v} TB at +{t//60}h{t%60:02d}m"/>' for t, v in by)}
<text x="{tml+6}" y="{tmt+14}" class="note">target TB completed (measured checkpoints)</text>
</svg>'''

r = S["red_pcts"]
g = S["get_rate_pcts"]; c = S["compress_rate_pcts"]; v_ = S["verify_rate_pcts"]; u = S["upload_rate_pcts"]

html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TensorDex full-hub compression campaign</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ margin:0; font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--surface-2); color: var(--text-primary); }}
.viz-root {{
  --surface-1:#fcfcfb; --surface-2:#f4f4f2; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --text-muted:#84837e; --line:#e3e2de;
  --series-1:#2a78d6; --series-2:#eb6834; --good:#008300;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    --surface-1:#1a1a19; --surface-2:#111110; --text-primary:#fff;
    --text-secondary:#c3c2b7; --text-muted:#8f8e86; --line:#33322f;
    --series-1:#3987e5; --series-2:#d95926; --good:#4caf50;
  }}
}}
:root[data-theme="dark"] .viz-root {{
  --surface-1:#1a1a19; --surface-2:#111110; --text-primary:#fff;
  --text-secondary:#c3c2b7; --text-muted:#8f8e86; --line:#33322f;
  --series-1:#3987e5; --series-2:#d95926; --good:#4caf50;
}}
main {{ max-width: 960px; margin: 0 auto; padding: 28px 20px 60px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 18px; margin: 40px 0 10px; }}
.sub {{ color: var(--text-secondary); margin-bottom: 24px; }}
.tiles {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap:12px; }}
.tile {{ background: var(--surface-1); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.tile .v {{ font-size:28px; font-weight:650; letter-spacing:-.5px; }}
.tile .k {{ color: var(--text-secondary); font-size:13px; }}
.tile .d {{ color: var(--text-muted); font-size:12px; margin-top:2px; }}
.card {{ background: var(--surface-1); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin-top:12px; overflow-x:auto; }}
svg {{ width:100%; height:auto; display:block; }}
.grid {{ stroke: var(--line); stroke-width:1; }}
.axis {{ fill: var(--text-muted); font-size:12px; }}
.lbl {{ fill: var(--text-secondary); font-size:13px; }}
.val {{ fill: var(--text-primary); font-size:13px; font-weight:600; }}
.note {{ fill: var(--text-secondary); font-size:12px; }}
.s1 {{ stroke: var(--series-1); }} .s2 {{ stroke: var(--series-2); }}
.s1f {{ fill: var(--series-1); }} .dot2 {{ fill: var(--series-2); stroke: var(--surface-1); stroke-width:2; }}
.refline {{ stroke: var(--text-muted); stroke-dasharray: 4 4; }}
.cross {{ stroke: var(--text-muted); stroke-width:1; }}
.legend {{ display:flex; gap:18px; font-size:13px; color:var(--text-secondary); margin:6px 0 0 52px; }}
.sw {{ display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }}
table {{ border-collapse: collapse; width:100%; font-size:14px; }}
th, td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); }}
th:first-child, td:first-child {{ text-align:left; }}
th {{ color: var(--text-secondary); font-weight:600; }}
.ok {{ color: var(--good); font-weight:650; }}
#tip {{ position:fixed; pointer-events:none; background:var(--surface-1); border:1px solid var(--line);
  border-radius:6px; padding:4px 8px; font-size:12px; visibility:hidden; z-index:9; color:var(--text-primary); }}
ol li, ul li {{ margin: 5px 0; }}
code {{ background: var(--surface-2); padding:1px 5px; border-radius:4px; font-size:13px; }}
.small {{ font-size:13px; color:var(--text-secondary); }}
</style></head>
<body class="viz-root"><div id="tip"></div><main>
<h1>TensorDex full-hub compression campaign</h1>
<div class="sub">FlexSplit + TensorX (level 1) over the live origin store — 2,892 models · 642,964 physical tensors · 34.81&nbsp;TB —
executed 2026-08-04/05 on one i4i.8xlarge · <b>0 errors</b> · every pair roundtrip-verified byte-exact</div>

<div class="tiles">
  <div class="tile"><div class="v">60.9%</div><div class="k">dataset reduction ratio</div><div class="d">physical-unique bytes: 34.81 → 13.61 TB stored (0.39×)</div></div>
  <div class="tile"><div class="v">65.2%</div><div class="k">reduction, logical accounting</div><div class="d">dedup savings included (39.1 TB logical basis)</div></div>
  <div class="tile"><div class="v">5.9 h</div><div class="k">compression wall time</div><div class="d">581,274 pairs · 34.2 TB read · 10.9 TB written</div></div>
  <div class="tile"><div class="v">66.0%</div><div class="k">reduction on delta&#8209;compressed tensors</div><div class="d">weighted; 32.12 → 10.91 TB (tratio 0.3395)</div></div>
</div>

<h2>1 · Per-tensor reduction distribution</h2>
<div class="card">
{cdf_svg}
<div class="legend"><span><span class="sw" style="background:var(--series-1)"></span>share of tensors (n = {S['n']:,})</span>
<span><span class="sw" style="background:var(--series-2)"></span>share of bytes (32.12 TB)</span></div>
<table style="margin-top:14px"><tr><th>percentile</th><th>p5</th><th>p10</th><th>p25</th><th>p50</th><th>p75</th><th>p90</th><th>p95</th><th>p99</th></tr>
<tr><td>reduction</td><td>{r['5']*100:.0f}%</td><td>{r['10']*100:.0f}%</td><td>{r['25']*100:.0f}%</td><td>{r['50']*100:.0f}%</td><td>{r['75']*100:.0f}%</td><td>{r['90']*100:.0f}%</td><td>{r['95']*100:.0f}%</td><td>{r['99']*100:.1f}%</td></tr></table>
<p class="small">{S['frac_ge_0.5']*100:.0f}% of tensors ({S['bytes_ge_0.5']*100:.0f}% of bytes) shrink by ≥50%; {S['frac_ge_0.9']*100:.1f}% shrink by ≥90%.
Raw tensors (5,368 bases + 56,097 unattached/skipped, 2.69&nbsp;TB) sit at 0% reduction and are included in the dataset-level 60.9% but not in this per-pair distribution.
Plan provenance matters: pairs from the paper&rsquo;s FlexSplit plan realize weighted CR 0.330; the 28k fallback pairs (incremental planner) realize 0.554 — base selection, not the codec, is the dominant lever.</p>
</div>

<h2>2 · Runtime and progress</h2>
<div class="card">{tl1}{tl2}
<p class="small">Wall time 21,228 s (5 h 54 m). The byte curve is front-loaded (largest base-groups scheduled first);
the pair curve accelerates in the small-tensor tail where the run becomes request-rate-bound rather than bandwidth-bound.</p></div>

<h2>3 · Where the time went (per-pair stage timing, summed)</h2>
<div class="card">{stage_svg}
<p class="small">S3 I/O (fetch + upload) is 87% of aggregate stage time; TensorX compute is {S['stage_hours']['compress']/tot_h*100:.0f}%.
Compression at hub scale is I/O-bound — the paper&rsquo;s claim, observed in the wild.</p></div>

<h2>4 · Throughput</h2>
<div class="card">
<table>
<tr><th>path</th><th>p10</th><th>p50</th><th>p90</th><th>aggregate</th></tr>
<tr><td>S3 GET (per object, under ~112 concurrent streams)</td><td>{g['10']:.1f} MB/s</td><td>{g['50']:.0f} MB/s</td><td>{g['90']:.0f} MB/s</td><td><b>1.5–1.9 GB/s</b> sustained NIC (13–15 Gbps); 56 MB/s single-stream unloaded</td></tr>
<tr><td>TensorX compress (per core)</td><td>{c['10']:.0f} MB/s</td><td>{c['50']:.0f} MB/s</td><td>{c['90']:.0f} MB/s</td><td>{S['agg_compress_MBps_core']:.0f} MB/s·core mean; 28 workers ≫ wire rate</td></tr>
<tr><td>roundtrip verify = decompress + XXH3 (per core)</td><td>{v_['10']:.0f} MB/s</td><td>{v_['50']:.0f} MB/s</td><td>{v_['90']:.0f} MB/s</td><td>{S['agg_verify_MBps_core']:.0f} MB/s·core mean</td></tr>
<tr><td>S3 PUT (per delta blob)</td><td>{u['10']:.1f} MB/s</td><td>{u['50']:.0f} MB/s</td><td>{u['90']:.0f} MB/s</td><td>10.9 TB uploaded in-line</td></tr>
</table>
<p class="small">p10 per-object rates are latency-dominated small objects (a 100&nbsp;KB norm tensor &ldquo;transfers&rdquo; at &lt;1&nbsp;MB/s in one RTT).
<b>Local disk:</b> the data plane is deliberately disk-free — tensors stream S3 → RAM → codec → S3, so instance NVMe saw only metadata traffic:
node-DB recovery (120 GB in at ~450 MB/s), the 6.1 GB fingerprint arena (memmap), SQLite catalog builds, and ~0.5 GB of append-only ledgers.
Peak RAM ≈ 190 GB of 247 GB during the giant-tensor phase.</p>
</div>

<h2>5 · Validation and fidelity</h2>
<div class="card">
<ul>
<li><b>Per-pair integrity (100% coverage):</b> every one of the 581,274 deltas was decompressed against its base and XXH3-128-verified byte-exact <i>before</i> upload. <span class="ok">0 mismatches</span>.</li>
<li><b>Against the published artifact (TensorX):</b> on the 195,839 pairs (12.79 TB) where the paper&rsquo;s results.db holds a measured tratio for the identical (target, base): our weighted tratio 0.3063 vs published 0.3068; per-pair relative difference 0.0000% at p50 and p95; 98.2% byte-identical (the 1.77% that differ are sub-MB tensors, 91% smaller on our newer zstd).</li>
<li><b>Paper claims:</b> TensorDex-TX <b>65.1%</b> reproduced exactly offline from the AE cache; realized 60.9% here, the 4.2-point gap fully attributed to corpus decay (5.31 TB of registered tensors — including 1,058 of the paper&rsquo;s 7,948 bases — have no blobs in S3).</li>
<li><b>FM++ (measurement-only; hub stays TensorX):</b> 500-pair stratified sample — 500/500 roundtrips byte-exact, 400/400 published fratios reproduced <i>byte-identically</i>; sample-informed projection validates the <b>70.5%</b> claim at <b>71.1%</b> on the paper corpus (66.8% on the surviving corpus).</li>
<li><b>Whole-model reconstruction:</b> random fully-servable models rebuilt end-to-end from the hybrid layout (deltas from <code>compressed_full/</code> + bases from <code>tensordb/</code>), every tensor hash-verified — sampling run in progress at publish time; see repo report for the final stamp.</li>
</ul>
</div>

<h2>6 · Process</h2>
<div class="card"><ol>
<li><b>Census &amp; corpus truth.</b> Registry: 742,027 tensors / 40.1 TB / 2,894 models. Full-prefix LIST: 99,063 registered tensors (5.31 TB, 13.4%) have <i>no blob</i> — silent metadata/blob divergence at hub scale. Physical corpus: 642,964 tensors / 34.81 TB.</li>
<li><b>Fingerprint recovery — no 40 TB sketch pass.</b> Live master.db fps are an old 256-dim format; the paper-config BCS sketches (d=2, w=1024) were recovered from <code>backups/node_0..15</code> DBs, bit-exact vs today&rsquo;s kernel; 167 stragglers sketched fresh. 100% coverage of the physical corpus.</li>
<li><b>Planning.</b> A NumPy/BLAS reimplementation of the Rust incremental planner (verified bit-equivalent) plans 643k tensors in 25 s — but predicts weighted CR only 0.534. Diffing against the paper&rsquo;s shipped plan exposed the gap (11% same-base choices; theirs realizes 0.333), so the campaign <b>adopts the paper&rsquo;s own plan</b>: 552,781 surviving pairs + 28,718 fallback + AE conflict resolution, depth-1 everywhere.</li>
<li><b>Execution.</b> 28 processes × base-grouped 4 GB units: fetch → TensorX L1 → roundtrip verify → wrap (engine blob format) → upload. Resumable per-pair JSONL ledger; 245 cross-dtype pairs skipped raw.</li>
<li><b>Catalog.</b> <code>compressed_full/master.db</code>: full schema + <code>tensor_deltas</code> (581,254 edges), per-tensor storage_uri/physical/logical bytes, <code>missing://</code> flags for GC&rsquo;d tensors.</li>
</ol></div>

<h2>7 · Artifact layout (build on this)</h2>
<div class="card"><table>
<tr><th>artifact</th><th>location</th></tr>
<tr><td>581k TensorX delta blobs (self-describing headers)</td><td><code>s3://tensor-tingfeng/compressed_full/blobs/&#123;tid[:2]&#125;/&#123;tid&#125;.safetensors</code></td></tr>
<tr><td>raw bases + unattached tensors (untouched)</td><td><code>s3://tensor-tingfeng/tensordb/blobs/…</code></td></tr>
<tr><td>catalog (tensors · mappings · deltas · flags)</td><td><code>s3://tensor-tingfeng/compressed_full/master.db</code></td></tr>
<tr><td>plan + per-pair ledger + logs</td><td><code>s3://tensor-tingfeng/compressed_full/meta/campaign_meta.tar.gz</code></td></tr>
<tr><td>pipeline code</td><td><code>eval/scripts/full_hub/</code> (repo)</td></tr>
</table></div>

<p class="small" style="margin-top:26px">Generated 2026-08-05 · TensorDex FAST&rsquo;27 evaluation · companion doc: eval/results/FULL_HUB_CAMPAIGN.md</p>
</main>
<script>
const tip = document.getElementById('tip');
function showTip(e, text) {{
  tip.textContent = text; tip.style.visibility = 'visible';
  tip.style.left = Math.min(e.clientX + 12, innerWidth - 180) + 'px';
  tip.style.top = (e.clientY + 12) + 'px';
}}
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mousemove', e => showTip(e, el.dataset.tip));
  el.addEventListener('mouseleave', () => tip.style.visibility = 'hidden');
}});
// CDF crosshair
const svg = document.getElementById('cdfsvg');
const hit = document.getElementById('cdf-hit');
const cross = document.getElementById('cdf-cross');
const CDF_X = {json.dumps([round(x,4) for x in S['cdf_x']])};
const CDF_C = {json.dumps([round(y,4) for y in S['cdf_count']])};
const CDF_B = {json.dumps([round(y,4) for y in S['cdf_bytes']])};
if (svg && hit) hit.addEventListener('mousemove', e => {{
  const r = svg.getBoundingClientRect();
  const fx = ((e.clientX - r.left) / r.width * {W} - {ML}) / {PW};
  const x = Math.max(0, Math.min(1, fx));
  const i = Math.round(x * (CDF_X.length - 1));
  const px = {ML} + x * {PW};
  cross.setAttribute('x1', px); cross.setAttribute('x2', px);
  cross.setAttribute('visibility', 'visible');
  showTip(e, `reduction ≤ ${{(x*100).toFixed(0)}}%: ${{(CDF_C[i]*100).toFixed(1)}}% of tensors · ${{(CDF_B[i]*100).toFixed(1)}}% of bytes`);
}});
if (hit) hit.addEventListener('mouseleave', () => {{
  cross.setAttribute('visibility', 'hidden'); tip.style.visibility = 'hidden';
}});
</script>
</body></html>'''

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({len(html)/1024:.0f} KB)")
