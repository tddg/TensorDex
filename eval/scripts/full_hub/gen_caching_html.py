#!/usr/bin/env python3
"""Generate the DAG-aware caching direction doc (self-contained HTML)."""
import json
import math

S = json.load(open("/mnt/nvme0/campaign/plan/cache_direction_stats.json"))
OUT = "/home/ubuntu/TensorDex/eval/results/caching_direction.html"

# concentration chart: log-x capacity GB vs cumulative coverage
W, H, ML, MR, MT, MB = 860, 320, 52, 18, 16, 44
PW, PH = W - ML - MR, H - MT - MB
XMIN, XMAX = 1.0, 405.1


def px(gb):
    return ML + (math.log10(max(gb, XMIN)) - 0) / (math.log10(XMAX)) * PW


def poly(ys_key):
    pts = []
    for gb, y in zip(S["cap_gb"], S[ys_key]):
        if gb < XMIN:
            continue
        pts.append(f"{px(gb):.1f},{MT + PH - y * PH:.1f}")
    return " ".join(pts)


gridx = "".join(
    f'<line x1="{px(g):.0f}" y1="{MT}" x2="{px(g):.0f}" y2="{MT+PH}" class="grid"/>'
    f'<text x="{px(g):.0f}" y="{MT+PH+18}" class="axis" text-anchor="middle">{lbl}</text>'
    for g, lbl in [(1, "1"), (10, "10"), (100, "100"), (405, "405 GB")])
gridy = "".join(
    f'<line x1="{ML}" y1="{MT + PH - f*PH:.0f}" x2="{ML+PW}" y2="{MT + PH - f*PH:.0f}" class="grid"/>'
    f'<text x="{ML-8}" y="{MT + PH - f*PH + 4:.0f}" class="axis" text-anchor="end">{int(f*100)}%</text>'
    for f in (0, .25, .5, .75, 1))
marks = "".join(
    f'<circle cx="{px(g):.1f}" cy="{MT + PH - f*PH:.1f}" r="4.5" class="dot1" data-tip="{lbl}"/>'
    for g, f, lbl in [(29.2, .5, "29 GB of bases → 50% of all deltas"),
                      (106, .8, "106 GB → 80%"),
                      (162.4, .9, "162 GB → 90%")])
conc_svg = f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="Cumulative delta coverage vs base cache capacity">
{gridx}{gridy}
<polyline points="{poly('cum_deltas')}" fill="none" class="s1" stroke-width="2"/>
<polyline points="{poly('cum_delta_bytes')}" fill="none" class="s2" stroke-width="2"/>
{marks}
<text x="{ML+PW-4}" y="{MT+PH+36}" class="axis" text-anchor="end">cumulative base capacity, bases sorted by fan-in (log scale)</text>
</svg>'''

html = f'''<style>
:root {{ color-scheme: light dark; }}
.viz-root {{
  --surface-1:#fcfcfb; --surface-2:#f4f4f2; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --text-muted:#84837e; --line:#e3e2de;
  --series-1:#2a78d6; --series-2:#eb6834; --good:#008300; --warn:#eda100;
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--surface-2); color: var(--text-primary);
  display:block; margin:0;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    --surface-1:#1a1a19; --surface-2:#111110; --text-primary:#fff;
    --text-secondary:#c3c2b7; --text-muted:#8f8e86; --line:#33322f;
    --series-1:#3987e5; --series-2:#d95926; --good:#4caf50; --warn:#c98500;
  }}
}}
:root[data-theme="dark"] .viz-root {{
  --surface-1:#1a1a19; --surface-2:#111110; --text-primary:#fff;
  --text-secondary:#c3c2b7; --text-muted:#8f8e86; --line:#33322f;
  --series-1:#3987e5; --series-2:#d95926; --good:#4caf50; --warn:#c98500;
}}
.viz-root main {{ max-width: 960px; margin: 0 auto; padding: 28px 20px 60px; }}
.viz-root h1 {{ font-size: 24px; margin: 0 0 4px; text-wrap: balance; }}
.viz-root h2 {{ font-size: 18px; margin: 38px 0 10px; }}
.viz-root .sub {{ color: var(--text-secondary); margin-bottom: 22px; }}
.viz-root .tiles {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap:12px; }}
.viz-root .tile {{ background: var(--surface-1); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.viz-root .tile .v {{ font-size:26px; font-weight:650; letter-spacing:-.5px; }}
.viz-root .tile .k {{ color: var(--text-secondary); font-size:13px; }}
.viz-root .card {{ background: var(--surface-1); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin-top:12px; overflow-x:auto; }}
.viz-root svg {{ width:100%; height:auto; display:block; }}
.viz-root .grid {{ stroke: var(--line); stroke-width:1; }}
.viz-root .axis {{ fill: var(--text-muted); font-size:12px; }}
.viz-root .s1 {{ stroke: var(--series-1); }} .viz-root .s2 {{ stroke: var(--series-2); }}
.viz-root .dot1 {{ fill: var(--series-1); stroke: var(--surface-1); stroke-width:2; }}
.viz-root .legend {{ display:flex; gap:18px; font-size:13px; color:var(--text-secondary); margin:6px 0 0 52px; flex-wrap:wrap; }}
.viz-root .sw {{ display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }}
.viz-root table {{ border-collapse: collapse; width:100%; font-size:14px; }}
.viz-root th, .viz-root td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
.viz-root th {{ color: var(--text-secondary); font-weight:600; }}
.viz-root td.num, .viz-root th.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
.viz-root code {{ background: var(--surface-2); padding:1px 5px; border-radius:4px; font-size:13px; }}
.viz-root .small {{ font-size:13px; color:var(--text-secondary); }}
.viz-root .spec {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin-top:8px; }}
.viz-root .spec > div {{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:var(--surface-2); }}
.viz-root .spec b {{ display:block; font-size:13.5px; }}
.viz-root .spec span {{ font-size:12.5px; color:var(--text-secondary); }}
.viz-root ol li, .viz-root ul li {{ margin: 5px 0; }}
#ctip {{ position:fixed; pointer-events:none; background:var(--surface-1); border:1px solid var(--line);
  border-radius:6px; padding:4px 8px; font-size:12px; visibility:hidden; z-index:9; color:var(--text-primary); }}
</style>
<div class="viz-root"><div id="ctip"></div><main>
<h1>Caching a derived-data DAG: the TensorDex hot-cache research direction</h1>
<div class="sub">Why classic eviction optimizes the wrong objective for a delta-compressed model hub — evidence from the
full-hub campaign (581,254 deltas · 7,169 active bases · 34.8 TB corpus) and the experiment plan to test it.</div>

<div class="tiles">
  <div class="tile"><div class="v">1.16%</div><div class="k">of corpus bytes (405 GB) are bases — yet every delta decode needs one</div></div>
  <div class="tile"><div class="v">29 GB</div><div class="k">of top bases discounts misses for 50% of all deltas</div></div>
  <div class="tile"><div class="v">7,459</div><div class="k">deltas hang off the single hottest base (fan-in p50 = 25)</div></div>
  <div class="tile"><div class="v">4×</div><div class="k">origin-traffic gap between a cheap miss and an expensive one</div></div>
</div>

<h2>1 · The problem: misses have state-dependent costs</h2>
<div class="card">
<p>The hot cache stores <i>reconstructed</i> tensors. A request for tensor <i>x</i> (logical size L, delta size 0.34&thinsp;L on average) that misses is served by fetching from origin and decoding at the cache. What that miss <b>costs</b> depends on what else is resident:</p>
<table>
<tr><th>cache state at miss time</th><th class="num">origin bytes</th><th class="num">cache CPU</th><th>note</th></tr>
<tr><td>x&rsquo;s base resident (pinned, or hot as a servable tensor)</td><td class="num"><b>0.34 L</b></td><td class="num">1 decode</td><td>cheap miss</td></tr>
<tr><td>base not resident</td><td class="num"><b>1.34 L</b></td><td class="num">1 decode</td><td>expensive miss — and whether the base <i>stays</i> decides the cost of every sibling&rsquo;s future miss</td></tr>
<tr><td>x is raw (base / unattached)</td><td class="num">1.0 L</td><td class="num">0</td><td>classic miss</td></tr>
</table>
<p>Hit-rate-maximizing policies (LRU, LFU, size-aware variants) optimize hit fraction. But the tier&rsquo;s throughput and latency are governed by
<b>origin bytes moved + decodes performed per unit time</b>. Those are different objectives, and the gap between them grows with the DAG&rsquo;s fan-in skew.
A resident base is never &ldquo;hit&rdquo; by most requests — it silently converts expensive misses into cheap ones for thousands of descendants.</p>
</div>

<h2>2 · Evidence: fan-in concentration in the real hub</h2>
<div class="card">
{conc_svg}
<div class="legend"><span><span class="sw" style="background:var(--series-1)"></span>share of deltas whose base is resident</span>
<span><span class="sw" style="background:var(--series-2)"></span>share of delta <i>bytes</i> covered</span></div>
<p class="small">Bases sorted by descendant count, x = cumulative base capacity. The knee is extreme: 29 GB of base capacity discounts misses for half of all
581k deltas; 162 GB covers 90%; the full base set is 405 GB — about 6% of one i4i.8xlarge&rsquo;s NVMe. Fan-in: p50 25, p99 589, max 7,459.</p>
</div>

<h2>3 · The policy spectrum to evaluate</h2>
<div class="card">
<div class="spec">
<div><b>LRU / size-aware LRU</b><span>hit-rate anchor; DAG-blind. Expected to waste capacity on giant reconstructions whose bases then get evicted.</span></div>
<div><b>Base-pinned + LRU remainder</b><span>lower-bound anchor: pin all 405 GB of bases; every delta request misses cheap. Maximum miss rate, minimum per-miss cost.</span></div>
<div><b>GDSF (cost-aware)</b><span>classic miss-cost caching with <i>static</i> per-item cost. The strongest prior-art baseline — but costs here aren&rsquo;t static.</span></div>
<div><b>dag_value (ours)</b><span>value(x) = own traffic + Σ resident-descendant miss discount, recomputed as residency changes. Greedy knapsack over coupled utilities.</span></div>
</div>
<p class="small" style="margin-top:10px"><b>What&rsquo;s new vs prior art:</b> cost-aware caching (GreedyDual-Size family) assumes each item has a fixed miss cost.
Here the miss cost of x is a function of the cache&rsquo;s current contents (base resident or not), utilities couple through the DAG (non-modular value),
and items are dual-role — every base is also a servable model tensor, so its value is own-traffic + descendant-discount. Offline oracle (ILP/greedy on the trace) bounds the achievable gap.</p>
</div>

<h2>4 · Hypotheses</h2>
<div class="card"><ol>
<li><b>H1:</b> at small-to-mid capacity (cache ≪ working set), DAG-aware allocation beats LRU on origin traffic by a large factor (the 29 GB → 50% knee makes the head bases nearly free to keep); the policies converge as capacity approaches the working set.</li>
<li><b>H2:</b> hit rate and byte-weighted miss cost <i>rank policies differently</i> — the experiment that shows a policy with a worse hit rate delivering better e2e latency is the paper&rsquo;s key figure.</li>
<li><b>H3:</b> at high concurrency, the decode-CPU term of cheap misses becomes visible, merging the eviction question with the store-format question (cache deltas vs reconstructions) into a single capacity-vs-CPU frontier.</li>
<li><b>H4:</b> ancestor pre-warming (serving model A warms bases shared with model B) yields measurable cross-model latency wins under popularity-correlated workloads — the effect already observed accidentally in the i4i campaign (§4).</li>
</ol></div>

<h2>5 · Experiment design (summary)</h2>
<div class="card">
<table>
<tr><th>axis</th><th>values</th></tr>
<tr><td>workload</td><td>popularity-weighted model pulls over the 2,530 servable models (HF download counts where available, Zipf α∈{{0.8, 1.1}} synthetic); closed-loop client swarm; fixed seeds; 2 h steady-state per cell after warmup</td></tr>
<tr><td>concurrency</td><td>8 · 32 · 128 concurrent model pulls (client swarm on small instances + one large client as control)</td></tr>
<tr><td>cache capacity</td><td>50 GB · 200 GB · 800 GB · 3.2 TB (enforced by the evictor, single i4i.8xlarge node)</td></tr>
<tr><td>policy</td><td>LRU · size-aware · GDSF · base-pinned+LRU · dag_value · offline oracle (replayed)</td></tr>
<tr><td>store format</td><td>reconstructed (default); delta+decode-on-read and hybrid at one capacity point (H3)</td></tr>
<tr><td>metrics</td><td><b>origin bytes/hour</b> (primary) · byte-weighted miss cost · e2e model latency p50/p99 · TTFB · hit rate (reported to show it misleads) · decode core-s/hour · NVMe/NIC utilization</td></tr>
</table>
<p class="small">Full protocol, harness changes (evictor hooks, workload generator, per-request cost accounting), and run book:
<code>eval/CACHING_EXPERIMENTS.md</code> in the repo.</p>
</div>

<p class="small" style="margin-top:26px">TensorDex FAST&rsquo;27 · 2026-08-05 · companion docs: FULL_HUB_CAMPAIGN.md · full_hub_campaign_report.html</p>
</main></div>
<script>
const ctip = document.getElementById('ctip');
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mousemove', e => {{
    ctip.textContent = el.dataset.tip; ctip.style.visibility = 'visible';
    ctip.style.left = Math.min(e.clientX + 12, innerWidth - 220) + 'px';
    ctip.style.top = (e.clientY + 12) + 'px';
  }});
  el.addEventListener('mouseleave', () => ctip.style.visibility = 'hidden');
}});
</script>'''

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({len(html)/1024:.0f} KB)")
