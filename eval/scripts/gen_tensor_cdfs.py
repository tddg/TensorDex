#!/usr/bin/env python3
"""Build per-model per-tensor-fetch-latency CDF panels (inline SVG) from
event traces: i4i cold (r0) / i4i warm (r1) / direct-S3 td (pooled reps)."""
import glob
import json
import math
import os
import sys

I4I = "/home/ubuntu/TensorDex/eval/results/i4i_server"
SC = "/home/ubuntu/TensorDex/eval/results/single_client"
OUT = sys.argv[1] if len(sys.argv) > 1 else "cdf_snippet.html"

MODELS = [  # (trace key fragment, display name)
    ("AhmedSSoliman_llama-3.2-3b-chat-doctor", "chat-doctor 3B"),
    ("Gabbar01_llama-3.2-3b-GSOC-DATASET", "GSOC 3B"),
    ("BanglaLLM_BanglaLLama-3.2-3b", "Bangla 3B"),
    ("KPEP_krx-qwen-2.5-7b", "krx-qwen 7B"),
    ("HPAI-BSC_Qwen2.5-7B", "Egida-DPO 7B"),
    ("mlfoundations-dev_hp_ablations", "hp-ablations 7B"),
    ("princeton-nlp_gemma-2-9b-it-SimPO", "SimPO 9B"),
    ("TongZheng1999_gemma-2-9b", "Tong-star 9B"),
    ("AlexBefest_WoonaV1.2-9b", "Woona 9B"),
]
SERIES = [("s3", "direct S3", "--s1"), ("cold", "cold · i4i", "--s3"),
          ("warm", "warm · i4i", "--s4")]


def parse(path):
    starts, lats, events = {}, [], []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            ev = e["event"]
            if ev == "s3_request_start":
                rid = e["request_id"]
                if rid not in starts:          # first attempt only
                    starts[rid] = e["ts_ns"]
                    events.append((e["ts_ns"], 1))
            elif ev == "s3_request_end":
                rid = e["request_id"]
                if rid in starts:
                    lats.append((e["ts_ns"] - starts[rid]) / 1e6)
                    events.append((e["ts_ns"], -1))
    cur = peak = 0
    for _, d in sorted(events):
        cur += d
        peak = max(peak, cur)
    return lats, peak


def find(pattern):
    return sorted(glob.glob(pattern))


def quantile(sorted_v, q):
    i = q * (len(sorted_v) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (i - lo)


def fmt(ms):
    return f"{ms/1000:.1f} s" if ms >= 1000 else (
        f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms")


data = {}
for frag, disp in MODELS:
    d = {}
    cold = find(f"{I4I}/events_tensordex_srv-{frag}*-r0-*.jsonl")
    warm = find(f"{I4I}/events_tensordex_srv-{frag}*-r1-*.jsonl")
    s3 = find(f"{SC}/events_tensordex-{frag}*.jsonl")
    assert len(cold) == 1 and len(warm) == 1 and s3, (frag, cold, warm, s3)
    lats, pk = parse(cold[0])
    d["cold"] = (sorted(lats), pk, 1)
    lats, pk = parse(warm[0])
    d["warm"] = (sorted(lats), pk, 1)
    pool, pks = [], []
    for p in s3:
        lats, pk = parse(p)
        if pk > 10:      # early aborted-protocol runs used x32; keep only
            continue     # runs concurrency-matched to the i4i protocol
        pool += lats
        pks.append(pk)
    d["s3"] = (sorted(pool), max(pks), len(pks))
    data[disp] = d

gmin = min(v[0][0] for d in data.values() for v in d.values())
gmax = max(v[0][-1] for d in data.values() for v in d.values())
print(f"global latency range: {gmin:.2f} ms .. {gmax:.0f} ms")
for disp, d in data.items():
    row = "  ".join(
        f"{k}: n={len(v[0])} p50={fmt(quantile(v[0],.5))} "
        f"p99={fmt(quantile(v[0],.99))} maxinflight={v[1]}"
        for k, v in d.items())
    print(f"{disp:16s} {row}")

# ---- SVG ----
LMIN, LMAX = math.log10(1), math.log10(gmax * 1.1)
PX0, PX1, PY0, PY1 = 40, 332, 14, 172   # plot box in a 340x212 viewBox
DEC = [1, 10, 100, 1000, 10000, 100000]
DECL = ["1ms", "10ms", "0.1s", "1s", "10s", "100s"]


def X(ms):
    return PX0 + (math.log10(max(ms, 1)) - LMIN) / (LMAX - LMIN) * (PX1 - PX0)


def Y(f):
    return PY1 - f * (PY1 - PY0)


def curve(vals):
    n = 240
    pts = [(X(quantile(vals, i / n)), Y(i / n)) for i in range(n + 1)]
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)


panels = []
for pi, (frag, disp) in enumerate(MODELS):
    d = data[disp]
    g = []
    for dec, lab in zip(DEC, DECL):
        if dec > 10 ** LMAX:
            continue
        x = X(dec)
        g.append(f'<line x1="{x:.1f}" y1="{PY0}" x2="{x:.1f}" y2="{PY1}" '
                 f'stroke="var(--line-soft)" stroke-width="1"/>')
        g.append(f'<text x="{x:.1f}" y="{PY1+14}" text-anchor="middle" '
                 f'font-size="9" fill="var(--ink-3)">{lab}</text>')
    for f, lab in [(0, "0"), (.5, "50%"), (1, "100%")]:
        y = Y(f)
        g.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" '
                 f'stroke="var(--line-soft)" stroke-width="1"/>')
        g.append(f'<text x="{PX0-5}" y="{y+3:.1f}" text-anchor="end" '
                 f'font-size="9" fill="var(--ink-3)">{lab}</text>')
    for key, name, var in SERIES:
        vals, pk, nrep = d[key]
        p50, p90, p99 = (quantile(vals, q) for q in (.5, .9, .99))
        tip = (f"{disp} — {name}: p50 {fmt(p50)} · p90 {fmt(p90)} · "
               f"p99 {fmt(p99)} · max {fmt(vals[-1])} · n={len(vals)}"
               + (f" (pooled {nrep} reps)" if nrep > 1 else "")
               + f" · max in-flight {pk}")
        pts = curve(vals)
        g.append(f'<polyline points="{pts}" fill="none" '
                 f'stroke="var({var})" stroke-width="2" '
                 f'stroke-linejoin="round"/>')
        g.append(f'<polyline points="{pts}" fill="none" stroke="transparent" '
                 f'stroke-width="12" class="cdftrack" data-tip="{tip}" '
                 f'style="pointer-events:stroke"/>')
    if pi == 0:  # direct labels once
        for key, name, var, fy in [("s3", "direct S3", "--s1", .30),
                                   ("cold", "cold", "--s3", .58),
                                   ("warm", "warm", "--s4", .86)]:
            vals = d[key][0]
            x = X(quantile(vals, fy)) + 7
            y = Y(fy)
            g.append(f'<circle cx="{x-3:.1f}" cy="{y-3:.1f}" r="3" '
                     f'fill="var({var})"/>')
            g.append(f'<text x="{x+3:.1f}" y="{y:.1f}" font-size="9.5" '
                     f'font-weight="600" fill="var(--ink-2)">{name}</text>')
    panels.append(
        f'<div><div class="panel-name">{disp}</div>'
        f'<svg viewBox="0 0 340 212" role="img" '
        f'aria-label="Per-tensor fetch latency CDF, {disp}">'
        + "".join(g) + "</svg></div>")

rows = []
for frag, disp in MODELS:
    cells = [f'<td class="t">{disp}</td>']
    for key, _, _ in SERIES:
        vals = data[disp][key][0]
        cells.append("<td>" + " / ".join(
            fmt(quantile(vals, q)) for q in (.5, .9, .99)) + "</td>")
    rows.append("<tr>" + "".join(cells) + "</tr>")

html = f"""
  <div class="fig" id="fig-cdf">
    <p class="fig-title">Per-tensor fetch latency CDFs, per model</p>
    <p class="fig-sub">Latency of each individual object GET (request issue &rarr; last byte,
    log scale) as the client actually experienced it — including queueing at the client's
    request concurrency (max in-flight &asymp;10 in every run, annotated per curve in the
    tooltips). Direct S3 pools both Stage&nbsp;1 reps and fetches unique dedup'd blobs; the
    serving-tier curves are single runs fetching tensor objects. Hover a curve for
    percentiles.</p>
    <div class="legend">
      <span class="key"><span class="swatch" style="background:var(--s1)"></span>direct S3 (t3, Stage 1)</span>
      <span class="key"><span class="swatch" style="background:var(--s3)"></span>cold · i4i (demand-miss materialization)</span>
      <span class="key"><span class="swatch" style="background:var(--s4)"></span>warm · i4i (NVMe cache hit)</span>
    </div>
    <div class="cdfgrid">
{chr(10).join(panels)}
    </div>
    <div class="tablewrap" style="margin-top:14px"><table>
      <thead><tr><th style="text-align:left">model</th><th>direct S3 p50/p90/p99</th><th>i4i cold p50/p90/p99</th><th>i4i warm p50/p90/p99</th></tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table></div>
    <p class="fig-sub" style="margin-top:10px">Two readings. <strong>Warm medians are
    7&ndash;9&times; below direct S3</strong> (10&ndash;12 ms vs 71&ndash;89 ms on the 3Bs;
    45&ndash;77 ms vs 251&ndash;373 ms on 7B/9B): an NVMe hit over one VPC hop beats an S3 GET
    at every percentile, and the warm p99 (~0.5 s on 3B, ~3 s on 7B/9B) is just the largest
    shard-sized tensors draining through the client pipeline. <strong>The 3B cold e2e spread
    (9.9 / 28.4 / 43.2 s) is not in the distribution body</strong> &mdash; cold p50s are a
    uniform 206&ndash;228 ms; the gap is one 752 MiB embedding tensor per model, served in
    1.8 s (chat-doctor) vs 21.5 s (GSOC) vs 15&ndash;20 s &times;2 (Bangla, which has two).
    That is the &gt;512 MB demand-budget spill path (&sect;3's open item) caught in the act:
    the whole family's cold story is a single-tensor tail, not a distribution shift.</p>
  </div>
"""
with open(OUT, "w") as f:
    f.write(html)
print("wrote", OUT, len(html), "bytes")
