# Server-path download: i4i.2xlarge deployment vs t3 prototype

2026-08-03. New results from the 18-run matrix in this directory
(`model_runs.jsonl`, per-run traces, `PROTOCOL.md`); old results from
Stage 1 (`../server_client/`, 2026-08-01). Tables are kept SIDE BY SIDE,
never merged — the two campaigns measured different deployments.

## 1. The two deployments

| | OLD: t3 prototype (Stage 1) | NEW: i4i deployment (this matrix) |
|---|---|---|
| topology | client + both servers on the SAME t3.xlarge (localhost) | dedicated i4i.2xlarge server; t3.xlarge client over VPC (private IP) |
| server HW | 4 vCPU / 16 GB shared with client; gp3 EBS cache | 8 vCPU / 64 GB dedicated; 1.8 TB instance-store NVMe |
| cache front | Python http.server, per-tensor demand materialize | nginx sendfile + X-Accel-Redirect; FastAPI materializer |
| serving code | eval/server/*.py prototypes | server/ package @ 5b9f5fb (spill + keepalive + 502 fixes) |
| client↔cache link | loopback (no NIC) | VPC network — t3 NIC (~145 MB/s sustained, burst higher) |
| cold protocol | /admin/clear before each model | full clear once, then natural first touch (see §4 caveat) |

Implication for reading the tables: OLD warm numbers measured EBS/page
cache with zero network; NEW warm numbers include a real network hop and
are bounded by the CLIENT's NIC, not by the cache. OLD cold was
per-tensor unpipelined; NEW cold is demand-miss driven by 10 parallel
client streams against the fixed keepalive path.

## 2. Side-by-side: per-model e2e seconds (median over reps)

### OLD — t3 prototype (server on same box, EBS cache)

| model | logical GB | srv cold | srv warm |
|---|---|---|---|
| AhmedSSoliman/llama-3.2-3b-chat-doctor | 1.67 | 25.8 | 3.2 |
| Gabbar01/llama-3.2-3b-GSOC-DATASET | 1.67 | 23.7 | 3.9 |
| BanglaLLM/BanglaLLama-3.2-3b | 1.77 | 38.5 | 7.1 |
| KPEP/krx-qwen-2.5-7b-v1.4.4 | 15.23 | 276.7 | 198.3 |
| HPAI-BSC/Qwen2.5-7B-Egida-DPO | 15.23 | 259.5 | 204.4 |
| mlfoundations-dev/hp_ablations_qwen | 15.23 | 290.6 | 205.1 |
| princeton-nlp/gemma-2-9b-it-SimPO | 18.48 | 299.4 | 250.7 |
| TongZheng1999/gemma-2-9b-star | 18.48 | 397.8 | 260.0 |
| AlexBefest/WoonaV1.2-9b | 18.48 | 415.5 | 258.4 |

### NEW — i4i deployment (dedicated server, NVMe, over network)

| model | logical GB | srv cold | srv warm | cold pre-cached bytes |
|---|---|---|---|---|
| AhmedSSoliman/llama-3.2-3b-chat-doctor | 1.67 | 9.9 | 3.5 | 0.3% |
| Gabbar01/llama-3.2-3b-GSOC-DATASET | 1.67 | 28.4 | 3.5 | 0% (ran first) |
| BanglaLLM/BanglaLLama-3.2-3b | 1.77 | 43.2 | 6.2 | 0.2% |
| KPEP/krx-qwen-2.5-7b-v1.4.4 | 15.23 | 117.4 | 102.3 | 2.5% |
| HPAI-BSC/Qwen2.5-7B-Egida-DPO | 15.23 | 129.2 | 101.8 | 2.1% |
| mlfoundations-dev/hp_ablations_qwen | 15.23 | 156.3 | 103.0 | 0.2% |
| princeton-nlp/gemma-2-9b-it-SimPO | 18.48 | 128.3 | 127.9 | 7.2% |
| TongZheng1999/gemma-2-9b-star | 18.48 | 186.0 | 128.1 | 0.3% |
| AlexBefest/WoonaV1.2-9b | 18.48 | 137.4 | 129.8 | 0.0% |

(pre-cached bytes: share of served bytes that were already resident from
earlier runs' shared ancestors, measured from client TTFB<50 ms — see §4.)

### Family medians, side by side

| family | OLD cold | NEW cold | OLD warm | NEW warm | ref: direct S3 (td) | ref: client decode (td_c) |
|---|---|---|---|---|---|---|
| llama-3.2-3B | 25.8 | 28.4 | 3.9 | 3.5 | 12.4 | 18.8 |
| Qwen2.5-7B | 276.7 | 129.2 | 204.4 | 102.3 | 113.1 | 138.3 |
| gemma-2-9B | 397.8 | 137.4 | 258.4 | 128.1 | 141.3 | 158.0 |

(td/td_c reference columns are Stage 1 t3 client-path baselines,
unchanged by this campaign; included for ratio context only.)

## 3. What changed and why

1. **Cold: 2.1–2.9x faster on 7B/9B; now at direct-S3 parity.**
   Old cold left ~60% of the NIC idle (unpipelined per-tensor
   materialization + a bug that burned a keepalive connection per miss).
   New cold runs 93–155 MB/s: Woona's 26.9 GB depth-2 closure
   materializes in 137 s (old: 415 s). SimPO's shallow closure
   materializes faster than the client NIC drains it — its cold equals
   its warm (both NIC-bound), which is the end state a serving tier
   wants: reconstruction fully hidden behind the wire.
2. **Warm: 2.0x faster on big models, and the bottleneck moved OFF the
   server.** Old warm was EBS-bound at ~75 MB/s for models too big for
   page cache. New warm pins every big model at 136–143 MB/s = the
   CLIENT NIC sustained rate (3B models, running during burst credits,
   hit 432–459 MB/s; on-box smoke hit 878 MB/s). The cache tier is no
   longer measurable from a t3 client — finding its ceiling needs
   Stage 3 (concurrent clients) or a bigger-NIC client.
3. **llama-family cold is the one soft spot: 9.9 / 28.4 / 43.2 s across
   three near-identical 1.7 GB models** (vs 12.4 s direct S3). All
   three are >99% miss-served, so this is NOT cache pollution. The 3B
   workload is latency-bound (≈400 tensors, ~4.5 MB avg: per-miss
   S3-GET + hash + write round trips dominate; nothing pipelines).
   Variance suspects, needing server-side log correlation: first-run
   effects (Gabbar ran seconds after the materializer restart) and
   spill behavior around the >512 MB shared embedding chains
   (per-run spill isolation from 5b9f5fb re-materializes shared spilled
   bases instead of reusing a sibling's). Open item.

## 4. Protocol caveat discovered post-run: closures DO overlap

The pre-run assumption "no closure shares a tid across models" was
wrong. Measured from the manifests: Woona's closure contains 265 of
SimPO's 464 params; Tong's contains 259; the three Qwens share 100+
closure tids pairwise; even chat-doctor/GSOC share 11. Materializing a
model caches its closure ancestors (content-addressed), so later cold
runs found some tensors resident.

Observed impact (client TTFB<50 ms hit detection): small — the shared
tids are overwhelmingly small tensors (norms, biases). Pre-cached BYTES:
7.2% for SimPO, ≤2.5% everywhere else, 0.0–0.3% for five of nine.
The e2e conclusions are unaffected at these magnitudes; per-run numbers
above carry the annotation instead of being discarded.

Research note, not just a caveat: this is cross-model ancestor sharing
at the origin tier doing exactly what a DAG-aware cache should do —
serving Woona pre-warms SimPO. It is the serving-side face of the
delta-graph locality that the dag_value eviction policy wants to
exploit, and it deserves a measured experiment of its own (serve the
base, then time the fine-tune) rather than being an accident in a cold
protocol.

## 5. Detailed per-run metrics (this matrix)

See `per_run_metrics.csv` / `per_model_medians.csv`; generated tables
below.

| model | sys | e2e_s | s3_GB | read_amp | resolve_ms | net_MBps | ttfb_ms_p50 | goodput_MBps | rss_peak_GB |
|---|---|---|---|---|---|---|---|---|---|
| chat-doctor | cold | 9.89 | 0.89 | 0.53 | 10 | 89.7 | 205.7 | 169.3 | 1.18 |
| chat-doctor | warm | 3.47 | 0.89 | 0.53 | 9 | 436.4 | 4.2 | 481.8 | 1.17 |
| BanglaLLama | cold | 43.16 | 1.77 | 1.00 | 11 | 42.1 | 227.6 | 41.0 | 1.21 |
| BanglaLLama | warm | 6.17 | 1.77 | 1.00 | 11 | 350.9 | 4.9 | 287.2 | 1.32 |
| GSOC | cold | 28.44 | 0.89 | 0.53 | 11 | 34.5 | 214.6 | 58.8 | 1.13 |
| GSOC | warm | 3.49 | 0.89 | 0.53 | 9 | 432.2 | 4.5 | 479.8 | 1.21 |
| HPAI Egida | cold | 129.20 | 15.23 | 1.00 | 10 | 123.2 | 280.2 | 117.9 | 4.16 |
| HPAI Egida | warm | 101.85 | 15.23 | 1.00 | 10 | 162.4 | 3.2 | 149.6 | 3.00 |
| KPEP krx | cold | 117.37 | 15.23 | 1.00 | 9 | 131.0 | 859.6 | 129.8 | 3.14 |
| KPEP krx | warm | 102.33 | 15.23 | 1.00 | 9 | 162.3 | 2.9 | 148.8 | 4.16 |
| mlf hp_abl | cold | 156.30 | 15.23 | 1.00 | 10 | 97.9 | 1710.5 | 97.5 | 4.05 |
| mlf hp_abl | warm | 102.96 | 15.23 | 1.00 | 11 | 160.9 | 3.3 | 147.9 | 3.05 |
| Woona | cold | 137.40 | 18.48 | 1.00 | 12 | 142.1 | 1293.9 | 134.5 | 3.83 |
| Woona | warm | 129.78 | 18.48 | 1.00 | 12 | 153.1 | 3.9 | 142.4 | 3.90 |
| Tong star | cold | 186.05 | 18.48 | 1.00 | 12 | 99.5 | 2359.8 | 99.4 | 5.02 |
| Tong star | warm | 128.10 | 18.48 | 1.00 | 13 | 155.5 | 3.9 | 144.3 | 5.36 |
| SimPO | cold | 128.32 | 18.48 | 1.00 | 12 | 155.2 | 23.9 | 144.0 | 3.95 |
| SimPO | warm | 127.95 | 18.48 | 1.00 | 11 | 155.7 | 3.6 | 144.5 | 4.04 |

Notes: resolve_ms 9–13 ms against the new manifest server (old
prototype: 26–122 ms; old local SQLite: 250–380 ms). Cold ttfb_ms_p50
tracks materialization queueing (24 ms SimPO → 2.4 s Tong) and is the
cleanest per-run indicator of how hard the materializer worked.

## 6. Hugging Face hub baseline (2026-08-03)

Same models, downloaded from huggingface.co the way a real user does:
`hf download <repo> --include "*.safetensors*"` — huggingface_hub
1.26.0 with the hf_xet backend (parallel chunked transfer, HF's current
default), fresh cache, same t3.xlarge client, 1 rep per model, one
model per family. HF's weight files are byte-identical in size to our
logical bytes, so e2e times are directly comparable. Raw numbers:
`eval/results/hf_baseline.json`.

| model | weights | HF e2e | HF goodput | NIC bytes | vs direct S3 | vs i4i cold | vs i4i warm |
|---|---|---|---|---|---|---|---|
| chat-doctor 3B | 1.56 GiB | 24.3 s | 66 MiB/s | 1.61 GiB | 2.0x slower | 2.5x slower | 7.0x slower |
| Egida-DPO 7B | 14.19 GiB | 125.6 s | 116 MiB/s | 12.72 GiB | 1.12x slower | ~parity (0.97x) | 1.23x slower |
| SimPO 9B | 17.21 GiB | 149.7 s | 118 MiB/s | 15.54 GiB | 1.06x slower | 1.17x slower | 1.17x slower |

Reading:

1. **TStore's serving tier beats the actual Hugging Face hub on this
   client for every regime except one** (Egida cold, where they tie).
   Warm serving wins 1.2x on big models and 7x on the 3B; even cold
   demand-miss reconstruction matches or beats HF.
2. **HF runs below the client NIC** (116–118 MiB/s vs the 136–143
   sustained our warm path pins): CDN + Xet chunk assembly overhead
   costs it ~15–20% of line rate on big models, and the single-file 3B
   adapter gets only 66 MiB/s.
3. **Xet's wire savings are real but modest**: NIC bytes ran 8–10%
   under the written bytes on 7B/9B (chunk-level dedup/compression).
   Compare: our store saves 35% at rest; our server path ships amp-1.0
   logical bytes; our client-decode path ships 0.52–1.45x depending on
   closure. HF's transfer dedup and our storage dedup are different
   layers — a hub could do both.
4. Protocol caveats: 1 rep, public CDN performance varies by time of
   day and region; our numbers come from a private single-tenant tier —
   this is a sanity anchor against the real-world incumbent, not a
   controlled A/B.
