# Compressed vs uncompressed download (single client)

> **2026-08-05:** the whole origin store is now delta-compressed —
> 581k TensorX deltas at `s3://tensor-tingfeng/compressed_full/`
> (60.9% reduction physical / 65.2% logical; FM++ 70.5% claim validated).
> Full campaign report: [FULL_HUB_CAMPAIGN.md](FULL_HUB_CAMPAIGN.md).

Medians over reps (min reps per cell: 1). tensordex = uncompressed dedup store (`tensordb/`), tensordex_c = delta-compressed hub (`compressed_eval/`).

## Origin store census (`s3://tensor-tingfeng/tensordb/`, measured 2026-08-04)

Registered store, per the metadata DB (`master.db` snapshot of 2026-08-01;
tables `model_meta` / `model_mappings` / `tensors`):

| stat | value |
|---|---|
| models | **2,894** (2,892 ready · 1 ingesting · 1 failed) |
| named parameters (model→tensor mappings) | **964,011** |
| unique tensors (content-addressed blobs) | **742,027** |
| unique bytes (physical, registered) | **40.1 TB** |
| logical bytes (sum over all models) | **45.0 TB** |
| identical-tensor dedup saving | **4.8 TB (10.8%)** — 22,784 tensors shared by >1 model |
| avg tensors / model · avg tensor size | 333 · 54 MB |

This store is **uncompressed originals only** — `tensor_deltas` is empty;
delta compression lives in the separate 9-model `compressed_eval/` hub
(69 GB, 35% saved, byte-exact). Bucket-level inventory is larger than the
registered store: `blobs/` holds **1,775,896** `.safetensors` objects,
**71.9 TB** total — i.e. ~1.03M blobs / 31.8 TB are not referenced by the
current metadata DB (earlier/unregistered ingests; every key is a
one-per-tid tensor blob, no sidecar files). All eval models resolve
against the registered set.

| model | sys | e2e_s | logical_GB | s3_GB | gets | read_amp | resolve_ms | fetch_span_s | net_MBps | decode_cpu_ms | tail_s | ttfb_ms_p50 | goodput_MBps | rss_peak_GB | proc_cpu_pct_mean | disk_write_MBps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AhmedSSoliman/llama-3.2-3b-chat-doctor | td | 12.05 | 1.67 | 0.89 | 393 | 0.53 | 254 | 10.35 | 85.51 | 0 | 1.45 | 66.48 | 138.82 | 1.58 | 58.79 | 114.82 |
| AhmedSSoliman/llama-3.2-3b-chat-doctor | td_c | 13.54 | 1.67 | 0.87 | 393 | 0.52 | 40 | 11.97 | 75.18 | 119 | 1.52 | 63.65 | 126.65 | 1.57 | 36.46 | 128.49 |
| AhmedSSoliman/llama-3.2-3b-chat-doctor | td_srv_cold | 25.78 | 1.67 | 0.89 | 393 | 0.53 | 96 | 19.50 | 45.39 | 0 | 6.18 | 164.46 | 64.90 | 1.94 | 15.43 | 65.56 |
| AhmedSSoliman/llama-3.2-3b-chat-doctor | td_srv_warm | 3.16 | 1.67 | 0.89 | 393 | 0.53 | 14 | 1.25 | 710.15 | 0 | 1.90 | 3.78 | 529.92 | 1.94 | 94.19 | 698.29 |
| BanglaLLM/BanglaLLama-3.2-3b-unolp-culturax-base-v0.0.1 | td | 23.49 | 1.77 | 1.77 | 394 | 1.00 | 272 | 22.09 | 80.25 | 0 | 1.13 | 69.21 | 75.44 | 1.51 | 51.30 | 68.16 |
| BanglaLLM/BanglaLLama-3.2-3b-unolp-culturax-base-v0.0.1 | td_c | 21.55 | 1.77 | 1.44 | 394 | 0.81 | 66 | 20.92 | 68.90 | 2,449 | 0.56 | 67.59 | 82.18 | 2.18 | 40.84 | 81.81 |
| BanglaLLM/BanglaLLama-3.2-3b-unolp-culturax-base-v0.0.1 | td_srv_cold | 38.51 | 1.77 | 1.77 | 394 | 1.00 | 53 | 34.63 | 51.13 | 0 | 3.82 | 172.42 | 45.98 | 2.83 | 12.37 | 45.77 |
| BanglaLLM/BanglaLLama-3.2-3b-unolp-culturax-base-v0.0.1 | td_srv_warm | 7.07 | 1.77 | 1.77 | 394 | 1.00 | 13 | 3.36 | 526.42 | 0 | 3.69 | 4.35 | 250.51 | 2.83 | 58.39 | 260.83 |
| Gabbar01/llama-3.2-3b-GSOC-DATASET | td | 12.37 | 1.67 | 0.89 | 393 | 0.53 | 263 | 10.66 | 83.44 | 0 | 1.45 | 70.39 | 135.68 | 1.60 | 55.96 | 112.07 |
| Gabbar01/llama-3.2-3b-GSOC-DATASET | td_c | 18.77 | 1.67 | 0.88 | 404 | 0.52 | 67 | 15.57 | 56.26 | 1,559 | 3.14 | 66.79 | 89.16 | 2.33 | 29.00 | 88.61 |
| Gabbar01/llama-3.2-3b-GSOC-DATASET | td_srv_cold | 23.70 | 1.67 | 0.89 | 393 | 0.53 | 49 | 16.25 | 54.48 | 0 | 7.40 | 155.18 | 70.61 | 1.94 | 15.78 | 71.94 |
| Gabbar01/llama-3.2-3b-GSOC-DATASET | td_srv_warm | 3.94 | 1.67 | 0.89 | 393 | 0.53 | 14 | 1.27 | 698.97 | 0 | 2.66 | 3.70 | 424.78 | 1.94 | 70.08 | 534.92 |
| HPAI-BSC/Qwen2.5-7B-Instruct-Egida-DPO | td | 112.42 | 15.23 | 15.23 | 339 | 1.00 | 252 | 104.77 | 145.38 | 0 | 7.40 | 98.98 | 135.50 | 3.60 | 110.49 | 135.87 |
| HPAI-BSC/Qwen2.5-7B-Instruct-Egida-DPO | td_c | 138.34 | 15.23 | 12.47 | 407 | 0.82 | 70 | 131.55 | 94.79 | 35,387 | 6.72 | 85.32 | 110.10 | 5.05 | 73.67 | 110.40 |
| HPAI-BSC/Qwen2.5-7B-Instruct-Egida-DPO | td_srv_cold | 259.53 | 15.23 | 15.23 | 339 | 1.00 | 45 | 251.09 | 60.66 | 0 | 8.39 | 950.17 | 58.69 | 4.23 | 15.98 | 58.83 |
| HPAI-BSC/Qwen2.5-7B-Instruct-Egida-DPO | td_srv_warm | 204.38 | 15.23 | 15.23 | 339 | 1.00 | 27 | 197.16 | 77.25 | 0 | 7.20 | 1.20 | 74.52 | 3.15 | 18.09 | 74.63 |
| KPEP/krx-qwen-2.5-7b-v1.4.4 | td | 113.24 | 15.23 | 15.23 | 339 | 1.00 | 246 | 105.73 | 144.10 | 0 | 7.26 | 98.67 | 134.53 | 3.08 | 108.47 | 134.85 |
| KPEP/krx-qwen-2.5-7b-v1.4.4 | td_c | 123.78 | 15.23 | 12.46 | 406 | 0.82 | 94 | 115.42 | 108.17 | 37,813 | 8.26 | 95.49 | 123.20 | 5.06 | 93.86 | 123.44 |
| KPEP/krx-qwen-2.5-7b-v1.4.4 | td_srv_cold | 276.74 | 15.23 | 15.23 | 339 | 1.00 | 122 | 265.46 | 57.38 | 0 | 11.15 | 398.09 | 55.04 | 3.12 | 15.40 | 55.03 |
| KPEP/krx-qwen-2.5-7b-v1.4.4 | td_srv_warm | 198.25 | 15.23 | 15.23 | 339 | 1.00 | 34 | 188.62 | 80.75 | 0 | 9.59 | 1.16 | 76.83 | 4.25 | 18.77 | 76.97 |
| mlfoundations-dev/hp_ablations_qwen_adambeta2_0.95_dcftv1.2 | td | 113.12 | 15.23 | 15.23 | 339 | 1.00 | 268 | 105.99 | 143.76 | 0 | 6.86 | 96.58 | 134.71 | 3.24 | 109.07 | 135.03 |
| mlfoundations-dev/hp_ablations_qwen_adambeta2_0.95_dcftv1.2 | td_c | 148.13 | 15.23 | 13.98 | 466 | 0.92 | 104 | 143.56 | 97.42 | 37,776 | 4.47 | 90.69 | 102.83 | 5.07 | 75.37 | 102.25 |
| mlfoundations-dev/hp_ablations_qwen_adambeta2_0.95_dcftv1.2 | td_srv_cold | 290.62 | 15.23 | 15.23 | 339 | 1.00 | 43 | 283.59 | 53.71 | 0 | 6.99 | 1056.53 | 52.41 | 3.02 | 14.27 | 52.50 |
| mlfoundations-dev/hp_ablations_qwen_adambeta2_0.95_dcftv1.2 | td_srv_warm | 205.14 | 15.23 | 15.23 | 339 | 1.00 | 26 | 197.04 | 77.30 | 0 | 8.08 | 1.26 | 74.25 | 3.03 | 18.00 | 74.35 |
| AlexBefest/WoonaV1.2-9b | td | 139.80 | 18.48 | 18.48 | 464 | 1.00 | 356 | 131.52 | 140.54 | 0 | 7.93 | 97.33 | 132.21 | 4.94 | 102.72 | 132.48 |
| AlexBefest/WoonaV1.2-9b | td_c | 176.77 | 18.48 | 26.89 | 726 | 1.45 | 144 | 174.00 | 154.52 | 48,207 | 2.63 | 99.17 | 104.57 | 5.61 | 149.19 | 104.02 |
| AlexBefest/WoonaV1.2-9b | td_srv_cold | 415.45 | 18.48 | 18.48 | 464 | 1.00 | 61 | 407.23 | 45.39 | 0 | 8.16 | 2595.10 | 44.49 | 4.01 | 11.36 | 44.55 |
| AlexBefest/WoonaV1.2-9b | td_srv_warm | 258.36 | 18.48 | 18.48 | 464 | 1.00 | 32 | 248.64 | 74.34 | 0 | 9.69 | 1.19 | 71.54 | 5.93 | 17.03 | 71.62 |
| TongZheng1999/gemma-2-9b-it-star-truth_table-OP-final_1-2-3Rounds-iter-1 | td | 155.24 | 18.48 | 18.48 | 464 | 1.00 | 379 | 147.87 | 125.03 | 0 | 7.00 | 100.15 | 119.08 | 4.75 | 91.93 | 119.30 |
| TongZheng1999/gemma-2-9b-it-star-truth_table-OP-final_1-2-3Rounds-iter-1 | td_c | 158.02 | 18.48 | 20.93 | 667 | 1.13 | 96 | 152.26 | 137.47 | 38,173 | 5.67 | 102.50 | 116.99 | 5.50 | 129.80 | 117.19 |
| TongZheng1999/gemma-2-9b-it-star-truth_table-OP-final_1-2-3Rounds-iter-1 | td_srv_cold | 397.84 | 18.48 | 18.48 | 464 | 1.00 | 64 | 388.15 | 47.62 | 0 | 9.62 | 1168.10 | 46.46 | 3.76 | 12.68 | 46.50 |
| TongZheng1999/gemma-2-9b-it-star-truth_table-OP-final_1-2-3Rounds-iter-1 | td_srv_warm | 259.97 | 18.48 | 18.48 | 464 | 1.00 | 37 | 249.89 | 73.97 | 0 | 10.04 | 1.20 | 71.10 | 5.20 | 17.38 | 71.13 |
| princeton-nlp/gemma-2-9b-it-SimPO | td | 141.31 | 18.48 | 18.48 | 464 | 1.00 | 331 | 132.39 | 139.65 | 0 | 8.59 | 100.83 | 130.80 | 3.84 | 102.54 | 131.06 |
| princeton-nlp/gemma-2-9b-it-SimPO | td_c | 144.58 | 18.48 | 17.09 | 469 | 0.92 | 68 | 135.47 | 126.17 | 11,305 | 9.05 | 92.13 | 127.90 | 4.88 | 75.18 | 128.44 |
| princeton-nlp/gemma-2-9b-it-SimPO | td_srv_cold | 299.36 | 18.48 | 18.48 | 464 | 1.00 | 58 | 289.69 | 63.80 | 0 | 9.61 | 783.84 | 61.74 | 5.71 | 16.26 | 61.81 |
| princeton-nlp/gemma-2-9b-it-SimPO | td_srv_warm | 250.69 | 18.48 | 18.48 | 464 | 1.00 | 26 | 241.43 | 76.56 | 0 | 9.23 | 1.25 | 73.73 | 3.95 | 17.46 | 73.79 |

## Family aggregates (median of per-run values)

| family | sys | e2e_s | s3_GB | net_MBps | goodput_MBps | decode_cpu_ms |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | td | 112.4 | 15.23 | 145 | 135 | 0 |
| Qwen2.5-7B | td_c | 138.3 | 12.47 | 97 | 110 | 37,762 |
| Qwen2.5-7B | td_srv_cold | 276.7 | 15.23 | 57 | 55 | 0 |
| Qwen2.5-7B | td_srv_warm | 204.4 | 15.23 | 77 | 75 | 0 |
| gemma-2-9B | td | 141.3 | 18.48 | 139 | 131 | 0 |
| gemma-2-9B | td_c | 158.0 | 20.93 | 137 | 117 | 38,173 |
| gemma-2-9B | td_srv_cold | 397.8 | 18.48 | 48 | 46 | 0 |
| gemma-2-9B | td_srv_warm | 258.4 | 18.48 | 74 | 72 | 0 |
| llama-3.2-3B | td | 12.6 | 0.89 | 84 | 133 | 0 |
| llama-3.2-3B | td_c | 18.8 | 0.88 | 65 | 89 | 1,559 |
| llama-3.2-3B | td_srv_cold | 25.8 | 0.89 | 51 | 65 | 0 |
| llama-3.2-3B | td_srv_warm | 3.9 | 0.89 | 699 | 425 | 0 |

## Bottleneck analysis (single client, t3.xlarge, us-east-1, same-region S3)

Regimes: `td` = uncompressed dedup store, direct S3; `td_c` = compressed
hub, client-side delta decode; `td_srv_cold/warm` = backend
reconstruction via colocated metadata (:8701) + cache (:8702) servers.
All runs byte-exact (per-tensor XXH3 verified end-to-end).

### Ranked bottlenecks

1. **Client NIC line rate (~145 MB/s).** `td` saturates it; its e2e is
   pure transfer time. Nothing beats it cold unless bytes on the wire
   actually shrink.
2. **Delta-closure read amplification (`td_c`).** Transfer = deltas +
   all transitive bases: amp 0.52 (twin llamas) … 0.92 (SimPO,
   base-heavy) … **1.45 (Woona, depth-2 chains — 26.9 GB moved for an
   18.5 GB model)**. Chain depth converts storage savings into extra
   traffic; only genuinely delta-heavy models (Bangla 0.81) win e2e.
3. **EBS throughput (~50–130 MB/s, shared).** Dominates both server
   paths and caps everything else:
   - warm big models: server disk-read + client shard-write contend →
     74–77 MB/s, i.e. **warm cache is ~1.8x slower than direct S3**
     (Qwen 204 s vs 112 s). Only models that fit page cache (3B family)
     see RAM-speed serving: 700 MB/s, 3–7 s e2e, 4x faster than any
     S3 path.
   - cold server: sequential per-tensor materialize (S3 fetch → decode
     → EBS write → serve) with no cross-tensor S3 parallelism →
     45–65 MB/s, 2–3.5x slower than `td`; TTFB p50 up to 2.6 s while a
     chain materializes.
4. **TensorX decode CPU: never the bottleneck.** Server-side
   fetch:decode ≈ 7:1 on big tensors (2.3 s vs 0.33 s p50); client-side
   ~38 s CPU on a 138 s run, largely overlapped with fetch (it shows up
   as effective network dropping 145 → 97–137 MB/s, not as added wall
   time).
5. **Metadata: solved by the server.** Per-model manifest over HTTP:
   26–122 ms (32 MB DB); local SQLite over the 1.35 GB whole-catalog
   copy: 250–380 ms; warm-hit TTFB via cache server: 1.2 ms. The
   metadata tier is nowhere near any critical path.

### Implications for the production design

- A reconstruction cache tier is only worth deploying on hardware whose
  serving medium beats the client's S3 path: RAM/NVMe, not gp3 EBS on a
  small instance. On this VM the cache helps small models (page cache)
  and *hurts* large ones.
- The cold server path must pipeline: prefetch the whole closure with
  parallel S3 GETs and decode concurrently (reuse the client adapter's
  head-of-line-bounded pipeline server-side). Today it leaves ~60% of
  the NIC idle.
- Cap delta-chain depth at 1 for distribution (Woona's depth-2 chains
  are the only amp > 1 case); depth-2+ is fine for cold archival tiers
  behind a reconstruction cache, where amplification is paid
  server-side once.
- Client-side decode remains the right *opt-in* for warm fleets (Stage
  2 will quantify: with cached bases, a new fine-tune costs only delta
  bytes).
- Engineering note: three OOMs during this eval came from the same
  pattern — fast producer (S3/cache) + slow consumer (decode/hash/EBS)
  with implicit buffering (futures retaining results, budget valves
  that trickle). Streaming tensor clients need explicit end-to-end
  byte-budget back-pressure; peak RSS went 15 GB → 5.4 GB once
  enforced.

### Run inventory

54 successful runs: 36 matrix (9 models x {td, td_c} x 2 reps) +
18 server (9 models x {cold, warm}). Traces: per-run event JSONL +
model_runs.jsonl under eval/results/{single_client,server_client}/,
server-side logs under eval/results/server/. Per-run metrics:
eval/results/single_client/per_run_metrics.csv.
