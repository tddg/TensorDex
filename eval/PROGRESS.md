# TensorDex Eval — Progress Report (2026-08-01)

Session summary: FAST'27 decision-study groundwork (Stage 0) plus construction
and upload of a delta-compressed TensorDex hub built from the live S3 store,
following the SOSP'26 AE process. VM was shut down at the end of this session.

## 1. What exists now

### Compressed hub on S3 (new)
- **Location:** `s3://tensor-tingfeng/compressed_eval/` (write-only prefix;
  the source store `tensordb/` was never modified)
  - `compressed_eval/blobs/{tid[:2]}/{tid}.safetensors` — 3,456 blobs
  - `compressed_eval/master.db` — hub metadata (32 MB): `tensors`,
    `model_mappings`, `tensor_deltas` (the delta→base graph the live store
    lacks)
- **Contents:** 9 models, 3 families × 3 variants, built via
  ingest → TensorSketch fingerprints → FlexSplit attach planning (threshold
  0.70, cross-model bases) → TensorX deltas (zstd L1) → per-tensor
  byte-exact verification (XXH3-128).

| family | models | logical | stored | saved | deltas/tensors |
|---|---|---|---|---|---|
| llama-3.2-3B | AhmedSSoliman/llama-3.2-3b-chat-doctor · Gabbar01/llama-3.2-3b-GSOC-DATASET · BanglaLLM/BanglaLLama-3.2-3b-unolp-culturax-base-v0.0.1 | 5.12 GB | 2.40 GB | **53.1%** | 1168/1180 |
| Qwen2.5-7B | KPEP/krx-qwen-2.5-7b-v1.4.4 · HPAI-BSC/Qwen2.5-7B-Instruct-Egida-DPO · mlfoundations-dev/hp_ablations_qwen_adambeta2_0.95 | 45.68 GB | 30.98 GB | **32.2%** | ~1013 tensors |
| gemma-2-9B | princeton-nlp/gemma-2-9b-it-SimPO · TongZheng1999/gemma-2-9b-it-star-truth_table-OP · AlexBefest/WoonaV1.2-9b | 55.45 GB | 35.69 GB | **35.6%** | ~1263 tensors |
| **Total** | 9 models | **106.26 GB** | **69.07 GB** | **35.0%** | 3026/3456 |

Every model verified byte-exact (streaming per-tensor re-hash after delta
decode). Near-twin fine-tunes attach at ~10× per-tensor deltas; unrelated
same-architecture fine-tunes land at ~0.55–0.7 CR — matching the paper's
distance-vs-ratio behavior. Upload: 3,457 objects / 69.10 GB @ 152 MB/s.

### Eval harness (repo, `eval/`)
- Instrumentation: JSONL event recorder, per-request S3 tracing
  (headers/first-byte/body), SQLite query tracing, psutil sampler.
- Adapters: tensordex (implements the cloud pull TensorDex itself lacks:
  manifest closure → parallel GETs → bottom-up delta decode → hash verify),
  zipnn, raw. Driver `run_single.py`, `summarize.py`.
- Configs: `object_layout.yaml` (bucket layout), `models.yaml` (19 live
  models, stratified).
- Scripts: `inventory_s3.py`, `inspect_metadata.py`, `select_models.py`,
  `build_compressed_hub.py`, `upload_compressed_hub.py`.
- First instrumented benchmark run (llama-3.2-3B, 1.67 GB): 14.05 s e2e,
  read amplification 0.53 (tied-embedding dedup), critical path = single
  788 MB blob (10.9 of 14.0 s). Traces in `eval/results/single_client/`.

## 2. Key findings

1. **The live S3 store (`tensordb/`) is dedup-only.** No `tensor_deltas`
   table in any metadata DB in the bucket; every blob is a raw tensor whose
   bytes hash to its key. The delta-compressed store now exists only under
   `compressed_eval/`.
2. **TensorDex's S3 backend cannot decode deltas** (local-backend-only in
   `engine.py`) — any cloud-pull benchmark of `compressed_eval/` needs the
   eval adapter (or that gap fixed).
3. **Bucket:** 1.78 M objects / 72.03 TB, but live `master.db` catalogs only
   742 K tensors / 40.1 TB / 2,893 models — ~1 M blobs (~32 TB) are
   orphaned/pre-GC. Always HEAD-preflight (e.g., `jinoo/gemma2-9b` is
   'ready' in metadata but missing 417/464 blobs).
4. **Corpus is size-degenerate:** p25–p99 ≈ 14.5–18.5 GB (7–9B fine-tunes).
5. Incremental (AE CLI) compression requires the working set on local disk;
   a Phase A/B split (global sketch-only planning + streaming pair
   execution) scales without that. Discussed with user; deferred — current
   run used the incremental process (user decision).

## 3. Operational notes (this VM: t3.xlarge, us-east-1d, i-06c568e186e13a1df)

- Python: **use `.venv`** (3.11). System Python 3.14 cannot build TensorDex
  (PyO3 ≤3.12). Rust toolchain was REMOVED for disk space — reinstall
  rustup before rebuilding the extension.
- AWS: profile `tensordex-eval` in `~/.aws` (IAM user `tingfeng`; no
  CreateBucket, no EC2). Keys were shared in chat — consider rotating.
- Deleted for disk space (all re-downloadable): `ae/cache` (results.db,
  sample_blobs — `make ae-cache` restores), `target/`, `~/.rustup`,
  `~/.cargo/registry`, benchmark pull outputs.
- Local hub copy: `eval/raw/compressed_hub/` (65 GB) — safe to delete once
  `compressed_eval/` is accepted; also `eval/raw/master.db` (1.35 GB copy).
- Failures survived during the build (all fixed in
  `build_compressed_hub.py`): WAL visibility race (→ in-process
  auto_compress), OOM on whole-model verify (→ streaming per-tensor
  verify), missing blobs (→ HEAD preflight), disk-full (→ chunked
  stage→ingest→compress, 4 GB chunks).

## 4. Stage 1 download measurements — DONE (second session, 2026-08-01)

54 byte-exact instrumented runs across four regimes; full tables and
bottleneck analysis in **eval/results/DOWNLOAD_COMPARISON.md**; design
context in **eval/DESIGN_NOTES.md**. Headlines:

- Uncompressed direct S3 saturates the t3.xlarge NIC (~145 MB/s); its
  e2e is pure transfer time and it beats client-side compressed pulls
  on 8/9 models (compressed pays closure amplification 0.52–1.45 —
  Woona's depth-2 chains move 26.9 GB for 18.5 GB logical — plus
  pipeline drag).
- Backend reconstruction (deployed metadata server :8701 + disk cache
  server :8702, prototypes in eval/server/): warm cache is 4x FASTER
  than S3 for models that fit page cache (3–7 s for 3B) and 1.8x
  SLOWER than S3 for 15–18 GB models — gp3 EBS, not network, is the
  cache tier's bottleneck. Cold reconstruction is 2–3.5x slower than
  direct (unpipelined per-tensor materialization).
- TensorX decode CPU is never the bottleneck (server fetch:decode 7:1;
  client decode overlaps fetch). Metadata is negligible once served
  per-model (26–122 ms vs 250–380 ms local whole-catalog SQLite).
- Adapter engineering: streaming shard-writer client with head-of-line
  byte-budget back-pressure (three OOMs traced to implicit buffering;
  peak RSS 15 GB -> 5.4 GB). Blob codec metadata is a nested JSON
  string under __metadata__[name] — item_size matters for f32 deltas.

## 5. Next steps

1. Stage 2: base-cache reuse (warm fleet client) — the delta-side win
   the cold measurements can't show.
2. ZipNN + raw-safetensors baselines (still absent from bucket) for the
   original 3-way comparison; Stage 3 concurrent clients.
3. Rerun the server path on RAM/NVMe-backed hardware before concluding
   on the cache tier; pipeline the cold materialization path.
4. Decision memo per the plan's Outcome A–E criteria.
