# Stage 0 status — 2026-08-01

## Update after S3 access (credentials via AWS_PROFILE=tensordex-eval)

- Inventory complete: **1,775,939 objects / 72.03 TB** under `tensordb/`
  (blobs 71.86 TB; the rest is master.db + db backups + checkpoint +
  node backups). Live `master.db` catalogs 742 K tensors / 40.1 TB /
  2,893 models → ~1.03 M blobs (~32 TB) are orphaned/pre-GC.
- **No `tensor_deltas` in master.db and no zipnn/ or raw/ layouts in the
  bucket** — confirmed: the deployed store is dedup-only; both baselines
  and the delta layout must be constructed before Stage 1.
- Model set re-frozen from the live catalog (19 models, eval/config/models.yaml);
  the AE metadata.db is a pre-GC superset (only 8/22 original picks were live).
- **Run 1 done (TensorDex path)**: llama-3.2-3b-chat-doctor, 1.67 GB /
  394 tensors in 14.05 s on t3.xlarge, all tensors hash-verified.
  Read amplification 0.53 (tied 788 MB embedding deduped); GET
  first-header p50 93 ms; critical path = the single 788 MB blob (10.9 s
  of 14.0 s). Traces in eval/results/single_client/.
- User confirmed t3.xlarge is the intended client-simulation machine.


## Done

- **Environment** (t3.xlarge, us-east-1d): Python 3.11 venv (`.venv`),
  tensordex 0.1.0 built from source (Rust 1.97), torch 2.13 CPU,
  boto3/zipnn/pyarrow/pandas/psutil installed. `make check`: 18 passed.
  Note: system Python 3.14 is unsupported by the pinned PyO3 (≤3.12) —
  always use `.venv`.
- **AE cache** (from HF dataset `tensordex/tensordex-ae-cache`, not S3):
  `results.db` (5.2 GB), `data/tensordb_s3/metadata.db` (567 MB),
  `sample_blobs/` (2 GB).
- **Schemas** (`eval/raw/metadata_schema.json`): metadata.db =
  `tensors(id, shape, dtype, size_bytes)` ×1.76 M +
  `model_mappings(model_name, param_name, tensor_id)` ×2.28 M, 6,501 models.
  results.db = `compression_results(target_id, base_id, …ratios/entropy)`.
- **Local validation**: 200/200 sample blobs parse via the eval blob parser
  and hash-verify (XXH3-128 == blob id); TensorX compress→decompress
  round-trips byte-exactly through the freshly built kernel.
- **Model set** (`eval/config/models.yaml`, 22 models): 16 stratified
  (size × cross-model tensor-sharing), 4 size extremes, 2 small complete
  models for Run 1 (`AhmedSSoliman/llama-3.2-3b-chat-doctor` 1.67 GB / 394
  tensors is the primary Run-1 candidate).
- **Harness scaffolding** (`eval/src`, `eval/scripts`): event recorder,
  S3 request/body tracing, SQLite query tracing, resource monitor,
  tensordex/zipnn/raw adapters, run_single driver, summarizer. All compile;
  none exercised against S3 yet.

## Findings that change the plan

1. **TensorDex S3 + delta is not implemented in the package.** Delta decode
   is local-backend-only (`engine.py` raises for non-local backends). The
   eval adapter therefore implements the cloud pull itself:
   `manifest_blobs()` closure → parallel content-addressed GETs
   (`{prefix}/blobs/{tid[:2]}/{tid}.safetensors`) → bottom-up
   `decompress_tensorx_rust` with shared-base cache → hash verify.
   With the SQL manifest, all keys are known up front, so
   `sequential_fetch_rounds == 1` by construction; header-driven discovery
   (what the local path does) is the multi-round variant.
2. **The live S3 store is probably dedup-only (no delta blobs).** Evidence:
   the published metadata has no `tensor_deltas` table; `verify_s3_ids.py`
   assumes every S3 object's bytes hash to its key (only true for raw
   tensors); the S3 pull path can't decode deltas anyway. If confirmed, the
   delta-retrieval experiment requires *building* the delta-compressed S3
   layout (assign bases via results.db, compress, upload) before measuring
   it — a scoping decision, not a wrapper.
3. **The model corpus is size-degenerate**: p25–p99 ≈ 14.5–18.5 GB (7–9B
   fine-tunes); only 24 models < 5 GB, one 48 GB. Percentile size strata
   barely differ; conclusions about "size scaling" will need the extreme
   models and possibly out-of-corpus additions.
4. Metadata for the S3 hub lives at `s3://{bucket}/{prefix}/master.db`
   (fallback `metadata.db`) per `ae/run_full.py`; metadata may be a
   superset of live blobs (gc'd/compacted) — model selection must preflight
   HEAD every blob before freezing the set.

## Blocked

- **AWS credentials** (blocks tasks: S3 inventory, object_layout.yaml,
  ZipNN/raw baseline availability, all three download paths, Stage 0 exit
  criterion). No instance profile, no `~/.aws`; bucket denies anonymous.
  Provision an instance profile or `AWS_PROFILE=tensordex-eval`.
- **ZipNN + raw artifacts may not exist in the bucket at all** — the AE
  flow only shows tensor blobs + master.db. If absent, baselines must be
  generated and uploaded (compress originals with zipnn; fetch originals
  from HF) before any comparison.
- Performance runs need the c6a.48xlarge-class machine; this t3.xlarge is
  for correctness only.

## Next actions (once credentials exist)

1. `.venv/bin/python eval/scripts/inventory_s3.py` → fill
   `eval/config/object_layout.yaml`; confirm whether delta blobs and
   zipnn/raw artifacts exist.
2. Pull `master.db`; dump `tensor_deltas`; recompute dependency strata in
   `models.yaml`; preflight HEAD all selected models' blobs.
3. Run 1 (correctness): llama-3.2-3b model through all three adapters,
   byte-exact check; then Runs 2–3 per plan §15.
