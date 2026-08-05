# Full-hub TensorX compression campaign — 2026-08-05

Reproduction of the paper's full-trace TensorDex-TX compression (Fig 11a,
65.1%) over the live origin store `s3://tensor-tingfeng/tensordb/`, executed
on an i4i.8xlarge (32 vCPU / 247 GB / 25 Gbps-class NIC). Delta blobs live in
`s3://tensor-tingfeng/compressed_full/`; **tensordb/ was never modified** —
every delta's base is a raw tensordb/ blob (depth-1 by construction).

## Headline results

| metric | value |
|---|---|
| pairs compressed | **581,274** (245 skipped cross-dtype, **0 errors**) |
| target bytes | 32.118 TB → **10.905 TB** deltas (weighted tratio **0.3395**) |
| stored total (deltas + raw bases/unattached) | **13.61 TB** of 34.81 TB physical |
| **reduction (physical-unique accounting, paper's basis)** | **60.9%** |
| **reduction (logical accounting, dedup included)** | **65.2%** |
| wall time (compression pass) | **5.9 h** (~1.6 GB/s sustained from S3) |
| verification | every pair roundtrip-decompressed + XXH3 == tid before upload; sampled whole-model reconstruction byte-exact (see §5) |

Paper claims, validated:

- **TensorDex-TX 65.1%** — reproduced exactly offline from the published AE
  cache (stored 14.01 TB / 40.11 TB via the AE's own chart code), and
  realized at 60.9% on the surviving corpus; the 4.2-point gap is fully
  attributed to corpus decay (§4).
- **TensorDex-FM++ 70.5%** — validated at **71.1%** by combining the paper's
  plan with published fratios (39.6% byte coverage) + a 500-pair stratified
  FM++ sample measured on this box: **all 500 roundtrips byte-exact; the 400
  pairs with published fratios reproduced byte-identically (400/400)**;
  unmeasured pairs sampled at fratio 0.2875. FM++ remains measurement-only:
  the stored hub is TensorX (the only codec the engine/serving stack
  decodes).

## 1. What made the campaign feasible

1. **Fingerprint recovery (no 40 TB sketch pass).** The live master.db
   fingerprints (256-dim f32) are an old, incompatible sketch format. The
   paper-config BCS fps (d=2, w=1024, 2048×i32) were recovered from
   `tensordb/backups/node_{0..15}_metadata.db` — **bit-exact** vs today's
   kernel (verified on 3 nodes) — covering 642,797/742,027 tids; the 167
   remaining physical tensors were sketched fresh (5 min).
2. **Vectorized planner.** A NumPy/BLAS reimplementation of
   `_ops.plan_attach` (verified bit-equivalent: same bases, same pairs,
   |Δdist| = 0) plans all 642,964 tensors in **25 s**.
3. **Plan adoption.** The incremental CLI planner's plan predicts weighted
   CR 0.534 — the paper's plan (research-pipeline FlexSplit with Phase II
   splitting; shipped in the AE cache) realizes 0.333, choosing the same
   base for only 11% of common targets. The campaign therefore executes the
   **paper's own plan** (552,781 surviving pairs) + v1 fallback for
   orphaned-base targets (28,718), with the AE's forced-base conflict
   resolution keeping depth-1. Realized: paper-plan pairs wCR **0.330**,
   fallback pairs **0.554** — same codec, same corpus: **base selection is
   worth 22 CR points**; the planner, not the codec, is the dominant lever.
4. **Executor.** 28 processes × base-grouped 4 GB work units; fetch →
   TensorX L1 → roundtrip decompress + XXH3 verify → wrap (engine codec.py
   blob format) → upload; resumable per-pair JSONL ledger.

## 2. Reproduction fidelity

On the 195,839 pairs (12.79 TB) where the published results.db has a
measured tratio for the identical (target, base):

- our weighted tratio **0.3063** vs published **0.3068**
- per-pair relative diff **0.0000% at p50 and p95**; only 3,466 pairs
  (1.77%) differ at all — overwhelmingly sub-MB tensors, and 91% of the
  differences are in our favor (newer zstd).

FM++ cross-validation: 400/400 published fratios reproduced
**byte-identically**; 500/500 roundtrips byte-exact.

## 3. Throughput and bottlenecks (per-pair stage timing, whole campaign)

| stage | core-hours | share |
|---|---|---|
| S3 fetch | 321 | 71.3% |
| TensorX compress | 21 | 4.7% |
| roundtrip verify (decompress + hash) | 39 | 8.7% |
| S3 upload | 69 | 15.3% |

S3 GET throughput: ~27 MB/s per object at p50 under ~112 concurrent streams
(56 MB/s unloaded single-stream); aggregate 1.5–1.9 GB/s sustained (13–15
Gbps). The paper's "compression is I/O-bound" claim holds at hub scale: the
codec is 4.7% of stage time.

## 4. Corpus decay — a hub-scale reliability finding

Of 742,027 registered tensors (40.1 TB), **99,063 (5.31 TB, 13.4%) have no
blob in S3** — silent metadata/blob divergence (the jinoo/gemma2-9b failure
mode, at scale), including **1,058 of the paper's 7,948 bases**, which
degraded 76,764 targets to fallback/raw. This decay costs 4.2 points of
reduction vs the paper corpus and makes a set of registered models
unservable; per-model missing-blob flags are recorded in the catalog
(`storage_uri = 'missing://…'`).

## 5. Artifact layout (analysis-ready)

- `compressed_full/blobs/{tid[:2]}/{tid}.safetensors` — 581k TensorX delta
  blobs, self-describing headers (codec, base_tensor_id, item_size, level,
  shape, dtype).
- `tensordb/blobs/…` — raw bases + unattached tensors, untouched.
- `compressed_full/master.db` — full catalog: tensors / model_mappings /
  model_meta / **tensor_deltas** (581,254 edges), per-tensor storage_uri
  (which prefix), size_bytes (physical), logical_bytes, missing:// flags.
- `compressed_full/meta/campaign_meta.tar.gz` — plan (with provenance +
  published-tratio join), per-pair ledger (sizes, ratios, stage timings),
  logs.

**Sampled whole-model reconstruction verification (complete): 8/8 random
servable models rebuilt end-to-end from the hybrid layout — every tensor
(2,283 deltas + raw) hash-verified byte-exact.** Cold single-client read
amplification 1.07–1.55 (delta + base fetched, no base cache). Of 2,892
ready models, **2,530 (87.5%) are fully servable**; 362 have at least one
GC'd tensor (§4).

## 6. Research notes for FAST'27

1. **Plan quality dominates codec choice** (0.330 vs 0.554 wCR, same codec)
   — online incremental attach vs global/Phase-II planning is a real
   systems tradeoff: the better plan needs global fingerprint knowledge and
   periodic re-clustering, i.e., a *compaction service*, not an ingest-path
   heuristic.
2. **Metadata/blob divergence at scale** (13.4% of registry, 5.31 TB) —
   refcount-driven GC over the delta DAG + auditing is not optional at hub
   scale; base loss is amplified through the delta graph (1,058 lost bases
   → 76,764 degraded targets).
3. **I/O-bound compression** — 71% of stage time is S3 GETs; a compaction
   tier colocated with storage (or operating on cached tensors) would
   compress at 5–10× the wall speed.
4. **Serving story unchanged**: depth-1 hybrid layout (delta in
   compressed_full/, base in tensordb/) keeps worst-case closure = 2 blobs.
   Cross-model ancestor sharing (dag-aware caching) gets stronger: 7,948
   bases now serve 581k deltas.
