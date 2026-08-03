# TensorDex productionization — design notes (2026-08-01)

Distilled from design discussion (user + Claude) during the FAST'27 eval
sessions. Priority ordering set by user: **(1) performance evaluation —
pinpoint bottlenecks (metadata server + colocated reconstruction cache,
TensorDex vs baselines); (2) tradeoff analysis afterwards.**

## 1. Metadata: no server exists today

TensorDex today has NO metadata service. Metadata is a whole-catalog
SQLite file stored as an S3 object next to the blobs:

- `compressed_eval/master.db` — 32 MB (9 models; includes tensor_deltas),
  0.28 s cold fetch.
- `tensordb/master.db` — 1.35 GB (2,893 models, dedup-only). Unrealistic
  to ship per client; 40x size gap shows whole-catalog SQLite does not
  scale as a client-facing interface.

Clients download the DB once, then resolve locally per model
(model_mappings -> iterative tensors+tensor_deltas closure, 5–176 ms).
All S3 traffic after resolve is data: 1 GET per blob in the closure,
single fetch round (the SQL manifest gives every key up front). The
alternative — discovering bases from blob headers — costs
dependency_depth+1 sequential rounds.

## 2. Production architecture (control plane / data plane)

**Control plane** — stateless metadata API replicas over a replicated
OLTP DB (Postgres-class) holding tensors / model_mappings /
tensor_deltas, PLUS (missing from today's schema): per-tensor refcounts
and real model/version lifecycle states.

- `resolve(model) -> manifest`: full delta closure (keys, sizes,
  dtype/shape, base graph) in one response; immutable per model version
  => trivially CDN/edge-cacheable. Per-model manifests replace
  ship-the-whole-catalog.
- `begin_upload / commit_upload`: blobs first (orphans harmless),
  metadata commit last in one transaction, idempotent retries.
- Optionally presigned S3 URLs so the metadata tier never carries data
  bytes.

**Data plane** — content-addressed blobs on S3 (raw or TensorX delta).
Client: manifest -> parallel GETs -> bottom-up delta decode -> content
hash verify (keep; free end-to-end integrity).

**Background services** — ingest workers (hash -> dedup -> sketch ->
put -> commit); async compactor (FlexSplit pairing + TensorX encode;
never blocks upload; bound chain depth <=2–3); GC/auditor (refcount
mark-and-sweep with grace period + periodic metadata<->S3 HEAD sampling).

## 3. "S3 is HA so no extra fault tolerance" — rejected

S3 protects bytes, not cross-object invariants. Evidence from our own
bucket:

1. Metadata tier is the crux: content-addressed keys are opaque —
   lose metadata and 72 TB of durable blobs are garbage. Metadata needs
   replication/backup/PITR more than blobs need anything.
2. Half-finished ingests leak blobs: ~1M orphans / ~32 TB (44% of
   bucket) in the live store. Fix = write ordering + GC, not S3.
3. Inverse failure is worse: jinoo/gemma2-9b 'ready' in metadata,
   417/464 blobs GC'd. With deltas this sharpens: a base may serve
   OTHER models' deltas -> deletion must be refcount-driven over the
   delta DAG, never per-model.
4. AE-style in-place blob rewrite during compression is a race under
   concurrent readers. Production: write delta as NEW object, flip
   pointer transactionally, GC old after grace period.

## 4. Client-side vs backend reconstruction (tradeoff, analyze later)

Client-side decode facts (measured, single cold client, t3.xlarge):

- Decode CPU is trivial: ~8.5 s CPU for an 18.5 GB gemma (vs 348 s e2e).
- The real tax is the dependency closure: cold pull of
  gemma-2-9b-it-SimPO fetched 17.09 GB for 18.48 GB logical (read amp
  0.92) — storage saving nearly vanished on the wire because SimPO is
  mostly base tensors. Delta-heavy variants (Woona, Tong) fetch far
  less. Docker-precedent doesn't apply: per-blob zstd is
  self-contained; delta-vs-base is not.
- Who has the bases decides the winner:
  - Cold one-shot client: backend reconstruction strictly dominates
    (also transparent — plain safetensors, no client changes).
  - Warm fleet client (repeated fine-tune pulls, same family): local
    base cache => new fine-tune costs only delta bytes (~10x less
    traffic than ANY backend scheme, which must ship full logical bytes
    on the last hop). Backend reconstruction forfeits this win.
- Egress economics: same-region S3->EC2 is free (closure costs time not
  money); a reconstruction tier serving logical bytes maximizes egress
  on the hop that costs money.
- A reconstruction cache that stores decoded tensors = tiered storage
  ("compress the cold tier"), roughly HF Xet/CAS shape. Per-request
  decode instead would burn CPU on every hot download.

**Production answer: both, per client.** `resolve()` returns either
(a) presigned URLs for reconstructed artifacts via the cache tier
(default, transparent) or (b) the delta closure; smart clients advertise
held bases and receive only missing keys. Smart path degrades to (a).

## 5. Current eval status (feeds the tradeoff analysis)

- Stage 1 matrix running: 9 models x {tensordex_c = compressed hub
  client-side decode, tensordex = uncompressed dedup store} x 2 reps,
  single client, full instrumentation. One OOM lesson: depth-2 delta
  chains stack ~2 GB decoded bases; fetch budget 1.25 GB + 10 workers
  fits 16 GB RAM.
- Next: metadata server + colocated reconstruction cache on this VM;
  measure server-reconstructed downloads vs tensordex_c vs uncompressed;
  bottleneck breakdown (resolve / fetch span / decode / write).
- Then: Stage 2 base-cache reuse (warm-client win), tradeoff memo.
