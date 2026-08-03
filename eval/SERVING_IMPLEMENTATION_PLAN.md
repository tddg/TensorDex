# TensorDex Serving: metadata server + NVMe hot cache — implementation plan

Target deployment: one i4i.2xlarge (8 vCPU, 64 GB RAM, 1.8 TB instance-store
NVMe, 12.5 Gbps), colocated metadata + cache. Owner deploys; this repo
supplies the code, config, and bootstrap.

Research context (FAST'27): the server is the substrate for the paper's
contributions — state-negotiated transfer planning (§P1.4) and DAG-weighted
caching (§P2.5) are built in from the start, not retrofitted. See
"Novelty framing" at the bottom.

---

## Phase 0 — shared library (½ day)

`server/tensordex_serving/` package, importable by both services and tests:

- `manifest.py` — manifest schema + builder: read master.db, emit one
  immutable JSON manifest per model version:
  `{model, version, params[{name, tid, dtype, shape, logical_bytes}],
    closure[{tid, key, stored_bytes, codec, base_id, depth}]}`.
  `version` = content hash of the canonicalized manifest. Golden-tested
  against `eval/src/adapters/tensordex.py::resolve` (must agree on all 9
  eval models, incl. Woona's depth-2 chains).
- `pipeline.py` — the head-of-line byte-budget fetch/decode engine, ported
  from the eval adapter (proven: NIC-saturating at 5.4 GB RSS). Parameters:
  inflight budget, decoded-base RAM cap + NVMe spill dir.
- `blob.py` — safetensors blob parsing incl. the nested
  `__metadata__[name]` JSON-string codec metadata (`item_size`!).

## Phase 1 — metadata server (1 day)

FastAPI + uvicorn (2 workers is plenty; measured cost is 26–122 ms/manifest
and most of that disappears with precomputation).

1. **Manifest store.** Offline job (`build_manifests.py`) materializes all
   manifests to `/srv/manifests/{model_slug}@{version}.json` + a `latest`
   symlink per model. Runs at ingest/compaction; serving never computes.
2. **API.**
   - `GET /v1/models` — list w/ versions, logical/stored sizes.
   - `GET /v1/manifest?model=` — serve prebuilt file; `ETag: <version>`,
     `Cache-Control: immutable`. SQLite fallback if missing.
   - `GET /v1/closure?tid=` — for the cache's miss path.
   - `POST /v1/plan` — **negotiation (research hook)**: body
     `{model, have: [tid...]}` → `{fetch: [closure entries minus anything
     derivable from haves], reconstructed_fallback: bool}`. v1 semantics:
     a have satisfies its entire subtree requirement (client holds decoded
     bytes). Pure function over the manifest — no DB writes.
   - `GET /healthz`, `GET /metrics` (Prometheus text).
3. **SQLite hygiene.** Read-only URI conns per worker, WAL, mmap. Catalog +
   manifests live on the EBS root volume (durable), never on instance store.
4. **Tests.** Manifest golden tests; plan() correctness: have=∅ ⇒ full
   closure; have={bases} ⇒ deltas only; have={delta's parent chain} ⇒ leaf
   only; cross-model haves (llama shared embedding) honored.

## Phase 2 — hot cache (2–3 days)

Data path rule: Python never copies hot bytes; nginx serves from NVMe.

1. **Layout.** `/mnt/nvme/cache/{tid[:2]}/{tid}` = decoded bytes, atomic
   tmp+rename, hash-verified before rename. Content-addressed ⇒ no
   invalidation, cross-model dedup for free.
2. **nginx front** (`server/deploy/nginx.conf`):
   `GET /v1/tensor/{tid}` → `try_files` off NVMe → miss proxies to the
   materializer, which writes the entry and replies `X-Accel-Redirect`
   so nginx still does the send (sendfile, zero-copy).
3. **Materializer** (FastAPI, same host):
   - singleflight per tid (in-proc locks; the eval prototype's tid_locks).
   - **whole-closure pipelining**: on first miss for a model, warm the
     full manifest through `pipeline.py` in background (the measured cold
     penalty — 45–65 MB/s, 60% NIC idle — came from per-tensor
     demand-miss materialization; this is its fix). Explicit
     `POST /v1/warm?model=` too.
   - RAM: one byte-budget semaphore across all materializations
     (8 GB cap; leave ~50 GB to page cache — page cache IS the fast tier,
     it's what served 700 MB/s).
4. **Eviction daemon.** Watermarks on NVMe usage (evict at 90% → down to
   80%). v1: LRU by atime at tid granularity, never evict tids pinned by
   in-flight materializations.
5. **DAG-weighted eviction hook (research).** Score interface
   `score(tid) -> float` with pluggable policies: v1 `lru`, v2
   `dag_value` = re-materialization cost saved, summed over resident
   dependents (needs the reverse-dependency index from manifests). Ship
   both behind a flag + counters so policies are A/B-comparable — this is
   a paper experiment, not just ops.
6. **Metrics/logs.** Keep the prototype's per-request JSONL (hit, fetch_ms,
   decode_ms, TTFB, bytes) + `/metrics` aggregates. These logs were how
   every bottleneck in the study was found; they are non-optional.
7. **Tests.** Cold/warm correctness (hash at client), concurrent-miss
   coalescing, eviction under synthetic fill, empty-cache restart, big-chain
   RAM ceiling (Woona under 8 GB budget).

## Phase 3 — client + Stage 2 harness (1 day)

- Extend `eval/src/adapters/tensordex_srv.py`: resolve via `/v1/plan` with
  a local base-cache dir advertised as haves; fetch the want-list (mixed:
  tensors from cache server, or presigned raw blobs). This adapter **is**
  the Stage 2 (warm fleet) experiment when pointed at successive fine-tunes.
- Keep the existing harness (`run_single.py`, `run_server_eval.py`,
  `compare_download.py`) pointed at the new host via `--metadata-url/--cache-url`.

## Phase 4 — deployment on i4i.2xlarge (½ day, mostly yours)

`server/deploy/bootstrap.sh` (idempotent, run as root on first boot):

1. Format + mount instance store: xfs on `/mnt/nvme`, `noatime` is NOT set
   (eviction needs atime; use `relatime`).
2. Install nginx + Python venv (Python ≤3.12 for the tensordex wheel —
   same PyO3 constraint as this VM); pip install the repo + built wheel.
3. systemd units: `td-metadata.service`, `td-materializer.service`,
   `td-evictor.service` (+ nginx). Restart=always; JSONL logs to the EBS
   volume.
4. Auth: **instance profile (IAM role) with read-only S3 scope** — no
   static keys on the box (and the keys pasted in chat earlier should be
   rotated regardless).
5. Config file `serving.yaml`: bucket/prefix, watermarks, RAM budgets,
   eviction policy flag.
6. Smoke: `curl /healthz` ×2, one 3B manifest + cold pull + warm pull via
   the existing harness (single model; not the full matrix).

Sizing defaults for i4i.2xlarge: cache watermarks 1.45 TB / 1.25 TB;
materializer budget 8 GB; uvicorn 2+2 workers; nginx sendfile on,
`output_buffers` default.

## Phase 5 — DAG-transactional write path (paper spine; after deployment)

Moves from non-goals into scope: with TDN owning geo-distribution, the
FAST'27 paper is the HUB — the storage service layer — and its core
contribution is lifecycle correctness over the delta DAG:

1. **Upload commit protocol**: blobs first (orphans harmless), one
   metadata transaction last; idempotent retries via content addressing;
   crash at any step leaves the store consistent.
2. **Refcount GC**: mark-and-sweep with grace period over the DAG; a base
   referenced by other models'/tenants' deltas is never collectable;
   auditor reconciles metadata vs S3 (HEAD sampling).
3. **Copy-on-write compaction**: delta written as a NEW object, pointer
   flip in one transaction, old blob reaped by GC — never in-place
   (readers holding old manifests stay correct).
4. **Online ingest**: synchronous dedup at upload; sketch + pairing
   async; temperature-driven storage form (hot raw/shallow, cold deep) as
   an origin-local compaction policy; bounded-working-set global pairing
   (the deferred Phase A/B design).
Fault-injection harness (kill ingest/compaction mid-flight, then audit
invariants) is the paper's correctness evaluation.

## Explicit non-goals (v1 deployment)

- No multi-node cache, no consistent hashing — one box.
- Phase 5 write path is post-deployment scope (paper artifact), not in
  the first i4i bring-up.
- No TLS/authn on the services (private VPC assumption) — note in README.
- No Postgres. SQLite + precomputed manifests until write concurrency
  demands otherwise.
- **No geo-anything**: placement, edge caching, cross-site egress, client
  steering are TDN's territory (companion work) — cited, never claimed.

## Novelty framing (FAST'27) — the hub between TensorDex and TDN

Layer map: TensorDex (SOSP'26) = the encoder, offline/single-writer.
TDN = the geo-distributed delivery network (owns placement + egress
optimization). FAST'27 = the missing layer between them: the hub as a
real storage service. Thesis: **in a derivative-aware store, every
ordinary hub operation becomes a DAG transaction.**

1. **C1 — transactional lifecycle** (Phase 5): upload/GC/compaction with
   cross-object DAG invariants. Motivated by live-bucket forensics: 44%
   of 72 TB orphaned by partial ingests; 'ready' models with GC'd blobs.
2. **C2 — online ingest + storage-form scheduling**: when/what to
   (re)compress; hot raw / cold deep as origin-local tiering. The
   storage-form residue of "storage-optimal ≠ distribution-optimal"
   (measured: 35% smaller store, 8/9 slower cold, amp ≤1.45) — placement
   itself belongs to TDN.
3. **C3 — the DAG-aware serving interface as narrow waist**: manifests,
   closure, /v1/plan have/want, reconstruct-on-read. FAST defines it;
   TDN consumes it (an edge = a client with a large have-list).
4. **C4 — DAG-weighted origin caching**: keep ONLY if TDN does not claim
   graph-aware eviction at edges — needs a written boundary agreement
   with the TDN authors (do this early; shared reviewer pool).

Not claimed as novel: manifests, CAS layout, back-pressure clients,
nginx serving — implementation, cited as such. Geo-distribution: TDN's.
