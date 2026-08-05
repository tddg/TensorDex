# DAG-aware caching experiments — design & run book

Companion to `eval/results/caching_direction.html` (motivation, cost model,
hypotheses H1–H4). This document is the operational design: workload, policy
implementations, deployment, sweep matrix, metrics, protocol, and analysis
plan. Substrate: the compressed hub built 2026-08-05 — 581,254 TensorX deltas
(`compressed_full/`) over 7,169 active bases in `tensordb/`, catalog at
`compressed_full/master.db`, 2,530 fully-servable models.

## 0. The one-sentence objective

Show that for a cache fronting a delta-compressed model hub, policies that
optimize **byte-weighted miss cost over the delta DAG** beat hit-rate-optimal
policies on origin traffic and end-to-end latency at realistic capacities —
and quantify by how much, where, and why.

## 1. Workload

**Request unit:** a model pull (client resolves manifest, fetches all tensors
of the model through the cache tier, verifies, discards). Tensor-level
requests derive from the model's manifest — this preserves the real
co-access structure (a model's 300–700 tensors always travel together),
which is what makes ancestor sharing exploitable.

**Popularity:** two generators, both seeded:

- `zipf`: model i ranked by a permutation drawn per seed; P(i) ∝ 1/rank^α,
  α ∈ {0.8, 1.1} (bracketing observed hub skews).
- `hf-weighted`: HuggingFace 30-day download counts for the same repo names
  (crawled once, checked into `eval/config/model_popularity.json`); models
  missing from HF get the median weight. Preferred for headline results;
  zipf for sensitivity.

**Arrival process:** closed-loop swarm (each client: pull, verify, sleep
exp(λ=5 s), repeat) — matches how registries are actually hammered, and
makes offered load a function of concurrency C. One open-loop cell
(Poisson arrivals at 0.8× measured closed-loop throughput) as a check
that conclusions aren't closed-loop artifacts.

**Trace freeze:** each (seed, α, duration) generates a concrete request
trace file *before* any run; all policies replay identical traces —
policy comparisons are paired, never sampled independently.

## 2. Policies under test

Implemented in the cache server's evictor (nginx serves; the FastAPI
materializer owns admission/eviction; capacity enforced logically):

| id | policy | notes |
|---|---|---|
| `lru` | LRU over reconstructed tensors | today's behavior; hit-rate anchor |
| `slru` | size-aware LRU (evict-largest-first among cold) | cheap classic upgrade |
| `gdsf` | GreedyDual-Size-Frequency, cost = origin bytes at last miss | strongest prior art; cost is *static* per item |
| `basepin` | pin all 7,169 bases (405 GB), LRU for the rest | lower-bound anchor: max miss rate, min per-miss cost; invalid below 405 GB capacity (use top-k bases by fan-in that fit half the budget) |
| `dagv` | dag_value greedy: value(x)/size(x) eviction, value(x) = EWMA own-hit bytes + Σ_{d ∈ resident-or-recently-requested descendants(x)} (base-refetch bytes avoided) | the contribution; value updates on residency change of descendants — O(fan-in) amortized via lazy recompute |
| `oracle` | offline: knapsack/greedy on the *frozen trace* with full future knowledge, replayed as a static allocation + LRU for the remainder | upper bound; gap to `dagv` measures headroom |

Decode-on-miss path is identical across policies; only admission/eviction
differ. `dagv` and `basepin` need the delta graph — served to the evictor
once from master.db (`tensor_deltas` join, 22 MB in memory).

## 3. Deployment

- **Cache node:** the i4i.8xlarge (32 vCPU / 247 GB / 2×3.4 TB NVMe),
  nginx sendfile + X-Accel front, materializer with the tid-scoped
  refcounted spill fix (the 5b9f5fb spill-collision lesson). Cache dir on
  one NVMe; the other NVMe holds traces/logs.
- **Metadata:** manifest server colocated (9–13 ms resolve is noise at
  n=1; E5 separates it later).
- **Clients:** 8 × c6a.xlarge swarm nodes (4 vCPU each, ~12.5 Gbps
  aggregate — enough to exceed the cache NIC), driven by the existing
  `tensordex_srv` adapter in lightweight verify-only mode (hash, no shard
  write — we are measuring the tier, not client disks; the full
  verify+write mode runs in one anchor cell to keep continuity with the
  Stage-1 numbers).
- **Origin:** S3 `compressed_full/` + `tensordb/`, untouched.

## 4. Sweep matrix

Primary sweep (policy × capacity), at C = 32 clients, hf-weighted, 2 h
steady-state after a 30 min warmup, 1 rep + a second rep on the
headline cells:

| capacity | lru | slru | gdsf | basepin | dagv | oracle |
|---|---|---|---|---|---|---|
| 50 GB | ● | ● | ● | ●(top-k) | ● | ● |
| 200 GB | ● | ● | ● | ● | ● | ● |
| 800 GB | ● | ● | ● | ● | ● | ● |
| 3.2 TB | ● | — | ● | ● | ● | — |

Secondary sweeps:

- **Concurrency:** C ∈ {8, 128} × {lru, dagv} × {200 GB, 800 GB} (H3's
  CPU term; find the ceiling from E1 in the same runs).
- **Store format (H3):** at 200 GB × C=32: reconstructed vs
  delta+decode-on-read vs hybrid (bases raw, targets stored as deltas) ×
  {lru, dagv}.
- **Skew sensitivity:** α ∈ {0.8, 1.1} × {lru, dagv} at 200 GB.
- **Ancestor pre-warming (H4):** paired-model protocol — serve family
  sibling A cold, then B; vs B cold with no prior A — measure B's
  latency delta as a function of shared-ancestor bytes. 20 family pairs
  sampled from the manifest overlap matrix.

≈ 30 cells × ~2.5 h ≈ 3 machine-days on one cache node; capacity cells
are independent and can interleave overnight.

## 5. Metrics (per run, from existing JSONL instrumentation + new evictor log)

Primary:
- **origin bytes per hour** (the tier's S3 bill and its scaling limit)
- **byte-weighted miss cost**: Σ over requests of origin bytes incurred
- e2e model latency p50 / p95 / p99; TTFB per tensor

Secondary / explanatory:
- hit rate by requests and by bytes (reported to demonstrate it
  mis-ranks policies — H2's figure)
- decode core-seconds per hour; NVMe read/write MB/s; NIC in/out
- evictor accounting: residency-seconds by role (base / reconstruction /
  delta), eviction churn, dag_value recompute overhead
- per-miss classification: cheap (base resident) vs expensive (base
  fetched) vs raw — the direct test of the cost model

## 6. Protocol details & controls

1. Trace frozen per seed before runs; identical across policies (paired
   comparison; report per-trace deltas, not just means).
2. Cache cleared + materializer restarted between cells; S3 has no
   cache-state (origin is cold by construction — verified in the i4i
   campaign).
3. Client verify-only mode pins the client wall well above the tier
   ceiling (transport-only probe showed 590 MB/s/client available); one
   anchor cell runs full verify+write for continuity.
4. Run order randomized within capacity blocks; wall-clock and instance
   metadata pinned in every run record.
5. Failure budget: any cell with >0.1% request errors is rerun; errors
   never silently dropped.

## 7. Analysis plan → figures

- **F1 (headline):** origin GB/h vs capacity, one line per policy —
  expected: dagv ≪ lru at 50–200 GB, converging by 3.2 TB (H1).
- **F2:** hit-rate vs e2e-p99 scatter per policy-cell — the "hit rate
  mis-ranks" figure (H2).
- **F3:** stacked per-miss cost decomposition (cheap/expensive/raw) per
  policy at 200 GB — mechanism evidence.
- **F4:** decode core-s/h vs concurrency for reconstructed vs
  delta-store formats (H3 frontier).
- **F5:** pre-warming: sibling-B latency vs shared-ancestor bytes (H4).
- **T1:** oracle-gap table: dagv captures X% of oracle's improvement
  over lru.

## 8. Build list (order of work)

1. Evictor: policy plugin interface + capacity enforcement + role-aware
   accounting log (extend `server/` cache materializer).
2. Workload generator + trace freezer (`eval/scripts/caching/gen_trace.py`);
   HF popularity crawl → `eval/config/model_popularity.json`.
3. dag_value + gdsf + basepin implementations (delta-graph feed from
   master.db).
4. Oracle solver (offline greedy over frozen trace).
5. Swarm driver (extend `run_server_eval.py` to closed-loop multi-client)
   + verify-only client mode.
6. Smoke cell (50 GB, C=8, lru) → sanity dashboards → full matrix.

## 9. Risks / open questions

- **Client wall reappearing at C=8**: mitigated by verify-only mode; if
  the 8-node swarm can't exceed the cache NIC, add nodes before blaming
  the tier.
- **dag_value recompute cost** at 581k-edge scale: lazy/EWMA design
  should be O(1) amortized per request; measure and report its overhead
  as part of the results (a policy that's too expensive to run is a
  finding, not an embarrassment).
- **Popularity realism**: HF download counts are repo-level and
  long-tailed; sensitivity sweep over α covers mis-estimation.
- **Single-node scope**: placement (E7) is explicitly out of scope here;
  the DAG-aware *placement* question inherits this machinery later.
