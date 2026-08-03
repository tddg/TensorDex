# i4i server-path matrix — protocol (2026-08-03, relaunch after 5b9f5fb)

Server: i4i.2xlarge (172.31.93.29), metadata :8701 + nginx cache front
:8700 (materializer loopback-only), serving code at 5b9f5fb. Client:
t3.xlarge (172.31.45.64), same VPC, private IP, adapter `tensordex_srv`
(path-style URLs + 180 s read timeout + 3-attempt re-issue), 10 workers,
harness `run_single.py`.

## Scope: ALL 9 models, rep0 cold / rep1 warm

Cache was fully cleared server-side (0 entries, pins + spill wiped,
materializer restarted on 5b9f5fb, fresh metrics) immediately before
this run — so all 9 models are honestly cold, including the two 3B
llamas measured during deployment smoke. The earlier smoke traces in
`eval/results/i4i_smoke/` remain valid as pre-fix reference points but
this matrix supersedes them.

## Cold protocol: pure demand-miss, client-parallel x10

- NO `/v1/warm` calls anywhere. Cold = the client's parallel
  `GET /v1/tensor/{tid}` stream (10 workers) drives demand-miss
  materialization; concurrency comes from the client. (On-box
  sequential demand smoke: 62.7 MB/s — different protocol, do not
  mix in tables.)
- Zero tid overlap across all 9 manifests (measured server-side), so
  run order cannot contaminate cold runs; the driver shuffles model
  order and that is fine.
  **POST-RUN CORRECTION (see I4I_COMPARISON.md §4): this premise was
  wrong** — manifests DO share closure tids heavily within families
  (e.g. Woona's closure contains 265 of SimPO's 464 params), and
  materialization caches ancestors, so later cold runs found shared
  tensors resident. Measured impact on served BYTES is small (7.2%
  SimPO, ≤2.5% all others); per-run contamination is annotated in
  I4I_COMPARISON.md rather than the runs being discarded.
- Eviction never triggers (~99 GB decoded vs 1.7 TB NVMe): this
  matrix measures the cache, not the eviction policy.

## Comparability notes vs the smoke runs

- 5b9f5fb fixed a server bug where EVERY cache miss sent a malformed
  Content-Length on the X-Accel-Redirect reply, aborting the upstream
  connection mid-response: files were still served byte-exact, but
  each miss burned its keepalive connection. In this rerun the client
  connection pool is actually reused on the cold path for the first
  time — if cold numbers improve vs the 12.60 s GSOC smoke, that is
  the fix, not noise.
- 5b9f5fb also fixed a concurrent-spill collision (shared spill dir,
  chains through the same >512 MB spilled base racing on unlink) that
  produced the one `tensor_error` + hung response which wedged the
  first matrix attempt (see events_mlfoundations_r0_STALLED.jsonl.aside;
  root cause: FileNotFoundError on a depth-2 qwen delta whose shared
  base spill file was unlinked by a sibling chain).
- Materialization failures now return 502 fast (404 for unknown tid);
  client retries up to 3x with backoff (`cache_retry` events in traces).

## Caveats

- Client NIC (t3.xlarge, ~145 MB/s sustained, higher in burst) is the
  delivery ceiling for warm runs; on-box warm was 878 MB/s. Watch for
  t3 burst-credit depletion late in the matrix (7B/9B warm reps).
- nginx proxy_read_timeout is 600 s by design (Woona-scale cold chains
  legitimately run minutes); with fast-fail 502s this should only be
  reached by genuinely long materializations.
- Static keys still in the materializer env (pre-IAM-role); systemd
  migration pending — irrelevant to these measurements.
