"""Materializer service (plan Phase 2.2–2.3) — FastAPI, port 8702.

The miss path behind nginx: nginx `try_files` serves cache hits off
NVMe directly (sendfile); a miss proxies here. We materialize the
tensor (fetch compressed blobs, decode bottom-up, hash-verify, atomic
cache insert) and reply with ``X-Accel-Redirect`` so nginx still does
the actual send — Python never copies hot bytes. Without nginx (dev
mode, no ``X-Accel-Expected`` header) the file is streamed directly.

    GET  /v1/tensor/{tid}     materialize-if-missing, then serve
    POST /v1/warm             {"model": ...} — whole-closure pipelined
                              warm in background (the fix for Stage 1's
                              per-tensor demand-miss cold penalty:
                              45–65 MB/s at 60% NIC idle)
    GET  /v1/warm/{model_slug}  warm job status
    GET  /healthz, /metrics

Concurrency/RAM discipline:
  - singleflight per tid (in-proc locks) — concurrent misses coalesce;
  - chain materializations bounded by decode slots; chains whose target
    exceeds BIG_THRESHOLD run exclusively (the prototype's OOM guard);
  - one warm job per model at a time; pipeline budgets are sized so
    warm + demand worst case stays under the configured RAM cap;
  - every decoded tensor is pinned against eviction while its chain is
    in flight.

Config env (set by main() for uvicorn workers): TDM_METADATA_URL,
TDM_BUCKET, TDM_CACHE_DIR, TDM_SPILL_DIR, TDM_LOG, TDM_S3_POOL,
TDM_DECODE_SLOTS, TDM_WARM_INFLIGHT, TDM_WARM_BASE_RAM,
TDM_DEMAND_INFLIGHT, TDM_DEMAND_BASE_RAM.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse

import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from .cache import TensorCache
from .manifest import model_slug
from .pipeline import ClosurePipeline

METADATA_URL = os.environ.get("TDM_METADATA_URL", "http://127.0.0.1:8701")
BUCKET = os.environ.get("TDM_BUCKET", "tensor-tingfeng")
CACHE_DIR = os.environ.get("TDM_CACHE_DIR", "/mnt/nvme/cache")
SPILL_DIR = os.environ.get("TDM_SPILL_DIR", "/mnt/nvme/spill")
LOG_PATH = os.environ.get("TDM_LOG", "logs/materializer.jsonl")
S3_POOL = int(os.environ.get("TDM_S3_POOL", "32"))
DECODE_SLOTS_N = int(os.environ.get("TDM_DECODE_SLOTS", "4"))
WARM_INFLIGHT = int(os.environ.get("TDM_WARM_INFLIGHT",
                                   str(2 * 1024 ** 3)))
WARM_BASE_RAM = int(os.environ.get("TDM_WARM_BASE_RAM",
                                   str(2 * 1024 ** 3)))
DEMAND_INFLIGHT = int(os.environ.get("TDM_DEMAND_INFLIGHT",
                                     str(256 * 1024 ** 2)))
DEMAND_BASE_RAM = int(os.environ.get("TDM_DEMAND_BASE_RAM",
                                     str(512 * 1024 ** 2)))
BIG_THRESHOLD = 256 * 1024 ** 2

_LOG_LOCK = threading.Lock()
_LOG_FH = None
_METRICS_LOCK = threading.Lock()
_COUNTERS: dict[str, float] = {}

cache = TensorCache(CACHE_DIR)
_S3 = None
_S3_LOCK = threading.Lock()
DECODE_SLOTS = threading.BoundedSemaphore(DECODE_SLOTS_N)
BIG_LOCK = threading.Lock()
TID_LOCKS: dict[str, threading.Lock] = {}
TID_LOCKS_GUARD = threading.Lock()
# model -> {"state": running|done|error, ...}
WARM_JOBS: dict[str, dict] = {}
WARM_GUARD = threading.Lock()


def _log(rec: dict):
    global _LOG_FH
    with _LOG_LOCK:
        if _LOG_FH is None:
            os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
            _LOG_FH = open(LOG_PATH, "a", buffering=1)
        _LOG_FH.write(json.dumps(rec, separators=(",", ":")) + "\n")


def _count(name: str, delta: float = 1):
    with _METRICS_LOCK:
        _COUNTERS[name] = _COUNTERS.get(name, 0) + delta


def s3():
    global _S3
    with _S3_LOCK:
        if _S3 is None:
            import boto3
            from botocore.config import Config
            _S3 = boto3.client("s3", config=Config(
                max_pool_connections=S3_POOL,
                retries={"mode": "standard", "max_attempts": 5}))
        return _S3


def tid_lock(tid: str) -> threading.Lock:
    with TID_LOCKS_GUARD:
        if tid not in TID_LOCKS:
            TID_LOCKS[tid] = threading.Lock()
        return TID_LOCKS[tid]


# Test seams: unit tests monkeypatch these two + s3().
def fetch_closure(tid: str) -> list[dict]:
    r = requests.get(f"{METADATA_URL}/v1/closure",
                     params={"tid": tid}, timeout=30)
    r.raise_for_status()
    return r.json()["chain"]


def fetch_plan(model: str, have) -> dict:
    r = requests.post(f"{METADATA_URL}/v1/plan",
                      json={"model": model, "have": sorted(have)},
                      timeout=60)
    r.raise_for_status()
    return r.json()


def _pipeline(inflight: int, base_ram: int) -> ClosurePipeline:
    return ClosurePipeline(
        s3(), BUCKET, inflight_budget=inflight, base_ram_budget=base_ram,
        spill_dir=os.path.join(SPILL_DIR, f"pid{os.getpid()}"))


def materialize(tid: str, stats: dict) -> str:
    """Ensure decoded bytes for tid are cached; returns the entry path."""
    path = cache.path(tid)
    if os.path.exists(path):
        stats["cache_hit"] = True
        return path
    stats["cache_hit"] = False
    with tid_lock(tid):
        if os.path.exists(path):  # materialized while we waited
            stats["cache_hit"] = True
            return path
        chain = fetch_closure(tid)  # tid first, root base last
        # Cut the chain at the deepest already-cached ancestor — the
        # pipeline resolves it through base_lookup instead of S3.
        entries = []
        for ent in chain:
            if cache.has(ent["tid"]):
                break
            entries.append(ent)
        big = chain[0]["logical_bytes"] > BIG_THRESHOLD
        gate = BIG_LOCK if big else DECODE_SLOTS
        with gate, cache.pinned([e["tid"] for e in entries]):
            pipe = _pipeline(DEMAND_INFLIGHT, DEMAND_BASE_RAM)
            st = pipe.run(entries,
                          lambda e, data: cache.put(e["tid"], data),
                          base_lookup=cache.get_bytes)
        for k in ("s3_gets", "s3_bytes", "fetch_ms", "decode_ms"):
            stats[k] = stats.get(k, 0) + st[k]
    return path


def _warm_job(model: str):
    job = WARM_JOBS[model]
    t0 = time.perf_counter()
    try:
        plan = fetch_plan(model, cache.tids())
        entries = plan["fetch"]
        job.update(fetch=len(entries),
                   fetch_bytes=plan["fetch_bytes"])
        tids = [e["tid"] for e in entries]
        with cache.pinned(tids):
            pipe = _pipeline(WARM_INFLIGHT, WARM_BASE_RAM)
            st = pipe.run(entries,
                          lambda e, data: cache.put(e["tid"], data),
                          base_lookup=cache.get_bytes)
        job.update(state="done", stats=st,
                   duration_s=round(time.perf_counter() - t0, 3))
        _count("warm_done")
    except Exception as e:  # noqa: BLE001
        job.update(state="error", error=repr(e),
                   duration_s=round(time.perf_counter() - t0, 3))
        _count("warm_error")
    _log({"ts_ns": time.perf_counter_ns(), "event": "warm",
          "model": model, **{k: v for k, v in job.items()}})


app = FastAPI(title="tensordex-materializer", docs_url=None,
              redoc_url=None)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/metrics")
def metrics():
    out = ["# TYPE tdm_counter counter"]
    with _METRICS_LOCK:
        for name, v in sorted(_COUNTERS.items()):
            out.append(f'tdm_counter{{name="{name}"}} {v}')
    return Response("\n".join(out) + "\n", media_type="text/plain")


@app.get("/v1/tensor/{tid}")
def get_tensor(tid: str, request: Request):
    t0 = time.perf_counter_ns()
    stats: dict = {}
    try:
        path = materialize(tid, stats)
    except Exception as e:  # noqa: BLE001
        _count("tensor_error")
        _log({"ts_ns": time.perf_counter_ns(), "tid": tid, "status": 500,
              "error": repr(e),
              "duration_ms": (time.perf_counter_ns() - t0) / 1e6})
        return JSONResponse({"error": repr(e)}, status_code=500)
    _count("tensor_hit" if stats.get("cache_hit") else "tensor_miss")
    size = os.path.getsize(path)
    _log({"ts_ns": time.perf_counter_ns(), "tid": tid, "status": 200,
          "serve_bytes": size,
          "duration_ms": (time.perf_counter_ns() - t0) / 1e6, **stats})
    headers = {"ETag": tid, "Content-Length": str(size),
               "X-Cache": "hit" if stats.get("cache_hit") else "miss"}
    if request.headers.get("x-accel-expected"):
        # nginx does the send: zero bytes cross this process.
        headers["X-Accel-Redirect"] = \
            f"/_cache/{urllib.parse.quote(cache.rel_path(tid))}"
        return Response(status_code=200, headers=headers)
    return FileResponse(path, media_type="application/octet-stream",
                        headers=headers)


@app.post("/v1/warm")
async def warm(request: Request):
    body = await request.json()
    model = body.get("model", "")
    if not model:
        return JSONResponse({"error": "model required"}, status_code=400)
    with WARM_GUARD:
        job = WARM_JOBS.get(model)
        if job and job["state"] == "running":
            return {"model": model, "state": "running", "dedup": True}
        WARM_JOBS[model] = {"state": "running",
                            "started_ns": time.perf_counter_ns()}
        threading.Thread(target=_warm_job, args=(model,),
                         daemon=True).start()
    _count("warm_started")
    return {"model": model, "state": "running"}


@app.get("/v1/warm/{slug}")
def warm_status(slug: str):
    for model, job in WARM_JOBS.items():
        if model_slug(model) == slug or model == slug:
            return {"model": model, **job}
    return JSONResponse({"error": "no warm job"}, status_code=404)


def main(argv=None) -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-url", default=METADATA_URL)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--spill-dir", default=SPILL_DIR)
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8702)
    args = ap.parse_args(argv)
    os.environ.update(
        TDM_METADATA_URL=args.metadata_url, TDM_BUCKET=args.bucket,
        TDM_CACHE_DIR=args.cache_dir, TDM_SPILL_DIR=args.spill_dir,
        TDM_LOG=args.log)
    # Single worker on purpose: singleflight + RAM budgets are in-proc.
    # Throughput scales via nginx (hits) and pipeline threads (misses).
    uvicorn.run("tensordex_serving.materializer:app", host=args.host,
                port=args.port, workers=1, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
