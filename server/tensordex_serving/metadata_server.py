"""TensorDex metadata server (plan Phase 1) — FastAPI, port 8701.

Serves prebuilt immutable manifests off the EBS root; SQLite is a
fallback for manifests not yet materialized. Stateless: one read-only
WAL/mmap connection per worker thread, no writes anywhere.

    GET  /v1/models              list w/ versions + logical/stored sizes
    GET  /v1/manifest?model=     prebuilt manifest; ETag = version,
                                 Cache-Control: immutable; 304 on match
    GET  /v1/closure?tid=        delta chain, tid first, root base last
                                 (the cache's miss path)
    POST /v1/plan                {model, have:[tid...]} -> fetch list
                                 (state negotiation, §P1.4)
    GET  /healthz, /metrics      liveness + Prometheus text

Config via environment (set by main() from CLI args so uvicorn workers
inherit it): TDS_DB, TDS_MANIFEST_ROOT, TDS_PREFIX, TDS_LOG.

Every request logs one JSONL line (ts_ns, path, ident, status,
resp_bytes, duration_ms) — same format the Stage 1 bottleneck analysis
consumed; non-optional.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from . import manifest as M
from .blob import logical_nbytes

DB_PATH = os.environ.get("TDS_DB", "eval/raw/hub_compressed/metadata.db")
MANIFEST_ROOT = os.environ.get("TDS_MANIFEST_ROOT", "/srv/manifests")
PREFIX = os.environ.get("TDS_PREFIX", "compressed_eval")
LOG_PATH = os.environ.get("TDS_LOG", "logs/metadata_server.jsonl")

_LOCAL = threading.local()
_LOG_LOCK = threading.Lock()
_LOG_FH = None

# Per-worker Prometheus-style counters ({(path, status): count}, ms sums).
_METRICS_LOCK = threading.Lock()
_REQ_COUNT: dict[tuple[str, int], int] = {}
_REQ_MS: dict[str, float] = {}


def _log_fh():
    global _LOG_FH
    if _LOG_FH is None:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        _LOG_FH = open(LOG_PATH, "a", buffering=1)
    return _LOG_FH


def db() -> sqlite3.Connection:
    con = getattr(_LOCAL, "db", None)
    if con is None:
        con = M.connect_ro(DB_PATH)
        con.execute("PRAGMA mmap_size=268435456")
        _LOCAL.db = con
    return con


app = FastAPI(title="tensordex-metadata", docs_url=None, redoc_url=None)


@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter_ns()
    response = await call_next(request)
    dur_ms = (time.perf_counter_ns() - t0) / 1e6
    path = request.url.path
    with _METRICS_LOCK:
        key = (path, response.status_code)
        _REQ_COUNT[key] = _REQ_COUNT.get(key, 0) + 1
        _REQ_MS[path] = _REQ_MS.get(path, 0.0) + dur_ms
    ident = (request.query_params.get("model")
             or request.query_params.get("tid") or "")
    line = json.dumps({
        "ts_ns": time.perf_counter_ns(), "path": path, "ident": ident,
        "status": response.status_code,
        "resp_bytes": int(response.headers.get("content-length") or 0),
        "duration_ms": dur_ms}, separators=(",", ":"))
    with _LOG_LOCK:
        _log_fh().write(line + "\n")
    return response


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/metrics")
def metrics():
    out = ["# TYPE tds_requests_total counter"]
    with _METRICS_LOCK:
        for (path, status), n in sorted(_REQ_COUNT.items()):
            out.append(f'tds_requests_total{{path="{path}",'
                       f'status="{status}"}} {n}')
        out.append("# TYPE tds_request_ms_sum counter")
        for path, ms in sorted(_REQ_MS.items()):
            out.append(f'tds_request_ms_sum{{path="{path}"}} {ms:.3f}')
    return Response("\n".join(out) + "\n", media_type="text/plain")


@app.get("/v1/models")
def models():
    result = []
    for model in M.list_models(db()):
        man = M.load_manifest(MANIFEST_ROOT, model)
        if man is not None:
            result.append({
                "model": model, "version": man["version"],
                "params": len(man["params"]),
                "logical_bytes": sum(p["logical_bytes"]
                                     for p in man["params"]),
                "stored_bytes": sum(c["stored_bytes"]
                                    for c in man["closure"])})
        else:
            result.append({"model": model, "version": None})
    return {"models": result}


def _manifest_for(model: str) -> dict:
    """Prebuilt manifest, or SQLite fallback if not materialized."""
    man = M.load_manifest(MANIFEST_ROOT, model)
    if man is None:
        man = M.build_manifest(db(), model, PREFIX)
    return man


@app.get("/v1/manifest")
def get_manifest(model: str, request: Request):
    try:
        man = _manifest_for(model)
    except KeyError:
        return JSONResponse({"error": f"unknown model: {model}"},
                            status_code=404)
    if request.headers.get("if-none-match") == man["version"]:
        return Response(status_code=304)
    return JSONResponse(man, headers={
        "ETag": man["version"],
        "Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/v1/closure")
def closure(tid: str):
    """Delta chain for one tid (tid first, root base last)."""
    con = db()
    chain, cur, seen = [], tid, set()
    while cur:
        if cur in seen:
            return JSONResponse({"error": f"delta cycle at {cur}"},
                                status_code=500)
        seen.add(cur)
        row = con.execute(
            "SELECT id, shape, dtype, size_bytes FROM tensors "
            "WHERE id=?", (cur,)).fetchone()
        if row is None:
            return JSONResponse({"error": f"unknown tid: {cur}"},
                                status_code=404)
        shape = json.loads(row[1])
        r = con.execute(
            "SELECT base_tensor_id, codec FROM tensor_deltas "
            "WHERE tensor_id=?", (cur,)).fetchone()
        base, codec = r if r else (None, None)
        chain.append({"tid": cur, "key": M.blob_key(PREFIX, cur),
                      "stored_bytes": row[3],
                      "logical_bytes": logical_nbytes(row[2], shape),
                      "dtype": row[2], "shape": shape,
                      "codec": codec, "base_id": base})
        cur = base
    return {"chain": chain}


@app.post("/v1/plan")
async def plan(request: Request):
    body = await request.json()
    model = body.get("model", "")
    have = set(body.get("have", ()))
    try:
        man = _manifest_for(model)
    except KeyError:
        return JSONResponse({"error": f"unknown model: {model}"},
                            status_code=404)
    return M.plan(man, have)


def main(argv=None) -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--manifest-root", default=MANIFEST_ROOT)
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    os.environ.update(TDS_DB=args.db, TDS_MANIFEST_ROOT=args.manifest_root,
                      TDS_PREFIX=args.prefix, TDS_LOG=args.log)
    uvicorn.run("tensordex_serving.metadata_server:app", host=args.host,
                port=args.port, workers=args.workers, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
