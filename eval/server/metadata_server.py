#!/usr/bin/env python3
"""TensorDex metadata server (production-design prototype, plan: deploy a
standalone metadata service instead of shipping the whole SQLite catalog).

Serves per-model manifests and per-tensor delta closures over a hub
metadata.db (read-only). Stateless; one SQLite read-only connection per
request thread.

Endpoints:
    GET /healthz
    GET /v1/models                     -> {"models": [...]}
    GET /v1/manifest?model=<id>        -> {"model", "params": [...],
                                           "blobs": [...]}   (full closure)
    GET /v1/closure?tid=<tid>          -> {"chain": [...]}    (tid first,
                                           root base last)

Every request is logged as one JSONL line (ts_ns, path, model/tid,
duration_ms, status, bytes) so server-side metadata cost is measurable.

Usage:
    python eval/server/metadata_server.py --db eval/raw/hub_compressed/metadata.db \
        --bucket tensor-tingfeng --prefix compressed_eval --port 8701
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SQL_CHUNK = 500
DTYPE_ITEM = {
    "torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4,
    "torch.float64": 8, "torch.int64": 8, "torch.int32": 4,
    "torch.int16": 2, "torch.int8": 1, "torch.uint8": 1, "torch.bool": 1,
}

ARGS = None
LOG_LOCK = threading.Lock()
LOG_FH = None
LOCAL = threading.local()


def db() -> sqlite3.Connection:
    if getattr(LOCAL, "db", None) is None:
        LOCAL.db = sqlite3.connect(
            f"file:{ARGS.db}?mode=ro", uri=True, check_same_thread=False)
    return LOCAL.db


def blob_key(tid: str) -> str:
    return f"{ARGS.prefix}/blobs/{tid[:2]}/{tid}.safetensors"


def has_deltas(con) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='tensor_deltas'").fetchone() is not None


def manifest(model: str) -> dict:
    con = db()
    mapping = con.execute(
        "SELECT param_name, tensor_id FROM model_mappings "
        "WHERE model_name = ?", (model,)).fetchall()
    if not mapping:
        raise KeyError(model)
    deltas_on = has_deltas(con)
    blobs: dict[str, dict] = {}
    frontier = sorted({t for _, t in mapping})
    while frontier:
        nxt = set()
        for i in range(0, len(frontier), SQL_CHUNK):
            chunk = frontier[i:i + SQL_CHUNK]
            ph = ",".join("?" * len(chunk))
            rows = con.execute(
                f"SELECT id, shape, dtype, size_bytes FROM tensors "
                f"WHERE id IN ({ph})", chunk).fetchall()
            if len(rows) != len(chunk):
                missing = sorted(set(chunk) - {r[0] for r in rows})
                raise ValueError(f"missing tensors: {missing[:3]}")
            dmap = dict(con.execute(
                f"SELECT tensor_id, base_tensor_id FROM tensor_deltas "
                f"WHERE tensor_id IN ({ph})", chunk).fetchall()) \
                if deltas_on else {}
            for tid, shape_json, dtype, stored in rows:
                shape = json.loads(shape_json)
                nel = 1
                for s in shape:
                    nel *= s
                base = dmap.get(tid)
                blobs[tid] = {
                    "tid": tid, "key": blob_key(tid),
                    "stored_bytes": stored, "shape": shape, "dtype": dtype,
                    "logical_bytes": nel * DTYPE_ITEM[dtype],
                    "base_id": base}
                if base and base not in blobs:
                    nxt.add(base)
        frontier = sorted(nxt - set(blobs))

    depth: dict[str, int] = {}

    def d(tid):
        if tid not in depth:
            b = blobs[tid]["base_id"]
            depth[tid] = 0 if not b else 1 + d(b)
        return depth[tid]

    for t in blobs:
        d(t)
    for t, b in blobs.items():
        b["depth"] = depth[t]
    params = [{"name": p, "tid": t, "dtype": blobs[t]["dtype"],
               "shape": blobs[t]["shape"],
               "logical_bytes": blobs[t]["logical_bytes"]}
              for p, t in sorted(mapping)]
    return {"model": model, "params": params,
            "blobs": sorted(blobs.values(), key=lambda b: b["tid"])}


def closure(tid: str) -> dict:
    con = db()
    deltas_on = has_deltas(con)
    chain = []
    cur = tid
    seen = set()
    while cur:
        if cur in seen:
            raise ValueError(f"delta cycle at {cur}")
        seen.add(cur)
        row = con.execute(
            "SELECT id, shape, dtype, size_bytes FROM tensors WHERE id=?",
            (cur,)).fetchone()
        if row is None:
            raise KeyError(cur)
        shape = json.loads(row[1])
        nel = 1
        for s in shape:
            nel *= s
        base = None
        if deltas_on:
            r = con.execute(
                "SELECT base_tensor_id FROM tensor_deltas "
                "WHERE tensor_id=?", (cur,)).fetchone()
            base = r[0] if r else None
        chain.append({"tid": cur, "key": blob_key(cur),
                      "stored_bytes": row[3], "shape": shape,
                      "dtype": row[2],
                      "logical_bytes": nel * DTYPE_ITEM[row[2]],
                      "base_id": base})
        cur = base
    return {"chain": chain}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet default stderr logging
        pass

    def _json(self, code: int, obj: dict) -> int:
        body = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return len(body)

    def do_GET(self):  # noqa: N802
        t0 = time.perf_counter_ns()
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        code, nbytes, ident = 200, 0, ""
        try:
            if url.path == "/healthz":
                nbytes = self._json(200, {"ok": True})
            elif url.path == "/v1/models":
                con = db()
                models = [r[0] for r in con.execute(
                    "SELECT DISTINCT model_name FROM model_mappings")]
                nbytes = self._json(200, {"models": models})
            elif url.path == "/v1/manifest":
                ident = q.get("model", [""])[0]
                nbytes = self._json(200, manifest(ident))
            elif url.path == "/v1/closure":
                ident = q.get("tid", [""])[0]
                nbytes = self._json(200, closure(ident))
            else:
                code = 404
                nbytes = self._json(404, {"error": "not found"})
        except KeyError as e:
            code = 404
            nbytes = self._json(404, {"error": f"unknown: {e}"})
        except Exception as e:  # noqa: BLE001
            code = 500
            nbytes = self._json(500, {"error": repr(e)})
        if LOG_FH:
            line = json.dumps({
                "ts_ns": time.perf_counter_ns(), "path": url.path,
                "ident": ident, "status": code, "resp_bytes": nbytes,
                "duration_ms": (time.perf_counter_ns() - t0) / 1e6},
                separators=(",", ":"))
            with LOG_LOCK:
                LOG_FH.write(line + "\n")


def main() -> int:
    global ARGS, LOG_FH
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--bucket", default="tensor-tingfeng")
    ap.add_argument("--prefix", default="compressed_eval")
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--log", default="eval/results/server/metadata_server.jsonl")
    ARGS = ap.parse_args()
    os.makedirs(os.path.dirname(ARGS.log) or ".", exist_ok=True)
    LOG_FH = open(ARGS.log, "a", buffering=1)
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"metadata server on :{ARGS.port} db={ARGS.db}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
