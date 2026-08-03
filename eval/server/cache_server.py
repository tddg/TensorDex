#!/usr/bin/env python3
"""TensorDex reconstruction cache server (production-design prototype).

Colocated with the metadata server. Serves *reconstructed* (raw) tensor
bytes: on cache miss it asks the metadata server for the tensor's delta
closure, fetches the needed compressed blobs from S3, decodes TensorX
bottom-up, verifies the content hash, caches every decoded tensor on
disk, and streams the requested tensor back. Clients therefore see plain
bytes with read amplification exactly 1.0 — the closure cost lives here.

Endpoints:
    GET  /healthz
    GET  /v1/tensor?tid=<tid>   -> application/octet-stream (decoded bytes)
    POST /admin/clear           -> drop the entire disk cache + stats

Cache layout: {cache_dir}/{tid[:2]}/{tid}  (decoded bytes, atomic rename)

Every request logs one JSONL line: cache_hit, s3_gets, s3_bytes,
fetch_ms, decode_ms, serve_bytes, duration_ms — the server-side half of
the bottleneck breakdown.

Usage:
    python eval/server/cache_server.py --metadata-url http://127.0.0.1:8701 \
        --bucket tensor-tingfeng --cache-dir eval/raw/reconstruct_cache \
        --port 8702
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402

from eval.src.adapters.tensordex import parse_safetensors_bytes  # noqa: E402

ARGS = None
S3 = None
LOG_FH = None
LOG_LOCK = threading.Lock()
# Limit concurrent materializations (each may hold a decoded chain).
DECODE_SLOTS = None
# Chains whose target tensor exceeds this decode exclusively — several
# multi-GB chains in flight would OOM a 16 GB host.
BIG_THRESHOLD = 256 * 1024 ** 2
BIG_LOCK = threading.Lock()
TID_LOCKS: dict[str, threading.Lock] = {}
TID_LOCKS_GUARD = threading.Lock()
CHUNK = 8 * 1024 * 1024


def cache_path(tid: str) -> str:
    return os.path.join(ARGS.cache_dir, tid[:2], tid)


def tid_lock(tid: str) -> threading.Lock:
    with TID_LOCKS_GUARD:
        if tid not in TID_LOCKS:
            TID_LOCKS[tid] = threading.Lock()
        return TID_LOCKS[tid]


def fetch_closure(tid: str) -> list[dict]:
    url = f"{ARGS.metadata_url}/v1/closure?tid={urllib.parse.quote(tid)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())["chain"]


def s3_get(key: str) -> bytes:
    resp = S3.get_object(Bucket=ARGS.bucket, Key=key)
    n = int(resp["ContentLength"])
    buf = bytearray(n)
    view = memoryview(buf)
    off = 0
    for chunk in resp["Body"].iter_chunks(CHUNK):
        view[off:off + len(chunk)] = chunk
        off += len(chunk)
    return bytes(buf)


def put_cache(tid: str, data) -> None:
    path = cache_path(tid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def materialize(tid: str, stats: dict) -> str:
    """Ensure decoded bytes for tid exist in the cache; returns path.

    Walks the delta chain root-first, reusing any already-cached decoded
    ancestor as the starting base.
    """
    from tensordex import _ops

    path = cache_path(tid)
    if os.path.exists(path):
        stats["cache_hit"] = True
        return path
    stats["cache_hit"] = False
    with tid_lock(tid):
        if os.path.exists(path):  # decoded while we waited on the lock
            stats["cache_hit"] = True
            return path
        chain = fetch_closure(tid)  # tid first, root last
        big = chain[0]["logical_bytes"] > BIG_THRESHOLD
        with BIG_LOCK if big else DECODE_SLOTS:
            # Deepest ancestor already decoded in cache (or root).
            start = len(chain)  # index of first element to decode from raw
            base_bytes = None
            for i in range(1, len(chain)):
                p = cache_path(chain[i]["tid"])
                if os.path.exists(p):
                    with open(p, "rb") as fh:
                        base_bytes = fh.read()
                    start = i
                    break
            # Decode from the bottom of the un-cached suffix upward.
            for i in range(min(start, len(chain)) - 1, -1, -1):
                ent = chain[i]
                t0 = time.perf_counter_ns()
                raw = s3_get(ent["key"])
                stats["s3_gets"] += 1
                stats["s3_bytes"] += len(raw)
                stats["fetch_ms"] += (time.perf_counter_ns() - t0) / 1e6
                meta, _k, (_dt, _sh, payload) = parse_safetensors_bytes(raw)
                t0 = time.perf_counter_ns()
                if ent["base_id"] is None:
                    out = bytes(payload)
                else:
                    if base_bytes is None:
                        raise RuntimeError(
                            f"no base bytes for delta {ent['tid']}")
                    item_size = int(meta.get("item_size", 2))
                    out = bytes(_ops.decompress_tensorx_rust(
                        bytes(payload), base_bytes, item_size))
                stats["decode_ms"] += (time.perf_counter_ns() - t0) / 1e6
                del raw, payload
                got = _ops.content_hash(out)
                if got != ent["tid"]:
                    raise RuntimeError(
                        f"hash mismatch decoding {ent['tid']}: got {got}")
                put_cache(ent["tid"], out)
                base_bytes = out
            del base_bytes
    return path


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path == "/admin/clear":
            n = 0
            if os.path.isdir(ARGS.cache_dir):
                for entry in os.listdir(ARGS.cache_dir):
                    shutil.rmtree(os.path.join(ARGS.cache_dir, entry),
                                  ignore_errors=True)
                    n += 1
            self._json(200, {"cleared": n})
        else:
            self._json(404, {"error": "not found"})

    def do_GET(self):  # noqa: N802
        t0 = time.perf_counter_ns()
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path == "/healthz":
            self._json(200, {"ok": True})
            return
        if url.path != "/v1/tensor":
            self._json(404, {"error": "not found"})
            return
        tid = q.get("tid", [""])[0]
        stats = {"s3_gets": 0, "s3_bytes": 0, "fetch_ms": 0.0,
                 "decode_ms": 0.0, "cache_hit": False}
        code, served = 200, 0
        try:
            path = materialize(tid, stats)
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb", buffering=0) as fh:
                while True:
                    chunk = fh.read(CHUNK)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    served += len(chunk)
        except BrokenPipeError:
            code = 499
        except Exception as e:  # noqa: BLE001
            code = 500
            try:
                self._json(500, {"error": repr(e)})
            except Exception:  # noqa: BLE001
                pass
        if LOG_FH:
            line = json.dumps({
                "ts_ns": time.perf_counter_ns(), "tid": tid,
                "status": code, "serve_bytes": served,
                "duration_ms": (time.perf_counter_ns() - t0) / 1e6,
                **stats}, separators=(",", ":"))
            with LOG_LOCK:
                LOG_FH.write(line + "\n")


def main() -> int:
    global ARGS, S3, LOG_FH, DECODE_SLOTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-url", default="http://127.0.0.1:8701")
    ap.add_argument("--bucket", default="tensor-tingfeng")
    ap.add_argument("--cache-dir", default="eval/raw/reconstruct_cache")
    ap.add_argument("--port", type=int, default=8702)
    ap.add_argument("--s3-pool", type=int, default=32)
    ap.add_argument("--decode-slots", type=int, default=4,
                    help="max concurrent chain materializations")
    ap.add_argument("--log", default="eval/results/server/cache_server.jsonl")
    ARGS = ap.parse_args()
    os.makedirs(ARGS.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ARGS.log) or ".", exist_ok=True)
    LOG_FH = open(ARGS.log, "a", buffering=1)
    DECODE_SLOTS = threading.BoundedSemaphore(ARGS.decode_slots)
    S3 = boto3.client("s3", config=Config(
        region_name="us-east-1", max_pool_connections=ARGS.s3_pool,
        retries={"mode": "standard", "max_attempts": 5}))
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"cache server on :{ARGS.port} cache={ARGS.cache_dir} "
          f"meta={ARGS.metadata_url}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
