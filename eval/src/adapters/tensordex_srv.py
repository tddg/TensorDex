"""TensorDex server-reconstruction adapter.

The client talks to the deployed services instead of S3:
  resolve:  GET {metadata_url}/v1/manifest?model=... — per-model manifest
            from the standalone metadata server (production design), no
            local SQLite.
  download: GET {cache_url}/v1/tensor/{tid} for each *logical* tensor
            only — deltas/bases are invisible; the cache server fetches
            compressed blobs from S3, decodes and caches. Client-side
            read amplification is exactly 1.0 by construction.

Events use the s3_request_* names (operation="cache_get") so the same
summarizers apply; first-byte comes from urllib3's response headers.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import threading
import time
import urllib.parse
import uuid

import urllib3

from ..recorder import EventRecorder
from .base import DownloadAdapter, DownloadPlan, DownloadResult
from .tensordex import _ShardWriter

CHUNK = 8 * 1024 * 1024


class TensorDexServerAdapter(DownloadAdapter):
    system = "tensordex_srv"

    def __init__(self, metadata_url: str, cache_url: str,
                 max_workers: int = 10,
                 inflight_budget: int = 1280 * 1024 ** 2,
                 max_shard_bytes: int = 10 * 1024 ** 3):
        self.metadata_url = metadata_url.rstrip("/")
        self.cache_url = cache_url.rstrip("/")
        self.max_workers = max_workers
        self.inflight_budget = inflight_budget
        self.max_shard_bytes = max_shard_bytes
        self.http = urllib3.PoolManager(
            maxsize=max_workers + 4,
            timeout=urllib3.Timeout(connect=10, read=180),
            retries=urllib3.Retry(total=3, backoff_factor=0.5))

    def resolve(self, model_id: str) -> DownloadPlan:
        url = (f"{self.metadata_url}/v1/manifest?"
               f"model={urllib.parse.quote(model_id, safe='')}")
        r = self.http.request("GET", url)
        if r.status != 200:
            raise ValueError(f"manifest {r.status}: {r.data[:200]}")
        man = json.loads(r.data)
        tensors = [{"tensor_name": p["name"], "tid": p["tid"],
                    "logical_bytes": p["logical_bytes"],
                    "dtype": p["dtype"], "shape": p["shape"],
                    "dependency_depth": 0, "base_chain": [],
                    "object_fetch_count": 1}
                   for p in man["params"]]
        uniq = {}
        for t in tensors:
            uniq[t["tid"]] = t["logical_bytes"]
        objects = [{"key": f"/v1/tensor/{tid}", "tid": tid,
                    "size": size, "role": "tensor", "depth": 0}
                   for tid, size in sorted(uniq.items())]
        return DownloadPlan(model_id=model_id, system=self.system,
                            tensors=tensors, objects=objects,
                            extra={"manifest": man})

    # ---- download ----------------------------------------------------
    def _fetch(self, tid: str, size: int, pos: int, budget,
               recorder) -> bytes:
        # Head-of-line admission (same as TensorDexAdapter): only the
        # blob the consume loop is blocked on may exceed the budget,
        # otherwise completed buffers accumulate without bound.
        while True:
            with budget["lock"]:
                if (budget["bytes"] + size <= self.inflight_budget
                        or pos <= budget["head"]):
                    budget["bytes"] += size
                    break
            with budget["cv"]:
                budget["cv"].wait(0.05)
        url = f"{self.cache_url}/v1/tensor/{tid}"
        # Read timeout + explicit re-issue: a materialization that dies
        # server-side can leave the response stream open forever, and
        # urllib3 cannot resume a body that died mid-stream. Retrying is
        # safe (content-addressed, idempotent GET); the singleflight on
        # the server dedupes concurrent re-materializations.
        last_exc = None
        for attempt in range(3):
            rid = uuid.uuid4().hex[:16]
            recorder.emit("s3_request_start", request_id=rid, key=url,
                          operation="cache_get", tensor_name=tid,
                          attempt=attempt)
            r = None
            try:
                r = self.http.request("GET", url, preload_content=False)
                recorder.emit("s3_headers", request_id=rid,
                              http_status=r.status)
                if r.status != 200:
                    raise ValueError(f"cache_get {tid} -> {r.status}")
                n = int(r.headers.get("Content-Length", size))
                buf = bytearray(n)
                view = memoryview(buf)
                off = 0
                first = True
                for chunk in r.stream(CHUNK):
                    if first:
                        recorder.emit("s3_first_byte", request_id=rid)
                        first = False
                    view[off:off + len(chunk)] = chunk
                    off += len(chunk)
                r.release_conn()
                if off != n:
                    raise ValueError(f"short read {tid}: {off} != {n}")
                recorder.emit("s3_request_end", request_id=rid,
                              payload_bytes=n)
                return buf
            except Exception as exc:  # noqa: BLE001 — retry then re-raise
                last_exc = exc
                if r is not None:
                    try:
                        r.close()
                    except Exception:
                        pass
                recorder.emit("cache_retry", request_id=rid,
                              tensor_name=tid, attempt=attempt,
                              error=repr(exc))
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"cache_get {tid} failed after 3 attempts") from last_exc

    def download(self, plan: DownloadPlan, output_dir: str,
                 recorder: EventRecorder) -> DownloadResult:
        from tensordex import _ops

        res = DownloadResult(model_id=plan.model_id, system=self.system,
                             ok=False)
        recorder.emit("model_start", n_tensors=len(plan.tensors),
                      n_objects=len(plan.objects))
        writer = None
        try:
            params = [{"name": t["tensor_name"], "tid": t["tid"],
                       "dtype": t["dtype"], "shape": t["shape"],
                       "nbytes": t["logical_bytes"]}
                      for t in sorted(plan.tensors,
                                      key=lambda x: x["tensor_name"])]
            writer = _ShardWriter(output_dir, params, self.max_shard_bytes)

            budget = {"bytes": 0, "head": 0,
                      "lock": threading.Lock(),
                      "cv": threading.Condition()}
            with cf.ThreadPoolExecutor(self.max_workers) as pool:
                futs = [(o["tid"], pool.submit(
                    self._fetch, o["tid"], o["size"], pos, budget,
                    recorder))
                    for pos, o in enumerate(plan.objects)]
                for pos in range(len(futs)):
                    tid, fut = futs[pos]
                    with budget["lock"]:
                        budget["head"] = pos
                    with budget["cv"]:
                        budget["cv"].notify_all()
                    out = bytes(fut.result())  # _ops needs PyBytes
                    # Drop the future's internal reference to its result;
                    # otherwise every fetched buffer stays alive until
                    # the loop ends (a whole-model leak).
                    futs[pos] = None
                    fut = None
                    res.s3_get_count += 1
                    res.s3_payload_bytes += len(out)
                    got = _ops.content_hash(out)
                    if got != tid:
                        raise ValueError(
                            f"hash mismatch: expected {tid}, got {got}")
                    written = writer.write(tid, out)
                    res.logical_model_bytes += written
                    recorder.emit("tensor_ready", tid=tid,
                                  logical_bytes=len(out),
                                  dependency_depth=0,
                                  copies=written // max(len(out), 1))
                    with budget["lock"]:
                        budget["bytes"] -= len(out)
                    with budget["cv"]:
                        budget["cv"].notify_all()
                    out = None
            recorder.emit("model_assembly_end",
                          bytes_written=writer.bytes_written,
                          shards=len(writer.paths), decode_cpu_ms=0.0)
            res.output_paths.extend(writer.paths)
            recorder.emit("model_ready",
                          logical_model_bytes=res.logical_model_bytes)
            res.ok = True
        except Exception as e:  # noqa: BLE001
            res.error = repr(e)
            recorder.emit("error", where="tensordex_srv.download",
                          exception=res.error)
        finally:
            if writer is not None:
                writer.close()
        return res
