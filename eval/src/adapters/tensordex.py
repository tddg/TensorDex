"""TensorDex S3 adapter (plan §2.1) — streaming, memory-bounded.

The stock TensorDex S3 backend cannot decode delta blobs (engine.py raises
NotImplementedError for deltas on non-local backends), so this adapter
implements the cloud pull itself using the same primitives:

  resolve:  model_mappings -> direct tensor ids, then an iterative
            tensors+tensor_deltas closure over the delta graph (direct
            SQL, read-only; works on both the dedup-only live store and
            the delta-compressed hub).
  download: fetch every blob in the closure, decode deltas bottom-up
            (_ops.decompress_tensorx_rust), verify by content hash, and
            pwrite payloads into preallocated safetensors shards with
            true dtype/shape headers.

Memory bounds (16 GB VM, 18 GB models):
  - fetches are submitted in DFS order over the delta forest and their
    raw bytes count against an in-flight budget; the submitter blocks
    until decode consumes earlier blobs,
  - decoded bytes are kept only while a not-yet-decoded delta still
    references them (refcount), so the live set is roughly the current
    root-to-leaf path, not the model.

Because the SQL closure yields every key up front, all S3 fetches are
issuable in one round (sequential_fetch_rounds == 1); dependency depth
still constrains decode order and is recorded per tensor.

The compressed hub's tensors.size_bytes is the *stored* (post-compress)
size — logical sizes are always recomputed from shape x dtype.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sqlite3
import struct
import threading

from ..recorder import EventRecorder
from .base import DownloadAdapter, DownloadPlan, DownloadResult

CHUNK = 8 * 1024 * 1024
SQL_CHUNK = 500

# torch dtype -> (safetensors dtype, item size)
DTYPE_MAP = {
    "torch.bfloat16": ("BF16", 2), "torch.float16": ("F16", 2),
    "torch.float32": ("F32", 4), "torch.float64": ("F64", 8),
    "torch.int64": ("I64", 8), "torch.int32": ("I32", 4),
    "torch.int16": ("I16", 2), "torch.int8": ("I8", 1),
    "torch.uint8": ("U8", 1), "torch.bool": ("BOOL", 1),
}


def parse_safetensors_bytes(data: bytes):
    """Return (metadata_dict, real_key, (dtype, shape, payload_view)).

    A TensorDex blob holds `_fingerprint`, the tensor under its own name,
    and `<name>_fingerprint` (see ae/_blobs.py); only the real tensor key
    is returned. The payload is a memoryview into `data` (no copy).
    """
    (hlen,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + hlen])
    meta = header.pop("__metadata__", {}) or {}
    body = memoryview(data)[8 + hlen:]
    for name, info in header.items():
        if name == "_fingerprint" or name.endswith("_fingerprint"):
            continue
        # Per-tensor codec metadata is a JSON string keyed by the tensor
        # name inside __metadata__ (e.g. {"tensor": "{\"codec\":...}"}).
        tmeta = meta.get(name, {})
        if isinstance(tmeta, str):
            tmeta = json.loads(tmeta)
        b0, b1 = info["data_offsets"]
        return tmeta, name, (info["dtype"], info["shape"], body[b0:b1])
    raise ValueError("no real tensor key in blob header")


class _ShardWriter:
    """Preallocated multi-shard safetensors writer (out-of-order pwrite).

    Shapes/dtypes/sizes are known from metadata, so headers are written
    up front and each tensor payload lands at its offset when decoded.
    """

    def __init__(self, output_dir: str, params: list[dict],
                 max_shard_bytes: int = 10 * 1024 ** 3):
        os.makedirs(output_dir, exist_ok=True)
        groups: list[list[dict]] = [[]]
        acc = 0
        for p in params:
            if groups[-1] and acc + p["nbytes"] > max_shard_bytes:
                groups.append([])
                acc = 0
            groups[-1].append(p)
            acc += p["nbytes"]

        n = len(groups)
        self.paths: list[str] = []
        self.fds: list[int] = []
        # tid -> [(fd, abs_offset, nbytes)]
        self.slots: dict[str, list[tuple[int, int, int]]] = {}
        self.bytes_written = 0
        self._lock = threading.Lock()
        for i, group in enumerate(groups):
            name = ("model.safetensors" if n == 1 else
                    f"model-{i + 1:05d}-of-{n:05d}.safetensors")
            path = os.path.join(output_dir, name)
            header, off = {}, 0
            for p in group:
                st_dtype, _ = DTYPE_MAP[p["dtype"]]
                header[p["name"]] = {
                    "dtype": st_dtype, "shape": p["shape"],
                    "data_offsets": [off, off + p["nbytes"]]}
                off += p["nbytes"]
            hjson = json.dumps(header, separators=(",", ":")).encode()
            pad = (8 - len(hjson) % 8) % 8  # keep payloads 8-aligned
            hjson += b" " * pad
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.write(fd, struct.pack("<Q", len(hjson)) + hjson)
            base = 8 + len(hjson)
            os.truncate(fd, base + off)
            for p in group:
                o0 = header[p["name"]]["data_offsets"][0]
                self.slots.setdefault(p["tid"], []).append(
                    (fd, base + o0, p["nbytes"]))
            self.paths.append(path)
            self.fds.append(fd)

    def write(self, tid: str, payload) -> int:
        """Write payload to every param slot mapped to tid; returns bytes."""
        n = 0
        for fd, off, nbytes in self.slots.get(tid, ()):
            if len(payload) != nbytes:
                raise ValueError(f"size mismatch for {tid}: "
                                 f"{len(payload)} != {nbytes}")
            os.pwrite(fd, payload, off)
            n += nbytes
        with self._lock:
            self.bytes_written += n
        return n

    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds = []


class _BaseStore:
    """Decoded-base store with a RAM cap; overflow spills to disk.

    Delta chains of ~1.8 GB embedding tensors stack several decoded
    copies (root + intermediate + output + codec transients) — more than
    a 16 GB client can hold in RAM alongside the fetch pipeline. Bases
    that fit under ram_budget stay in memory; larger residents spill to
    a scratch dir and are re-read on use.
    """

    def __init__(self, spill_dir: str, ram_budget: int = 4 * 1024 ** 3):
        self.spill_dir = spill_dir
        self.ram_budget = ram_budget
        self.ram: dict[str, bytes] = {}
        self.ram_bytes = 0
        self.on_disk: set[str] = set()

    def _path(self, tid: str) -> str:
        return os.path.join(self.spill_dir, tid)

    def put(self, tid: str, data: bytes):
        if self.ram_bytes + len(data) <= self.ram_budget:
            self.ram[tid] = data
            self.ram_bytes += len(data)
        else:
            os.makedirs(self.spill_dir, exist_ok=True)
            with open(self._path(tid), "wb") as fh:
                fh.write(data)
            self.on_disk.add(tid)

    def get(self, tid: str):
        if tid in self.ram:
            return self.ram[tid]
        if tid in self.on_disk:
            with open(self._path(tid), "rb") as fh:
                return fh.read()
        return None

    def drop(self, tid: str):
        data = self.ram.pop(tid, None)
        if data is not None:
            self.ram_bytes -= len(data)
        if tid in self.on_disk:
            self.on_disk.discard(tid)
            try:
                os.unlink(self._path(tid))
            except OSError:
                pass

    def clear(self):
        self.ram.clear()
        self.ram_bytes = 0
        for tid in list(self.on_disk):
            self.drop(tid)
        try:
            os.rmdir(self.spill_dir)
        except OSError:
            pass


class TensorDexAdapter(DownloadAdapter):
    system = "tensordex"

    def __init__(self, s3_client, bucket: str, prefix: str,
                 metadata_db_dir: str, max_workers: int = 10,
                 inflight_budget: int = 1280 * 1024 ** 2,
                 max_shard_bytes: int = 10 * 1024 ** 3,
                 base_ram_budget: int = 2560 * 1024 ** 2,
                 base_cache: dict | None = None):
        """metadata_db_dir: dir containing metadata.db (a master.db copy).
        base_cache: optional persistent {tid: decoded_bytes} cache shared
        across model pulls (Stage 2 shared-base experiments)."""
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.max_workers = max_workers
        self.inflight_budget = inflight_budget
        self.base_ram_budget = base_ram_budget
        self.max_shard_bytes = max_shard_bytes
        self.base_cache = base_cache
        self.db_path = os.path.join(metadata_db_dir, "metadata.db")

    # ---- resolve -----------------------------------------------------
    def _blob_key(self, tid: str) -> str:
        return f"{self.prefix}/blobs/{tid[:2]}/{tid}.safetensors"

    def resolve(self, model_id: str) -> DownloadPlan:
        db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            mapping = dict(db.execute(
                "SELECT param_name, tensor_id FROM model_mappings "
                "WHERE model_name = ?", (model_id,)).fetchall())
            if not mapping:
                raise ValueError(f"model not in metadata: {model_id}")
            has_deltas = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='tensor_deltas'").fetchone() is not None

            # Iterative closure over the delta graph.
            blobs: dict[str, dict] = {}
            frontier = sorted(set(mapping.values()))
            while frontier:
                nxt: set[str] = set()
                for i in range(0, len(frontier), SQL_CHUNK):
                    chunk = frontier[i:i + SQL_CHUNK]
                    ph = ",".join("?" * len(chunk))
                    rows = db.execute(
                        f"SELECT id, shape, dtype, size_bytes FROM tensors "
                        f"WHERE id IN ({ph})", chunk).fetchall()
                    if len(rows) != len(chunk):
                        missing = set(chunk) - {r[0] for r in rows}
                        raise ValueError(f"tensors missing from metadata: "
                                         f"{sorted(missing)[:5]}...")
                    deltas = dict(db.execute(
                        f"SELECT tensor_id, base_tensor_id FROM "
                        f"tensor_deltas WHERE tensor_id IN ({ph})",
                        chunk).fetchall()) if has_deltas else {}
                    for tid, shape_json, dtype, stored in rows:
                        shape = json.loads(shape_json)
                        nel = 1
                        for d in shape:
                            nel *= d
                        logical = nel * DTYPE_MAP[dtype][1]
                        base = deltas.get(tid)
                        blobs[tid] = {"tid": tid, "stored": stored,
                                      "logical": logical, "shape": shape,
                                      "dtype": dtype, "base_id": base}
                        if base and base not in blobs:
                            nxt.add(base)
                frontier = sorted(nxt - set(blobs))
        finally:
            db.close()

        # Dependency depth per tid (0 = direct storage).
        depth: dict[str, int] = {}

        def d(tid: str, seen=()) -> int:
            if tid in depth:
                return depth[tid]
            if tid in seen:
                raise ValueError(f"delta cycle at {tid}")
            b = blobs[tid]["base_id"]
            depth[tid] = 0 if not b else 1 + d(b, seen + (tid,))
            return depth[tid]

        for t in blobs:
            d(t)

        tensors = []
        for param, tid in mapping.items():
            info = blobs[tid]
            chain = []
            cur = info["base_id"]
            while cur:
                chain.append(cur)
                cur = blobs[cur]["base_id"]
            tensors.append({
                "tensor_name": param,
                "tid": tid,
                "logical_bytes": info["logical"],
                "dtype": info["dtype"],
                "shape": info["shape"],
                "dependency_depth": depth[tid],
                "base_chain": chain,
                "object_fetch_count": 1 + len(chain),
            })

        objects = [{"key": self._blob_key(t), "tid": t, "size": b["stored"],
                    "role": "delta" if b["base_id"] else "blob",
                    "depth": depth[t]}
                   for t, b in blobs.items()]
        return DownloadPlan(model_id=model_id, system=self.system,
                            tensors=tensors, objects=objects,
                            extra={"blobs": blobs, "mapping": mapping,
                                   "depth": depth})

    # ---- download ----------------------------------------------------
    def _fetch_blob(self, tid: str, size: int, pos: int, budget,
                    recorder) -> bytes:
        # Admission: fetched-but-unconsumed bytes stay counted until the
        # decode loop consumes them, so total buffered bytes are bounded
        # by inflight_budget. Only the head-of-line blob (the one the
        # decode loop is blocked on) may exceed the budget — anything
        # looser lets completed blobs accumulate without bound while the
        # head decodes.
        while True:
            with budget["lock"]:
                if (budget["bytes"] + size <= self.inflight_budget
                        or pos <= budget["head"]):
                    budget["bytes"] += size
                    break
            with budget["cv"]:
                budget["cv"].wait(0.05)
        try:
            self.s3.tensor_name = tid
            resp = self.s3.get_object(Bucket=self.bucket,
                                      Key=self._blob_key(tid))
            # Preallocate at the exact object size (no bytearray-growth or
            # final bytes() copy — a 2 GB blob would transiently double).
            n = int(resp.get("ContentLength", size))
            buf = bytearray(n)
            view = memoryview(buf)
            off = 0
            for chunk in resp["Body"].iter_chunks(CHUNK):
                view[off:off + len(chunk)] = chunk
                off += len(chunk)
            if off != n:
                raise ValueError(f"short read for {tid}: {off} != {n}")
            return buf
        finally:
            pass

    def download(self, plan: DownloadPlan, output_dir: str,
                 recorder: EventRecorder) -> DownloadResult:
        from tensordex import _ops

        res = DownloadResult(model_id=plan.model_id, system=self.system,
                             ok=False)
        blobs = plan.extra["blobs"]
        mapping = plan.extra["mapping"]
        recorder.emit("model_start", n_tensors=len(plan.tensors),
                      n_objects=len(plan.objects))
        writer = None
        try:
            # DFS pre-order over the delta forest: a base is decoded
            # immediately before its dependents, so decoded bytes stay
            # live only along the current root-to-leaf path.
            children: dict[str, list[str]] = {}
            roots = []
            for tid, b in sorted(blobs.items()):
                if b["base_id"]:
                    children.setdefault(b["base_id"], []).append(tid)
                else:
                    roots.append(tid)
            order: list[str] = []
            stack = list(reversed(roots))
            while stack:
                tid = stack.pop()
                order.append(tid)
                stack.extend(reversed(children.get(tid, ())))
            if len(order) != len(blobs):
                raise ValueError("delta forest traversal incomplete")
            refcount = {t: len(children.get(t, ())) for t in order}

            params = [{"name": p, "tid": tid,
                       "dtype": blobs[tid]["dtype"],
                       "shape": blobs[tid]["shape"],
                       "nbytes": blobs[tid]["logical"]}
                      for p, tid in sorted(mapping.items())]
            writer = _ShardWriter(output_dir, params, self.max_shard_bytes)

            budget = {"bytes": 0, "head": 0, "lock": threading.Lock(),
                      "cv": threading.Condition()}
            decoded = _BaseStore(os.path.join(output_dir, ".basespill"),
                                 ram_budget=self.base_ram_budget)
            t_decode_ns = 0

            with cf.ThreadPoolExecutor(self.max_workers) as pool:
                futs = {}
                for pos, tid in enumerate(order):
                    if self.base_cache is not None and \
                            tid in self.base_cache:
                        continue
                    futs[tid] = pool.submit(
                        self._fetch_blob, tid, blobs[tid]["stored"], pos,
                        budget, recorder)

                import time as _time
                for pos, tid in enumerate(order):
                    info = blobs[tid]
                    with budget["lock"]:
                        budget["head"] = pos
                    with budget["cv"]:
                        budget["cv"].notify_all()
                    if self.base_cache is not None and \
                            tid in self.base_cache:
                        out = self.base_cache[tid]
                        recorder.emit("s3_request_end",
                                      key=self._blob_key(tid),
                                      cache_hit=True, payload_bytes=0)
                    else:
                        raw = futs.pop(tid).result()
                        res.s3_get_count += 1
                        res.s3_payload_bytes += len(raw)
                        meta, _k, (_dt, _sh, payload) = \
                            parse_safetensors_bytes(raw)
                        base_id = meta.get("base_tensor_id") or \
                            info["base_id"]
                        if not base_id:
                            out = bytes(payload)
                        else:
                            base_bytes = decoded.get(base_id)
                            if base_bytes is None and self.base_cache:
                                base_bytes = self.base_cache.get(base_id)
                            if base_bytes is None:
                                raise ValueError(
                                    f"base {base_id} not decoded "
                                    f"before delta {tid}")
                            item_size = int(meta.get("item_size", 2))
                            recorder.emit("decode_start", tensor_name=tid,
                                          compressed_bytes=len(payload))
                            t0 = _time.perf_counter_ns()
                            out = bytes(_ops.decompress_tensorx_rust(
                                bytes(payload), base_bytes, item_size))
                            base_bytes = None
                            t_decode_ns += _time.perf_counter_ns() - t0
                            recorder.emit("decode_end", tensor_name=tid,
                                          logical_bytes=len(out))
                        del raw, payload
                        with budget["lock"]:
                            budget["bytes"] -= info["stored"]
                        with budget["cv"]:
                            budget["cv"].notify_all()

                    got = _ops.content_hash(out)
                    if got != tid:
                        raise ValueError(f"hash mismatch: expected {tid}, "
                                         f"got {got}")
                    written = writer.write(tid, out)
                    if written:
                        res.logical_model_bytes += written
                        recorder.emit(
                            "tensor_ready", tid=tid,
                            logical_bytes=info["logical"],
                            dependency_depth=plan.extra["depth"][tid],
                            copies=written // info["logical"])
                    if refcount[tid] > 0:
                        decoded.put(tid, out)
                        if self.base_cache is not None:
                            self.base_cache[tid] = out
                    base_id = info["base_id"]
                    if base_id in refcount:
                        refcount[base_id] -= 1
                        if refcount[base_id] == 0:
                            decoded.drop(base_id)
                    out = None  # noqa: F841 — release before next iteration

            decoded.clear()
            recorder.emit("model_assembly_end",
                          bytes_written=writer.bytes_written,
                          shards=len(writer.paths),
                          decode_cpu_ms=t_decode_ns / 1e6)
            res.output_paths.extend(writer.paths)
            recorder.emit("model_ready",
                          logical_model_bytes=res.logical_model_bytes)
            res.ok = True
        except Exception as e:  # noqa: BLE001
            res.error = repr(e)
            recorder.emit("error", where="tensordex.download",
                          exception=res.error)
        finally:
            if writer is not None:
                writer.close()
        return res
