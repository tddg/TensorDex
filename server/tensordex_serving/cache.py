"""NVMe hot-cache core (plan Phase 2.1): layout, atomic writes, pins.

Layout: ``{root}/{tid[:2]}/{tid}`` = decoded tensor bytes. Writes are
tmp+rename and hash-verified *before* the rename, so a crash can only
leave ``.tmp`` litter, never a wrong entry. Content addressing means no
invalidation and cross-model dedup for free.

Pins live at ``{root}/.pins/{tid}`` so the evictor (a separate process)
can see in-flight materializations. A pin older than PIN_STALE_SECS is
treated as leaked (materializer crash) and ignored.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time

PIN_STALE_SECS = 3600


class TensorCache:
    def __init__(self, root: str):
        self.root = root
        self.pin_dir = os.path.join(root, ".pins")
        os.makedirs(self.pin_dir, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, tid: str) -> str:
        return os.path.join(self.root, tid[:2], tid)

    def rel_path(self, tid: str) -> str:
        """Path relative to root (what X-Accel-Redirect appends)."""
        return f"{tid[:2]}/{tid}"

    def has(self, tid: str) -> bool:
        return os.path.exists(self.path(tid))

    def get_bytes(self, tid: str):
        """Whole-entry read (base_lookup for the pipeline); None on miss."""
        try:
            with open(self.path(tid), "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return None

    def put(self, tid: str, data, verify: bool = True) -> str:
        """Atomic hash-verified insert; returns the entry path."""
        if verify:
            from tensordex import _ops
            got = _ops.content_hash(bytes(data))
            if got != tid:
                raise ValueError(f"refusing cache write: content hash "
                                 f"{got} != tid {tid}")
        path = self.path(tid)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        return path

    # ---- pins (cross-process, crash-tolerant) ------------------------
    def _pin_path(self, tid: str) -> str:
        return os.path.join(self.pin_dir, tid)

    @contextlib.contextmanager
    def pinned(self, tids):
        """Pin a set of tids for the duration of a materialization."""
        paths = []
        for tid in tids:
            p = self._pin_path(tid)
            with open(p, "a"):
                os.utime(p)
            paths.append(p)
        try:
            yield
        finally:
            for p in paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def pinned_tids(self) -> set:
        """Live pins (stale ones from crashed writers excluded)."""
        now = time.time()
        out = set()
        try:
            with os.scandir(self.pin_dir) as it:
                for e in it:
                    try:
                        if now - e.stat().st_mtime < PIN_STALE_SECS:
                            out.add(e.name)
                    except FileNotFoundError:
                        pass
        except FileNotFoundError:
            pass
        return out

    # ---- enumeration / usage -----------------------------------------
    def entries(self):
        """Yield (tid, size_bytes, atime) for every cached entry."""
        try:
            shards = [e for e in os.scandir(self.root)
                      if e.is_dir() and len(e.name) == 2]
        except FileNotFoundError:
            return
        for shard in shards:
            with os.scandir(shard.path) as it:
                for e in it:
                    if e.name.endswith(".tmp") or ".tmp." in e.name:
                        continue
                    try:
                        st = e.stat()
                    except FileNotFoundError:
                        continue
                    yield e.name, st.st_size, st.st_atime

    def tids(self) -> set:
        return {tid for tid, _, _ in self.entries()}

    def used_bytes(self) -> int:
        return sum(size for _, size, _ in self.entries())

    def fs_usage(self):
        """(used_bytes, total_bytes) of the underlying filesystem."""
        st = os.statvfs(self.root)
        total = st.f_blocks * st.f_frsize
        return total - st.f_bavail * st.f_frsize, total
