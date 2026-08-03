"""Append-only JSONL event recorder shared by all storage adapters.

One recorder instance per benchmark run. Events are written as they happen
(line-buffered) so a crashed run still leaves a usable trace. Convert to
Parquet after the run with eval/scripts/summarize.py.

All timestamps are time.perf_counter_ns() (monotonic). Each recorder also
stores one wall-clock anchor (time_ns) at construction so traces can be
aligned across client instances.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

EVENT_TYPES = {
    "model_start",
    "metadata_query_start",
    "metadata_query_end",
    "tensor_resolved",
    "s3_request_start",
    "s3_headers",
    "s3_first_byte",
    "s3_request_end",
    "decode_start",
    "decode_end",
    "tensor_ready",
    "model_assembly_start",
    "model_assembly_end",
    "model_ready",
    "resource_sample",
    "error",
}


class EventRecorder:
    def __init__(self, out_path: str, run_id: str | None = None,
                 system: str = "", model_id: str = ""):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.system = system
        self.model_id = model_id
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self._fh = open(out_path, "a", buffering=1)
        self._lock = threading.Lock()
        # Wall-clock anchor for cross-client alignment.
        self.emit("recorder_anchor",
                  wall_time_ns=time.time_ns(),
                  pid=os.getpid())

    def now(self) -> int:
        return time.perf_counter_ns()

    def emit(self, event: str, **fields) -> int:
        """Record an event; returns the monotonic timestamp used."""
        ts = time.perf_counter_ns()
        rec = {
            "ts_ns": ts,
            "event": event,
            "run_id": self.run_id,
            "system": self.system,
            "model_id": self.model_id,
        }
        rec.update(fields)
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
        return ts

    def close(self):
        with self._lock:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
