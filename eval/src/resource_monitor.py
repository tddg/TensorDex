"""Background resource sampler (plan §9.3): psutil-based, 100-500 ms cadence.

Emits resource_sample events into the shared EventRecorder so resource
timelines align with per-tensor and per-request events on the same
monotonic clock.
"""
from __future__ import annotations

import threading

import psutil

from .recorder import EventRecorder


class ResourceMonitor:
    def __init__(self, recorder: EventRecorder, interval_s: float = 0.25):
        self._rec = recorder
        self._interval = interval_s
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        # Prime cpu_percent so the first sample is meaningful.
        self._proc.cpu_percent(None)
        psutil.cpu_percent(None)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self._interval):
            try:
                with self._proc.oneshot():
                    mem = self._proc.memory_info()
                    io = self._proc.io_counters() if hasattr(
                        self._proc, "io_counters") else None
                    net = psutil.net_io_counters()
                    self._rec.emit(
                        "resource_sample",
                        proc_cpu_pct=self._proc.cpu_percent(None),
                        sys_cpu_pct=psutil.cpu_percent(None),
                        rss_bytes=mem.rss,
                        disk_read_bytes=io.read_bytes if io else None,
                        disk_write_bytes=io.write_bytes if io else None,
                        net_recv_bytes=net.bytes_recv,
                        net_sent_bytes=net.bytes_sent,
                        threads=self._proc.num_threads(),
                        open_fds=self._proc.num_fds(),
                    )
            except psutil.Error:
                pass

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
