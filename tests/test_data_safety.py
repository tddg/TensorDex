"""Regression tests for the four data-safety fixes.

1. Remote pull mirrors delta edges — gc on a pulled hub must not delete
   bases (previously: silent, permanent data loss).
2. ``compress_pair`` records the protective edge before the blob rewrite
   and retracts it if the rewrite fails.
3. ``verify=True`` is enforced on non-local backends (previously a
   silent no-op).
4. The catalog sets a SQLite busy timeout so a second writer process
   queues instead of failing immediately with SQLITE_BUSY.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import requests
import torch
import uvicorn
from safetensors.torch import save_file

from tensordex.client.remote import pull_remote
from tensordex.core import engine as engine_mod
from tensordex.core.codec import probe_codec
from tensordex.core.engine import IntegrityError, TensorDex
from tensordex.server.app import build_app


def _ingest_one(hub: TensorDex, model: str, name: str, tensor: torch.Tensor, tmp: Path) -> str:
    shard = tmp / f"{model.replace('/', '_')}.safetensors"
    save_file({name: tensor}, str(shard))
    hub.init_model(model)
    mapping = hub.ingest([str(shard)], model)
    hub.commit_model(model)
    return mapping[name]


def _make_hub_with_delta(tmp_path: Path):
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    base = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    target = base + 0.01
    base_id = _ingest_one(hub, "org/base", "w", base, tmp_path)
    target_id = _ingest_one(hub, "org/target", "w", target, tmp_path)
    res = hub.compress_pair(target_id, base_id, codec="tensorx")
    assert res["status"] == "ok"
    return hub, base_id, target_id, target


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _live_server(hub: TensorDex) -> Iterator[str]:
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            build_app(hub), host="127.0.0.1", port=port,
            log_level="warning", lifespan="off",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/healthz", timeout=0.2).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Test server did not start")
    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# 1. Remote pull mirrors delta edges; gc must not delete pulled bases
# ---------------------------------------------------------------------------

def test_pulled_hub_gc_preserves_delta_bases(tmp_path: Path) -> None:
    source, base_id, target_id, target = _make_hub_with_delta(tmp_path)
    dest = TensorDex(str(tmp_path / "dest_hub"), hydrate_all=False)

    # Pull ONLY the compressed model: its manifest closure carries the base
    # blob, but the base is not a parameter of org/target, so it gets no
    # mapping row in the destination hub.
    with _live_server(source) as endpoint:
        pull_remote(
            "org/target", endpoint=endpoint, local_hub=dest,
            output_dir=str(tmp_path / "out"), filename="t.safetensors",
            workers=2,
        )

    # The delta edge must have been mirrored...
    assert base_id in dest.metadata.protected_base_ids()

    # ...so gc must NOT reclaim the base blob.
    dest.gc()
    assert dest.storage_backend.blob_path_for_id(base_id).exists()
    reconstructed = dest.get_tensor(tensor_id=target_id, verify=True)
    assert torch.equal(reconstructed, target)


# ---------------------------------------------------------------------------
# 2. compress_pair: edge precedes rewrite; failed rewrite retracts it
# ---------------------------------------------------------------------------

def test_failed_compress_retracts_protective_edge(tmp_path: Path, monkeypatch) -> None:
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    base = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    target = base + 0.01
    base_id = _ingest_one(hub, "org/base", "w", base, tmp_path)
    target_id = _ingest_one(hub, "org/target", "w", target, tmp_path)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(engine_mod, "save_compressed", _boom)
    with pytest.raises(OSError, match="disk full"):
        hub.compress_pair(target_id, base_id, codec="tensorx")

    # Blob untouched (still raw), and the stale protective edge retracted.
    assert probe_codec(hub.storage_backend.blob_path_for_id(target_id)) is None
    assert base_id not in hub.metadata.protected_base_ids()
    # Target still reads back as its raw self.
    assert torch.equal(hub.get_tensor(tensor_id=target_id, verify=True), target)


# ---------------------------------------------------------------------------
# 3. verify=True is enforced on non-local backends
# ---------------------------------------------------------------------------

class _CorruptingBackend:
    """Stands in for a remote backend that returns wrong bytes."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def load_tensor(self, uri: str) -> torch.Tensor:
        return self._tensor


def test_verify_enforced_on_non_local_backend(tmp_path: Path) -> None:
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    good = torch.arange(64, dtype=torch.float32)
    tid = _ingest_one(hub, "org/m", "w", good, tmp_path)

    corrupt = good.clone()
    corrupt[0] = -1.0
    hub.storage_backend = _CorruptingBackend(corrupt)

    # Unverified read returns whatever the backend served.
    assert torch.equal(hub.get_tensor(tensor_id=tid), corrupt)
    # Verified read must now actually check — and fail loudly.
    with pytest.raises(IntegrityError):
        hub.get_tensor(tensor_id=tid, verify=True)


# ---------------------------------------------------------------------------
# 4. busy_timeout: a second writer waits for a lock instead of erroring
# ---------------------------------------------------------------------------

def test_second_writer_waits_for_locked_catalog(tmp_path: Path) -> None:
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    a = _ingest_one(hub, "org/a", "w", torch.zeros(8), tmp_path)
    b = _ingest_one(hub, "org/b", "w", torch.ones(8), tmp_path)
    db = str(Path(hub.storage_dir) / "metadata.db")

    hold = 1.2  # seconds the other process holds the write lock
    locker = subprocess.Popen(
        [
            sys.executable, "-c",
            "import sqlite3, sys, time\n"
            "conn = sqlite3.connect(sys.argv[1])\n"
            "conn.execute('BEGIN IMMEDIATE')\n"
            "print('locked', flush=True)\n"
            f"time.sleep({hold})\n"
            "conn.commit()\n",
            db,
        ],
        stdout=subprocess.PIPE, text=True,
    )
    assert locker.stdout is not None
    assert locker.stdout.readline().strip() == "locked"

    # Without a busy timeout this raised SQLITE_BUSY immediately; with it,
    # the write blocks until the other process commits, then succeeds.
    t0 = time.monotonic()
    hub.metadata.set_tensor_delta(b, a, "tensorx")
    elapsed = time.monotonic() - t0
    locker.wait(timeout=10)

    assert elapsed > 0.4, f"writer did not wait for the lock ({elapsed:.3f}s)"
    assert a in hub.metadata.protected_base_ids()
