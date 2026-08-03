"""Unified adapter interface (plan §9) so timer boundaries are identical
across TensorDex, ZipNN, and raw safetensors.

Lifecycle per measured request:
    adapter.resolve(model_id)      -> DownloadPlan   (metadata phase)
    adapter.download(plan, outdir, recorder) -> DownloadResult

resolve() must perform ALL metadata work needed to know what to fetch;
download() must not silently re-resolve. The recorder captures phase
events; DownloadResult carries the summary counters that go into
model_runs.jsonl.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from ..recorder import EventRecorder


@dataclasses.dataclass
class DownloadPlan:
    model_id: str
    system: str
    # Logical tensors in completion-independent declaration order.
    tensors: list[dict] = dataclasses.field(default_factory=list)
    # S3 objects to fetch: [{key, size, role: blob|delta|base|artifact}]
    objects: list[dict] = dataclasses.field(default_factory=list)
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DownloadResult:
    model_id: str
    system: str
    ok: bool
    logical_model_bytes: int = 0
    s3_payload_bytes: int = 0
    s3_get_count: int = 0
    s3_head_count: int = 0
    s3_retry_count: int = 0
    output_paths: list[str] = dataclasses.field(default_factory=list)
    verify_digest: str = ""
    error: str = ""


class DownloadAdapter:
    system: str = "abstract"

    def resolve(self, model_id: str) -> DownloadPlan:
        raise NotImplementedError

    def download(self, plan: DownloadPlan, output_dir: str,
                 recorder: EventRecorder) -> DownloadResult:
        raise NotImplementedError
