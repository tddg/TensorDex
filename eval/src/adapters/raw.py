"""Raw uncompressed-safetensors baseline (plan §2.3).

Downloads the original model file or shards straight from S3 and writes
them locally. No decompression phase; sharded models issue one GET per
shard (plus one for the HF-style index when present).
"""
from __future__ import annotations

import json
import os

from ..recorder import EventRecorder
from .base import DownloadAdapter, DownloadPlan, DownloadResult

CHUNK = 8 * 1024 * 1024


class RawAdapter(DownloadAdapter):
    system = "raw"

    def __init__(self, s3_client, bucket: str, model_prefix: str,
                 layout_index: dict[str, list[dict]]):
        """layout_index: model_id -> [{key, size}] from eval/config/models.yaml."""
        self.s3 = s3_client
        self.bucket = bucket
        self.model_prefix = model_prefix
        self.layout_index = layout_index

    def resolve(self, model_id: str) -> DownloadPlan:
        objs = self.layout_index[model_id]
        return DownloadPlan(
            model_id=model_id, system=self.system,
            objects=[{"key": o["key"], "size": o.get("size"),
                      "role": "artifact"} for o in objs])

    def download(self, plan: DownloadPlan, output_dir: str,
                 recorder: EventRecorder) -> DownloadResult:
        os.makedirs(output_dir, exist_ok=True)
        res = DownloadResult(model_id=plan.model_id, system=self.system,
                             ok=False)
        recorder.emit("model_start", n_objects=len(plan.objects))
        try:
            for obj in plan.objects:
                self.s3.tensor_name = None
                resp = self.s3.get_object(Bucket=self.bucket, Key=obj["key"])
                out_path = os.path.join(output_dir,
                                        os.path.basename(obj["key"]))
                recorder.emit("model_assembly_start", key=obj["key"])
                with open(out_path, "wb") as fh:
                    for chunk in resp["Body"].iter_chunks(CHUNK):
                        fh.write(chunk)
                    fh.flush()
                    os.fsync(fh.fileno())
                recorder.emit("model_assembly_end", key=obj["key"])
                res.output_paths.append(out_path)
                res.s3_get_count += 1
                res.logical_model_bytes += os.path.getsize(out_path)
            res.s3_payload_bytes = res.logical_model_bytes
            recorder.emit("model_ready",
                          logical_model_bytes=res.logical_model_bytes)
            res.ok = True
        except Exception as e:  # noqa: BLE001
            res.error = repr(e)
            recorder.emit("error", where="raw.download", exception=res.error)
        return res
