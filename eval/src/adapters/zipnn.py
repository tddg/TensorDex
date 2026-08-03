"""ZipNN baseline (plan §2.2): download compressed artifact(s) from S3,
decompress locally, write the logical model.

Decompression uses the `zipnn` package. The compressed layout (single
artifact vs per-shard) comes from eval/config/models.yaml once the S3
inventory identifies it.
"""
from __future__ import annotations

import os

from ..recorder import EventRecorder
from .base import DownloadAdapter, DownloadPlan, DownloadResult

CHUNK = 8 * 1024 * 1024


class ZipNNAdapter(DownloadAdapter):
    system = "zipnn"

    def __init__(self, s3_client, bucket: str, model_prefix: str,
                 layout_index: dict[str, list[dict]]):
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
        from zipnn import ZipNN

        os.makedirs(output_dir, exist_ok=True)
        res = DownloadResult(model_id=plan.model_id, system=self.system,
                             ok=False)
        recorder.emit("model_start", n_objects=len(plan.objects))
        try:
            for obj in plan.objects:
                self.s3.tensor_name = None
                resp = self.s3.get_object(Bucket=self.bucket, Key=obj["key"])
                buf = bytearray()
                for chunk in resp["Body"].iter_chunks(CHUNK):
                    buf += chunk
                res.s3_get_count += 1
                res.s3_payload_bytes += len(buf)

                recorder.emit("decode_start", key=obj["key"],
                              compressed_bytes=len(buf))
                # zipnn's file naming: foo.safetensors.znn -> foo.safetensors
                zc = ZipNN(is_streaming=True)
                decompressed = zc.decompress(bytes(buf))
                recorder.emit("decode_end", key=obj["key"],
                              logical_bytes=len(decompressed))

                base = os.path.basename(obj["key"])
                if base.endswith(".znn"):
                    base = base[: -len(".znn")]
                out_path = os.path.join(output_dir, base)
                recorder.emit("model_assembly_start", key=obj["key"])
                with open(out_path, "wb") as fh:
                    fh.write(decompressed)
                    fh.flush()
                    os.fsync(fh.fileno())
                recorder.emit("model_assembly_end", key=obj["key"])
                res.output_paths.append(out_path)
                res.logical_model_bytes += len(decompressed)
            recorder.emit("model_ready",
                          logical_model_bytes=res.logical_model_bytes)
            res.ok = True
        except Exception as e:  # noqa: BLE001
            res.error = repr(e)
            recorder.emit("error", where="zipnn.download",
                          exception=res.error)
        return res
