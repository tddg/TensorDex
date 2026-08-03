"""Per-request S3 instrumentation.

Wraps a boto3 client so every GET/HEAD produces s3_request_start /
s3_headers / s3_first_byte / s3_request_end events with byte counts,
HTTP status, the AWS request id, and attempt number. Timing get_object()
alone only covers request-to-headers, so the StreamingBody is wrapped to
observe first payload byte and body-complete separately (plan §9.2).
"""
from __future__ import annotations

import itertools
import threading

from botocore.config import Config

from .recorder import EventRecorder

_req_counter = itertools.count()
_counter_lock = threading.Lock()


def make_config(max_pool_connections: int = 256, max_attempts: int = 5,
                region: str = "us-east-1") -> Config:
    return Config(
        region_name=region,
        max_pool_connections=max_pool_connections,
        retries={"mode": "standard", "max_attempts": max_attempts},
    )


class TracedBody:
    """Wraps a botocore StreamingBody; emits first-byte and completion."""

    def __init__(self, body, recorder: EventRecorder, request_id: int,
                 context: dict):
        self._body = body
        self._rec = recorder
        self._rid = request_id
        self._ctx = context
        self._bytes = 0
        self._first_byte_seen = False

    def read(self, amt=None):
        chunk = self._body.read(amt)
        if chunk and not self._first_byte_seen:
            self._first_byte_seen = True
            self._rec.emit("s3_first_byte", request_id=self._rid, **self._ctx)
        self._bytes += len(chunk) if chunk else 0
        if chunk == b"" or (amt is None and chunk is not None):
            self._rec.emit("s3_request_end", request_id=self._rid,
                           payload_bytes=self._bytes, **self._ctx)
        return chunk

    def iter_chunks(self, chunk_size=1024 * 1024):
        for chunk in self._body.iter_chunks(chunk_size):
            if chunk and not self._first_byte_seen:
                self._first_byte_seen = True
                self._rec.emit("s3_first_byte", request_id=self._rid,
                               **self._ctx)
            self._bytes += len(chunk)
            yield chunk
        self._rec.emit("s3_request_end", request_id=self._rid,
                       payload_bytes=self._bytes, **self._ctx)

    def close(self):
        self._body.close()


class TracedS3Client:
    """Thin instrumented facade over a boto3 s3 client.

    Only the operations the benchmark needs (get_object, head_object,
    list_objects_v2) are wrapped; anything else falls through untimed.
    """

    def __init__(self, client, recorder: EventRecorder,
                 tensor_name: str | None = None):
        self._c = client
        self._rec = recorder
        self.tensor_name = tensor_name  # mutable; set per fetch by adapters

    def _ctx(self, bucket, key, operation, range_=None):
        return {
            "bucket": bucket,
            "key": key,
            "operation": operation,
            "range": range_,
            "tensor_name": self.tensor_name,
        }

    def get_object(self, Bucket, Key, Range=None, **kw):
        with _counter_lock:
            rid = next(_req_counter)
        ctx = self._ctx(Bucket, Key, "get_object", Range)
        self._rec.emit("s3_request_start", request_id=rid, **ctx)
        params = dict(Bucket=Bucket, Key=Key, **kw)
        if Range is not None:
            params["Range"] = Range
        try:
            resp = self._c.get_object(**params)
        except Exception as e:  # noqa: BLE001 - recorded then re-raised
            self._rec.emit("error", request_id=rid,
                           exception=repr(e), **ctx)
            raise
        meta = resp.get("ResponseMetadata", {})
        self._rec.emit("s3_headers", request_id=rid,
                       http_status=meta.get("HTTPStatusCode"),
                       aws_request_id=meta.get("RequestId"),
                       retry_attempts=meta.get("RetryAttempts", 0),
                       content_length=resp.get("ContentLength"),
                       **ctx)
        resp["Body"] = TracedBody(resp["Body"], self._rec, rid, ctx)
        return resp

    def head_object(self, Bucket, Key, **kw):
        with _counter_lock:
            rid = next(_req_counter)
        ctx = self._ctx(Bucket, Key, "head_object")
        self._rec.emit("s3_request_start", request_id=rid, **ctx)
        try:
            resp = self._c.head_object(Bucket=Bucket, Key=Key, **kw)
        except Exception as e:  # noqa: BLE001
            self._rec.emit("error", request_id=rid, exception=repr(e), **ctx)
            raise
        meta = resp.get("ResponseMetadata", {})
        self._rec.emit("s3_request_end", request_id=rid,
                       http_status=meta.get("HTTPStatusCode"),
                       aws_request_id=meta.get("RequestId"),
                       retry_attempts=meta.get("RetryAttempts", 0),
                       payload_bytes=0, **ctx)
        return resp

    def __getattr__(self, name):
        return getattr(self._c, name)
