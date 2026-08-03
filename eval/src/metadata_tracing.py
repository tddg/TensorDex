"""SQLite metadata-query instrumentation.

Wraps a sqlite3 connection so every query emits metadata_query_start /
metadata_query_end events with a query-class label, row count, and
serialized result size estimate (plan §5.4). Query classes are matched by
substring against the normalized SQL; unmatched queries get class "other"
and should be triaged before the performance runs.
"""
from __future__ import annotations

import sqlite3

from .recorder import EventRecorder

# Query classes from plan §5.4, matched to the real hub schema
# (src/rust/metadata/schema.rs: tensors, model_mappings, model_meta,
# tensor_deltas). First matching pattern wins.
QUERY_CLASSES = [
    ("model_to_tensors", ("model_mappings",)),
    ("delta_to_base", ("tensor_deltas",)),
    ("model_state", ("model_meta",)),
    ("tensor_meta", ("from tensors",)),
]


def classify(sql: str) -> str:
    s = sql.lower()
    for name, needles in QUERY_CLASSES:
        if all(n in s for n in needles):
            return name
    return "other"


class TracedCursor:
    def __init__(self, cursor, recorder: EventRecorder):
        self._cur = cursor
        self._rec = recorder

    def execute(self, sql, params=()):
        self._rec.emit("metadata_query_start", sql_class=classify(sql),
                       sql=sql.strip()[:300])
        self._cur.execute(sql, params)
        return self

    def fetchall(self):
        rows = self._cur.fetchall()
        est = sum(len(repr(r)) for r in rows[:100])
        if len(rows) > 100:
            est = est * len(rows) // 100
        self._rec.emit("metadata_query_end", rows=len(rows),
                       approx_bytes=est)
        return rows

    def fetchone(self):
        row = self._cur.fetchone()
        self._rec.emit("metadata_query_end", rows=0 if row is None else 1,
                       approx_bytes=len(repr(row)) if row else 0)
        return row

    def __getattr__(self, name):
        return getattr(self._cur, name)

    def __iter__(self):
        return iter(self._cur)


class TracedConnection:
    def __init__(self, path: str, recorder: EventRecorder):
        self._conn = sqlite3.connect(path)
        self._rec = recorder

    def cursor(self):
        return TracedCursor(self._conn.cursor(), self._rec)

    def execute(self, sql, params=()):
        return self.cursor().execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._conn, name)
