#!/usr/bin/env python3
"""Build compressed_full/master.db from the live catalog + executed plan.

- tensors/model_mappings/model_meta copied from the live master.db
  (registered-but-blobless tensors kept, flagged via storage_uri='missing://'
  so consumers can distinguish "raw in tensordb" from "gone").
- tensor_deltas: one row per successfully executed pair (codec=tensorx).
- For compressed tensors: size_bytes = physical delta blob size,
  storage_uri -> compressed_full/blobs/...; logical size preserved in a new
  column logical_bytes (for read-amp / manifest math).
- Raw tensors (bases + unattached + skipped/failed pairs) keep their
  tensordb/ URI and sizes.

Output: /mnt/nvme0/campaign/master_compressed.db (upload separately).
"""
import glob
import json
import pickle
import sqlite3
import time

OUT = "/mnt/nvme0/campaign/master_compressed.db"
SRC = "/mnt/nvme0/master.db"
LOGS = "/mnt/nvme0/campaign/logs"
PLAN = "/mnt/nvme0/campaign/plan"
DST_URI = "s3://tensor-tingfeng/compressed_full/blobs/{p}/{tid}.safetensors"


def main():
    t0 = time.time()
    existing = pickle.load(open(f"{PLAN}/existing_blobs.pkl", "rb"))

    ok = {}      # tid -> (base_tid, blob_bytes)
    skipped = 0
    for path in sorted(glob.glob(f"{LOGS}/exec_done_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r.get("status") == "ok":
                ok[r["tid"]] = (r["base"], int(r["blob"]))
            else:
                skipped += 1
    print(f"executed ok={len(ok)} skipped={skipped} ({time.time()-t0:.0f}s)")

    import shutil
    shutil.copyfile(SRC, OUT)
    con = sqlite3.connect(OUT)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS tensor_deltas (
        tensor_id TEXT PRIMARY KEY,
        base_tensor_id TEXT NOT NULL,
        codec TEXT NOT NULL)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tensor_deltas_base "
                "ON tensor_deltas(base_tensor_id)")
    try:
        con.execute("ALTER TABLE tensors ADD COLUMN logical_bytes INTEGER")
    except sqlite3.OperationalError:
        pass
    con.execute("UPDATE tensors SET logical_bytes = size_bytes")

    con.executemany(
        "INSERT OR REPLACE INTO tensor_deltas VALUES (?, ?, 'tensorx')",
        [(t, b) for t, (b, _) in ok.items()])
    con.executemany(
        "UPDATE tensors SET size_bytes = ?, storage_uri = ? WHERE id = ?",
        [(blob, DST_URI.format(p=t[:2], tid=t), t)
         for t, (_b, blob) in ok.items()])
    # flag registered-but-blobless tensors
    missing = [(f"missing://{t}", t) for (t,) in
               con.execute("SELECT id FROM tensors")
               if t not in existing]
    con.executemany("UPDATE tensors SET storage_uri = ? WHERE id = ?",
                    missing)
    con.commit()

    n_d = con.execute("SELECT COUNT(*) FROM tensor_deltas").fetchone()[0]
    stored = con.execute(
        "SELECT SUM(size_bytes) FROM tensors "
        "WHERE storage_uri NOT LIKE 'missing://%'").fetchone()[0]
    logical = con.execute(
        "SELECT SUM(logical_bytes) FROM tensors "
        "WHERE storage_uri NOT LIKE 'missing://%'").fetchone()[0]
    print(f"deltas={n_d} missing={len(missing)}")
    print(f"stored {stored/1e12:.2f} TB vs logical-unique {logical/1e12:.2f} "
          f"TB -> reduction {1-stored/logical:.1%}")
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
