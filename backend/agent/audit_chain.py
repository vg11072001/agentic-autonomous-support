"""
tamper-evident, hash-chained audit log.
"""
from __future__ import annotations
import json
import hashlib

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    seq         INT PRIMARY KEY,
    prev_hash   STRING NOT NULL,
    record_hash STRING NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);
"""


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(prev_hash: str, payload: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(payload)).encode()).hexdigest()


def append_audit(conn, payload: dict) -> dict:
    """Append one hash-chained record. Uses a transaction + the current max seq;
    audit appends are low-rate. For high concurrency, take an advisory lock on a
    head row or use a dedicated sequence table (noted)."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute("SELECT seq, record_hash FROM audit_log ORDER BY seq DESC LIMIT 1")
        row = cur.fetchone()
        seq = (row[0] + 1) if row else 1
        prev = row[1] if row else GENESIS
        rec = compute_hash(prev, payload)
        cur.execute("INSERT INTO audit_log (seq, prev_hash, record_hash, payload) "
                    "VALUES (%s,%s,%s,%s)", (seq, prev, rec, json.dumps(payload, default=str)))
    conn.commit()
    return {"seq": seq, "prev_hash": prev, "record_hash": rec}


def verify_chain(conn) -> dict:
    from psycopg.rows import dict_row
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT seq, prev_hash, record_hash, payload FROM audit_log ORDER BY seq ASC")
        rows = cur.fetchall()
    return _verify_rows(rows)


def _verify_rows(rows) -> dict:
    prev = GENESIS
    for r in rows:
        payload = r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])
        if r["prev_hash"] != prev:
            return {"ok": False, "broken_at": r["seq"], "why": "prev_hash mismatch", "count": len(rows)}
        if compute_hash(prev, payload) != r["record_hash"]:
            return {"ok": False, "broken_at": r["seq"], "why": "record_hash mismatch (record altered)",
                    "count": len(rows)}
        prev = r["record_hash"]
    return {"ok": True, "count": len(rows), "head": prev if rows else GENESIS}


def latest_head(conn) -> str:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute("SELECT record_hash FROM audit_log ORDER BY seq DESC LIMIT 1")
        r = cur.fetchone()
    return r[0] if r else GENESIS


if __name__ == "__main__":
    # in-memory demo of chaining + tamper detection (no DB)
    rows, prev = [], GENESIS
    for i, ev in enumerate([
        {"turn": 1, "action": "refund", "ceiling": "propose_only", "escalated": False},
        {"turn": 2, "action": "inquiry", "ceiling": "act", "escalated": False},
        {"turn": 3, "action": "refund", "ceiling": "escalate", "escalated": True},
    ], 1):
        h = compute_hash(prev, ev)
        rows.append({"seq": i, "prev_hash": prev, "record_hash": h, "payload": ev})
        prev = h
    print("clean chain:", _verify_rows(rows))
    # tamper: someone edits turn 2's ceiling from act -> (hide it)
    rows[1]["payload"]["ceiling"] = "escalate"
    print("after tampering record 2:", _verify_rows(rows))

