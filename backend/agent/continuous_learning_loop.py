"""
Continuous Learning Loop -- weekly job.
"""
import os
import sys
import json
import argparse
from datetime import datetime, timedelta, timezone
import urllib.parse
from dotenv import load_dotenv
try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    sys.exit("Run: pip install 'psycopg[binary]' --break-system-packages")
def _with_query_param(url, key, value):
    parts = urllib.parse.urlsplit(url)
    params = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(params), parts.fragment))
load_dotenv()

CRDB_URL = os.environ.get("CRDB_URL")
if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)

SCHEMA_ADDITIONS = """
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    rating INT,               -- 1-5, or -1/+1 thumbs
    comment STRING,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""

OUT_PATH = "data/retraining_export.jsonl"


def pull_low_scoring_evaluations(conn, days):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT e.conversation_id, e.retrieval_correctness, e.response_accuracy,
                      e.grammar_language_ok, e.coherence_to_context, e.relevance_to_request,
                      e.judge_reasoning, c.summary
               FROM evaluation_results e
               JOIN conversations c ON c.id = e.conversation_id
               WHERE e.created_at > %s
                 AND e.is_simulated = false
                 AND (e.retrieval_correctness = false OR e.response_accuracy = false
                      OR e.coherence_to_context = false OR e.relevance_to_request = false)""",
            (cutoff,),
        )
        return cur.fetchall()


def pull_negative_feedback(conn, days):
    # TODO: Inlcude feedback also for this
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT f.conversation_id, f.rating, f.comment, c.summary
               FROM feedback f
               JOIN conversations c ON c.id = f.conversation_id
               WHERE f.created_at > %s AND f.rating <= 2""",
            (cutoff,),
        )
        return cur.fetchall()


def pull_case_state_for(conn, conversation_id):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT state FROM case_state WHERE conversation_id = %s
               ORDER BY turn_number DESC LIMIT 1""",
            (conversation_id,),
        )
        row = cur.fetchone()
        return row["state"] if row else None


def build_export(conn, days):
    """ 
    TODO:
    """
    rows = []
    for e in pull_low_scoring_evaluations(conn, days):
        rows.append({
            "conversation_id": e["conversation_id"], "source": "low_llm_judge_score",
            "failed_dimensions": [k for k in
                ("retrieval_correctness", "response_accuracy", "grammar_language_ok",
                 "coherence_to_context", "relevance_to_request") if e[k] is False],
            "judge_reasoning": e["judge_reasoning"], "summary": e["summary"],
            "case_state": pull_case_state_for(conn, e["conversation_id"]),
        })
    for f in pull_negative_feedback(conn, days):
        rows.append({
            "conversation_id": f["conversation_id"], "source": "negative_feedback",
            "rating": f["rating"], "comment": f["comment"], "summary": f["summary"],
            "case_state": pull_case_state_for(conn, f["conversation_id"]),
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    if not CRDB_URL:
        sys.exit("Set CRDB_URL env var first.")

    with psycopg.connect(CRDB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_ADDITIONS)
        conn.commit()

        rows = build_export(conn, args.days)

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"Exported {len(rows)} low-signal cases -> {OUT_PATH}")
    print("Next: feed this into a Bedrock fine-tuning job, or use as a "
          "prompt/KB-gap review queue (manual first, automate once volume justifies it).")
