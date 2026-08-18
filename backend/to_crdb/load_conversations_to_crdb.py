"""
Load data/conversations.jsonl into the EXISTING conversations table
"""
import os
import sys
import json

from dotenv import load_dotenv

try:
    import psycopg
except ImportError:
    sys.exit("Run: pip install 'psycopg[binary]' --break-system-packages")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run: pip install sentence-transformers --break-system-packages")

load_dotenv()
import urllib.parse  # noqa: E402
def _with_query_param(url, key, value):
    parts = urllib.parse.urlsplit(url)
    params = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(params), parts.fragment))

CRDB_URL = os.environ.get("CRDB_URL")
if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)


def build_summary(turns):
    return " ".join(t["text"] for t in turns if t["role"] == "customer")

def to_vector_literal(vec):
    """CockroachDB's VECTOR type expects '[v1,v2,...]' syntax.
    psycopg's default list adapter produces Postgres array syntax
    '{v1,v2,...}' instead, which VECTOR rejects."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"

if __name__ == "__main__":
    if not CRDB_URL:
        sys.exit("Set CRDB_URL env var first.")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("2_conversation_data/conversations.jsonl") as f:
        records = [json.loads(line) for line in f]

    summaries = [build_summary(r["turns"]) for r in records]
    embeddings = model.encode(summaries, normalize_embeddings=True)

    with psycopg.connect(CRDB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ticket_id FROM support_tickets")
            valid_ticket_ids = {row[0] for row in cur.fetchall()}

            missing = 0
            for r, summary, emb in zip(records, summaries, embeddings):
                ticket_id = r["ticket_id"]
                if ticket_id not in valid_ticket_ids:
                    missing += 1
                    ticket_id = None

                cur.execute(
                    """INSERT INTO conversations
                       (id, customer_id, ticket_id, status, summary,
                        summary_embedding, escalated, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO NOTHING""",
                    (r["conversation_id"], r["customer_id"], ticket_id,
                     "escalated" if not r["resolved"] else "resolved",
                     summary, to_vector_literal(emb.tolist()), not r["resolved"], r["created_at"]),
                )
        conn.commit()

    if missing:
        print(f"Warning: {missing} conversation(s) referenced a ticket_id not in support_tickets; set to NULL.")
    print(f"Loaded {len(records)} conversations -> conversations table")
