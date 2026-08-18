"""
Retrieval Agent.
"""
import os
import sys
import json

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    sys.exit("Run: pip install 'psycopg[binary]' --break-system-packages")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run: pip install sentence-transformers --break-system-packages")

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
def _with_query_param(url, key, value):
    parts = urlsplit(url)
    params = parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))

CRDB_URL = os.environ.get("CRDB_URL")
if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)
def to_vector_literal(vec):
    """CockroachDB's VECTOR type expects '[v1,v2,...]' syntax.
    psycopg's default list adapter produces Postgres array syntax
    '{v1,v2,...}' instead, which VECTOR rejects."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class RetrievalAgent:
    def __init__(self, conn):
        self.conn = conn
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    # -- KB search: always available, every segment uses this -----------
    def search_kb(self, query, k=3):
        emb = to_vector_literal(self.embed(query))
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id, doc_id, title, content,
                          embedding <-> %s AS distance
                   FROM kb_chunks
                   ORDER BY embedding <-> %s
                   LIMIT %s""",
                (emb, emb, k),
            )
            return cur.fetchall()

    # -- individual history: vector search partitioned by customer_id ---
    # (this is what the (customer_id, summary_embedding) VECTOR INDEX
    # prefix column buys you -- search stays fast per-customer regardless
    # of total corpus size)
    def search_individual_history(self, customer_id, query, k=3):
        emb = to_vector_literal(self.embed(query))
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id, summary, created_at,
                          summary_embedding <-> %s AS distance
                   FROM conversations
                   WHERE customer_id = %s AND summary_embedding IS NOT NULL
                   ORDER BY summary_embedding <-> %s
                   LIMIT %s""",
                (emb, customer_id, emb, k),
            )
            return cur.fetchall()

    # -- cohort aggregate: population pattern, NOT a vector search ------
    def cohort_pattern(self, category, tenure_bucket):
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT resolution_type, frequency
                   FROM cohort_resolution_patterns
                   WHERE category = %s AND tenure_bucket = %s
                   ORDER BY frequency DESC""",
                (category, tenure_bucket),
            )
            return cur.fetchall()

    # -- cross-customer similar RESOLVED cases (past memory across customers)
    def search_similar_cases(self, query, k=3):
        emb = to_vector_literal(self.embed(query))
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id, customer_id, summary, status,
                          summary_embedding <-> %s AS distance
                   FROM conversations
                   WHERE summary_embedding IS NOT NULL AND status = 'resolved'
                   ORDER BY summary_embedding <-> %s
                   LIMIT %s""",
                (emb, emb, k),
            )
            return cur.fetchall()

    def _tenure_bucket(self, signals):
        d = signals.tenure_days
        return "new" if d < 30 else "young" if d < 180 else "established"

    # -- main entrypoint: SIGNAL-routed retrieval (no segment label) --------
    # Which memory sources fire is decided by customer_signals.retrieval_plan
    # from data facts (has_history? active fraud hold?), not a 4-way bucket.
    def retrieve(self, customer_id, query, category="REFUND"):
        from backend.utilis.customer_signals import build_signals, retrieval_plan
        signals = build_signals(self.conn, customer_id)
        plan = retrieval_plan(signals)

        result = {
            "signals": signals.as_dict(),
            "plan": plan,
            "kb": self.search_kb(query) if plan["kb"] else [],
            "individual_history": None,
            "cohort_pattern": None,
            "similar_cases": None,
            "note": plan.get("note"),
        }
        if plan["individual_history"]:
            result["individual_history"] = self.search_individual_history(customer_id, query)
        if plan["cohort"]:
            result["cohort_pattern"] = self.cohort_pattern(category, self._tenure_bucket(signals))
        if plan["similar_cases"]:
            result["similar_cases"] = self.search_similar_cases(query)
        return result


if __name__ == "__main__":
    if not CRDB_URL:
        sys.exit("Set CRDB_URL env var first.")
    with psycopg.connect(CRDB_URL) as conn:
        agent = RetrievalAgent(conn)
        # demo against a real customer_id from your loaded customer_profile
        with conn.cursor() as cur:
            cur.execute("SELECT customer_id FROM customer_profile WHERE fraud_flag = true LIMIT 1")
            row = cur.fetchone()
            print(row)
        if row:
            out = agent.retrieve(row[0], "Can I get a refund for my last order?")
            print(json.dumps(out, indent=2, default=str))
        else:
            print("No fraud-flagged customer found -- did you run 06_load_tabular_to_crdb.py?")

