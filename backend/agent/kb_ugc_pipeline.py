"""
KB gap-filling from escalated conversations (UGC pipeline).

Usage:
  export CRDB_URL="postgresql://..."
  python3 17_kb_ugc_pipeline.py detect                 # find gap clusters, write drafts
  python3 17_kb_ugc_pipeline.py review                 # list pending drafts
  python3 17_kb_ugc_pipeline.py approve <draft_id>      # publish a draft into kb_chunks
"""
import os
import sys
import json
import uuid

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    sys.exit("Run: pip install 'psycopg[binary]' --break-system-packages")

try:
    from sklearn.cluster import KMeans
    import numpy as np
except ImportError:
    sys.exit("Run: pip install scikit-learn --break-system-packages")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run: pip install sentence-transformers --break-system-packages")

CRDB_URL = os.environ.get("CRDB_URL")

SCHEMA_ADDITIONS = """
CREATE TABLE IF NOT EXISTS kb_draft_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_cluster_id STRING,
    escalation_count INT,
    draft_content STRING,
    status STRING DEFAULT 'pending_review',
    reviewer STRING,
    published_chunk_id UUID REFERENCES kb_chunks(id),
    created_at TIMESTAMPTZ DEFAULT now()
);
"""

MIN_CLUSTER_SIZE = 5          # don't draft an article for a one-off complaint
GAP_DISTANCE_THRESHOLD = 0.4  # min distance from nearest existing kb_chunk to count as a "gap"
N_CLUSTERS = 8


def _parse_embedding(v) -> np.ndarray:
    """CockroachDB VECTOR columns come back as strings like '[-0.07,0.12,...]'.
    Handle that, plus plain lists and numpy arrays."""
    if isinstance(v, np.ndarray):
        return v.astype(float)
    if isinstance(v, list):
        return np.array(v, dtype=float)
    if isinstance(v, str):
        return np.array(json.loads(v), dtype=float)
    return np.array(v, dtype=float)


def _vector_literal(arr) -> str:
    """Format a numpy array as a CockroachDB VECTOR literal '[x,y,...]'."""
    return "[" + ",".join(f"{float(x):.8g}" for x in arr) + "]"


def fetch_escalated_embeddings(conn):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, summary, summary_embedding
               FROM conversations
               WHERE escalated = true AND summary_embedding IS NOT NULL"""
        )
        return cur.fetchall()


def nearest_kb_distance(conn, embedding) -> float:
    """embedding is a numpy array or list; formats it as a VECTOR literal for CockroachDB."""
    vec = _vector_literal(embedding)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT embedding <-> %s::VECTOR AS distance FROM kb_chunks
               ORDER BY distance LIMIT 1""",
            (vec,),
        )
        row = cur.fetchone()
        return float(row["distance"]) if row else 999.0


def draft_article_llm(cluster_summaries):
    """Real AWS Bedrock (Claude) call: summarize a cluster of escalated
    conversations into a candidate KB article. Always writes to
    kb_draft_articles as `pending_review` — a human approves before it is
    embedded and inserted into kb_chunks. Degrades to a labelled offline
    stub if Bedrock is unreachable (still human-gated, so safe)."""
    from backend.utilis.bedrock_client import get_llm
    joined = "\n- ".join(cluster_summaries[:8])
    system = (
        "You draft internal knowledge-base policy articles for an e-commerce "
        "support team. Given a cluster of customer issues that currently have "
        "NO covering policy (they keep escalating), draft ONE concise policy "
        "article that would let an agent resolve them. Mark any specific "
        "numbers/thresholds as [TO CONFIRM] — do not invent policy values."
    )
    user = f"Recurring escalated issues in this cluster:\n- {joined}\n\nDraft the article."
    return get_llm().complete(system, user, max_tokens=600).text


def detect_gaps(conn):
    rows = fetch_escalated_embeddings(conn)
    if len(rows) < N_CLUSTERS:
        print(f"Not enough escalated conversations ({len(rows)}) to cluster meaningfully.")
        return

    X = np.array([_parse_embedding(r["summary_embedding"]) for r in rows])
    km = KMeans(n_clusters=min(N_CLUSTERS, len(rows)), random_state=0, n_init=10).fit(X)

    clusters = {}
    for row, label in zip(rows, km.labels_):
        clusters.setdefault(label, []).append(row)

    drafts_written = 0
    with conn.cursor() as cur:
        for label, members in clusters.items():
            if len(members) < MIN_CLUSTER_SIZE:
                continue
            centroid = km.cluster_centers_[label].tolist()
            gap_distance = nearest_kb_distance(conn, centroid)
            if gap_distance < GAP_DISTANCE_THRESHOLD:
                continue  # KB already covers this cluster reasonably well

            summaries = [m["summary"] for m in members]
            draft = draft_article_llm(summaries)
            cur.execute(
                """INSERT INTO kb_draft_articles
                   (source_cluster_id, escalation_count, draft_content, status)
                   VALUES (%s, %s, %s, 'pending_review')""",
                (f"cluster_{label}", len(members), draft),
            )
            drafts_written += 1
            print(f"cluster_{label}: {len(members)} escalations, "
                  f"gap_distance={gap_distance:.3f} -> draft written")

            # A recurring, uncovered escalation cluster is also a missing
            # PLAYBOOK. Propose a DRAFT skill (human-reviewed before it can
            # ever be equipped) — the "extract skill from completed tasks,
            # share after review" loop. Best-effort; never blocks the article.
            try:
                from importlib import import_module
                skills_mod = import_module("23_skills")
                embed_fn = import_module("08_retrieval_agent").RetrievalAgent(conn).embed
                skills_mod.propose_skill_from_cluster(
                    conn, "\n- ".join(summaries[:8]),
                    category="UNKNOWN", embed_fn=_vector_literal(embed_fn))
            except Exception as e:
                print(f"  (skill proposal skipped: {e})")
    conn.commit()
    print(f"\n{drafts_written} draft article(s) written to kb_draft_articles (status=pending_review)")


def list_pending(conn):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, source_cluster_id, escalation_count, draft_content "
                     "FROM kb_draft_articles WHERE status = 'pending_review'")
        for row in cur.fetchall():
            print(json.dumps(row, indent=2, default=str))


def approve_draft(conn, draft_id, reviewer="human_reviewer", model=None):
    model = model or SentenceTransformer("all-MiniLM-L6-v2")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM kb_draft_articles WHERE id = %s", (draft_id,))
        draft = cur.fetchone()
    if not draft:
        print("Draft not found.")
        return

    emb = _vector_literal(model.encode([draft["draft_content"]], normalize_embeddings=True)[0].tolist())
    chunk_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO kb_chunks (id, doc_id, version, title, category, content, embedding)
               VALUES (%s, %s, 1, %s, 'ugc_derived', %s, %s)""",
            (chunk_id, draft["source_cluster_id"], f"UGC gap-fill: {draft['source_cluster_id']}",
             draft["draft_content"], emb),
        )
        cur.execute(
            """UPDATE kb_draft_articles SET status = 'approved', reviewer = %s,
               published_chunk_id = %s WHERE id = %s""",
            (reviewer, chunk_id, draft_id),
        )
    conn.commit()
    print(f"Published draft {draft_id} -> kb_chunks {chunk_id}")


def escalation_rate_impact(conn, category):
    """Compare escalation rate before/after a KB publish for a category --
    run this some time after approving a draft to measure real impact."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT date_trunc('week', created_at) AS week,
                      avg(CASE WHEN resolution_type = 'escalated_to_human' THEN 1.0 ELSE 0.0 END) AS escalation_rate
               FROM support_tickets WHERE category = %s
               GROUP BY week ORDER BY week""",
            (category,),
        )
        return cur.fetchall()


if __name__ == "__main__":
    if not CRDB_URL:
        sys.exit("Set CRDB_URL env var first.")
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    with psycopg.connect(CRDB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_ADDITIONS)
        conn.commit()

        cmd = sys.argv[1]
        if cmd == "detect":
            detect_gaps(conn)
        elif cmd == "review":
            list_pending(conn)
        elif cmd == "approve":
            approve_draft(conn, sys.argv[2])
        else:
            sys.exit(__doc__)
