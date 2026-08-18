"""
Benchmark: CockroachDB distributed vector index vs Qdrant
"""
import os
import sys
import json
import time
import ast
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run: pip install sentence-transformers --break-system-packages")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
except ImportError:
    sys.exit("Run: pip install qdrant-client --break-system-packages")

import urllib.parse
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
    
QDRANT_URL = os.environ.get("QDRANT_URL")  # unset -> in-memory, good for a quick local test
COLLECTION = "kb_chunks_benchmark"
TOP_K = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "3_knowledge_base")
KB_DIR = DATA_DIR
MANIFEST = os.path.join(DATA_DIR, "kb_manifest.json")


TEST_QUERIES = [
    "Can I get a refund for my last order?",
    "My package is late, what happens?",
    "The item I received is broken",
    "Is my account under review?",
    "Can I cancel my order?",
]


def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)["articles"]


def load_kb_chunks_from_manifest(model):
    """Same source of truth as 07_chunk_and_embed_kb.py -- re-embeds from
    the manifest so this script can run standalone without needing to
    read back from CockroachDB first."""
    manifest = load_manifest()
    chunks = []
    for article in manifest:
        with open( os.path.join(KB_DIR, article["path"]))  as f:
            content = f.read()
        body = content.split("\n\n", 1)[-1].strip()
        emb = model.encode([body], normalize_embeddings=True)[0]
        
        chunks.append({
            "id": article["doc_id"], "title": article["title"],
            "content": body, "embedding": to_vector_literal(emb.tolist()),
        })
    return chunks

def to_vector_literal(vec):
    """CockroachDB's VECTOR type expects '[v1,v2,...]' syntax.
    psycopg's default list adapter produces Postgres array syntax
    '{v1,v2,...}' instead, which VECTOR rejects."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def setup_qdrant(chunks, dim=384):
    client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(":memory:")
    if client.collection_exists(COLLECTION):
    #     client.delete_collection(COLLECTION)
    # client.create_collection(
    #     collection_name=COLLECTION,
    #     vectors_config=VectorParams(size=dim, distance=Distance.COSINE),)
    # points = []
    # for i, c in enumerate(chunks):
    #     embedding = c["embedding"]
    #     # Convert string representation of list to actual list
    #     if isinstance(embedding, str):
    #         embedding = ast.literal_eval(embedding)
    #     embedding = [float(x) for x in embedding]
    #     if len(embedding) != dim:
    #         raise ValueError(
    #             f"Embedding dimension mismatch for chunk {i}: "
    #             f"expected {dim}, got {len(embedding)}"
    #         )
    #     points.append(
    #         PointStruct(
    #             id=i,
    #             vector=embedding,
    #             payload={
    #                 "doc_id": c["id"],
    #                 "title": c["title"],
    #                 "content": c["content"],
    #             },
    #         )
    #     )
    # client.upsert(
    #     collection_name=COLLECTION,
    #     points=points,
    # )
    # return client
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=c["embedding"],
                        payload={"doc_id": c["id"], "title": c["title"], "content": c["content"]})
            for i, c in enumerate(chunks)
        ],
    )
    return client

def normalize_embedding(embedding):
    if isinstance(embedding, str):
        embedding = ast.literal_eval(
            embedding
        )
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    return [
        float(x)
        for x in embedding
    ]
    
def query_qdrant(client, query_emb, k=TOP_K):
    query_emb = normalize_embedding(query_emb)
    t0 = time.perf_counter()
    resp = client.query_points(collection_name=COLLECTION, query=query_emb, limit=k)
    latency_ms = (time.perf_counter() - t0) * 1000
    return [(p.payload["doc_id"], p.score) for p in resp.points], latency_ms


def query_crdb(conn, query_emb, k=TOP_K):
    from psycopg.rows import dict_row
    t0 = time.perf_counter()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT doc_id, embedding <-> %s AS distance
               FROM kb_chunks ORDER BY embedding <-> %s LIMIT %s""",
            (query_emb, query_emb, k),
        )
        rows = cur.fetchall()
    latency_ms = (time.perf_counter() - t0) * 1000
    return [(r["doc_id"], r["distance"]) for r in rows], latency_ms


def overlap_at_k(crdb_results, qdrant_results):
    crdb_ids = {r[0] for r in crdb_results}
    qdrant_ids = {r[0] for r in qdrant_results}
    if not crdb_ids or not qdrant_ids:
        return None
    return len(crdb_ids & qdrant_ids) / len(crdb_ids | qdrant_ids)


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = load_kb_chunks_from_manifest(model)
    print(f"Loaded {len(chunks)} KB chunks for benchmarking\n")

    qdrant_client = setup_qdrant(chunks)
    print(f"Qdrant collection ready ({'remote: ' + QDRANT_URL if QDRANT_URL else 'in-memory'})\n")

    crdb_conn = None
    if CRDB_URL:
        import psycopg
        crdb_conn = psycopg.connect(CRDB_URL)
    else:
        print("CRDB_URL not set -- skipping CockroachDB side, Qdrant-only run\n")

    results = []
    for query in TEST_QUERIES:
        emb = model.encode([query], normalize_embeddings=True)[0].tolist()
        emb = to_vector_literal(emb)
        qdrant_res, qdrant_ms = query_qdrant(qdrant_client, emb)
        row = {"query": query, "qdrant_top": qdrant_res, "qdrant_ms": round(qdrant_ms, 2)}

        if crdb_conn:
            crdb_res, crdb_ms = query_crdb(crdb_conn, emb)
            row["crdb_top"] = crdb_res
            row["crdb_ms"] = round(crdb_ms, 2)
            row["overlap_at_k"] = overlap_at_k(crdb_res, qdrant_res)

        results.append(row)

    if crdb_conn:
        crdb_conn.close()

    print(json.dumps(results, indent=2, default=str))

    if crdb_conn is not None:
        avg_overlap = sum(r["overlap_at_k"] for r in results if r["overlap_at_k"] is not None) / len(results)
        avg_crdb_ms = sum(r["crdb_ms"] for r in results) / len(results)
        avg_qdrant_ms = sum(r["qdrant_ms"] for r in results) / len(results)
        print(f"\nAvg overlap@{TOP_K}: {avg_overlap:.2f}  "
              f"(should be ~1.0 -- same embeddings, same math, just different engines; "
              f"anything lower signals a bug, not a real quality difference)")
        print(f"Avg latency -- CockroachDB: {avg_crdb_ms:.2f}ms  Qdrant: {avg_qdrant_ms:.2f}ms")
