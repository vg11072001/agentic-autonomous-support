"""
Chunk the Phase-2 KB markdown docs, embed with all-MiniLM-L6-v2
load into CockroachDB's kb_chunks table.
"""
import os
import sys
import json
import re
from sentence_transformers import SentenceTransformer
import psycopg
from dotenv import load_dotenv
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
def _with_query_param(url, key, value):
    parts = urlsplit(url)
    params = parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))
load_dotenv()  

CRDB_URL = os.environ.get("CRDB_URL")
if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "backend/data/3_knowledge_base")
KB_DIR = DATA_DIR
MANIFEST = os.path.join(DATA_DIR, "kb_manifest.json")

CHUNK_TOKENS = 150  
OVERLAP = 20

def simple_chunk(text, chunk_words=CHUNK_TOKENS, overlap=OVERLAP):
    """Word-based chunking -- swap for a tokenizer-aware splitter (e.g. from
    the docx/pdf skill's chunking helper) once real PDFs replace these
    markdown stand-ins."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_words]
        chunks.append(" ".join(chunk))
        if i + chunk_words >= len(words):
            break
        i += chunk_words - overlap
    return chunks

def to_vector_literal(vec):
    """CockroachDB's VECTOR type expects '[v1,v2,...]' syntax.
    psycopg's default list adapter produces Postgres array syntax
    '{v1,v2,...}' instead, which VECTOR rejects."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"

def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)["articles"]


if __name__ == "__main__":
    if not CRDB_URL:
        sys.exit("Set CRDB_URL env var first.")

    print("Loading embedding model (all-MiniLM-L6-v2, 384 dims)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    manifest = load_manifest()
    rows_to_insert = []

    for article in manifest:
        with open( os.path.join(KB_DIR, article["path"])) as f:
            content = f.read()
        body = re.sub(r"^# .*\n\n", "", content).strip()
        chunks = simple_chunk(body)
        embeddings = model.encode(chunks, normalize_embeddings=True)

        for chunk_text, emb in zip(chunks, embeddings):
            rows_to_insert.append((
                article["doc_id"], article["version"], article["title"],
                article["category"], chunk_text, emb.tolist(),
            ))
        print(f"  {article['doc_id']}: {len(chunks)} chunk(s)")

    print(f"\nTotal chunks to insert: {len(rows_to_insert)}")

    with psycopg.connect(CRDB_URL) as conn:
        with conn.cursor() as cur:
            for doc_id, version, title, category, content, emb in rows_to_insert:
                cur.execute(
                    """INSERT INTO kb_chunks (doc_id, version, title, category, content, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (doc_id, version, title, category, content, to_vector_literal(emb)),
                )
        conn.commit()

    print("Loaded into kb_chunks.")
    print("\nVerify with:")
    print("  SELECT doc_id, title, count(*) FROM kb_chunks GROUP BY doc_id, title;")
