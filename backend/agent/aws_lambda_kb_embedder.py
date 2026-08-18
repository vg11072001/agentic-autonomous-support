
"""
AWS Lambda background job.
"""
import os
import sys
import json
import logging

log = logging.getLogger("kb_embedder")
logging.basicConfig(level=logging.INFO)

CRDB_URL = os.environ.get("CRDB_URL")
KB_BUCKET = os.environ.get("KB_BUCKET", "")

CHUNK_CHARS = 1200        # policy docs are short; one or few chunks each
CHUNK_OVERLAP = 150


def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def _embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _to_vector_literal(vec):
    # CockroachDB VECTOR expects '[v1,v2,...]', not Postgres array '{...}'.
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def upsert_chunks(conn, doc_id, title, category, chunks, embeddings):
    """Idempotent per (doc_id, chunk index): re-running on an updated doc
    replaces its chunks instead of duplicating them."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM kb_chunks WHERE doc_id = %s", (doc_id,))
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """INSERT INTO kb_chunks (doc_id, version, title, category, content, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (f"{doc_id}#{idx}", 1, title, category, chunk, _to_vector_literal(emb)),
            )
    conn.commit()


def process_document(conn, model, doc_id, title, category, raw_text):
    chunks = chunk_text(raw_text)
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
    upsert_chunks(conn, doc_id, title, category, chunks, embeddings)
    log.info("embedded doc_id=%s chunks=%d", doc_id, len(chunks))
    return {"doc_id": doc_id, "chunks": len(chunks)}


# -- Lambda entrypoint ------------------------------------------------------
def handler(event, context):
    """S3 PutObject trigger. Reads the object, embeds it, upserts to CRDB."""
    import boto3
    import psycopg

    s3 = boto3.client("s3")
    results = []
    with psycopg.connect(CRDB_URL) as conn:
        model = _embed_model()
        for rec in event.get("Records", []):
            bucket = rec["s3"]["bucket"]["name"]
            key = rec["s3"]["object"]["key"]
            obj = s3.get_object(Bucket=bucket, Key=key)
            raw = obj["Body"].read().decode("utf-8")
            doc_id = os.path.splitext(os.path.basename(key))[0]
            title = doc_id.replace("_", " ").title()
            category = doc_id.split("_")[1] if "_" in doc_id else "general"
            results.append(process_document(conn, model, doc_id, title, category, raw))
    return {"statusCode": 200, "body": json.dumps({"embedded": results})}


if __name__ == "__main__":
    # local smoke test against a file path (no AWS/S3 needed)
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 22_aws_lambda_kb_embedder.py <policy.md>  (needs CRDB_URL + deps)")
    if not CRDB_URL:
        sys.exit("Set CRDB_URL to run the local test.")
    import psycopg
    path = sys.argv[1]
    with open(path) as f:
        raw = f.read()
    doc_id = os.path.splitext(os.path.basename(path))[0]
    with psycopg.connect(CRDB_URL) as conn:
        model = _embed_model()
        print(process_document(conn, model, doc_id, doc_id.replace("_", " ").title(),
                               "general", raw))


