

"""
02b_ingest_public_kb.py — turn PUBLIC support datasets into citable KB.
"""
import os
import json
import csv
import glob
import argparse
import hashlib

OUT = "backend/data/3_knowledge_base/kb_docs"
PUBLIC = "backend/data/public_data"
MANIFEST = "backend/data/3_knowledge_base/kb_manifest.json"

# our KB category vocabulary (matches the policy docs' `category`)
CATS = ["returns", "refunds", "shipping", "payment", "cancellation",
        "account", "order_status", "damaged_item", "general"]

# keyword -> category, used when a FAQ row has no category of its own
KW = {
    "return": "returns", "refund": "refunds", "money back": "refunds",
    "ship": "shipping", "deliver": "shipping", "track": "order_status",
    "where is my order": "order_status", "status": "order_status",
    "pay": "payment", "card": "payment", "invoice": "payment", "wallet": "payment",
    "cancel": "cancellation", "account": "account", "login": "account",
    "password": "account", "sign up": "account", "damage": "damaged_item",
    "broken": "damaged_item", "warranty": "damaged_item",
}
# Bitext's own categories -> ours
BITEXT_MAP = {
    "REFUND": "refunds", "ORDER": "order_status", "DELIVERY": "shipping",
    "SHIPPING_ADDRESS": "shipping", "CANCEL": "cancellation",
    "CANCELLATION_FEE": "cancellation", "PAYMENT": "payment", "INVOICE": "payment",
    "ACCOUNT": "account", "REFUND_POLICY": "refunds", "RETURN": "returns",
}


def categorize(text, given=None):
    if given:
        g = str(given).upper()
        if g in BITEXT_MAP:
            return BITEXT_MAP[g]
        gl = str(given).lower()
        if gl in CATS:
            return gl
    t = (text or "").lower()
    for kw, cat in KW.items():
        if kw in t:
            return cat
    return "general"


def load_pairs():
    """Return list of (question, answer, category, source_tag)."""
    pairs = []

    def add_json_qa(path, tag):
        raw = open(path, encoding="utf-8").read().strip()
        rows = []
        if raw.startswith("["):
            rows = json.loads(raw)
        else:  # jsonl
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        for r in rows:
            q = r.get("question") or r.get("instruction") or r.get("query")
            a = r.get("answer") or r.get("response") or r.get("output")
            if q and a:
                pairs.append((q.strip(), a.strip(), categorize(q, r.get("category")), tag))

    for fname, tag in [("maktek_faqs.json", "maktek"),
                       ("ecommerce_faq.json", "ecommerce_faq"),
                       ("qgyd_faq.json", "qgyd")]:
        p = os.path.join(PUBLIC, fname)
        if os.path.exists(p):
            add_json_qa(p, tag)
            print(f"  loaded {p}")

    for p in glob.glob(os.path.join(PUBLIC, "bitext*ecommerce*.csv")) + \
             glob.glob(os.path.join(PUBLIC, "bitext*.csv")):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                q, a = r.get("instruction"), r.get("response")
                if q and a:
                    pairs.append((q.strip(), a.strip(),
                                  categorize(q, r.get("category")), "bitext"))
        print(f"  loaded {p}")
        break
    return pairs


def dedupe(pairs):
    seen, out = set(), []
    for q, a, c, tag in pairs:
        key = hashlib.sha1((q.lower()[:80]).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append((q, a, c, tag))
    return out


def write_category_docs(pairs, per_category):
    by_cat = {}
    for q, a, c, tag in pairs:
        by_cat.setdefault(c, []).append((q, a, tag))

    manifest_add = []
    for cat, rows in by_cat.items():
        rows = rows[:per_category]
        doc_id = f"faq_{cat}_v1"
        title = f"Community FAQ — {cat.replace('_', ' ').title()}"
        body = [f"# {title}", ""]
        for q, a, tag in rows:
            body += [f"### {q}", "", a, "", f"_source: {tag}_", ""]
        path = os.path.join(OUT, f"{doc_id}.md")
        open(path, "w", encoding="utf-8").write("\n".join(body))
        manifest_add.append({"doc_id": doc_id, "title": title, "category": cat,
                             "path": path, "version": 1, "kind": "faq"})
        print(f"  wrote {path}  ({len(rows)} Q/A)")
    return manifest_add


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=40,
                    help="cap FAQ entries per category (keeps chunk count sane)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pairs = dedupe(load_pairs())
    if not pairs:
        print("No public files found in public_data/. See this script's docstring "
              "for the exact HF/Kaggle datasets to download. Nothing to do.")
        raise SystemExit(0)
    print(f"loaded {len(pairs)} unique Q/A pairs")

    manifest_add = write_category_docs(pairs, args.per_category)

    # merge into the existing manifest so 07 picks these up unchanged
    mani = {"articles": [], "intentional_gap_topics": []}
    if os.path.exists(MANIFEST):
        mani = json.load(open(MANIFEST))
    existing = {a["doc_id"] for a in mani["articles"]}
    for a in manifest_add:
        if a["doc_id"] not in existing:
            mani["articles"].append(a)
    json.dump(mani, open(MANIFEST, "w"), indent=2)
    print(f"\nmanifest now has {len(mani['articles'])} articles "
          f"({sum(1 for a in mani['articles'] if a.get('kind')=='faq')} FAQ). "
          f"Run 07_chunk_and_embed_kb.py next.")

