
"""
measures retrieval actually finds the RIGHT KB chunk for a real question (relevancy), 
and whether it correctly
REFUSES to match questions with no policy (the gap topics).
"""
import os
import sys
import argparse
from  backend.agent.retrieval_agent import RetrievalAgent

CRDB_URL = os.environ.get("CRDB_URL")

# Labeled probe set: realistic phrasings a customer would actually type,
# each mapped to the KB category that SHOULD win. Deliberately includes
# near-neighbor pairs (refund vs return, shipping delay vs order status) to
# expose ranking mistakes, plus the seeded GAP topics that must NOT match.
PROBES = [
    # (query, expected_category, is_gap)
    ("I want my money back for an order that never arrived", "refunds", False),
    ("how long do refunds take to hit my card", "refunds", False),
    ("can I send this back, it doesn't fit", "returns", False),
    ("what's the return window for electronics", "returns", False),
    ("my parcel is 5 days late, where is it", "shipping", False),
    ("do you deliver internationally and how long", "shipping", False),
    ("track my package", "order_status", False),
    ("I need to cancel, I ordered by mistake", "cancellation", False),
    ("my payment failed but I was charged", "payment", False),
    ("which cards do you take", "payment", False),
    ("the item showed up smashed, what do I do", "damaged_item", False),
    ("my account is locked and under review", "fraud", False),
    # --- seeded gaps: NO policy exists; must not confidently match ---
    ("when do my loyalty points expire", None, True),
    ("can I move my gift card balance to another card", None, True),
    ("do you price match competitors", None, True),
]


def evaluate(agent, k, gap_threshold):
    hits1 = hits_k = 0.0
    mrr = 0.0
    non_gap = [p for p in PROBES if not p[2]]
    gap = [p for p in PROBES if p[2]]

    print(f"{'query':52s} {'expected':13s} {'top hit':16s} {'d':>5s}  rank")
    print("-" * 96)
    for query, expected, _ in non_gap:
        chunks = agent.search_kb(query, k=k)
        cats = [c.get("category") for c in chunks]
        top = chunks[0] if chunks else {}
        rank = (cats.index(expected) + 1) if expected in cats else 0
        if rank == 1:
            hits1 += 1
        if rank:
            hits_k += 1
            mrr += 1.0 / rank
        print(f"{query[:52]:52s} {str(expected):13s} "
              f"{str(top.get('category'))[:16]:16s} "
              f"{float(top.get('distance', 0)):5.2f}  {rank or '-'}")

    n = len(non_gap)
    print("-" * 96)
    print(f"recall@1={hits1/n:.2f}  recall@{k}={hits_k/n:.2f}  MRR={mrr/n:.2f}  (n={n})")

    # gap rejection
    print("\nGAP TOPICS — top-1 distance should be HIGH (no confident match):")
    ok = 0
    for query, _, _ in gap:
        chunks = agent.search_kb(query, k=1)
        d = float(chunks[0].get("distance", 99)) if chunks else 99
        rejected = d > gap_threshold
        ok += rejected
        print(f"  {query[:52]:52s} d={d:5.2f}  {'OK (rejected)' if rejected else 'RISK (looks like a match!)'}")
    print(f"gap rejection: {ok}/{len(gap)} above threshold {gap_threshold}")
    print("\nInterpretation: if recall@1 is ~1.00 with only the 7 policy docs, that's "
          "the corpus being trivially small — re-run after 02b_ingest_public_kb.py "
          "so the numbers reflect real ranking difficulty.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gap-threshold", type=float, default=0.55,
                    help="normalized L2 distance above which a top hit is treated "
                         "as 'no confident match' (tune to your embedding model)")
    args = ap.parse_args()
    if not CRDB_URL:
        sys.exit("Set CRDB_URL first (this eval queries the live kb_chunks).")

    import psycopg
    with psycopg.connect(CRDB_URL) as conn:
        evaluate(RetrievalAgent(conn), args.k, args.gap_threshold)
