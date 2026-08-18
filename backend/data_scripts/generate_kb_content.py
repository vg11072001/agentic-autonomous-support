"""
Phase 2 — Unstructured knowledge: policy docs, FAQs, manuals.
"""
import os
import json

OUT = "backend/data/3_knowledge_base/kb_docs"
os.makedirs(OUT, exist_ok=True)

ARTICLES = [
    {
        "doc_id": "policy_returns_v1",
        "title": "Return Policy",
        "category": "returns",
        "content": """
Items may be returned within 30 days of delivery if unused and in original
packaging. Electronics have a 15-day return window due to manufacturer
restocking terms. Return shipping is free for defective or incorrect items;
customer-initiated returns for change-of-mind incur a $5.99 shipping deduction
from the refund. Refunds are issued to the original payment method within
5-7 business days of the returned item passing warehouse inspection.
""",
    },
    {
        "doc_id": "policy_refunds_v1",
        "title": "Refund Policy",
        "category": "refunds",
        "content": """
Refunds are processed automatically for orders marked "never arrived" after
21 days from the estimated delivery date, provided tracking shows no
delivery scan. Refunds for "damaged" or "wrong item" claims require a photo
upload and are auto-approved for orders under $75; orders at or above $75
route to manual review. Refund requests from accounts flagged for fraud
review are never auto-approved regardless of order value and must be handled
by a human agent.
""",
    },
    {
        "doc_id": "policy_shipping_v1",
        "title": "Shipping & Delivery Policy",
        "category": "shipping",
        "content": """
Standard shipping takes 3-7 business days domestically. Orders delayed
beyond the estimated delivery window by more than 3 days automatically
qualify for a $10 shipping credit, applied to the customer's account without
requiring a support request. International orders may incur customs delays
outside our control; we do not offer delivery-time guarantees on
international shipments.
""",
    },
    {
        "doc_id": "policy_account_security_v1",
        "title": "Account Security & Fraud Review Policy",
        "category": "fraud",
        "content": """
Accounts showing rapid high-value order activity shortly after signup, or a
pattern of repeated refund claims, are automatically routed to fraud review.
While an account is under fraud review: no autonomous refunds, address
changes, or payment method changes may be processed by the automated support
system. All requests from a flagged account must be handled by a human
fraud specialist. Customers should never be told their account is
"flagged for fraud" directly -- use the neutral phrase "your request is
being reviewed by our specialist team."
""",
    },
    {
        "doc_id": "policy_cancellation_v1",
        "title": "Order Cancellation Policy",
        "category": "cancellation",
        "content": """
Orders can be cancelled free of charge within 1 hour of placement, before
warehouse processing begins. After processing begins, cancellation is not
possible; the customer should be directed to the return process once the
order is delivered.
""",
    },
    {
        "doc_id": "policy_payment_faq_v1",
        "title": "Payment FAQ",
        "category": "payment",
        "content": """
We accept credit card, debit card, PayPal, and store wallet balance. Failed
payments are retried automatically once after 30 minutes. If a payment
fails twice, the order is cancelled and the customer is notified by email.
Wallet balance never expires and can be combined with one other payment
method per order.
""",
    },
    {
        "doc_id": "policy_damaged_item_v1",
        "title": "Damaged Item Policy",
        "category": "damaged_item",
        "content": """
If an item arrives damaged, customers should upload a photo within 48 hours
of delivery. Damaged-item claims do not require the item to be returned for
orders under $50; a replacement or refund is issued directly. For orders
$50 and above, a prepaid return label is issued and the replacement/refund
is processed once the item is received at the warehouse.
""",
    },
]

GAP_TOPICS = [
    "loyalty_points_expiration",   # no article exists yet
    "gift_card_balance_transfer"    # no article exists yet
    "price_match_guarantee",   # no article exists yet
]

if __name__ == "__main__":
    manifest = []
    for a in ARTICLES:
        path = os.path.join(OUT, f"{a['doc_id']}.md")
        with open(path, "w") as f:
            f.write(f"# {a['title']}\n\n{a['content'].strip()}\n")
        manifest.append({"doc_id": a["doc_id"], "title": a["title"],
                          "category": a["category"], "path": path, "version": 1})
        print(f"wrote {path}")

    with open("backend/data/3_knowledge_base/kb_manifest.json", "w") as f:
        json.dump({"articles": manifest, "intentional_gap_topics": GAP_TOPICS}, f, indent=2)
    print(f"\n{len(manifest)} KB articles written. "
          f"{len(GAP_TOPICS)} intentional gap topics recorded in backend/data/3_knowledge_base/kb_manifest.json")
    print("Gap topics (no matching article -- used to validate the KB gap-filling pipeline):")
    for t in GAP_TOPICS:
        print("  -", t)
