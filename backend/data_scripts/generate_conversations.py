"""
Phase 3 — Conversation data 
"""
import os
import csv
import json
import glob
import random
import uuid
import argparse
import time
import pandas as pd

random.seed(7)

IN = "backend/data/1_transactional_data"
PUBLIC = "backend/data/public_data"
OUT = "backend/data/2_conversation_data"
KB_DIR = os.path.join("backend/data/3_knowledge_base", "kb_docs")

BITEXT_CATEGORY_MAP = {
    "ORDER_STATUS": ["ORDER"],
    "REFUND": ["REFUND"],
    "SHIPPING_DELAY": ["DELIVERY", "SHIPPING_ADDRESS"],
    "ACCOUNT": ["ACCOUNT"],
    "PAYMENT_ISSUE": ["PAYMENT", "INVOICE"],
    "CANCELLATION": ["CANCELLATION_FEE"],
    # DAMAGED_ITEM, FRAUD_REVIEW intentionally absent -- template bank only
}

CAT_TO_KB = {
    "REFUND": "policy_refunds", "ORDER_STATUS": "policy_shipping",
    "SHIPPING_DELAY": "policy_shipping", "DAMAGED_ITEM": "policy_damaged_item",
    "ACCOUNT": "policy_account_security", "PAYMENT_ISSUE": "policy_payment_faq",
    "FRAUD_REVIEW": "policy_account_security", "CANCELLATION": "policy_cancellation",
}

OPENERS = {
    "ORDER_STATUS": ["Where is my order? It's been days.", "Can you tell me the status of order #{order}?"],
    "REFUND": ["I want a refund for my last order, it never showed up.",
               "Requesting a refund - item was wrong."],
    "SHIPPING_DELAY": ["My package is way past the delivery date, what's going on?"],
    "DAMAGED_ITEM": ["The item I received is damaged, I have photos."],
    "ACCOUNT": ["I can't log into my account anymore."],
    "PAYMENT_ISSUE": ["My payment failed twice but I was still charged."],
    "FRAUD_REVIEW": ["Why is my account restricted? I need my refund processed now."],
    "CANCELLATION": ["I need to cancel my order, I just placed it by mistake."],
}

SEGMENT_TONE = {
    "new": ["polite", "unfamiliar with policies, asks clarifying questions"],
    "existing": ["neutral", "references past order without prompting"],
    "churn_risk": ["frustrated", "mentions considering leaving/competitor", "cites repeated past issues"],
    "fraud": ["insistent", "pushes for immediate refund/account change", "impatient with review process"],
}

FOLLOWUPS_BY_TONE = {
    "polite": ["Oh okay, thank you for explaining.", "Got it, what should I do next?"],
    "neutral": ["Alright, and how long will that take?", "Same as last time I guess."],
    "frustrated": ["This keeps happening, I'm honestly considering just using a different site next time.",
                   "This is the third time I've had to reach out about something like this."],
    "insistent": ["I don't understand why this needs review, just process it now.",
                  "Can you just make an exception this once?"],
}

RESOLUTION_SYSTEM = (
    "You are a senior e-commerce support agent writing ONE reply in a chat. "
    "Ground every claim in the POLICY excerpt provided; never invent a policy "
    "or a specific refund amount. If the customer is fraud-segmented, do not "
    "promise any autonomous action — say it's under specialist review. Match "
    "the customer's tone. 2-4 sentences, no greeting boilerplate."
)

def read_csv(name):
    with open(os.path.join(IN, name)) as f:
        return list(csv.DictReader(f))


def load_bitext_bank():
    for fname in ("bitext_retail_ecommerce.csv", "bitext_ecommerce.csv"):
        path = os.path.join(PUBLIC, fname)
        if os.path.exists(path):
            break
    else:
        return None

    df = pd.read_csv(path)
    assert {"instruction", "category"}.issubset(df.columns), \
        f"{path} missing expected Bitext columns (instruction, category)"

    raw_bank = {}
    for _, row in df.iterrows():
        raw_cat = str(row.get("category", "")).upper()
        raw_bank.setdefault(raw_cat, []).append(str(row.get("instruction", "")))

    bank = {}
    for our_cat, bitext_cats in BITEXT_CATEGORY_MAP.items():
        pool = []
        for bc in bitext_cats:
            pool.extend(raw_bank.get(bc, []))
        if pool:
            bank[our_cat] = pool

    total_categories = len(BITEXT_CATEGORY_MAP) + 2 
    print(f"Loaded Bitext dataset from {path}: "
          f"{sum(len(v) for v in bank.values())} usable openers across "
          f"{len(bank)} of our {total_categories} ticket categories")
    return bank


def gen_conversation(ticket, bitext_bank):
    behavior = ticket.get("gen_behavior", "existing")   
    cat = ticket["category"]
    tone = SEGMENT_TONE.get(behavior, ["neutral"])[0]

    if bitext_bank and cat in bitext_bank and bitext_bank[cat]:
        opener = random.choice(bitext_bank[cat])
    else:
        opener = random.choice(OPENERS.get(cat, ["I need help with my order."])).format(
            order=ticket["order_id"][:8]
        )

    turns = [{"role": "customer", "text": opener}]

    turns.append({"role": "agent", "text": "[AGENT_RESPONSE_PLACEHOLDER]",
                  "meta": {"expected_context": ["case_state", "kb_retrieval", "segment_policy"]}})

    if random.random() < 0.6:
        turns.append({"role": "customer", "text": random.choice(FOLLOWUPS_BY_TONE[tone])})
        turns.append({"role": "agent", "text": "[AGENT_RESPONSE_PLACEHOLDER]",
                      "meta": {"expected_context": ["case_state", "kb_retrieval", "segment_policy"]}})

    escalate = ticket["resolution_type"] == "escalated_to_human"
    if escalate:
        turns.append({"role": "system", "text": "[ESCALATED_TO_HUMAN]",
                      "meta": {"reason": "fraud_review" if behavior == "fraud" else "guardrail_tier2_failure_or_policy_limit"}})

    return {
        "conversation_id": str(uuid.uuid4()),
        "ticket_id": ticket["ticket_id"],
        "customer_id": ticket["customer_id"],
        "order_id": ticket["order_id"],
        "behavior": behavior,
        "category": cat,
        "tone": tone,
        "turns": turns,
        "resolved": not escalate,
        "created_at": ticket["created_at"],
    }

def load_kb():
    kb = {}
    for path in glob.glob(os.path.join(KB_DIR, "*.md")):
        stem = os.path.basename(path).rsplit("_v", 1)[0]
        kb[stem] = open(path, encoding="utf-8").read()
    return kb


def build_prompt(conv, kb, last_customer_text):
    doc = kb.get(CAT_TO_KB.get(conv["category"], "policy_refunds"), "")[:900]
    return (
        f"CUSTOMER BEHAVIOR: {conv.get('behavior','existing')}\n"
        f"CATEGORY: {conv['category']}\n"
        f"TONE: {conv.get('tone','neutral')}\n\n"
        f"RELEVANT POLICY EXCERPT:\n{doc or '(no policy on file for this topic)'}\n\n"
        f"CUSTOMER SAID:\n{last_customer_text}\n\nWrite the agent reply now."
    )


def backfill_direct(conversations, kb, limit, sleep):
    from backend.utilis.gemini_client import get_llm
    # from backend.utilis.bedrock_client import get_llm
    llm = get_llm()
    print(f"model live={llm.is_live}  (offline fallback if False)")
    done = 0
    for conv in conversations:
        if limit and done >= limit:
            break
        last_customer = conv["turns"][0]["text"]
        filled = False
        for turn in conv["turns"]:
            if turn["role"] == "customer":
                last_customer = turn["text"]
            if turn["role"] == "agent" and turn["text"] == "[AGENT_RESPONSE_PLACEHOLDER]":
                res = llm.complete(RESOLUTION_SYSTEM, build_prompt(conv, kb, last_customer),
                                   max_tokens=220, temperature=0.4)
                turn["text"] = res.text.strip()
                turn.setdefault("meta", {})["generated_by"] = res.get("source")
                filled = True
        if filled:
            done += 1
            if done % 50 == 0:
                print(f"  filled {done} conversations...")
            if sleep:
                time.sleep(sleep)
    return done

def write_csv(rows, name):
    if not rows:
        return
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):>6} rows -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backfill", action="store_true",
                     help="skip agent-turn generation, leave placeholders (old script-1-only behavior)")
    ap.add_argument("--limit", type=int, default=0, help="cap conversations to backfill (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="delay between backfill calls, for rate limits")
    args = ap.parse_args()

    tickets = read_csv("support_tickets.csv")
    bitext_bank = load_bitext_bank()
    print("Using Bitext public dataset for openers" if bitext_bank else
          "No public_data/bitext_ecommerce.csv found -- using template bank")

    conversations = [gen_conversation(t, bitext_bank) for t in tickets]

    if not args.no_backfill:
        kb = load_kb()
        print(f"loaded {len(kb)} KB docs")
        n = backfill_direct(conversations, kb, args.limit, args.sleep)
        print(f"filled {n} conversations with real agent turns")
    else:
        print("--no-backfill set: agent turns left as [AGENT_RESPONSE_PLACEHOLDER]")

    with open(os.path.join(OUT, "conversations.jsonl"), "w") as f:
        for c in conversations:
            f.write(json.dumps(c) + "\n")

    print(pd.DataFrame(conversations).head(3).to_string())
    write_csv(conversations, "conversations.csv")

    seg_counts = {}
    for c in conversations:
        seg_counts[c["behavior"]] = seg_counts.get(c["behavior"], 0) + 1
    print(f"wrote {len(conversations)} conversations -> {OUT}/conversations.jsonl")
    print("by behavior (generation-only):", json.dumps(seg_counts, indent=2))