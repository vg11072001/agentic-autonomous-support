"""
03b_backfill_agent_turns.py  — dataset quality fix

TWO MODES
  --mode direct       (default) direct model call per conversation, using the
                      KB docs on disk. Runnable BEFORE loading to CRDB, no DB.
  --mode orchestrator run the live LangGraph graph for max fidelity. Needs
                      CRDB_URL + already-loaded data (run after 13). Slower.
"""
import os
import json
import glob
import argparse
import time

OUT = "backend/data/2_conversation_data"
KB_DIR = os.path.join("backend/data/3_knowledge_base", "kb_docs")

CAT_TO_KB = {
    "REFUND": "policy_refunds", "ORDER_STATUS": "policy_shipping",
    "SHIPPING_DELAY": "policy_shipping", "DAMAGED_ITEM": "policy_damaged_item",
    "ACCOUNT": "policy_account_security", "PAYMENT_ISSUE": "policy_payment_faq",
    "FRAUD_REVIEW": "policy_account_security", "CANCELLATION": "policy_cancellation",
}

RESOLUTION_SYSTEM = (
    "You are a senior e-commerce support agent writing ONE reply in a chat. "
    "Ground every claim in the POLICY excerpt provided; never invent a policy "
    "or a specific refund amount. If the customer is fraud-segmented, do not "
    "promise any autonomous action — say it's under specialist review. Match "
    "the customer's tone. 2-4 sentences, no greeting boilerplate."
)


def load_kb():
    kb = {}
    for path in glob.glob(os.path.join(KB_DIR, "*.md")):
        stem = os.path.basename(path).rsplit("_v", 1)[0]  # policy_refunds_v1.md -> policy_refunds
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


def backfill_orchestrator(conversations, limit, sleep):
    """Max-fidelity mode: replay each opener through the live graph. Requires
    CRDB_URL + loaded data. Imported lazily so `direct` mode has no DB dep."""
    import psycopg
    from importlib import import_module
    orch = import_module("10_orchestrator")
    url = os.environ["CRDB_URL"]
    done = 0
    with psycopg.connect(url) as conn:
        from backend.utilis.crdb_checkpointer import CRDBCheckpointSaver
        with CRDBCheckpointSaver.from_conn_string(url) as cp:
            cp.setup()
            graph = orch.build_graph(conn).compile(checkpointer=cp)
            for conv in conversations:
                if limit and done >= limit:
                    break
                cid = orch.create_conversation(conn, conv["customer_id"])
                out = graph.invoke(
                    {"conversation_id": cid, "customer_id": conv["customer_id"],
                     "query": conv["turns"][0]["text"], "escalated": False},
                    config={"configurable": {"thread_id": cid}})
                reply = out.get("final_response") or "[escalated to human specialist]"
                for turn in conv["turns"]:
                    if turn["role"] == "agent" and turn["text"] == "[AGENT_RESPONSE_PLACEHOLDER]":
                        turn["text"] = reply
                        turn.setdefault("meta", {})["generated_by"] = "orchestrator"
                        break
                done += 1
                if sleep:
                    time.sleep(sleep)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "orchestrator"], default="direct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    convos = [json.loads(line) for line in open(os.path.join(OUT, "conversations.jsonl"))]
    kb = load_kb()
    print(f"loaded {len(convos)} conversations, {len(kb)} KB docs")

    if args.mode == "direct":
        n = backfill_direct(convos, kb, args.limit, args.sleep)
    else:
        n = backfill_orchestrator(convos, args.limit, args.sleep)

    out_path = os.path.join(OUT, "conversations.jsonl")
    with open(out_path, "w") as f:
        for c in convos:
            f.write(json.dumps(c) + "\n")
    print(f"filled {n} conversations -> {out_path}")
    print("Next: point 13_load_conversations_to_crdb.py at conversations.jsonl "
          "(or rename it over conversations.jsonl) so the embeddings carry real agent text.")
    import json
    import os
    import pandas as pd

    pd.DataFrame(convos).to_csv(os.path.join("2_conversation_data", "conversations.csv"), index=False)
