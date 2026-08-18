"""
24_working_memory.py — assemble a BOUNDED, layered working-memory header for
the Resolution Agent, before it reasons.

WHY (see MEMORY_DESIGN_TENCENTDB.md §4)
---------------------------------------
Retrieval was per-turn and flat: every turn re-pulled KB + history + similar +
cohort and concatenated with no bound, so a chatty customer could bloat context
and DEGRADE reasoning. This assembles memory in TencentDB's layered order and
caps it:

  L3 persona          durable facts about the customer (customer_notes) +
                      value/tenure  -> always-on, cheap
  L2 latest scenario  the most recent conversation summary -> cheap bootstrap
  L1/L0 budgeted facts specific KB/history facts, pulled ONLY as needed and
                      capped by (item count + char budget + timeout)

Most turns are well served by L3+L2 alone; L1/L0 is the fallback for specific
facts. A hard budget keeps memory from overwhelming the prompt — which measurably
helps both reasoning quality and cost.

Everything read lives in the one CockroachDB cluster (customer_profile,
customer_notes, conversations, kb_chunks).

Runs offline: `python3 24_working_memory.py` assembles from a synthetic dict.
"""
from __future__ import annotations
import time
import json

# budget knobs (TencentDB caps: item count + char budget + timeout)
WM_MAX_ITEMS = 6
WM_CHAR_BUDGET = 1800
WM_TIMEOUT_S = 2.0


def _persona(conn, customer_id):
    from psycopg.rows import dict_row
    persona = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""SELECT tenure_days, total_orders, total_spent
                       FROM customer_profile WHERE customer_id = %s""", (customer_id,))
        p = cur.fetchone() or {}
        persona["tenure_days"] = p.get("tenure_days")
        persona["order_count"] = p.get("total_orders")
        persona["lifetime_value"] = float(p.get("total_spent") or 0)
        cur.execute("""SELECT note FROM customer_notes WHERE customer_id = %s
                       ORDER BY created_at DESC LIMIT 4""", (customer_id,))
        persona["notes"] = [r["note"] for r in cur.fetchall()]
    return persona


def _latest_scenario(conn, customer_id):
    from psycopg.rows import dict_row
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""SELECT summary, status, escalated, created_at
                       FROM conversations
                       WHERE customer_id = %s AND summary IS NOT NULL
                       ORDER BY created_at DESC LIMIT 1""", (customer_id,))
        r = cur.fetchone()
    return {"summary": r["summary"], "status": r.get("status"),
            "escalated": r.get("escalated")} if r else None
def to_vector_literal(vec):
    """CockroachDB's VECTOR type expects '[v1,v2,...]' syntax.
    psycopg's default list adapter produces Postgres array syntax
    '{v1,v2,...}' instead, which VECTOR rejects."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"



def _budgeted_facts(conn, query, embed_fn, remaining_chars, max_items):
    """L1/L0 fallback: only the specific facts the question needs, capped."""
    if not embed_fn:
        return []
    from psycopg.rows import dict_row
    emb = to_vector_literal(embed_fn(query))
    facts = []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""SELECT title, content, embedding <-> %s AS d
                       FROM kb_chunks ORDER BY embedding <-> %s LIMIT %s""",
                    (emb, emb, max_items))
        for row in cur.fetchall():
            snippet = (row["content"] or "")[:280]
            if len(snippet) > remaining_chars:
                break
            facts.append({"title": row["title"], "text": snippet,
                          "distance": round(float(row["d"]), 3)})
            remaining_chars -= len(snippet)
    return facts


def assemble(conn, customer_id, query, embed_fn=None,
             max_items=WM_MAX_ITEMS, char_budget=WM_CHAR_BUDGET, timeout_s=WM_TIMEOUT_S):
    """Return a bounded working-memory bundle + meta. Enforces item count,
    char budget, and a soft timeout (best-effort between layers)."""
    t0 = time.time()
    used_chars = 0
    persona = _persona(conn, customer_id) if customer_id else {}
    used_chars += len(json.dumps(persona, default=str))

    latest = None
    if time.time() - t0 < timeout_s:
        latest = _latest_scenario(conn, customer_id) if customer_id else None
        if latest:
            used_chars += len(latest.get("summary") or "")

    facts = []
    if time.time() - t0 < timeout_s and used_chars < char_budget:
        facts = _budgeted_facts(conn, query, embed_fn,
                                remaining_chars=char_budget - used_chars,
                                max_items=max_items - (1 if latest else 0))
        used_chars += sum(len(f["text"]) for f in facts)

    return {
        "persona": persona,
        "latest_scenario": latest,
        "budgeted_facts": facts,
        "meta": {"chars_used": used_chars, "char_budget": char_budget,
                 "items": len(facts) + (1 if latest else 0), "max_items": max_items,
                 "elapsed_ms": round((time.time() - t0) * 1000),
                 "hit_budget": used_chars >= char_budget},
    }


def to_prompt(wm):
    """Compact text header for the resolution prompt."""
    p = wm.get("persona") or {}
    lines = ["WORKING MEMORY (bounded):"]
    lines.append(f"- customer: tenure {p.get('tenure_days','?')}d, "
                 f"{p.get('order_count','?')} orders, "
                 f"${round(p.get('lifetime_value',0))} lifetime")
    for n in (p.get("notes") or [])[:3]:
        lines.append(f"- note: {n}")
    ls = wm.get("latest_scenario")
    if ls:
        lines.append(f"- last interaction: {ls.get('summary')}"
                     + (" (escalated)" if ls.get("escalated") else ""))
    for f in wm.get("budgeted_facts") or []:
        lines.append(f"- policy [{f['title']}]: {f['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # offline demo of to_prompt() + budget accounting on a synthetic bundle
    demo = {
        "persona": {"tenure_days": 2100, "order_count": 42, "lifetime_value": 9400,
                    "notes": ["prefers email", "had a damaged-item claim in March"]},
        "latest_scenario": {"summary": "asked about a late delivery, resolved with credit",
                            "status": "resolved", "escalated": False},
        "budgeted_facts": [{"title": "policy_refunds", "text": "Refunds for undelivered "
                            "orders are issued after the carrier confirms non-delivery.",
                            "distance": 0.21}],
        "meta": {"chars_used": 320, "char_budget": 1800, "items": 2, "hit_budget": False},
    }
    print(to_prompt(demo))
    print("\nmeta:", demo["meta"])

