"""
23_skills.py — the "Skill" memory asset, adapted from TencentDB Agent Memory.
Credit : https://github.com/TencentCloud/TencentDB-Agent-Memory
"""
from __future__ import annotations
import json
import re

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id        STRING PRIMARY KEY,
    name            STRING NOT NULL,
    version         INT NOT NULL DEFAULT 1,
    category        STRING,                 -- issue type this playbook serves
    trigger_predicate JSONB,                -- {"requires_tags":[...], "max_ceiling":"..."}
    trigger_text    STRING,                 -- natural-language "when to use this"
    trigger_embedding VECTOR(384),          -- semantic match vs the customer message
    steps           JSONB NOT NULL,         -- ordered execution steps
    validation_rules JSONB,                 -- guardrail checks specific to this playbook
    status          STRING DEFAULT 'active',-- active | draft | retired (governance)
    visibility      STRING DEFAULT 'team',  -- team | private (governance)
    updated_at      TIMESTAMPTZ DEFAULT now(),
    VECTOR INDEX (trigger_embedding)
);
"""

# --- seed library (also used as the offline fallback when the DB is empty) ---
SEED_SKILLS = [
    {
        "skill_id": "refund_late_delivery", "name": "Refund — late/undelivered", "version": 1,
        "category": "REFUND", "trigger_predicate": {},
        "trigger_text": "customer wants a refund because an order is late or never arrived",
        "steps": [
            "Confirm the order and its latest tracking status from case_state.",
            "Cite the refunds/shipping policy for undelivered orders.",
            "If within the autonomy ceiling, offer the eligible remedy; otherwise say a specialist will confirm.",
        ],
        "validation_rules": ["must_cite_kb", "no_dollar_amount_unless_in_case_state"],
    },
    {
        "skill_id": "damaged_item_claim", "name": "Damaged item", "version": 1,
        "category": "DAMAGED_ITEM", "trigger_predicate": {},
        "trigger_text": "item arrived broken or damaged, customer may have photos",
        "steps": [
            "Acknowledge the damage and ask for/confirm photo evidence.",
            "Cite the damaged-item policy and the replacement/refund options.",
            "Route to the photo-evidence agent if photos are provided.",
        ],
        "validation_rules": ["must_cite_kb"],
    },
    {
        "skill_id": "fraud_review_hold", "name": "Account under fraud review", "version": 2,
        "category": "FRAUD_REVIEW",
        "trigger_predicate": {"requires_tags": ["fraud_hold"], "max_ceiling": "escalate"},
        "trigger_text": "account is under fraud review and customer pushes for a refund or account change",
        "steps": [
            "Do NOT promise or claim any autonomous action.",
            "State the account is under specialist review and set expectations.",
            "Escalate to a human specialist.",
        ],
        "validation_rules": ["no_autonomous_action", "must_escalate"],
    },
    {
        "skill_id": "cancellation_window", "name": "Cancel within window", "version": 1,
        "category": "CANCELLATION", "trigger_predicate": {},
        "trigger_text": "customer wants to cancel an order they just placed",
        "steps": [
            "Check whether the order is still within the cancellation window (case_state).",
            "Cite the cancellation policy; if eligible and within ceiling, proceed, else propose.",
        ],
        "validation_rules": ["must_cite_kb"],
    },
    {
        "skill_id": "kb_gap_defer", "name": "No policy on file — defer", "version": 1,
        "category": "UNKNOWN", "trigger_predicate": {},
        "trigger_text": "question about loyalty points, gift cards, or price matching with no policy",
        "steps": [
            "Do NOT invent a policy.",
            "Acknowledge you don't have a policy on file and escalate to a specialist.",
            "This conversation feeds the KB gap-filling job.",
        ],
        "validation_rules": ["must_not_invent_policy", "must_escalate"],
    },
]


def _predicate_ok(pred, signals):
    """Deterministic gate: a skill only applies if its signal predicate holds."""
    pred = pred or {}
    tags = set((signals or {}).get("tags", []))
    for t in pred.get("requires_tags", []):
        if t not in tags:
            return False
    return True


def select_skill(conn, category, query, signals, embed_fn=None):
    """Pick the best playbook. Rule-filter by category + signal predicate, then
    (if a DB + embedder are available) vector-rank remaining candidates by
    trigger_embedding vs the message; offline, fall back to keyword overlap on
    trigger_text. Returns the skill dict (or the safe kb_gap_defer default)."""
    candidates = []
    if conn is not None:
        from psycopg.rows import dict_row
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM skills WHERE status='active' AND "
                        "(category = %s OR category = 'UNKNOWN')", (category,))
            candidates = cur.fetchall()
    if not candidates:  # offline / empty table -> seed library
        candidates = [s for s in SEED_SKILLS
                      if s["category"] == category or s["category"] == "UNKNOWN"]

    candidates = [c for c in candidates if _predicate_ok(c.get("trigger_predicate"), signals)]
    if not candidates:
        return next(s for s in SEED_SKILLS if s["skill_id"] == "kb_gap_defer")

    # rank: prefer a category-specific playbook; break ties semantically
    exact = [c for c in candidates if c.get("category") == category]
    pool = exact or candidates

    if conn is not None and embed_fn is not None and len(pool) > 1:
        from psycopg.rows import dict_row
        emb = embed_fn(query)
        ids = [c["skill_id"] for c in pool]
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""SELECT skill_id, trigger_embedding <-> %s AS d
                           FROM skills WHERE skill_id = ANY(%s)
                           ORDER BY d LIMIT 1""", (emb, ids))
            best = cur.fetchone()
        return next(c for c in pool if c["skill_id"] == best["skill_id"])

    # offline tie-break: keyword overlap with trigger_text
    q = set(re.findall(r"[a-z]+", query.lower()))
    pool.sort(key=lambda c: len(q & set(re.findall(r"[a-z]+", c["trigger_text"].lower()))),
              reverse=True)
    return pool[0]


def skill_for_prompt(skill):
    """Compact block the Resolution Agent gets in its prompt."""
    return ("PLAYBOOK (follow these steps): "
            + f"[{skill['skill_id']} v{skill['version']}] "
            + " | ".join(skill["steps"]))


def seed_into_db(conn, embed_fn):
    """One-time load of the seed library into CockroachDB, with embeddings."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        for s in SEED_SKILLS:
            emb = embed_fn(s["trigger_text"])
            cur.execute("""INSERT INTO skills
                (skill_id,name,version,category,trigger_predicate,trigger_text,
                 trigger_embedding,steps,validation_rules)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (skill_id) DO UPDATE SET version=excluded.version,
                 steps=excluded.steps, validation_rules=excluded.validation_rules,
                 trigger_embedding=excluded.trigger_embedding, updated_at=now()""",
                (s["skill_id"], s["name"], s["version"], s["category"],
                 json.dumps(s["trigger_predicate"]), s["trigger_text"], emb,
                 json.dumps(s["steps"]), json.dumps(s["validation_rules"])))
    conn.commit()
    print(f"seeded {len(SEED_SKILLS)} skills")


# GOVERNANCE — draft -> active review flow.
# New playbooks are never auto-activated. The KB-gap job proposes a `draft`
# (private); a human reviews and promotes it to `active` before select_skill
# will ever equip it. This mirrors TencentDB's "skills extracted from completed
# tasks, shared only after review" — the safe default for a support agent.
# select_skill() already filters `status='active'`, so drafts can't leak into
# live turns until promoted.

def propose_skill_from_cluster(conn, cluster_summary, category, embed_fn, llm=None):
    """Draft a NEW playbook from a cluster of recurring escalations that no
    existing skill/KB covers. Uses Bedrock (Llama) to turn the cluster into
    steps + validation, inserts as status='draft', visibility='private'.
    Returns the draft skill_id. Human-gated — never activated here."""
    import hashlib
    skill_id = "draft_" + hashlib.sha1((category + cluster_summary[:120]).encode()).hexdigest()[:8]
    if llm is None:
        from backend.utilis.bedrock_client import get_llm
        llm = get_llm()
    system = ("You write internal support RESOLUTION PLAYBOOKS. Given a cluster "
              "of recurring escalations with no covering policy, output a JSON "
              "object: {\"name\": str, \"trigger_text\": str, \"steps\": [str,...], "
              "\"validation_rules\": [str,...]}. Steps must be safe defaults and "
              "must NOT invent specific policy numbers.")
    out = llm.complete_json(system, f"Category: {category}\nCluster:\n{cluster_summary[:1200]}",
                            schema_hint={"name": str, "trigger_text": str,
                                         "steps": list, "validation_rules": list},
                            tag="skill_proposal").get("json", {})
    name = out.get("name") or f"Draft playbook — {category}"
    trigger_text = out.get("trigger_text") or cluster_summary[:200]
    steps = out.get("steps") or ["Acknowledge, gather details, and escalate to a specialist."]
    rules = out.get("validation_rules") or ["must_escalate", "must_not_invent_policy"]

    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute("""INSERT INTO skills
            (skill_id,name,version,category,trigger_predicate,trigger_text,
             trigger_embedding,steps,validation_rules,status,visibility)
            VALUES (%s,%s,1,%s,%s,%s,%s,%s,%s,'draft','private')
            ON CONFLICT (skill_id) DO NOTHING""",
            (skill_id, name, category, json.dumps({}), trigger_text,
             embed_fn(trigger_text), json.dumps(steps), json.dumps(rules)))
    conn.commit()
    print(f"proposed DRAFT skill {skill_id} ({name}) — awaiting human review")
    return skill_id


def list_skills(conn, status=None):
    from psycopg.rows import dict_row
    q = "SELECT skill_id,name,version,category,status,visibility,updated_at FROM skills"
    params = ()
    if status:
        q += " WHERE status = %s"
        params = (status,)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(q + " ORDER BY status, updated_at DESC", params)
        return cur.fetchall()


def promote_skill(conn, skill_id, reviewer="human_reviewer", visibility="team"):
    """The review action: draft -> active. Bumps version so the change is
    traceable. Only after this can select_skill() equip the playbook."""
    with conn.cursor() as cur:
        cur.execute("""UPDATE skills SET status='active', visibility=%s,
                       version = version + 1, updated_at = now()
                       WHERE skill_id = %s AND status = 'draft'""",
                    (visibility, skill_id))
        n = cur.rowcount
    conn.commit()
    print(f"promoted {skill_id} -> active (by {reviewer})" if n else f"{skill_id} not a draft / not found")
    return n > 0


def retire_skill(conn, skill_id):
    with conn.cursor() as cur:
        cur.execute("UPDATE skills SET status='retired', updated_at=now() WHERE skill_id=%s", (skill_id,))
    conn.commit()


if __name__ == "__main__":
    # offline: select a playbook for a few messages / signal profiles
    trials = [
        ("REFUND", "my order never arrived, I want my money back", {"tags": ["standard"]}),
        ("FRAUD_REVIEW", "just process my refund now", {"tags": ["fraud_hold"]}),
        ("DAMAGED_ITEM", "it came smashed, I have pictures", {"tags": ["high_value"]}),
        ("UNKNOWN", "do my loyalty points expire?", {"tags": ["cold_start"]}),
    ]
    for cat, msg, sig in trials:
        sk = select_skill(None, cat, msg, sig)
        print(f"\n[{cat}] \"{msg}\"  tags={sig['tags']}")
        print(f"  -> {sk['skill_id']} v{sk['version']}  validation={sk['validation_rules']}")
        print(f"  {skill_for_prompt(sk)}")

