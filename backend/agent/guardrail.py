"""
Two-tier guardrail.

Tier 1: cheap, no model call. Pure SQL vector-distance + rule checks.
Tier 2: LLM evaluator, invoked ONLY when tier 1 fails. This is what keeps
        latency/cost bounded -- most responses never reach tier 2.
"""
import os
import sys
import re
import json
import hashlib
import time
from dotenv import load_dotenv
try:
    from psycopg.rows import dict_row
except ImportError:
    sys.exit("Run: pip install 'psycopg[binary]' --break-system-packages")
from sentence_transformers import SentenceTransformer

load_dotenv()  # Injects variables from .env into os.environ
import urllib.parse  # noqa: E402
def _with_query_param(url, key, value):
    parts = urllib.parse.urlsplit(url)
    params = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(params), parts.fragment))


def _get_crdb_url():
    url = os.environ.get("CRDB_URL") or os.environ.get("COCKROACH_URL")
    if url:
        return url

    try:
        import subprocess
        url = subprocess.check_output(
            ["ccloud", "cluster", "sql", "autosupport", "--database", "defaultdb", "--connection-url"],
            text=True,
        ).strip()
        if url:
            return url
    except Exception:
        pass
    return None

import psycopg  # noqa: E402
CRDB_URL = _get_crdb_url()

if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)
    
SCHEMA_ADDITIONS = """
CREATE TABLE IF NOT EXISTS guardrail_verdict_cache (
    signature   STRING PRIMARY KEY,
    verdict     JSONB NOT NULL,
    hit_count   INT DEFAULT 1,
    created_at  TIMESTAMPTZ DEFAULT now()
);
"""
CACHE_TTL_SECONDS = 3600  
TIER1_DISTANCE_THRESHOLD = 0.35


# very simple language-consistency check -- production version should use
# a real language-detection library (e.g. langdetect, fasttext lid.176)
def detect_language(text):
    """
    TODO: replace with a real language detection library. 
    """
    if re.search(r"[¿¡áéíóúñ]", text.lower()):
        return "es"
    return "en"
def to_vector_literal(vec):
    """CockroachDB's VECTOR type expects '[v1,v2,...]' syntax.
    psycopg's default list adapter produces Postgres array syntax
    '{v1,v2,...}' instead, which VECTOR rejects."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"



def tier1_groundedness_check(conn, response_text, cited_kb_chunk_ids, response_embedding_fn, escalated=False):
    """Is the response's meaning close to the KB chunks it's supposedly
    grounded in? Pure SQL vector distance, no model call."""
    print(f"tier1_groundedness_check: escalated={escalated}")
    if escalated:
        return {"pass": False, "reason": "escalated to human"}
    if not cited_kb_chunk_ids:
        return {"pass": False, "reason": "no_kb_chunks_cited"}

    response_emb = response_embedding_fn(response_text)
    # print(f"response_emb: {response_emb}")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, embedding <-> %s AS distance
            FROM kb_chunks WHERE id = ANY(%s)
            ORDER BY distance LIMIT 1""",
            (to_vector_literal(response_emb), cited_kb_chunk_ids),
        )
        row = cur.fetchone()
    print(row["distance"])
    if not row:
        return {"pass": False, "reason": "cited_chunks_not_found"}
    if row["distance"] < TIER1_DISTANCE_THRESHOLD:
        return {"pass": False, "reason": "ungrounded", "distance": row["distance"]}
    return {"pass": True, "distance": row["distance"]}


def tool_text_consistency_check(conn, conversation_id, response_text):
    """Cross-check any $ amount stated in the response against what
    tool_calls actually returned for this conversation. Pure SQL/regex,
    no model call. Kept as a fallback when verified_facts is unavailable."""
    dollar_amounts = re.findall(r"\$(\d+(?:\.\d{1,2})?)", response_text)
    if not dollar_amounts:
        return {"pass": True, "reason": "no_dollar_amount_claimed"}

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT output FROM tool_calls
               WHERE conversation_id = %s AND tool_name IN ('issue_refund', 'check_refund_eligibility', 'build_customer_signals')
               ORDER BY created_at DESC LIMIT 1""",
            (conversation_id,),
        )
        row = cur.fetchone()

    if not row:
        return {"pass": False, "reason": "dollar_amount_claimed_with_no_tool_call"}

    actual_amount = (row["output"] or {}).get("refund_amount") or (row["output"] or {}).get("amount")
    claimed = float(dollar_amounts[0])
    if actual_amount is not None and abs(claimed - float(actual_amount)) > 0.01:
        return {"pass": False, "reason": "tool_text_mismatch",
                "claimed": claimed, "actual": actual_amount}
    return {"pass": True}


# Status phrases that would contradict a known DB order status.
# Key = actual DB status; value = list of response phrases that would be false.
_STATUS_CONTRADICTIONS = {
    "delivered": [
        "not yet delivered", "hasn't arrived", "not arrived",
        "still in transit", "still being processed", "being prepared",
        "hasn't been shipped", "not yet shipped",
    ],
    "processing": [
        "has shipped", "has been shipped", "on its way",
        "out for delivery", "already delivered", "was delivered",
        "will arrive tomorrow",
    ],
    "shipped": [
        "not yet shipped", "hasn't shipped", "still processing",
        "already delivered", "waiting to be shipped",
    ],
    "cancelled": [
        "your order is on its way", "your order will arrive",
        "is being processed", "will be delivered",
    ],
}


def db_fact_consistency_check(response_text: str, verified_facts: dict) -> dict:
    """Validate factual claims in the draft response against the verified DB
    snapshot that case_state_builder fetched this turn.

    Checks:
    1. Dollar amounts — every $X.XX claimed must be within $0.02 of a known
       order total, refund amount, or payment amount.
    2. Order status — response must not describe an order as being in a state
       that contradicts what the DB says (e.g. "your order has shipped" when
       DB status = "processing").
    3. Fraud-hold gate — if fraud_hold=True in the DB, agent must not claim a
       refund has been issued/processed (that action requires human review).

    Returns {"pass": bool, "failures": [...], "reason": str|None}
    """
    if not verified_facts:
        return {"pass": True, "failures": [], "reason": None}

    rt = response_text.lower()
    failures = []

    # ── 1. Dollar amount validation ─────────────────────────────────────────
    claimed_amounts = [
        round(float(a), 2)
        for a in re.findall(r"\$(\d+(?:\.\d{1,2})?)", response_text)
    ]
    if claimed_amounts:
        known_amounts: set = set()
        for o in (verified_facts.get("recent_orders") or []):
            if o.get("total"):        
                known_amounts.add(round(float(o["total"]), 2))
            if o.get("refund_amount"): 
                known_amounts.add(round(float(o["refund_amount"]), 2))
        od = verified_facts.get("order_detail") or {}
        if od.get("total"):          
            known_amounts.add(round(float(od["total"]), 2))
        if od.get("refund_amount"):  
            known_amounts.add(round(float(od["refund_amount"]), 2))
        for r in (verified_facts.get("open_refunds") or []):
            if r.get("amount"):      
                known_amounts.add(round(float(r["amount"]), 2))

        if known_amounts:   # only fail if we actually have data to compare against
            for amt in claimed_amounts:
                if amt > 0.99 and not any(abs(amt - k) < 0.03 for k in known_amounts):
                    failures.append(f"amount_not_in_db:${amt}")

    # ── 2. Order status contradiction ───────────────────────────────────────
    od = verified_facts.get("order_detail") or {}
    actual_status = (od.get("status") or "").lower().strip()
    if actual_status and actual_status in _STATUS_CONTRADICTIONS:
        for phrase in _STATUS_CONTRADICTIONS[actual_status]:
            if phrase in rt:
                label = phrase.replace(" ", "_")[:45]
                failures.append(f"status_mismatch:db={actual_status},claimed={label}")

    # ── 3. Fraud-hold gate ───────────────────────────────────────────────────
    if verified_facts.get("fraud_hold"):
        action_phrases = [
            "refund of $", "your refund has been", "i've issued",
            "has been processed", "i have issued", "refund was issued",
            "issued a refund", "processed your refund",
        ]
        if any(p in rt for p in action_phrases):
            failures.append("fraud_hold_but_refund_action_claimed")

    return {
        "pass": not failures,
        "failures": failures,
        "reason": "; ".join(failures) if failures else None,
    }


def language_consistency_check(customer_message, response_text):
    return {"pass": detect_language(customer_message) == detect_language(response_text)}


def autonomy_enforcement_check(response_text, autonomy):
    """The response must not exceed the DETERMINISTIC autonomy ceiling that
    customer_signals.autonomy_ceiling() computed from the signal profile.
    This replaces the old fraud-only check and generalizes it: it doesn't
    matter WHY the ceiling was raised (fraud hold, high refund rate, amount
    over the manual-review line) — the response simply may not claim an
    action beyond what it's allowed to.
        escalate     -> must not claim/promise ANY action was taken
        propose_only -> must not say the action is already done/processed
        act          -> no restriction here
    """
    ceiling = (autonomy or {}).get("ceiling", "act")
    t = response_text.lower()
    done_phrases = [
        "has been processed", "have processed", "i've processed", "i have processed",
        "we've updated your address", "payment method has been changed",
        "refund of $", "your refund of", "i've issued", "has been issued",
        "i've refunded", "i have refunded", "is confirmed", "done for you",
        "successfully cancelled", "successfully canceled",
    ]
    claims_done = any(p in t for p in done_phrases)
    if ceiling in ("escalate", "propose_only") and claims_done:
        return {"pass": False,
                "reason": f"action_claimed_above_{ceiling}_ceiling"}
    return {"pass": True}


def build_cache_signature(ceiling, category, cited_kb_chunk_ids, failure_reasons):
    """Order-independent signature — caches on WHAT determines the verdict
    (autonomy ceiling, category, KB chunks involved, failure pattern), not on
    the exact response wording, since paraphrases of the same failure should
    hit the same cached verdict. Uses the autonomy ceiling in place of the old
    coarse segment: the ceiling is the decision-relevant summary of the
    customer's risk posture."""
    key = json.dumps({
        "ceiling": ceiling, "category": category,
        "chunks": sorted(str(x) for x in cited_kb_chunk_ids), "reasons": sorted(failure_reasons),
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def get_cached_verdict(conn, signature):
    if conn is None:
        return None
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT verdict, created_at FROM guardrail_verdict_cache WHERE signature = %s",
            (signature,),
        )
        row = cur.fetchone()
    if not row:
        return None
    age = time.time() - row["created_at"].timestamp()
    if age > CACHE_TTL_SECONDS:
        return None  # stale -- fall through to a real tier-2 call
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE guardrail_verdict_cache SET hit_count = hit_count + 1 WHERE signature = %s",
            (signature,),
        )
    conn.commit()
    return row["verdict"]


def store_verdict_cache(conn, signature, verdict):
    if conn is None:
        return
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO guardrail_verdict_cache (signature, verdict)
               VALUES (%s, %s) ON CONFLICT (signature) DO UPDATE SET verdict = %s, created_at = now()""",
            (signature, json.dumps(verdict), json.dumps(verdict)),
        )
    conn.commit()


TIER2_SYSTEM = """You are an adversarial safety evaluator for e-commerce
support replies. A cheap tier-1 check already FAILED on this response, so
your job is not to rubber-stamp it — actively try to find the strongest
reason it is wrong, ungrounded, non-compliant, or unsafe to send.

Return a JSON object with keys:
  pass                 (bool) — true ONLY if, despite the tier-1 flag, the
                         response is actually safe and grounded to send.
  reasoning            (str)  — a human-readable "why", shown to a human
                         reviewer in the UI's "Why?" panel. Never empty.
  adversarial_critique (str)  — the single strongest argument that this
                         response should NOT be sent.
Fail closed: when genuinely uncertain, set pass=false."""


def tier2_llm_evaluator(response_text, context, failure_reasons):
    """
    TODO: replace with a real Bedrock/Claude call. Prompt should ask the
    model to judge groundedness/coherence/compliance given `context`
    (the case_state + cited KB chunks) and return a structured verdict.
    Placeholder here returns a conservative "fail closed" result so the
    pipeline can be tested end-to-end before wiring a real model call.
    """
    """Real AWS Bedrock (Claude) evaluator, invoked ONLY after a tier-1
    failure. Combines two winner patterns the spec calls out: Globot's
    adversarial-critique step (try to break the response, don't just score
    it) and the Aegis/AgentGuard "Glass Box" rule (every escalation carries
    a legible reason, never a bare boolean). Degrades to a conservative
    fail-closed verdict if Bedrock is unreachable — a support decision must
    never silently pass because the evaluator was down."""
    from backend.utilis.bedrock_client import get_llm
    # from gemini_client import get_llm
    user = (
        f"TIER-1 FAILURES: {failure_reasons}\n"
        f"CONTEXT: {json.dumps(context, default=str)[:1500]}\n\n"
        f"RESPONSE UNDER REVIEW:\n{response_text}\n\n"
        f"Evaluate now."
    )
    out = get_llm().complete_json(
        TIER2_SYSTEM, user,
        schema_hint={"pass": bool, "reasoning": str, "adversarial_critique": str},
        tag="guardrail_tier2",
    ).get("json", {})
    return {
        "pass": bool(out.get("pass", False)) and not out.get("_parse_error"),
        "reasoning": out.get("reasoning")
                     or f"Tier-1 failed on {failure_reasons}; evaluator returned no reason — escalating.",
        "adversarial_critique": out.get("adversarial_critique", ""),
    }


def run_guardrail(conn=None, response_text="", cited_kb_chunk_ids=None,
                   autonomy=None, signals=None, conversation_id=None,
                   customer_message="", embed_fn=None, category="unknown",
                   escalated=False, verified_facts=None):
    cited_kb_chunk_ids = cited_kb_chunk_ids or []
    autonomy = autonomy or {}
    failures = []

    autonomy_check = autonomy_enforcement_check(response_text, autonomy)
    if not autonomy_check["pass"]:
        failures.append(autonomy_check["reason"])

    if conn and embed_fn:
        ground_check = tier1_groundedness_check(conn, response_text, cited_kb_chunk_ids, embed_fn, escalated=escalated)
        print(f"tier1_groundedness_check: {ground_check}")
        if not ground_check["pass"]:
            failures.append(ground_check["reason"])

        if conversation_id:
            if verified_facts:
                # Prefer DB fact check (more authoritative) over tool_call lookup
                db_check = db_fact_consistency_check(response_text, verified_facts)
                if not db_check["pass"]:
                    for f in db_check["failures"]:
                        failures.append(f)
            else:
                # Fallback: cross-check against tool_call outputs
                consistency = tool_text_consistency_check(conn, conversation_id, response_text)
                if not consistency["pass"]:
                    failures.append(consistency["reason"])

    if customer_message:
        lang_check = language_consistency_check(customer_message, response_text)
        if not lang_check["pass"]:
            failures.append("language_mismatch")

    # Globot-style "red-team every (high-stakes) decision": if the agent is
    # about to act AUTONOMOUSLY on money and claims it's done, force a tier-2
    # adversarial critique even though tier-1 passed. Scoped to money actions
    # so the extra LLM call only fires where the downside is real.
    _ceiling = (autonomy or {}).get("ceiling", "act")
    _rt = response_text.lower()
    _money_claim = ("refund of $" in _rt or "your refund of" in _rt
                    or "has been processed" in _rt or "i've issued" in _rt
                    or "i've refunded" in _rt)
    if not failures and _ceiling == "act" and _money_claim:
        failures.append("high_stakes_autonomous_money_action")

    if not failures:
        return {"tier": 1, "action": "respond", "reason": ", ".join(failures), "reason_detail": None}
    print(failures)
    # tier 1 failed -> check cache before paying for a real LLM call
    ceiling = autonomy.get("ceiling", "act")
    signature = build_cache_signature(ceiling, category, cited_kb_chunk_ids, failures)
    cached = get_cached_verdict(conn, signature) if conn else None
    if cached is not None:
        tier2 = cached
        tier2["_cache_hit"] = True
    else:
        context = {"autonomy": autonomy, "signal_tags": (signals or {}).get("tags")}
        tier2 = tier2_llm_evaluator(response_text, context=context, failure_reasons=failures)
        tier2["_cache_hit"] = False
        if conn:
            store_verdict_cache(conn, signature, {k: v for k, v in tier2.items() if k != "_cache_hit"})
    #tag_for_final
    if tier2["pass"] and escalated:
        return {"tier": 2, "action": "respond", "reason": tier2['reasoning'], "reason_detail": None,
                "cache_hit": tier2["_cache_hit"]}

    if tier2["pass"]:
        return {"tier": 2, "action": "respond", "reason": None, "reason_detail": None,
                "cache_hit": tier2["_cache_hit"]}

    return {
        "tier": 2, "action": "escalate", "reason": ", ".join(failures),
        "reason_detail": tier2.get("reasoning"),  # human-readable, shown in UI's "Why?" expander
        "adversarial_critique": tier2.get("adversarial_critique"),
        "cache_hit": tier2["_cache_hit"],
    }


if __name__ == "__main__":
    # standalone test against your validation_test_suite.json cases C, E, F
    with open("4_validcase_check_data/validation_test_suite.json") as f:
        tests = json.load(f)
    print("Loading embedding model (all-MiniLM-L6-v2, 384 dims)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_fn(text: str):
        vec = model.encode(text, normalize_embeddings=True)
        return to_vector_literal(vec)

    print("NOTE: run without CRDB_URL, this only exercises the checks that "
          "don't need a live DB (fraud-autonomy, language). Groundedness "
          "(case C) and tool/text consistency (case E) need kb_chunks/"
          "tool_calls and will silently no-op here -- pass a real `conn` "
          "and `embed_fn` to actually validate those two.\n")

    for t in tests:
        if t["group"] not in ("C_guardrail", "E_consistency", "F_language"):
            continue
        response = t.get("injected_agent_response_for_test", "")
        # autonomy ceiling for the case (C/E/F test cases exercise
        # groundedness/consistency/language, so 'act' is the right default)
        autonomy = t.get("expected", {}).get("autonomy") \
            if isinstance(t.get("expected"), dict) else None
        if not isinstance(autonomy, dict):
            autonomy = {"ceiling": "act"}
        with psycopg.connect(CRDB_URL) as conn:
            result = run_guardrail(conn = conn,
                response_text=response,
                autonomy=autonomy,
                customer_message=t["query"],embed_fn = embed_fn
            )
        needs_db_note = " [needs live DB to actually test]" if t["group"] in ("C_guardrail", "E_consistency") else ""
        print(f"{t['test_id']}: {result}{needs_db_note}")


