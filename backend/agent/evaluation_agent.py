"""
Evaluation Agent -- LLM Judge, 5 named dimensions (not one blended score),
plus a calibration loop against human-labeled samples.
"""
import os
import sys
import json
from dotenv import load_dotenv
try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    sys.exit("Run: pip install 'psycopg[binary]' --break-system-packages")
import urllib.parse
def _with_query_param(url, key, value):
    parts = urllib.parse.urlsplit(url)
    params = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(params), parts.fragment))


load_dotenv()

CRDB_URL = os.environ.get("CRDB_URL")
if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)

JUDGE_PROMPT_TEMPLATE = """You are evaluating a customer support conversation turn. \
Score each dimension as pass/fail with brief reasoning.

Conversation history (prior turns, oldest first):
{history}

Current customer query: {query}
Case state (agent's synthesized memory at time of response): {case_state}
KB / retrieval chunks used: {kb_chunks}
Agent's response: {response}

Score these 5 dimensions independently:
1. retrieval_correctness  -- were the right KB chunks / history retrieved for this query?
2. response_accuracy      -- does the response correctly reflect what was retrieved?
3. grammar_language_ok    -- is the response grammatically correct and in the customer's language?
4. coherence_to_context   -- does the response make sense given the full conversation so far?
5. relevance_to_request   -- does the response actually address what the customer asked?

Return ONLY valid JSON with no markdown fences:
{{"retrieval_correctness": bool, "response_accuracy": bool, \
"grammar_language_ok": bool, "coherence_to_context": bool, \
"relevance_to_request": bool, "reasoning": "one short paragraph"}}
"""


def call_llm_judge(query, case_state, kb_chunks, response, history=""):
    """Real AWS Bedrock judge. Returns the 5 named boolean dimensions
    (never one blended score) plus reasoning, so calibration can track
    precision/recall/F1 per dimension. Degrades gracefully if Bedrock
    is unreachable — returns None booleans so the UI shows '—' chips."""
    from backend.utilis.bedrock_client import get_llm
    llm = get_llm()
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        history=history or "(first turn — no prior context)",
        query=query,
        case_state=json.dumps(case_state, default=str)[:2000],  # guard token budget
        kb_chunks=kb_chunks or "(none recorded)",
        response=response,
    )
    try:
        raw = llm.complete_json(
            "You are a strict evaluation judge for e-commerce support replies. "
            "Return only the JSON object requested — no extra keys, no markdown.",
            prompt,
            schema_hint={"retrieval_correctness": bool, "response_accuracy": bool,
                         "grammar_language_ok": bool, "coherence_to_context": bool,
                         "relevance_to_request": bool, "reasoning": str},
        )
        out = raw.get("json") or raw  # some bedrock_client versions return the dict directly
    except Exception as exc:
        return {
            "retrieval_correctness": None, "response_accuracy": None,
            "grammar_language_ok": None, "coherence_to_context": None,
            "relevance_to_request": None,
            "reasoning": f"[judge call failed: {exc}]",
            "judge_model": getattr(llm, "model_id", "unknown"),
            "_prompt_used": prompt,
        }
    return {
        "retrieval_correctness": _as_bool(out.get("retrieval_correctness")),
        "response_accuracy":     _as_bool(out.get("response_accuracy")),
        "grammar_language_ok":   _as_bool(out.get("grammar_language_ok")),
        "coherence_to_context":  _as_bool(out.get("coherence_to_context")),
        "relevance_to_request":  _as_bool(out.get("relevance_to_request")),
        "reasoning":  out.get("reasoning") or "[no reasoning returned]",
        "judge_model": getattr(llm, "model_id", "unknown"),
        "_prompt_used": prompt,
    }


def _as_bool(val):
    """Coerce LLM output to True/False/None — handles string 'true'/'false' too."""
    if val is None: 
        return None
    if isinstance(val, bool): 
        return val
    if isinstance(val, str): 
        return val.strip().lower() == "true"
    return bool(val)



def evaluate_conversation(conn, conversation_id, judge_model=None, is_simulated=False):
    """Pull real conversation data from CRDB, call the LLM judge, persist result."""
    with conn.cursor(row_factory=dict_row) as cur:
        # All messages in order — used for history + extracting last query/response
        cur.execute(
            "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY created_at ASC",
            (conversation_id,),
        )
        all_msgs = cur.fetchall()

        # Last agent case_state snapshot (contains retrieval.kb if stored there)
        cur.execute(
            "SELECT state FROM case_state WHERE conversation_id = %s ORDER BY turn_number DESC LIMIT 1",
            (conversation_id,),
        )
        case_row = cur.fetchone()
        case_state = case_row["state"] if case_row else {}

        # KB chunk titles from case_state.retrieval.kb (preferred) or tool_calls fallback
        kb_chunks = _extract_kb_chunks(case_state, cur, conversation_id)

    # Derive query and response from messages
    customer_msgs = [m for m in all_msgs if m["role"] == "customer"]
    agent_msgs    = [m for m in all_msgs if m["role"] == "agent"]
    query    = customer_msgs[-1]["content"] if customer_msgs else "[no customer message found]"
    response = agent_msgs[-1]["content"]    if agent_msgs    else "[no agent response found]"

    # Prior conversation turns (everything except the last customer + agent pair)
    prior = all_msgs[:-2] if len(all_msgs) >= 2 else []
    history = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in prior) or ""

    verdict = call_llm_judge(
        query=query, case_state=case_state,
        kb_chunks=kb_chunks, response=response, history=history,
    )

    # judge_model: prefer what the LLM returned, fall back to param or "unknown"
    resolved_model = verdict.pop("judge_model", None) or judge_model or "unknown"

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO evaluation_results
               (conversation_id, retrieval_correctness, response_accuracy,
                grammar_language_ok, coherence_to_context, relevance_to_request,
                judge_reasoning, judge_model, is_simulated)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT DO NOTHING""",
            (conversation_id,
             verdict["retrieval_correctness"], verdict["response_accuracy"],
             verdict["grammar_language_ok"],   verdict["coherence_to_context"],
             verdict["relevance_to_request"],  verdict["reasoning"],
             resolved_model, is_simulated),
        )
    conn.commit()
    verdict["judge_model"] = resolved_model
    return verdict


def _extract_kb_chunks(case_state, cur, conversation_id):
    """Try case_state.retrieval.kb first, fall back to tool_calls rows."""
    if isinstance(case_state, dict):
        retr = case_state.get("retrieval") or {}
        kb = retr.get("kb") or []
        if kb:
            return "; ".join(c.get("title", str(c)) for c in kb[:6])
    # fallback: any tool_call with 'kb' in the name
    try:
        cur.execute(
            "SELECT tool_name, output FROM tool_calls "
            "WHERE conversation_id = %s AND tool_name ILIKE %s ORDER BY created_at ASC LIMIT 6",
            (conversation_id, "%kb%"),
        )
        rows = cur.fetchall()
        if rows:
            return "; ".join(str(r["tool_name"]) for r in rows)
    except Exception:
        pass
    return "(no KB chunks recorded)"


# ---------------------------------------------------------------------------
# Calibration: sample recent evaluations, compare judge verdict to a human
# label, compute precision/recall/F1 per dimension. Run this periodically
# (weekly) against a held-out human-reviewed sample -- NOT on every request.
# ---------------------------------------------------------------------------
def run_calibration(conn, human_labels, dimension):
    """
    TODO: Setup 
    human_labels: list of {"conversation_id": ..., "human_label": bool}
    Compares against the judge's stored verdict for the same dimension.
    """
    tp = fp = tn = fn = 0
    mismatches = []
    with conn.cursor(row_factory=dict_row) as cur:
        for hl in human_labels:
            cur.execute(
                f"SELECT {dimension}, judge_reasoning FROM evaluation_results "
                f"WHERE conversation_id = %s ORDER BY created_at DESC LIMIT 1",
                (hl["conversation_id"],),
            )
            row = cur.fetchone()
            if not row or row[dimension] is None:
                continue
            judge_val, human_val = row[dimension], hl["human_label"]
            if judge_val and human_val: 
                tp += 1
            elif judge_val and not human_val: 
                fp += 1
            elif not judge_val and not human_val: 
                tn += 1
            else: 
                fn += 1
            if judge_val != human_val:
                mismatches.append({"conversation_id": hl["conversation_id"],
                                    "judge": judge_val, "human": human_val,
                                    "judge_reasoning": row["judge_reasoning"]})

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO judge_calibration_runs
               (dimension, precision, recall, f1, sample_size, human_labels)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (dimension, precision, recall, f1, len(human_labels), json.dumps(mismatches)),
        )
    conn.commit()
    return {"dimension": dimension, "precision": precision, "recall": recall,
            "f1": f1, "n_mismatches": len(mismatches), "mismatches": mismatches}


if __name__ == "__main__":
    if not CRDB_URL:
        sys.exit("Set CRDB_URL env var first.")
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 12_evaluation_agent.py <conversation_id>")

    with psycopg.connect(CRDB_URL) as conn:
        result = evaluate_conversation(conn, sys.argv[1])
        print(json.dumps(result, indent=2, default=str))
