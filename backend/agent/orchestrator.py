"""
Orchestration.
"""
import re
import os
import sys
import json
import uuid
from typing import TypedDict, Optional
import urllib.parse
from dotenv import load_dotenv

try:
    import psycopg
except ImportError:
    sys.exit("Run: pip install 'psycopg[binary]' --break-system-packages")
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    sys.exit("Run: pip install langgraph langgraph-checkpoint-postgres")
sys.path.insert(0, os.path.dirname(__file__))

from backend.agent.retrieval_agent import RetrievalAgent
from backend.agent.case_state_builder import build_case_state, log_tool_call
import backend.agent.guardrail as guardrail_mod
import backend.agent.evaluation_agent as eval_mod
import backend.agent.skills as skills_mod
import backend.agent.working_memory as wmem_mod
import backend.agent.input_firewall as firewall_mod
from backend.utilis.crdb_checkpointer import CRDBCheckpointSaver
import time
from psycopg import errors as pg_errors

load_dotenv()
def _with_query_param(url, key, value):
    parts = urllib.parse.urlsplit(url)
    params = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(params), parts.fragment))

CRDB_URL = os.environ.get("CRDB_URL")
if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)



# Conversation lifecycle -- answers "where does the conversations row come
# from". Call this ONCE per new chat session, before anything else touches
# tool_calls or case_state.

def create_conversation(conn, customer_id, ticket_id=None, max_retries=5):
    conv_id = str(uuid.uuid4())

    for attempt in range(max_retries):
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (id, customer_id, ticket_id, status) "
                    "VALUES (%s, %s, %s, 'open')",
                    (conv_id, customer_id, ticket_id),
                )
            conn.commit()
            return conv_id

        except pg_errors.SerializationFailure:
            conn.rollback()  # required before retrying -- txn is dead, must clear it
            if attempt == max_retries - 1:
                raise
            backoff = min(0.1 * (2 ** attempt), 2.0)
            time.sleep(backoff)

    # unreachable, but keeps linters happy
    raise RuntimeError("create_conversation: exhausted retries")


def update_conversation_summary(conn, conversation_id, summary_text, embed_fn):
    """Called at conversation close (or periodically) -- refreshes the
    rolling summary_embedding that search_individual_history() queries
    against for THIS customer's future conversations."""
    emb = embed_fn(summary_text)
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE conversations SET summary = %s, summary_embedding = %s
               WHERE id = %s""",
            (summary_text, emb, conversation_id),
        )
    conn.commit()

def log_message(conn, conversation_id, role, content):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (conversation_id, role, content),
        )
    conn.commit()


# Graph state
class SupportState(TypedDict):
    conversation_id: str
    customer_id: str
    query: str
    signals: Optional[dict]
    plan: Optional[dict]
    autonomy: Optional[dict]
    skill: Optional[dict]
    working_memory: Optional[dict]
    firewall: Optional[dict]
    evaluation: Optional[dict]
    retrieval: Optional[dict]
    case_state: Optional[dict]
    draft_response: Optional[str]
    response_source: Optional[str]
    guardrail_result: Optional[dict]
    final_response: Optional[str]
    escalated: bool
    is_simulated: bool

# Nodes -- each is a plain function over SupportState. Kept deliberately
# thin here; the point of this file is the WIRING (checkpointer, case
# state, conversation lifecycle), not agent prompt engineering.
def make_planner_node(conn):
    def planner(state: SupportState) -> SupportState:
        # No segment lookup. The customer's signal profile is computed in the
        # retrieval node (which needs it to choose sources); the planner just
        # opens the turn.
        if not state.get("is_simulated"):
            log_message(conn, state["conversation_id"], "customer", state["query"])
        # INPUT firewall (AgentGuard-style) before anything reasons over the msg.
        import os as _os
        state["firewall"] = firewall_mod.check_input(
            state["query"], use_llm=_os.environ.get("FIREWALL_LLM", "0") == "1")
        return state
    return planner


def make_retrieval_node(conn):
    #TODO: add based on qdrant 
    agent = RetrievalAgent(conn)

    def retrieval(state: SupportState) -> SupportState:
        t0 = time.time()
        r = agent.retrieve(state["customer_id"], state["query"])
        latency_ms = int((time.time() - t0) * 1000)

        state["retrieval"] = r
        state["signals"] = r.get("signals")   # surfaced for resolution/guardrail/trace
        state["plan"] = r.get("plan")

        # audit trail: this is WHERE the signal profile gets created
        # (build_signals() runs inside RetrievalAgent.retrieve()). Logging it
        # here means every turn's signal snapshot is traceable later --
        # e.g. "did this customer's churn_risk_score/tags change between
        # turn 2 and turn 5 of this conversation" becomes a real query
        # instead of only ever knowing the current value.
        log_tool_call(
            conn, state["conversation_id"], agent_name="retrieval_agent",
            tool_name="build_customer_signals",
            input_={"customer_id": state["customer_id"], "query": state["query"]},
            output=state["signals"] or {},
            latency_ms=latency_ms,
        )
        return state
    return retrieval


def _parse_request(query):
    """Light intent/amount parse from the message, only to feed the
    deterministic autonomy rule (not the model)."""
    import re
    q = query.lower()
    action = ("refund" if "refund" in q or "money back" in q else
              "cancel" if "cancel" in q else
              "address_change" if "address" in q else
              "payment_method_change" if "payment" in q or "card" in q else "inquiry")
    m = re.search(r"\$?\s?(\d{2,5})(?:\.\d{1,2})?", q)
    amount = float(m.group(1)) if m else 0.0
    return {"action": action, "amount": amount}


def make_case_state_node(conn):
    def case_state_node(state: SupportState) -> SupportState:
        # Real tool calls (refund lookup, order status) happen here and are
        # logged via log_tool_call(conn, ...) BEFORE this builds the snapshot.
        result = build_case_state(conn, state["conversation_id"])
        state["case_state"] = result
        # Deterministic autonomy ceiling from the signal profile + the parsed
        # request. This is the safety gate — a rule, not an LLM judgment.
        from backend.utilis.customer_signals import signals_from_dict, autonomy_ceiling
        sig = signals_from_dict(state.get("signals") or {"customer_id": state["customer_id"]})
        state["autonomy"] = autonomy_ceiling(sig, _parse_request(state["query"]))
        fw = state.get("firewall") or {}
        if fw.get("verdict") == "block":
            state["autonomy"] = {"ceiling": "escalate",
                "reasons": [{"rule": f"input firewall: {fw.get('category')}", "-> ": "escalate"}]}
        elif fw.get("verdict") == "review" and state["autonomy"].get("ceiling") == "act":
            state["autonomy"] = {"ceiling": "propose_only",
                "reasons": [{"rule": f"input firewall review: {fw.get('category')}", "-> ": "propose_only"}]}
        category = (result.get("state", {}) or {}).get("issue_category") \
            or _parse_request(state["query"])["action"].upper()
        try:
            embed_fn = RetrievalAgent(conn).embed
        except Exception as e:
            print(f"[embed_fn] ERROR: {type(e).__name__}: {e}")
            embed_fn = None

        t0 = time.time()
        skill = skills_mod.select_skill(conn, category, state["query"],
                                        state.get("signals") or {}, embed_fn=embed_fn)
        state["skill"] = {"skill_id": skill["skill_id"], "version": skill["version"],
                          "steps": skill["steps"], "validation_rules": skill.get("validation_rules", [])}
        log_tool_call(
            conn, state["conversation_id"], agent_name="case_state_agent",
            tool_name="select_skill",
            input_={"category": category, "query": state["query"]},
            output={"skill_id": skill["skill_id"], "version": skill["version"]},
            latency_ms=int((time.time() - t0) * 1000),
        )

        # bounded, layered working memory (L3 persona + L2 scenario + budgeted
        # L1/L0 facts) assembled once here for the resolution prompt.
        t0 = time.time()
        try:
            state["working_memory"] = wmem_mod.assemble(
                conn, state["customer_id"], state["query"], embed_fn=embed_fn)
        except Exception as e:
            print(f"[working_memory] ERROR: {type(e).__name__}: {e}")
            state["working_memory"] = None
        log_tool_call(
            conn, state["conversation_id"], agent_name="case_state_agent",
            tool_name="assemble_working_memory",
            input_={"customer_id": state["customer_id"]},
            output={"assembled": state["working_memory"] is not None},
            latency_ms=int((time.time() - t0) * 1000),
        )
        return state
    return case_state_node

# -- the Resolution Agent: a REAL Bedrock call, grounded in memory ----------
# No segment switch. The model is told the customer's SIGNAL PROFILE and the
# deterministic AUTONOMY CEILING, and adapts tone/framing itself.
RESOLUTION_SYSTEM_TMPL = """You are a senior e-commerce customer-support agent.
Rules you must follow exactly:
- Answer ONLY from the CASE STATE and the KNOWLEDGE BASE passages provided.
- If the knowledge base does not cover the question, say you'll escalate to a
  human specialist/customer support/service team rather than inventing a policy. Never guess a policy.
- Never state a specific refund dollar amount unless it appears in CASE STATE.
- AUTONOMY CEILING = {ceiling}. Obey it:
    act          -> you may take the action and say it's done.
    propose_only -> you may OFFER the action but say a specialist will confirm;
                    do NOT say it is already done/processed.
    escalate     -> do NOT promise ANY action; say the request is going to a
                    human specialist/customer support/service team for review.
- Adapt your tone and empathy to the CUSTOMER SIGNALS (e.g. a high-value,
  at-risk, repeat-issue customer deserves extra care and acknowledgement).
- Match the customer's language. Be concise, warm, and specific.

OUTPUT FORMAT:
Respond with ONLY a JSON object, no other text, no markdown fences:
{{
  "response": "<the reply to send the customer>",
  "escalate": "<true or false (boolean, not a string) if this case needs a human specialist/customer support/service team review then true, else false>",
  "escalate_reason": "<short reason if escalate="True", else empty string>"
}}
"""


def _build_resolution_prompt(state):
    cs = (state.get("case_state") or {}).get("state", {})
    kb = (state.get("retrieval") or {}).get("kb", []) or []
    kb_text = "\n\n".join(
        f"[{c.get('title','KB')}] {c.get('content','')[:500]}" for c in kb[:3]
    ) or "(no relevant knowledge-base passage found)"
    sig = state.get("signals") or {}
    tags = sig.get("tags", [])
    return (
        f"CUSTOMER SIGNALS: tags={tags}, tenure_days={sig.get('tenure_days')}, "
        f"orders={sig.get('order_count')}, lifetime_value={sig.get('lifetime_value')}, "
        f"churn_risk={sig.get('churn_risk_score')}, repeat_issue={sig.get('repeat_issue')}\n\n"
        f"CASE STATE (synthesized memory — the only customer facts you may use):\n"
        f"{json.dumps(cs, default=str, indent=2)}\n\n"
        f"KNOWLEDGE BASE PASSAGES (the only policy you may cite):\n{kb_text}\n\n"
        f"CUSTOMER MESSAGE:\n{state['query']}\n\n"
        f"Write the support reply now."
    )

def make_resolution_node(conn):
    #TODO : add agent call to generate response based on case state and retrieval
    from backend.utilis.bedrock_client import get_llm
    llm = get_llm()

    def resolution(state: SupportState) -> SupportState:
        if (state.get("firewall") or {}).get("verdict") == "block":
            state["draft_response"] = ("I can't help with that request as written. "
                                       "I'm routing this to a human specialist.")
            state["response_source"] = "firewall_block"
            return state
        ceiling = (state.get("autonomy") or {}).get("ceiling", "propose_only")
        system = RESOLUTION_SYSTEM_TMPL.format(ceiling=ceiling)
        print("[retrival] state: ", state)
        prompt = _build_resolution_prompt(state)
        wm = state.get("working_memory")
        if wm:
            prompt = wmem_mod.to_prompt(wm) + "\n\n" + prompt
        skill = state.get("skill")
        if skill:
            prompt += "\n\n" + skills_mod.skill_for_prompt(skill)

        t0 = time.time()
        print("[retrival] state prompt: ",prompt)
        result = llm.complete(system, prompt, max_tokens=500, temperature=0.2, tag="resolution")
        latency_ms = int((time.time() - t0) * 1000)

        raw = (result.text or "").strip()
        # tolerate the model wrapping output in ```json fences despite instructions
        raw_clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        reply_text = raw
        escalate = False
        reasons = []
        try:
            parsed = json.loads(raw_clean)
            reply_text = parsed.get("response", raw)
            escalate = bool(parsed.get("escalate", False))
            if escalate:
                reason = parsed.get("escalate_reason") or "model_flagged_escalate"
                reasons = [reason]
        except (json.JSONDecodeError, AttributeError):
            # model didn't return valid JSON -- fall back to raw text as the
            # reply and leave escalate=False; visible in logs via parse_error
            reasons = ["escalate_parse_failed"]  # noqa: F841

        state["draft_response"] = reply_text
        state["response_source"] = result.get("source")  # 'bedrock' | 'offline'
        if escalate:
            state["escalated"] = escalate
        print(f"[resolution]:  {state}")
        print(f"[resolution escalated]:  {state['escalated']}")
        # state["draft_response"] = result.text
        # state["response_source"] = result.get("source")  # 'bedrock' | 'offline'

        # audit trail: source matters here specifically -- a demo/run that
        # silently degraded to the offline fallback should be visible in
        # tool_calls, not just in a console print you might not be watching.
        log_tool_call(
            conn, state["conversation_id"], agent_name="resolution_agent",
            tool_name="llm_complete",
            input_={"ceiling": ceiling, "prompt_chars": len(prompt)},
            output={"source": state["response_source"], "response_chars": len(result.text or "")},
            latency_ms=latency_ms,
        )
        return state
    return resolution


def make_guardrail_node(conn):
    # embed fn reused from the retrieval agent's model (loaded once)
    agent = RetrievalAgent(conn)

    def guardrail(state: SupportState) -> SupportState:
        kb = (state.get("retrieval") or {}).get("kb", []) or []
        # print(f"Guardrail: checking groundedness of {len(kb)} KB chunks...{kb}")
        print(f"[from resolution escalated]:  {state['escalated']}")
        print(f"[from resolution escalated]:  {state}")
        
        result = guardrail_mod.run_guardrail(
            conn=conn,
            response_text=state["draft_response"],
            cited_kb_chunk_ids=[c["id"] for c in kb],
            autonomy=state.get("autonomy") or {},
            signals=state.get("signals") or {},
            conversation_id=state["conversation_id"],
            customer_message=state["query"],
            embed_fn=agent.embed,
            category=(state.get("retrieval") or {}).get("category", "unknown"),
            escalated = (state.get("escalated")) 
        )
        state["guardrail_result"] = result
        if result.get("action") == "escalate":
            state["escalated"] = True
        state["final_response"] = None if state["escalated"] else state["draft_response"]
        print(f"[guardrail]:  {state}")
        if not state["escalated"] and not state.get("is_simulated"):
            log_message(conn, state["conversation_id"], "agent", state["final_response"])
        
        return state
    return guardrail


def make_evaluate_node(conn):
    """Post-turn LLM judge — runs after the customer-visible path so it never
    adds latency to the reply. Writes to evaluation_results. Gated by the
    EVAL_ENABLED env var (default on) and skipped for simulated turns (those
    are judged separately by run_simulation via eval_mod directly)."""
    def evaluate(state: SupportState) -> SupportState:
        if os.environ.get("EVAL_ENABLED", "1") != "1":
            return state
        if state.get("is_simulated"):
            return state
        try:
            state["evaluation"] = eval_mod.evaluate_conversation(
                conn, state["conversation_id"])
        except Exception as e:
            state["evaluation"] = {"error": str(e)}
        return state
    return evaluate


def route_after_guardrail(state: SupportState):
    return "escalate" if state["escalated"] else "respond"


def make_escalate_node(conn):
    def escalate(state: SupportState) -> SupportState:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO escalations (conversation_id, reason) VALUES (%s, %s)",
                (state["conversation_id"], state["guardrail_result"]["reason"]),
            )
            cur.execute(
                "UPDATE conversations SET escalated = true, status = 'escalated' WHERE id = %s",
                (state["conversation_id"],),
            )
        conn.commit()
        # build_case_state() counts tool_calls named 'escalate_to_human' toward
        # prior_resolution_attempts -- log it here so a customer who escalated
        # once and comes back has that reflected in later turns, not silently lost.
        log_tool_call(conn, state["conversation_id"], agent_name="escalate_agent",
                      tool_name="escalate_to_human",
                      input_={"reason": state["guardrail_result"]["reason"]},
                      output={"escalated": True})
        if not state.get("is_simulated"):
            log_message(conn, state["conversation_id"], "system",
                        "Escalated to a human specialist — someone will follow up shortly.")
        return state
    return escalate


def build_graph(conn):
    graph = StateGraph(SupportState)
    graph.add_node("planner", make_planner_node(conn))
    graph.add_node("retrieval", make_retrieval_node(conn))
    graph.add_node("case_state", make_case_state_node(conn))
    graph.add_node("resolution", make_resolution_node(conn))
    graph.add_node("guardrail", make_guardrail_node(conn))
    graph.add_node("escalate", make_escalate_node(conn))
    graph.add_node("evaluate", make_evaluate_node(conn))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "case_state")
    graph.add_edge("case_state", "resolution")
    graph.add_edge("resolution", "guardrail")
    graph.add_conditional_edges(
        "guardrail", route_after_guardrail, {"escalate": "escalate", "respond": "evaluate"}
    )
    graph.add_edge("escalate", "evaluate")
    graph.add_edge("evaluate", END)
    return graph


if __name__ == "__main__":
    if not CRDB_URL:
        sys.exit("Set CRDB_URL env var first.")
    if len(sys.argv) < 3:
        sys.exit('Usage: python orchestrator.py <customer_id> "<message>"')

    customer_id, query = sys.argv[1], sys.argv[2]

    with psycopg.connect(CRDB_URL, autocommit=True) as conn:
        conversation_id = create_conversation(conn, customer_id)
        print(f"Created conversation {conversation_id}")

        with CRDBCheckpointSaver.from_conn_string(CRDB_URL) as checkpointer:
            checkpointer.setup()
            graph = build_graph(conn).compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": conversation_id}}

            result = graph.invoke(
                {
                    "conversation_id": conversation_id,
                    "customer_id": customer_id,
                    "query": query,
                    "escalated": False,
                    "is_simulated": False,
                },
                config=config,
            )
            print(json.dumps(
                {k: v for k, v in result.items() if k != "retrieval"},
                indent=2, default=str,
            ))
