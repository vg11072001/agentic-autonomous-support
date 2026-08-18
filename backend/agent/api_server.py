"""
API server — wraps the LangGraph orchestrator behind HTTP and exposes the
live trace the Memory Inspector console renders.

Endpoints:
  GET  /api/health
  GET  /api/customers            list/search; optional ?case_type= ; conversation_count
  GET  /api/customers/suggested  one vivid customer per case type, preferring WITH history
  GET  /api/case-types           the case-type vocabulary + counts (for filter chips)
  POST /api/chat                 run one turn; returns response + full trace
  GET  /api/conversations        past chats for a customer (old chats)
  GET  /api/conversations/{id}/messages
  GET  /api/trace/{id}           full dev trace (messages + evaluation + tool_calls + case_state)
  GET  /api/audit/verify
  GET  /simulations/versions
  GET  /simulations/versions/{prompt_version}
  GET  /simulations/scenarios
  POST /simulations/run
  GET  /simulations/status
  POST /api/analyze-photo
  GET  /api/escalations              list escalated conversations with latest resolution note
  GET  /api/escalations/metrics      aggregate stats: total, platform breakdown, weekly trend
  POST /api/escalations/{id}/note    add human resolution note (platform, outcome, notes)
  GET  /api/escalations/{id}/notes   all notes for a conversation
  GET  /api/kb-gaps                  list kb_draft_articles (default: pending_review)
  POST /api/kb-gaps/detect           trigger detect_gaps background job
  GET  /api/kb-gaps/status           status of running detection job
  POST /api/kb-gaps/{id}/approve     approve a draft → publishes to kb_chunks
  POST /api/kb-gaps/{id}/reject      mark draft rejected
"""
import os
import sys
import time
import threading
import traceback
from typing import Optional, List

import urllib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
from backend.utilis.crdb_checkpointer import CRDBCheckpointSaver

import backend.utilis.bedrock_client as bedrock_client
import backend.utilis.gemini_client  # noqa: F401
import backend.agent.audit_chain as audit_mod
import backend.agent.orchestrator as orchestrator_mod
import backend.agent.simulation_flywheel as sim_mod
import backend.agent.kb_ugc_pipeline as kb_mod
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

def _with_query_param(url, key, value):
    parts = urllib.parse.urlsplit(url)
    params = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, 
                                    urllib.parse.urlencode(params), parts.fragment))
CRDB_URL = os.environ.get("CRDB_URL")
if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)

if not CRDB_URL:
    sys.exit("Set CRDB_URL env var first.")

app = FastAPI(title="Support Platform API")

# Dev-open CORS -- lock allow_origins to your actual storefront domain
# before production.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],)

_conn = psycopg.connect(CRDB_URL, autocommit=True) 
# with CRDBCheckpointSaver.from_conn_string(CRDB_URL) as checkpointer:
_checkpointer = CRDBCheckpointSaver.open(CRDB_URL)
_checkpointer.setup()
_compiled_graph = orchestrator_mod.build_graph(_conn).compile(checkpointer=_checkpointer)
_llm = bedrock_client.get_llm()

try:
    with _conn.cursor() as _cur:
        _cur.execute("""
            CREATE TABLE IF NOT EXISTS escalation_notes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                conversation_id UUID NOT NULL,
                platform STRING NOT NULL DEFAULT 'unknown',
                outcome STRING DEFAULT 'pending',
                human_notes TEXT DEFAULT '',
                reviewer STRING DEFAULT '',
                created_at TIMESTAMPTZ DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS kb_draft_articles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_cluster_id STRING,
                escalation_count INT DEFAULT 0,
                draft_content TEXT,
                status STRING DEFAULT 'pending_review',
                reviewer STRING,
                published_chunk_id UUID,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """)
except Exception:
    pass


CASE_TYPE_SQL = """
CASE
  WHEN coalesce(p.fraud_flag,false) OR coalesce(p.fraud_score,0) >= 0.5 THEN 'Fraud hold'
  WHEN coalesce(p.churn_risk_score,0) >= 0.6
       AND (coalesce(p.total_spent,0) >= 750 OR coalesce(p.total_orders,0) >= 15) THEN 'High-value, at-risk'
  WHEN coalesce(p.churn_risk_score,0) >= 0.6 THEN 'Churn risk'
  WHEN coalesce(p.total_spent,0) >= 750 OR coalesce(p.total_orders,0) >= 15 THEN 'High value'
  WHEN coalesce(p.tenure_days,0) < 30 OR coalesce(p.total_orders,0) <= 1 THEN 'Cold start'
  ELSE 'Standard'
END"""

CASE_TYPES = ["Fraud hold", "High-value, at-risk", "Churn risk", "High value",
              "Cold start", "Standard"]


class ChatRequest(BaseModel):
    customer_id: Optional[str] = None
    message: str


def _retrieval_trace(retr: dict) -> dict:
    if not retr:
        return {"tools_used": [], "kb": [], "history": None, "cohort": None, "similar_cases": None}
    plan = retr.get("plan") or {}
    tools = []
    if plan.get("kb"): 
        tools.append("kb_chunks")
    kb = [{"title": c.get("title"), "doc_id": c.get("doc_id"),
           "distance": round(float(c.get("distance", 0)), 3)} for c in (retr.get("kb") or [])]
    history = retr.get("individual_history")
    if history:
        tools.append("conversations (this customer)")
        history = [{"summary": (h.get("summary") or "")[:120],
                    "distance": round(float(h.get("distance", 0)), 3)} for h in history]
    cohort = retr.get("cohort_pattern")
    if cohort:
        tools.append("cohort_resolution_patterns")
        cohort = [{"resolution_type": c.get("resolution_type"), "frequency": c.get("frequency")}
                  for c in cohort]
    similar = retr.get("similar_cases")
    if similar:
        tools.append("conversations (similar resolved cases)")
        similar = [{"summary": (s.get("summary") or "")[:120],
                    "distance": round(float(s.get("distance", 0)), 3)} for s in similar]
    return {"tools_used": tools, "plan": plan, "kb": kb, "history": history,
            "cohort": cohort, "similar_cases": similar, "note": retr.get("note")}


def _tables_touched(retr: dict, escalated: bool) -> List[dict]:
    plan = (retr or {}).get("plan") or {}
    t = [
        {"name": "conversations", "op": "write", "why": "new conversation row"},
        {"name": "customer_profile", "op": "read", "why": "signal profile inputs"},
        {"name": "orders/refunds/fraud_flags", "op": "read", "why": "signal profile inputs"},
        {"name": "messages", "op": "write", "why": "logged customer turn"},
        {"name": "kb_chunks", "op": "read", "why": "vector KB search + groundedness"},
        {"name": "tool_calls", "op": "read", "why": "audit source for case_state"},
        {"name": "case_state", "op": "write", "why": "synthesized memory snapshot"},
        {"name": "guardrail_verdict_cache", "op": "read/write", "why": "verdict cache by signature"},
    ]
    if plan.get("individual_history") or plan.get("similar_cases"):
        t.append({"name": "conversations", "op": "read", "why": "vector history / similar cases"})
    if plan.get("cohort"):
        t.append({"name": "cohort_resolution_patterns", "op": "read", "why": "cohort pattern"})
    if escalated:
        t.append({"name": "escalations", "op": "write", "why": "escalation + reason"})
        t.append({"name": "messages", "op": "write", "why": "system escalation notice"})
    else:
        t.append({"name": "messages", "op": "write", "why": "logged agent reply"})
    return t


@app.get("/api/health")
def health():
    db_ok, db_err = True, None
    try:
        with _conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception as e:
        db_ok, db_err = False, str(e)
        try: 
            _conn.rollback()
        except Exception: 
            pass
    return {"status": "ok" if db_ok else "degraded", "db_connected": db_ok, "db_error": db_err,
            "llm_live": _llm.is_live,
            "llm_model": _llm.model_id if _llm.is_live else "offline-fallback"}


@app.get("/api/case-types")
def case_types():
    try:
        with _conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT {CASE_TYPE_SQL} AS case_type, count(*) AS n "
                        f"FROM customer_profile p GROUP BY 1")
            counts = {r["case_type"]: r["n"] for r in cur.fetchall()}
        return {"case_types": CASE_TYPES, "counts": counts}
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"case-type query failed: {e}")


@app.get("/api/customers")
def list_customers(search: Optional[str] = None, case_type: Optional[str] = None,
                   with_history: bool = False, limit: int = 50, offset: int = 0):
    where, params = [], []
    if search:
        where.append("(c.name ILIKE %s OR c.email ILIKE %s OR c.customer_id::text ILIKE %s)")
        params += [f"%{search}%", f"%{search}%", f"%{search}%"]
    having = ""
    if case_type:
        where.append(f"{CASE_TYPE_SQL} = %s")
        params.append(case_type)
    if with_history:
        having = "HAVING (SELECT count(*) FROM conversations cv WHERE cv.customer_id = c.customer_id) > 0"
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT c.customer_id, c.name, c.email, c.country,
               p.tenure_days, p.total_orders, p.total_spent, p.days_since_last_order,
               p.churn_risk_score, p.fraud_flag, p.fraud_score,
               {CASE_TYPE_SQL} AS case_type,
               (SELECT count(*) FROM conversations cv WHERE cv.customer_id = c.customer_id)
                 AS conversation_count
        FROM customers c JOIN customer_profile p USING (customer_id)
        {clause}
        GROUP BY c.customer_id, c.name, c.email, c.country, p.tenure_days, p.total_orders,
                 p.total_spent, p.days_since_last_order, p.churn_risk_score, p.fraud_flag, p.fraud_score
        {having}
        ORDER BY conversation_count DESC, p.total_orders DESC
        LIMIT %s OFFSET %s"""
    params += [min(limit, 200), offset]
    try:
        with _conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return {"customers": cur.fetchall()}
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"customer query failed: {e}")


@app.get("/api/customers/suggested")
def suggested_customers():
    """One vivid customer per case type, PREFERRING customers who already have
    conversation history — so the inspector immediately shows history/similar-
    case retrieval firing, and old chats are visible."""
    out = []
    try:
        for ct in CASE_TYPES:
            with _conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"""
                    SELECT c.customer_id, c.name, p.tenure_days, p.total_orders,
                           p.total_spent, p.churn_risk_score, p.fraud_flag,
                           {CASE_TYPE_SQL} AS case_type,
                           (SELECT count(*) FROM conversations cv WHERE cv.customer_id=c.customer_id)
                             AS conversation_count
                    FROM customers c JOIN customer_profile p USING (customer_id)
                    WHERE {CASE_TYPE_SQL} = %s
                    ORDER BY conversation_count DESC, p.total_orders DESC
                    LIMIT 1""", (ct,))
                r = cur.fetchone()
                if r: 
                    out.append(r)
        return {"suggested": out}
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"suggested query failed: {e}")


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Runs one turn through the graph and returns the response PLUS the full
    trace (retrieval sources + distances, case_state, guardrail verdict, the
    exact LLM calls with model/latency/source, and the CockroachDB tables
    touched). Any failure is caught and returned as a structured error so the
    console can display it instead of a bare 500."""
    if not req.customer_id:
        raise HTTPException(400, "customer_id is required")

    bedrock_client.start_trace()  # begin capturing LLM calls for THIS request
    t0 = time.time()
    conversation_id = None
    try:
        conversation_id = orchestrator_mod.create_conversation(_conn, req.customer_id)
        print(conversation_id) # too in uvicorn logs 
        result = _compiled_graph.invoke(
            {"conversation_id": conversation_id, "customer_id": req.customer_id,
             "query": req.message, "escalated": False, "is_simulated": False},
            config={"configurable": {"thread_id": conversation_id}})
        print(result)
        retr = result.get("retrieval") or {}
        gr = result.get("guardrail_result") or {}
        cs = (result.get("case_state") or {}).get("state", {})
        escalated = result.get("escalated", False)
        is_simulated = result.get("is_simulated", False)  # noqa: F841
        sig = result.get("signals") or {}
        # quantified $ impact (Globot): what's financially at stake this turn
        parsed = orchestrator_mod._parse_request(req.message)
        impact = {"amount_at_stake": parsed.get("amount", 0),
                  "lifetime_value_at_risk": round(float(sig.get("lifetime_value") or 0), 2),
                  "action": parsed.get("action")}
        # tamper-evident audit record for this decision
        audit = None
        try:
            audit = audit_mod.append_audit(_conn, {
                "conversation_id": conversation_id, "customer_id": req.customer_id,
                "firewall": (result.get("firewall") or {}).get("verdict"),
                "autonomy_ceiling": (result.get("autonomy") or {}).get("ceiling"),
                "escalated": escalated, "response_source": result.get("response_source"),
                "guardrail_reason": gr.get("reason"), "amount_at_stake": parsed.get("amount", 0)})
        except Exception:
            try: 
                _conn.rollback()
            except Exception: 
                pass
        return {
            "conversation_id": conversation_id,
            "response": result.get("final_response"),
            "escalated": escalated,
            "escalation_reason": gr.get("reason_detail") or gr.get("reason"),
            "evaluation": result.get("evaluation"),
            "error": None,
            "trace": {
                "nodes": ["planner", "retrieval", "case_state", "resolution", "guardrail",
                          "escalate" if escalated else "respond"],
                "signals": result.get("signals"),
                "autonomy": result.get("autonomy"),
                "skill": result.get("skill"),
                "working_memory": result.get("working_memory"),
                "draft_response": result.get("draft_response"),
                "retrieval": _retrieval_trace(retr),
                "case_state": cs,
                "guardrail": {"tier": gr.get("tier"), "action": gr.get("action"),
                              "reason": gr.get("reason"), "reason_detail": gr.get("reason_detail"),
                              "cache_hit": gr.get("cache_hit"),
                              "adversarial_critique": gr.get("adversarial_critique")},
                "firewall": result.get("firewall"),
                "audit": audit,
                "impact": impact,
                "response_source": result.get("response_source"),
                "llm_calls": bedrock_client.drain_calls(),
                "tables": _tables_touched(retr, escalated),
                "timing_ms": round((time.time() - t0) * 1000),
                "evaluation": result.get("evaluation"),
            },
        }
    except Exception as e:
        try: 
            _conn.rollback()
        except Exception: 
            pass
        return {"conversation_id": conversation_id, "response": None, "escalated": False,
                "escalation_reason": None, "evaluation": None,
                "error": {"type": e.__class__.__name__, "message": str(e),
                          "trace": traceback.format_exc().splitlines()[-6:]},
                "trace": {"llm_calls": bedrock_client.drain_calls(),
                          "timing_ms": round((time.time() - t0) * 1000)}}


@app.get("/api/conversations")
def list_conversations(customer_id: str):
    with _conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, customer_id, status, escalated, summary, created_at
               FROM conversations
               WHERE customer_id = %s
               ORDER BY created_at DESC LIMIT 50""",
            (customer_id,),
        )
        return cur.fetchall()


@app.get("/api/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str):
    with _conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""SELECT role, content, created_at FROM messages
                       WHERE conversation_id = %s ORDER BY created_at ASC""", (conversation_id,))
        return cur.fetchall()



@app.get("/api/trace/{conversation_id}")
def conversation_trace(conversation_id: str):
    """Full dev trace for the memory inspector console — messages, evaluation
    results, tool call audit trail, and the last case_state snapshot."""
    with _conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at ASC",
            (conversation_id,),
        )
        messages = cur.fetchall()

        cur.execute(
            """SELECT retrieval_correctness, response_accuracy, grammar_language_ok,
                      coherence_to_context, relevance_to_request, judge_reasoning AS reasoning,
                      judge_model, is_simulated, created_at
               FROM evaluation_results
               WHERE conversation_id = %s
               ORDER BY created_at DESC LIMIT 1""",
            (conversation_id,),
        )
        evaluation = cur.fetchone()

        cur.execute(
            """SELECT agent_name, tool_name, input, output, latency_ms, created_at
               FROM tool_calls
               WHERE conversation_id = %s
               ORDER BY created_at ASC""",
            (conversation_id,),
        )
        tool_calls = cur.fetchall()

        cur.execute(
            """SELECT state, turn_number, created_at
               FROM case_state
               WHERE conversation_id = %s
               ORDER BY turn_number DESC LIMIT 1""",
            (conversation_id,),
        )
        case_state = cur.fetchone()

    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "evaluation": evaluation,
        "tool_calls": tool_calls,
        "case_state": case_state,
    }
    
@app.get("/api/audit/verify")
def audit_verify():
    """Recompute the whole hash chain and report integrity (tamper-evidence)."""
    try:
        return audit_mod.verify_chain(_conn)
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"verify failed: {e}")


@app.get("/simulations/versions")
def list_prompt_versions():
    """Group simulation_runs by prompt_version -- one row per CI/flywheel batch."""
    with _conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT prompt_version,
                      COUNT(*) AS total_runs,
                      COUNT(*) FILTER (WHERE pass) AS passed_runs,
                      MIN(created_at) AS started_at,
                      MAX(created_at) AS finished_at
               FROM simulation_runs
               GROUP BY prompt_version
               ORDER BY MAX(created_at) DESC"""
        )
        rows = cur.fetchall()
    for r in rows:
        r["pass_rate"] = (r["passed_runs"] / r["total_runs"]) if r["total_runs"] else None
    return rows


@app.get("/simulations/versions/{prompt_version}")
def prompt_version_detail(prompt_version: str):
    """All scenario executions for a given prompt_version."""
    with _conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT r.id AS run_id, r.conversation_id, r.pass, r.created_at,
                      s.id AS scenario_id, s.customer_intent, s.customer_story,
                      s.customer_characteristics, s.source_conversation_id
               FROM simulation_runs r
               JOIN simulation_scenarios s ON s.id = r.scenario_id
               WHERE r.prompt_version = %s
               ORDER BY r.created_at DESC""",
            (prompt_version,),
        )
        runs = cur.fetchall()
    total = len(runs)
    passed = sum(1 for r in runs if r["pass"])
    return {
        "prompt_version": prompt_version,
        "total_runs": total,
        "passed_runs": passed,
        "pass_rate": (passed / total) if total else None,
        "runs": runs,
    }


@app.get("/simulations/scenarios")
def list_scenarios():
    """Scenario library -- browse or seed new flywheel runs from recent escalations."""
    with _conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, customer_intent, customer_story, customer_characteristics,
                      source_conversation_id, created_at
               FROM simulation_scenarios
               ORDER BY created_at DESC LIMIT 100"""
        )
        return cur.fetchall()


class SimRunRequest(BaseModel):
    n: int = 10
    prompt_version: str = "v1"
    baseline: float = 0.85


_sim_status: dict = {"running": False, "last_run": None, "error": None}


@app.post("/simulations/run")
def trigger_simulation(req: SimRunRequest):
    """Kick off a flywheel run in the background. Poll /simulations/status to track."""
    if _sim_status["running"]:
        raise HTTPException(409, "A simulation is already running")

    def _run():
        _sim_status["running"] = True
        _sim_status["error"] = None
        try:
            scenarios = sim_mod.generate_scenarios_from_history(_conn, n=req.n)
            results = [
                sim_mod.run_simulation(_conn, s, prompt_version=req.prompt_version)
                for s in scenarios
            ]
            pass_rate = sum(results) / len(results) if results else 0.0
            _sim_status["last_run"] = {
                "pass_rate": pass_rate,
                "n": len(results),
                "prompt_version": req.prompt_version,
                "passed": sum(results),
            }
        except Exception as exc:
            _sim_status["error"] = str(exc)
        finally:
            _sim_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "n": req.n, "prompt_version": req.prompt_version}


@app.get("/simulations/status")
def simulation_status():
    """Current state of the background simulation runner."""
    return _sim_status


# Escalation review endpoints
class EscalationNoteRequest(BaseModel):
    platform: str = "unknown"   # chat | call | email | social_media | in_store
    outcome: str = "pending"    # resolved | unresolved | pending | transferred
    human_notes: str = ""
    reviewer: str = ""


_ESC_CASE_SQL = f"""
    SELECT
        c.id                  AS conversation_id,
        c.customer_id,
        c.summary,
        c.status,
        c.created_at,
        cu.name               AS customer_name,
        {CASE_TYPE_SQL}       AS case_type,
        en.platform,
        en.outcome,
        en.human_notes,
        en.reviewer,
        en.created_at         AS noted_at,
        (SELECT content FROM messages m
            WHERE m.conversation_id = c.id
              AND m.role = 'system' AND m.content ILIKE 'Escalated%%'
            ORDER BY m.created_at ASC LIMIT 1) AS escalation_reason,
        (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
    FROM conversations c
    LEFT JOIN customers   cu ON cu.customer_id = c.customer_id
    LEFT JOIN customer_profile p ON p.customer_id = c.customer_id
    LEFT JOIN LATERAL (
        SELECT platform, outcome, human_notes, reviewer, created_at
        FROM escalation_notes
        WHERE conversation_id = c.id
        ORDER BY created_at DESC LIMIT 1
    ) en ON true
    WHERE c.escalated = true
"""


@app.get("/api/escalations")
def list_escalations(limit: int = 100, platform: Optional[str] = None,
                     reviewed: Optional[bool] = None):
    extra, params = [], []
    if platform:
        extra.append("AND en.platform = %s")
        params.append(platform)
    if reviewed is True:
        extra.append("AND en.id IS NOT NULL")
    elif reviewed is False:
        extra.append("AND en.id IS NULL")
    sql = _ESC_CASE_SQL + " ".join(extra) + " ORDER BY c.created_at DESC LIMIT %s"
    params.append(min(limit, 500))
    try:
        with _conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"escalation query failed: {e}")


@app.get("/api/escalations/metrics")
def escalation_metrics():
    try:
        with _conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT count(*) AS total FROM conversations WHERE escalated = true")
            total = cur.fetchone()["total"]

            cur.execute("""SELECT count(*) AS n FROM conversations
                           WHERE escalated = true AND created_at >= now() - INTERVAL '7 days'""")
            this_week = cur.fetchone()["n"]

            cur.execute("""SELECT count(*) AS n FROM conversations
                           WHERE escalated = true AND created_at >= now() - INTERVAL '14 days'
                             AND created_at < now() - INTERVAL '7 days'""")
            last_week = cur.fetchone()["n"]

            cur.execute("""SELECT platform, count(*) AS n FROM escalation_notes
                           GROUP BY platform ORDER BY n DESC""")
            platforms = cur.fetchall()

            cur.execute("""
                SELECT date_trunc('week', created_at) AS week, count(*) AS escalations
                FROM conversations WHERE escalated = true
                  AND created_at >= now() - INTERVAL '8 weeks'
                GROUP BY week ORDER BY week
            """)
            trend = cur.fetchall()

            cur.execute("""
                SELECT
                    count(*) FILTER (WHERE en.conversation_id IS NOT NULL)     AS reviewed,
                    count(*) FILTER (WHERE en.conversation_id IS NULL)         AS unreviewed
                FROM conversations c
                LEFT JOIN LATERAL (
                    SELECT conversation_id FROM escalation_notes
                    WHERE conversation_id = c.id LIMIT 1
                ) en ON true
                WHERE c.escalated = true
            """)
            rv = cur.fetchone()

        return {
            "total": total, "this_week": this_week, "last_week": last_week,
            "platforms": platforms, "trend": trend,
            "reviewed": rv["reviewed"], "unreviewed": rv["unreviewed"],
        }
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"metrics query failed: {e}")


@app.post("/api/escalations/{conversation_id}/note")
def add_escalation_note(conversation_id: str, req: EscalationNoteRequest):
    try:
        with _conn.cursor() as cur:
            cur.execute(
                """INSERT INTO escalation_notes
                   (conversation_id, platform, outcome, human_notes, reviewer)
                   VALUES (%s, %s, %s, %s, %s)""",
                (conversation_id, req.platform, req.outcome, req.human_notes, req.reviewer),
            )
        return {"ok": True}
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"note insert failed: {e}")


@app.get("/api/escalations/{conversation_id}/notes")
def get_escalation_notes(conversation_id: str):
    try:
        with _conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id, platform, outcome, human_notes, reviewer, created_at
                   FROM escalation_notes WHERE conversation_id = %s
                   ORDER BY created_at DESC""",
                (conversation_id,),
            )
            return cur.fetchall()
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"notes query failed: {e}")


_kb_gap_status: dict = {"running": False, "last_run": None, "error": None}


@app.get("/api/kb-gaps")
def list_kb_gaps(status: str = "pending_review"):
    try:
        with _conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id, source_cluster_id, escalation_count, draft_content,
                          status, reviewer, published_chunk_id, created_at
                   FROM kb_draft_articles WHERE status = %s ORDER BY created_at DESC""",
                (status,),
            )
            return cur.fetchall()
    except Exception:
        _conn.rollback()
        return []   # table may not exist until first detect run


@app.post("/api/kb-gaps/detect")
def trigger_kb_gap_detection():
    if _kb_gap_status["running"]:
        raise HTTPException(409, "Gap detection already running")

    def _run():
        _kb_gap_status["running"] = True
        _kb_gap_status["error"] = None
        try:
            kb_mod.detect_gaps(_conn)
            _kb_gap_status["last_run"] = {"status": "completed"}
        except Exception as exc:
            _kb_gap_status["error"] = str(exc)
        finally:
            _kb_gap_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@app.get("/api/kb-gaps/status")
def kb_gap_detection_status():
    return _kb_gap_status


@app.post("/api/kb-gaps/{draft_id}/approve")
def approve_kb_gap(draft_id: str):
    try:
        kb_mod.approve_draft(_conn, draft_id)
        return {"ok": True}
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"approve failed: {e}")


@app.post("/api/kb-gaps/{draft_id}/reject")
def reject_kb_gap(draft_id: str):
    try:
        with _conn.cursor() as cur:
            cur.execute(
                "UPDATE kb_draft_articles SET status = 'rejected' WHERE id = %s",
                (draft_id,),
            )
        return {"ok": True}
    except Exception as e:
        _conn.rollback()
        raise HTTPException(500, f"reject failed: {e}")
