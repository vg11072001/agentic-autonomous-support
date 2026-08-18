"""
Case State Builder.
"""
import os
import re
import sys
import json

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    sys.exit("Run: pip install 'psycopg[binary]' --break-system-packages")

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
def _with_query_param(url, key, value):
    parts = urlsplit(url)
    params = parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))

CRDB_URL = os.environ.get("CRDB_URL")
if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)


def _extract_order_ids(query: str) -> list:
    """Pull potential order IDs / ticket IDs mentioned in a customer message."""
    patterns = [
        r"\border(?:er)?[#\-\s]+([A-Za-z0-9\-]{4,20})\b",  # order #ORD-123
        r"#([A-Za-z0-9\-]{4,20})\b",                         # #ORD-123
        r"\b([A-Z]{2,4}[\-_]\d{4,12})\b",                    # ORD-123456
        r"\bticket[#\-\s]+(\d{4,12})\b",                     # ticket #12345
    ]
    ids = []
    for p in patterns:
        ids.extend(m.group(1) for m in re.finditer(p, query, re.IGNORECASE))
    return list(set(ids))


def _safe_float(v):
    try:
        return round(float(v), 2) if v is not None else None
    except (TypeError, ValueError):
        return None


def _fetch_customer_data(conn, customer_id: str, query: str) -> dict:
    """Query orders, refunds, payments, support_tickets, escalations directly.

    Returns a compact dict the resolution agent can cite verbatim and the
    guardrail can validate dollar amounts / statuses against. Wrapped entirely
    in try/except — a DB hiccup must never crash the pipeline.

    Tables used:
      orders, order_items, products  (order facts + line items)
      payments                       (payment status per order)
      refunds                        (refund status + amounts)
      support_tickets                (open tickets for this customer)
      escalations + conversations    (most recent escalation date)
      customer_profile               (fraud_hold flag)
    """
    data: dict = {
        "recent_orders": [],
        "order_detail": None,
        "open_refunds": [],
        "pending_tickets": [],
        "recent_escalation": None,
        "fraud_hold": False,
    }

    if not customer_id:
        return data

    mentioned_ids = _extract_order_ids(query)

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            # ── recent orders with payment + refund overlay ─────────────────
            cur.execute(
                """
                SELECT o.id, o.status, o.total, o.created_at::date AS order_date,
                       p.status  AS payment_status,
                       r.amount  AS refund_amount,
                       r.status  AS refund_status,
                       (SELECT count(*) FROM order_items oi WHERE oi.order_id = o.id)
                           AS item_count
                FROM orders o
                LEFT JOIN payments p ON p.order_id = o.id
                LEFT JOIN refunds  r ON r.order_id = o.id
                WHERE o.customer_id = %s
                ORDER BY o.created_at DESC LIMIT 5
                """,
                (customer_id,),
            )
            orders = cur.fetchall()

            for o in orders:
                data["recent_orders"].append({
                    "order_id":       str(o["id"]),
                    "status":         o["status"],
                    "total":          _safe_float(o["total"]),
                    "order_date":     str(o["order_date"]),
                    "payment_status": o["payment_status"],
                    "refund_amount":  _safe_float(o["refund_amount"]),
                    "refund_status":  o["refund_status"],
                    "item_count":     int(o["item_count"] or 0),
                })

            # ── pick which order to expand with line items ──────────────────
            target_id = None
            if mentioned_ids and orders:
                for mid in mentioned_ids:
                    for o in orders:
                        if mid.lower() in str(o["id"]).lower():
                            target_id = str(o["id"])
                            break
            if target_id is None and orders:
                target_id = str(orders[0]["id"])  # default: most recent

            if target_id:
                cur.execute(
                    """
                    SELECT oi.quantity, oi.unit_price, pr.name AS product_name
                    FROM order_items oi
                    JOIN products pr ON pr.id = oi.product_id
                    WHERE oi.order_id = %s LIMIT 8
                    """,
                    (target_id,),
                )
                items = cur.fetchall()
                base = next(
                    (o for o in data["recent_orders"] if o["order_id"] == target_id),
                    None,
                )
                if base:
                    data["order_detail"] = {
                        **base,
                        "items": [
                            {
                                "name":       i["product_name"],
                                "qty":        int(i["quantity"] or 1),
                                "unit_price": _safe_float(i["unit_price"]),
                            }
                            for i in items
                        ],
                    }

            # ── open / pending refunds ──────────────────────────────────────
            cur.execute(
                """
                SELECT r.order_id, r.amount, r.status, r.created_at::date AS refund_date
                FROM refunds r
                JOIN orders o ON o.id = r.order_id
                WHERE o.customer_id = %s
                  AND r.status NOT IN ('completed', 'cancelled')
                ORDER BY r.created_at DESC LIMIT 5
                """,
                (customer_id,),
            )
            data["open_refunds"] = [
                {
                    "order_id":    str(r["order_id"]),
                    "amount":      _safe_float(r["amount"]),
                    "status":      r["status"],
                    "refund_date": str(r["refund_date"]),
                }
                for r in cur.fetchall()
            ]

            # ── pending support tickets ─────────────────────────────────────
            cur.execute(
                """
                SELECT id, subject, status, priority, created_at::date AS ticket_date
                FROM support_tickets
                WHERE customer_id = %s
                AND status NOT IN ('closed', 'resolved')
                ORDER BY created_at DESC LIMIT 3
                """,
                (customer_id,),
            )
            data["pending_tickets"] = [
                {
                    "ticket_id":   str(t["id"]),
                    "subject":     t["subject"],
                    "status":      t["status"],
                    "priority":    t["priority"],
                    "ticket_date": str(t["ticket_date"]),
                }
                for t in cur.fetchall()
            ]

            # ── most recent escalation ──────────────────────────────────────
            cur.execute(
                """
                SELECT e.created_at::date AS esc_date
                FROM escalations e
                JOIN conversations c ON c.id = e.conversation_id
                WHERE c.customer_id = %s
                ORDER BY e.created_at DESC LIMIT 1
                """,
                (customer_id,),
            )
            esc = cur.fetchone()
            if esc:
                data["recent_escalation"] = str(esc["esc_date"])

            # ── fraud hold from profile ─────────────────────────────────────
            cur.execute(
                "SELECT fraud_flag FROM customer_profile WHERE customer_id = %s LIMIT 1",
                (customer_id,),
            )
            fp = cur.fetchone()
            if fp:
                data["fraud_hold"] = bool(fp["fraud_flag"])

    except Exception as exc:
        # Never crash the pipeline — surface the error for debugging only
        data["_fetch_error"] = str(exc)

    return data


def build_case_state(conn, conversation_id, query: str = ""):
    """
    Synthesize the latest case_state from tool_calls for this conversation,
    PLUS a `verified_facts` block fetched directly from the transactional
    tables (orders, refunds, payments, support_tickets, escalations).

    The resolution agent reads `verified_facts` for exact amounts and statuses;
    the guardrail validates its reply against the same block rather than
    trusting the model's memory.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, tool_name, input, output, created_at
               FROM tool_calls
               WHERE conversation_id = %s
               ORDER BY created_at ASC""",
            (conversation_id,),
        )
        calls = cur.fetchall()

        cur.execute(
            "SELECT customer_id FROM conversations WHERE id = %s", (conversation_id,)
        )
        conv = cur.fetchone()
        customer_id = conv["customer_id"] if conv else None

        cur.execute(
            "SELECT churn_risk_score FROM customer_profile WHERE customer_id = %s",
            (customer_id,),
        )
        profile = cur.fetchone() or {}

        cur.execute(
            "SELECT max(turn_number) AS n FROM case_state WHERE conversation_id = %s",
            (conversation_id,),
        )
        last_turn = (cur.fetchone() or {}).get("n")
        next_turn = (last_turn or 0) + 1

    # ---- tool_call audit trail summary ------------------------------------
    order_status, refund_eligibility, delivery_eta = None, None, None
    prior_resolution_attempts = 0
    call_ids = []

    for c in calls:
        call_ids.append(c["id"])
        out = c["output"] or {}
        if c["tool_name"] == "get_order_status":
            order_status = out.get("status")
            delivery_eta = out.get("estimated_delivery")
        elif c["tool_name"] == "check_refund_eligibility":
            refund_eligibility = out.get("eligible")
        elif c["tool_name"] in ("issue_refund", "escalate_to_human", "apply_credit"):
            prior_resolution_attempts += 1

    # signal profile
    try:
        from backend.utilis.customer_signals import build_signals, case_type as _case_type
        _sig = build_signals(conn, customer_id) if customer_id else None
        signal_tags = _sig.tags if _sig else []
        ct = _case_type(_sig) if _sig else None
    except Exception:
        signal_tags, ct = [], None

    # ---- verified_facts: real rows from DB --------------------------------
    verified_facts = _fetch_customer_data(conn, customer_id, query)

    # Prefer DB facts over tool_call-derived values when available
    od = verified_facts.get("order_detail") or {}
    if od.get("status"):
        order_status = od["status"]   # DB is authoritative over tool_calls
    if od.get("refund_amount") is not None:
        refund_eligibility = True     # a refund row exists → eligible

    state = {
        "issue_category": calls[0]["input"].get("category") if calls and calls[0]["input"] else None,
        "order_status": order_status,
        "delivery_eta": delivery_eta,
        "refund_eligibility": refund_eligibility,
        "prior_resolution_attempts": prior_resolution_attempts,
        "signal_tags": signal_tags,
        "case_type": ct,
        "churn_risk_score": profile.get("churn_risk_score"),
        "open_questions": [],
        "verified_facts": verified_facts,  # ground truth — passed to resolution + guardrail
    }

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO case_state (conversation_id, turn_number, state, built_from_tool_call_ids)
               VALUES (%s, %s, %s, %s)""",
            (conversation_id, next_turn, json.dumps(state, default=str), call_ids),
        )
    conn.commit()
    return {"turn_number": next_turn, "state": state}


def log_tool_call(conn, conversation_id, agent_name, tool_name, input_, output, latency_ms=None):
    """Called by the orchestrator on every real tool invocation — populates the
    audit trail that build_case_state() reads from."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO tool_calls (conversation_id, agent_name, tool_name, input, output, latency_ms)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (conversation_id, agent_name, tool_name, json.dumps(input_, default=str), json.dumps(output, default=str), latency_ms),
        )
        call_id = cur.fetchone()[0]
    conn.commit()
    return call_id


if __name__ == "__main__":
    if not CRDB_URL:
        sys.exit("Set CRDB_URL env var first.")
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 09_case_state_builder.py <conversation_id>")

    conversation_id = sys.argv[1]
    query_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    with psycopg.connect(CRDB_URL) as conn:
        result = build_case_state(conn, conversation_id, query=query_arg)
        print(json.dumps(result, indent=2, default=str))
