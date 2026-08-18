"""
customer_signals.py a rich, data-derived SIGNAL PROFILE, plus the deterministic 
decisions that must NOT be left to the LLM.

THE DESIGN
Three responsibilities :

  1. build_signals()      -> a multi-valued profile computed from the raw
                             memory tables (facts + scores). Many can be true at
                             once. This is the "let the data describe the
                             customer" layer.

  2. autonomy_ceiling()   -> a DETERMINISTIC rule over the signals + the current
                             request that decides act | propose_only | escalate.
                             This is safety-critical and auditable, so it is a
                             rule, NOT an LLM judgment (see the docstring note).

  3. retrieval_plan()     -> which memory sources to pull, derived from data
                             facts (does this customer HAVE history? is there a
                             fraud hold?) rather than a hardcoded label.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

 
# 1. THE SIGNAL PROFILE  (facts + scores, many simultaneously true)
@dataclass
class CustomerSignals:
    customer_id: str
    # value / tenure
    tenure_days: int = 0
    order_count: int = 0
    lifetime_value: float = 0.0
    avg_order_value: float = 0.0
    # engagement / retention
    days_since_last_order: int = 0
    churn_risk_score: float = 0.0
    # memory / history
    past_conversation_count: int = 0
    unresolved_prior_issues: int = 0
    repeat_issue: bool = False
    # risk (safety-relevant)
    active_fraud_hold: bool = False
    fraud_score: float = 0.0
    refund_count: int = 0
    refund_rate: float = 0.0          # refunds / orders
    is_high_value: bool = False
    is_new: bool = False
    is_at_risk: bool = False
    has_history: bool = False
    tags: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def _derive_booleans(s: CustomerSignals) -> CustomerSignals:
    s.has_history = s.past_conversation_count > 0
    s.is_new = s.tenure_days < 30 or s.order_count <= 1
    s.is_high_value = s.lifetime_value >= 750 or s.order_count >= 15
    s.is_at_risk = s.churn_risk_score >= 0.6
    tags = []
    if s.active_fraud_hold: 
        tags.append("fraud_hold")
    if s.fraud_score >= 0.5: 
        tags.append("elevated_fraud_score")
    if s.refund_rate >= 0.4 and s.order_count >= 3: 
        tags.append("high_refund_rate")
    if s.is_high_value: 
        tags.append("high_value")
    if s.is_at_risk: 
        tags.append("churn_risk")
    if s.repeat_issue or s.unresolved_prior_issues: 
        tags.append("repeat_issue")
    if s.is_new: 
        tags.append("cold_start")
    if not tags: 
        tags.append("standard")
    s.tags = tags
    return s


def build_signals(conn, customer_id: str) -> CustomerSignals:
    from psycopg.rows import dict_row
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""SELECT tenure_days, total_orders, total_spent,
                              days_since_last_order, churn_risk_score,
                              fraud_flag, fraud_score
                       FROM customer_profile WHERE customer_id = %s""", (customer_id,))
        p = cur.fetchone() or {}
        cur.execute("SELECT count(*) AS n FROM refunds r JOIN orders o USING (order_id) "
                    "WHERE o.customer_id = %s", (customer_id,))
        refunds = (cur.fetchone() or {}).get("n", 0)
        cur.execute("SELECT count(*) AS n, "
                    "count(*) FILTER (WHERE status <> 'resolved') AS open "
                    "FROM conversations WHERE customer_id = %s", (customer_id,))
        c = cur.fetchone() or {}
        cur.execute("SELECT count(*) AS n FROM fraud_flags "
                    "WHERE customer_id = %s AND coalesce(fraud_score,0) >= 0.5",
                    (customer_id,))
        active_holds = (cur.fetchone() or {}).get("n", 0)

    orders = p.get("total_orders", 0) or 0
    s = CustomerSignals(
        customer_id=customer_id,
        tenure_days=p.get("tenure_days", 0) or 0,
        order_count=orders,
        lifetime_value=float(p.get("total_spent", 0) or 0),
        avg_order_value=(float(p.get("total_spent", 0) or 0) / orders) if orders else 0.0,
        days_since_last_order=p.get("days_since_last_order", 0) or 0,
        churn_risk_score=float(p.get("churn_risk_score", 0) or 0),
        past_conversation_count=c.get("n", 0) or 0,
        unresolved_prior_issues=c.get("open", 0) or 0,
        repeat_issue=(c.get("n", 0) or 0) >= 2,
        active_fraud_hold=bool(active_holds) or bool(p.get("fraud_flag")),
        fraud_score=float(p.get("fraud_score", 0) or 0),
        refund_count=refunds,
        refund_rate=(refunds / orders) if orders else 0.0,
    )
    return _derive_booleans(s)


def signals_from_dict(d: dict) -> CustomerSignals:
    s = CustomerSignals(**{k: v for k, v in d.items()
                           if k in CustomerSignals.__dataclass_fields__})
    return _derive_booleans(s)


ACT = "act"                 # agent may take the action autonomously
PROPOSE = "propose_only"    # agent may draft/offer, but a human confirms
ESCALATE = "escalate"       # hand off to a human specialist now

# money threshold that forces manual review (matches policy_refunds/damaged_item)
MANUAL_REVIEW_AMOUNT = 75.0


def autonomy_ceiling(s: CustomerSignals, request: Optional[dict] = None) -> dict:
    """Returns {ceiling, reasons[]}. Highest-restriction rule wins, and every
    rule contributes a legible reason (Glass Box)."""
    request = request or {}
    reasons = []
    ceiling = ACT

    def raise_to(level, why):
        nonlocal ceiling
        order = {ACT: 0, PROPOSE: 1, ESCALATE: 2}
        if order[level] > order[ceiling]:
            ceiling = level
        reasons.append({"rule": why, "-> ": level})

    if s.active_fraud_hold:
        raise_to(ESCALATE, "active_fraud_hold: no autonomous money/account actions")
    if s.refund_rate >= 0.5 and s.order_count >= 3:
        raise_to(ESCALATE, f"refund_rate {s.refund_rate:.0%} on {s.order_count} orders")
    if s.fraud_score >= 0.7:
        raise_to(PROPOSE, f"elevated fraud_score {s.fraud_score:.2f}")
    amt = float(request.get("amount", 0) or 0)
    if amt >= MANUAL_REVIEW_AMOUNT:
        raise_to(PROPOSE, f"request amount ${amt:.0f} >= ${MANUAL_REVIEW_AMOUNT:.0f} manual-review line")
    if request.get("action") in {"address_change", "payment_method_change"} and s.fraud_score >= 0.3:
        raise_to(PROPOSE, "sensitive account change with non-trivial fraud score")

    if not reasons:
        reasons.append({"rule": "no restriction triggered", "-> ": ACT})
    return {"ceiling": ceiling, "reasons": reasons}

def retrieval_plan(s: CustomerSignals) -> dict:
    """Which memory sources to pull. Same safety intent as the old segment
    routing, but each decision traces to a fact:
      - fraud hold -> KB only (don't feed history/cohort that could bias the
        model toward leniency);
      - has history -> individual vector history;
      - no history -> cohort aggregate (cold start);
      - at-risk / repeat -> also pull similar RESOLVED cases to guide retention.
    """
    if s.active_fraud_hold:
        return {"kb": True, "individual_history": False, "cohort": False,
                "similar_cases": False,
                "note": "fraud hold: KB-only to avoid leniency bias"}
    plan = {"kb": True,
            "individual_history": s.has_history,
            "cohort": not s.has_history,
            "similar_cases": s.is_at_risk or s.repeat_issue,
            "note": "history from data (has_history), cohort on cold start"}
    return plan


# DISPLAY-ONLY case type. NOT used for routing or gating (those use the full
# signal set). This exists purely so a UI can group/label customers by the
# most salient thing about them — a human-friendly headline, not a class the
# logic branches on.
def case_type(s: CustomerSignals) -> str:
    if s.active_fraud_hold:
        return "Fraud hold"
    if "high_refund_rate" in s.tags:
        return "High refund rate"
    if s.is_high_value and s.is_at_risk:
        return "High-value, at-risk"
    if s.is_at_risk:
        return "Churn risk"
    if s.repeat_issue or s.unresolved_prior_issues:
        return "Repeat issue"
    if s.is_high_value:
        return "High value"
    if s.is_new:
        return "Cold start"
    return "Standard"


# offline self-test: the cases the 4-way enum could not express
if __name__ == "__main__":
    import json
    cases = {
        "loyal_high_value_but_churning": dict(customer_id="A", tenure_days=2100,
            order_count=42, lifetime_value=9400, days_since_last_order=95,
            churn_risk_score=0.72, past_conversation_count=6, fraud_score=0.05),
        "new_with_one_fraud_signal": dict(customer_id="B", tenure_days=3,
            order_count=5, lifetime_value=1800, churn_risk_score=0.1,
            past_conversation_count=0, active_fraud_hold=True, fraud_score=0.82),
        "serial_refunder_not_flagged": dict(customer_id="C", tenure_days=200,
            order_count=8, lifetime_value=300, refund_count=6, refund_rate=0.75,
            churn_risk_score=0.3, past_conversation_count=3),
        "ordinary_existing": dict(customer_id="D", tenure_days=400, order_count=12,
            lifetime_value=640, churn_risk_score=0.2, past_conversation_count=2),
    }
    for name, d in cases.items():
        s = signals_from_dict(d)
        print(f"\n=== {name} ===")
        print("tags:", s.tags)
        print("retrieval_plan:", retrieval_plan(s))
        print("autonomy (refund $120):",
              json.dumps(autonomy_ceiling(s, {"action": "refund", "amount": 120}), indent=0))
