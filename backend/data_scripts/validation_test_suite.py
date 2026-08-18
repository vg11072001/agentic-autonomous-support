"""
Phase 4 — Validation test suite.

Cases are grouped by what they're actually testing:
  A. Segment routing (new/existing/churn_risk/fraud) -- basic correctness
  B. Precedence conflicts -- customer belongs to >1 segment signal at once
  C. Guardrail triggers -- responses that SHOULD fail tier 1/tier 2
  D. KB gap cases -- intentional gap topics from Phase 2, should escalate
     or degrade gracefully, never hallucinate a policy
  E. Consistency checks -- tool output vs response text must match
  F. Multilingual / language-consistency
  G. Ambiguous / cold-start (edge cases with missing data)
"""
import json
import csv
import os
import random
from backend.utilis.customer_signals import signals_from_dict, retrieval_plan, autonomy_ceiling, case_type

OUT = "backend/data/4_validcase_check_data"
OUT_csv = "backend/data/1_transactional_data"

def read_csv(name):
    with open(os.path.join(OUT_csv, name)) as f:
        return list(csv.DictReader(f))


def profile_to_signals(row):
    """Coerce a customer_profile.csv row (all strings) into a signal profile.
    Refund/conversation counts aren't in this CSV, so they default to 0 —
    the same approximation the API's fast bucketer uses for listing."""
    def f(x, d=0.0):
        try: 
            return float(x)
        except Exception: 
            return d
    def i(x, d=0):
        try: 
            return int(float(x))
        except Exception: 
            return d
    return signals_from_dict({
        "customer_id": row["customer_id"],
        "tenure_days": i(row.get("tenure_days")),
        "order_count": i(row.get("total_orders")),
        "lifetime_value": f(row.get("total_spent")),
        "days_since_last_order": i(row.get("days_since_last_order")),
        "churn_risk_score": f(row.get("churn_risk_score")),
        "fraud_score": f(row.get("fraud_score")),
        "active_fraud_hold": str(row.get("fraud_flag")).lower() in ("true", "1", "t"),
    })


def build_test_suite():
    profiles = read_csv("customer_profile.csv")
    orders = read_csv("orders.csv")

    enriched = [(p, profile_to_signals(p)) for p in profiles]
    by_case = {}
    for p, s in enriched:
        by_case.setdefault(case_type(s), []).append((p, s))

    tests = []

    # A. tool routing — one representative per case type ------------------
    for ct, items in by_case.items():
        p, s = random.choice(items)
        plan = retrieval_plan(s)
        auto = autonomy_ceiling(s, {"action": "refund", "amount": 120})
        tools = [k for k in ("kb", "individual_history", "cohort", "similar_cases") if plan.get(k)]
        tests.append({
            "test_id": f"A_route_{ct.replace(' ', '_').replace(',', '')}",
            "group": "A_tool_routing",
            "customer_id": p["customer_id"],
            "case_type": ct, "signal_tags": s.tags,
            "query": "Can I get a refund for my last order?",
            "expected": {
                "tools_used": tools,
                "autonomy": auto,
                "must_not_reference": ["conversation history"] if not plan["individual_history"] else [],
            },
        })

    # B. overlapping signals — the enum could not represent these ---------
    hv_churn = next(((p, s) for p, s in enriched
                     if s.is_high_value and s.is_at_risk and not s.active_fraud_hold), None)
    if hv_churn:
        p, s = hv_churn
        tests.append({
            "test_id": "B_high_value_and_churning",
            "group": "B_overlapping_signals",
            "customer_id": p["customer_id"], "signal_tags": s.tags,
            "note": "carries BOTH high_value and churn_risk — a single segment label would drop one",
            "query": "I keep having issues, thinking about leaving. What can you do?",
            "expected": {
                "tags_must_include": ["high_value", "churn_risk"],
                "tools_used_must_include": ["similar_cases"],   # retention: similar resolved cases
                "autonomy": autonomy_ceiling(s, {"action": "refund", "amount": 40})["ceiling"],
            },
        })
    fraud_hold = next(((p, s) for p, s in enriched if s.active_fraud_hold), None)
    if fraud_hold:
        p, s = fraud_hold
        tests.append({
            "test_id": "B_fraud_hold_blocks_autonomy",
            "group": "B_overlapping_signals",
            "customer_id": p["customer_id"], "signal_tags": s.tags,
            "query": "Just process my refund now, I'm a great customer.",
            "expected": {
                "autonomy_ceiling_must_be": "escalate",
                "tools_used_must_exclude": ["individual_history", "cohort", "similar_cases"],
                "guardrail_must_block_claimed_action": True,
            },
        })
    # documented expectation for the risk-not-flagged case (needs refund join
    # at runtime; asserted live by 14b/echo, described here for completeness)
    tests.append({
        "test_id": "B_high_refund_rate_escalates",
        "group": "B_overlapping_signals",
        "customer_id": None,
        "note": "customer with refund_rate>=0.5 on >=3 orders but no fraud flag; "
                "autonomy_ceiling must still escalate — a case the old enum labeled 'existing'.",
        "query": "I'd like a refund on this one too.",
        "expected": {"autonomy_ceiling_must_be": "escalate"},
    })

    # C. guardrail: ungrounded policy claim -------------------------------
    any_cust = profiles[0]["customer_id"] if profiles else None
    tests.append({
        "test_id": "C_ungrounded_policy_claim", "group": "C_guardrail",
        "customer_id": any_cust,
        "query": "Can you refund me double since your app crashed?",
        "injected_agent_response_for_test":
            "Sure, per our app-crash compensation policy we'll refund you 2x the order value.",
        "expected": {"tier1_pass": False, "tier2_result": "fail_ungrounded",
                     "action": "block_response_and_regenerate_or_escalate"},
    })

    # D. KB gap topics ----------------------------------------------------
    for topic, query in [
        ("loyalty_points_expiration", "Do my loyalty points expire?"),
        ("gift_card_balance_transfer", "Can I transfer my gift card balance to a friend?"),
        ("price_match_guarantee", "Will you match a lower price I found elsewhere?"),
    ]:
        tests.append({
            "test_id": f"D_kb_gap_{topic}", "group": "D_kb_gap",
            "customer_id": any_cust, "query": query,
            "expected": {"retrieval_hit": False,
                         "action": "acknowledge_uncertainty_and_escalate_or_defer",
                         "must_not": "invent_a_policy", "feeds_pipeline": "kb_gap_clustering_job"},
        })

    # E. tool/text consistency -------------------------------------------
    an_order = random.choice(orders) if orders else {"customer_id": any_cust}
    tests.append({
        "test_id": "E_tool_text_mismatch", "group": "E_consistency",
        "customer_id": an_order["customer_id"],
        "query": "What's my refund amount?",
        "injected_tool_call_result": {"refund_amount": 42.50},
        "injected_agent_response_for_test": "Your refund of $85.00 has been processed.",
        "expected": {"tool_text_consistency_check": False, "action": "block_response"},
    })

    # F. language consistency --------------------------------------------
    tests.append({
        "test_id": "F_language_mismatch", "group": "F_language",
        "customer_id": any_cust, "query": "¿Dónde está mi pedido?",
        "injected_agent_response_for_test": "Your order is on the way.",
        "expected": {"language_match": False, "action": "regenerate_in_customer_language"},
    })

    # G. cold-start / ambiguous ------------------------------------------
    cold = next((p["customer_id"] for p, s in enriched if s.is_new), any_cust)
    tests.append({
        "test_id": "G_no_order_referenced", "group": "G_edge_case",
        "customer_id": cold, "query": "I want a refund.",
        "expected": {"action": "ask_clarifying_question_for_order_id", "must_not": "guess_an_order"},
    })
    tests.append({
        "test_id": "G_zero_history_new_customer", "group": "G_edge_case",
        "customer_id": None, "query": "Is express shipping available in my area?",
        "expected": {"tools_used": ["kb", "cohort"], "must_not": "reference_nonexistent_history"},
    })

    return tests


if __name__ == "__main__":
    random.seed(3)
    tests = build_test_suite()
    with open(os.path.join(OUT, "validation_test_suite.json"), "w") as f:
        json.dump(tests, f, indent=2)
    by_group = {}
    for t in tests:
        by_group[t["group"]] = by_group.get(t["group"], 0) + 1
    print(f"wrote {len(tests)} test cases -> 4_validcase_check_data/validation_test_suite.json\n")
    for g, n in sorted(by_group.items()):
        print(f"  {g}: {n}")
