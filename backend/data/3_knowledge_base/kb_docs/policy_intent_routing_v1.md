---
document_id: POL-RTR-001
title: Customer Support Intent Routing Policy
category: retrieval_and_routing
version: 1.0
effective_date: 2026-08-18
source_type: company_policy
priority: high
---

# Customer Support Intent Routing Policy

## Purpose

Route customer requests to the correct policy before retrieving general FAQ content.

The customer's wording is not the policy category. Route by the customer's **primary requested action and problem state**.

## Retrieval priority

Use this order:

1. Current company policy (`source_type: company_policy`, highest version/effective date).
2. Issue-specific support article.
3. Customer guidance article.
4. FAQ only when the first three sources do not answer the question.

FAQ documents are examples of customer wording, not authoritative policy.

## Intent rules

| Customer wording or situation | Primary intent | Retrieve first |
|---|---|---|
| "follow up on return", "item was damaged" | damaged return / return follow-up | POL-DMG-002 + POL-RET-002 |
| "change email", "current email does not work" | account email update | POL-ACC-002 |
| "shipping is continuously delayed", "I want to leave" | shipping delay + retention/escalation | POL-SHP-002 + POL-ESC-001 |
| "payment issue", "payment failed" | failed payment | POL-PAY-002 |
| "cancel because I found a better price" | cancellation / change of mind | POL-CAN-001 |
| "refund all my last 10 orders" | multi-order refund request | POL-MRF-001 + POL-REF-002 + POL-RET-002 |
| "refund because I placed the order by mistake" | cancellation first, refund second | POL-CAN-001; then POL-RET-002 if cancellation is unavailable |

## Multi-intent rule

If a request contains multiple issues, identify:

- the requested action;
- the affected object (order, return, account, payment);
- the current state;
- whether the customer is asking for information, an action, or escalation.

Example:

"I want to follow up on my return because the item was damaged."

Primary intent = return/refund follow-up.
Secondary context = damaged item.
Do not retrieve a generic returns FAQ before the damaged-item policy.

## Version rule

If two documents cover the same issue, use the document with the newest effective date and highest policy version.

The following older policies are superseded where their newer counterparts exist:

- `policy_returns_v1.md` → `POL-RET-002`
- `policy_refunds_v1.md` → `POL-REF-002`
- `policy_cancellation_v1.md` → `POL-CAN-001`
- `policy_shipping_v1.md` → `POL-SHP-002`
- `policy_payment_faq_v1.md` → `POL-PAY-002`
- `policy_account_security_v1.md` → `POL-SEC-002`

## FAQ restriction

Do not use FAQ content as the source of eligibility, refund amounts, cancellation windows, payment retry behavior, or escalation decisions.

Use FAQ material only for:

- query-language expansion;
- typo/noisy-language matching;
- alternate customer phrasing.

## Response rule

When policy and FAQ disagree, policy wins.

When no policy covers the customer's exact situation, do not infer a new entitlement from FAQ examples. State what is known and route to human review when required.
