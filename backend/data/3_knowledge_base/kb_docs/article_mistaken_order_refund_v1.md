---
document_id: ART-REF-005
title: Refund Request for an Order Placed by Mistake
category: cancellation_and_refunds
version: 1.0
effective_date: 2026-08-18
source_type: support_article
priority: high
---

# Refund Request for an Order Placed by Mistake

## When this article applies

Use this article when a customer says they placed an order by mistake and wants their money back.

Examples:

- "I placed the order by mistake. I need a refund."
- "I ordered accidentally. Can you refund me?"
- "I clicked buy by mistake."

## Decision order

Do not immediately route this to the refund policy.

First determine whether the order can still be cancelled under `POL-CAN-001`.

### If cancellation is available

If the order is within the 1-hour cancellation window and warehouse processing has not started, follow the cancellation workflow.

### If cancellation is no longer available

If warehouse processing has started or the cancellation window has passed, explain that cancellation is unavailable and evaluate the applicable return process under `POL-RET-002`.

## Important distinction

"Placed by mistake" describes the customer's reason. It does not create a special refund entitlement by itself.

Do not promise a refund solely because the order was accidental.

## Retrieval rule

For "ordered by mistake", retrieve `POL-CAN-001` first.

Retrieve `POL-RET-002` only when cancellation is unavailable or the customer is asking about returning an already-delivered order.
