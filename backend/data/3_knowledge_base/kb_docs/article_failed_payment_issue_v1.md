---
document_id: ART-PAY-003
title: Customer Payment Issue and Failed Payment Handling
category: payments
version: 1.0
effective_date: 2026-08-18
source_type: support_article
priority: high
---

# Customer Payment Issue and Failed Payment Handling

## When this article applies

Use this article for payment problems during checkout or after an attempted payment.

Examples:

- "I am facing a payment issue."
- "My payment failed."
- "I tried paying but it did not go through."
- "Why did my order get cancelled after payment failed?"

## Authoritative policy

`POL-PAY-002` controls accepted payment methods, payment retry behavior, and cancellation after repeated failure.

## Workflow

1. Determine whether the customer is reporting a failed payment, an unknown charge, or a refund/payment reversal.
2. If it is a failed payment, retrieve `POL-PAY-002`.
3. Check the order/payment state before claiming that the order was cancelled.
4. Do not treat a payment FAQ as evidence that a particular payment succeeded or failed.
5. If the issue concerns a charge that exists but the order was not created, route to payment investigation rather than using failed-payment guidance alone.

## Key policy behavior

A failed payment is retried automatically once after 30 minutes. If it fails a second time, the order is cancelled and the customer is notified by email.

## Retrieval note

Payment FAQs can supply alternate wording such as "card didn't work" or "payment would not go through", but `POL-PAY-002` is the authority for what happens next.
