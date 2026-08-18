---
document_id: POL-MRF-001
title: Multi-Order Refund Request Policy
category: refunds
version: 1.0
effective_date: 2026-08-18
source_type: company_policy
priority: high
---

# Multi-Order Refund Request Policy

## Purpose

Define how to handle a customer asking for refunds on many previous orders in one request.

Typical request:

"I want a refund on all of my last 10 orders. I returned all the items but did not get my refunds."

## Core rule

A multi-order request is not one refund decision.

Each order must be evaluated independently using its order state, return state, inspection state, payment state, and applicable policy.

## Required information

Retrieve the relevant order IDs when available.

If the customer says "last 10 orders" but the system does not provide those orders in context, do not assume which orders they mean. Retrieve the customer's recent orders or ask for the order IDs.

## Per-order checks

For every order:

1. Was the item returned?
2. Was the return received?
3. Has warehouse inspection completed?
4. Was the refund already issued?
5. If issued, has the applicable 5–7 business-day period elapsed?
6. Is manual review required?
7. Is there another policy-specific reason the refund cannot be automatically processed?

## Policy references

- `POL-RET-002` governs return eligibility and the 5–7 business-day refund timeline after inspection.
- `POL-REF-002` governs automatic refund eligibility and specialist review.
- `POL-SEC-002` governs specialist handling when account security review applies.

## Do not make these assumptions

- Do not assume all 10 orders are eligible because the customer returned them.
- Do not assume all 10 refunds failed for the same reason.
- Do not issue a blanket refund without evaluating each order.
- Do not disclose internal fraud/security-review status.

## Response behavior

If several orders are pending, explain that each order is being checked individually.

If some refunds were already issued and others are pending, separate those outcomes rather than giving one generic status.

If specialist review is required, use neutral language and follow the escalation policy.
