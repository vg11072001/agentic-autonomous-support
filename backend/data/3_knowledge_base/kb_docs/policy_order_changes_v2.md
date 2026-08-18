---
document_id: POL-ORD-002
title: Order Changes and Customer Request Routing Policy
category: orders
version: 2.0
effective_date: 2026-08-18
source_type: company_policy
---

# Order Changes and Customer Request Routing Policy

## Separate account changes from order changes
Updating an account profile field does not automatically update an existing order.

Examples:
- Changing an email address changes the account profile, not the order.
- Changing an account address does not automatically authorize changing the shipping address of an order.
- Cancelling an order is a separate order action.

## Intent routing examples

| Customer statement | Primary intent |
|---|---|
| "I need to change my email." | ACCOUNT_EMAIL_UPDATE |
| "I just placed the order by mistake." | ORDER_CANCELLATION |
| "I need to change my email before cancelling my order." | ACCOUNT_EMAIL_UPDATE + ORDER_CANCELLATION |
| "The order is already being processed; cancel it." | ORDER_CANCELLATION |
| "I don't want the item anymore and it has shipped." | RETURN |
| "You sent me the wrong item." | WRONG_ITEM / REFUND |
| "My package arrived damaged." | DAMAGED_ITEM |
| "My account was hacked." | ACCOUNT_SECURITY |

## Multi-intent requests
When a customer has multiple requests, identify each intent and handle them independently. Do not let an account-profile change bypass order eligibility rules.

## Source citation
For intent separation and routing, cite: **POL-ORD-002 — Order Changes and Customer Request Routing Policy**.
