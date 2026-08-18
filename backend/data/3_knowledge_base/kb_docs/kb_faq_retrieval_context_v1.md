---
document_id: KB-RET-002
title: FAQ Retrieval Context and Demotion Rules
category: knowledge_base
version: 1.0
effective_date: 2026-08-18
source_type: retrieval_policy
priority: high
---

# FAQ Retrieval Context and Demotion Rules

## Problem

Large community FAQ files contain many noisy paraphrases and generated answers. They can match a customer's wording strongly while providing weaker or generic guidance than the company policy.

This creates false retrieval matches such as:

- a cancellation request retrieving a generic order-status FAQ;
- a damaged-return follow-up retrieving a generic returns FAQ;
- an email-change request retrieving a broad account FAQ;
- a payment failure retrieving a generic payment-how-to FAQ.

## Rule: FAQs are language examples, not policy

FAQ documents should be treated as **low-authority query expansion data**.

They may help the retriever recognize:

- spelling mistakes;
- slang;
- profanity;
- informal wording;
- alternate ways to ask the same question.

They must not override a current company policy or issue-specific support article.

## Recommended ranking metadata

For each document, prefer these fields:

- `source_type`
- `priority`
- `document_id`
- `version`
- `effective_date`
- `category`
- `intent`
- `issue_state`
- `supersedes`

Recommended priority:

1. `company_policy`
2. `retrieval_policy`
3. `support_article`
4. `customer_guidance`
5. `faq`

## Query-context matching

The retriever should score more than semantic similarity.

Use these contextual dimensions when available:

- **intent**: refund, cancellation, return, payment, account, shipping;
- **object**: order, return, payment, account;
- **state**: failed, returned, damaged, delayed, pending, delivered;
- **requested action**: cancel, refund, change, follow up, escalate;
- **time context**: recently placed, already returned, last N orders;
- **scope**: one order vs multiple orders.

## Strong examples

Query:
"I want to follow up on return order, as my item was damaged."

Preferred:
1. `POL-DMG-002`
2. `ART-RET-004`
3. `POL-RET-002`

Avoid using `faq_returns_v1.md` as the primary source.

Query:
"I need refund as I placed order by mistake."

Preferred:
1. `POL-CAN-001`
2. `ART-REF-005`
3. `POL-RET-002` only if cancellation is unavailable.

Query:
"I want to change email because current one is not working."

Preferred:
1. `POL-ACC-002`
2. `ART-ACC-003`

Query:
"I want refund on my last 10 returned orders."

Preferred:
1. `POL-MRF-001`
2. `POL-REF-002`
3. `POL-RET-002`

## Conflict rule

If an FAQ answer conflicts with a policy, discard the FAQ answer.

If multiple policies conflict, select the newest effective policy/version and record the conflict for KB maintenance.

## Suggested retrieval implementation

Use a two-stage approach:

### Stage 1: intent/state routing

Classify the request into a compact structured representation:

`intent + object + state + requested_action + scope`

### Stage 2: policy-aware retrieval

Retrieve documents using semantic similarity plus metadata filters/boosts.

Conceptually:

`final_score = semantic_score + intent_match + state_match + policy_priority + recency`

Do not let a long FAQ file win solely because it contains many similar phrases.

## FAQ maintenance recommendation

Do not continuously add more FAQ question-answer pairs.

Instead:

- keep a small set of high-quality FAQs;
- move useful customer phrasings into retrieval metadata or intent examples;
- remove duplicate/generated answers;
- keep authoritative rules in policy documents;
- keep issue-specific procedures in support articles.

This reduces retrieval noise while preserving the value of real customer language.
