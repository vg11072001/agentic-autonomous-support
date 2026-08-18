# Example Queries — Non-Escalated Customer Conversations

Queries target CockroachDB (PostgreSQL-compatible). All tables match the Aurora Support schema
(`conversations`, `escalations`, `orders`, `refunds`, `support_tickets`, `kb_chunks`, etc.).

---

## 1. All conversations that were resolved without escalation

```sql
SELECT
    c.id            AS conversation_id,
    c.customer_id,
    c.platform,
    c.summary,
    c.created_at
FROM conversations c
WHERE c.escalated = false
ORDER BY c.created_at DESC
LIMIT 100;
```

---

## 2. Customers who have NEVER had an escalation

```sql
SELECT
    cp.customer_id,
    cp.churn_risk_score,
    cp.fraud_flag,
    COUNT(c.id) AS total_conversations
FROM customer_profile cp
JOIN conversations c ON c.customer_id = cp.customer_id
WHERE NOT EXISTS (
    SELECT 1
    FROM escalations e
    JOIN conversations ec ON ec.id = e.conversation_id
    WHERE ec.customer_id = cp.customer_id
)
GROUP BY cp.customer_id, cp.churn_risk_score, cp.fraud_flag
ORDER BY total_conversations DESC;
```

---

## 3. Non-escalated conversations by platform (breakdown)

Useful to compare which channels self-resolve most often.

```sql
SELECT
    c.platform,
    COUNT(*)                                           AS total_conversations,
    SUM(CASE WHEN c.escalated = false THEN 1 ELSE 0 END) AS resolved_count,
    ROUND(
        100.0 * SUM(CASE WHEN c.escalated = false THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                  AS resolution_pct
FROM conversations c
GROUP BY c.platform
ORDER BY resolution_pct DESC;
```

---

## 4. Cancellation-related conversations that resolved without escalation

Based on KB doc `policy_cancellation_v1` — orders cancelled within 1 hour are resolved by policy.
The filter looks for cancellation topics in the conversation summary.

```sql
SELECT
    c.id,
    c.customer_id,
    c.platform,
    c.summary,
    c.created_at
FROM conversations c
WHERE c.escalated = false
  AND c.summary ILIKE '%cancel%'
ORDER BY c.created_at DESC;
```

---

## 5. Refund requests resolved without escalation

Per `policy_refunds_v1`: auto-approved for orders under $75 for damaged/wrong-item.
This checks conversations mentioning refund that never hit the escalations table.

```sql
SELECT
    c.id            AS conversation_id,
    c.customer_id,
    o.id            AS order_id,
    o.total         AS order_total,
    r.amount        AS refund_amount,
    r.status        AS refund_status,
    c.summary
FROM conversations c
JOIN orders o      ON o.customer_id = c.customer_id
LEFT JOIN refunds r ON r.order_id = o.id
WHERE c.escalated = false
  AND c.summary ILIKE '%refund%'
  AND r.status IS NOT NULL
ORDER BY c.created_at DESC;
```

---

## 6. Return requests resolved without escalation

Per `policy_returns_v1`: 30-day window (15 days for electronics).

```sql
SELECT
    c.id,
    c.customer_id,
    c.summary,
    o.status        AS order_status,
    r.status        AS refund_status,
    c.created_at
FROM conversations c
JOIN orders o ON o.customer_id = c.customer_id
LEFT JOIN refunds r ON r.order_id = o.id
WHERE c.escalated = false
  AND (c.summary ILIKE '%return%' OR c.summary ILIKE '%wrong item%')
ORDER BY c.created_at DESC;
```

---

## 7. Shipping-delay conversations resolved without escalation

Per `policy_shipping_v1`: 3+ day delay triggers a $10 automatic credit (no support needed).
These conversations should have short turn counts.

```sql
SELECT
    c.id,
    c.customer_id,
    c.summary,
    c.created_at,
    o.status AS order_status
FROM conversations c
JOIN orders o ON o.customer_id = c.customer_id
WHERE c.escalated = false
  AND (
      c.summary ILIKE '%shipping%'
   OR c.summary ILIKE '%delayed%'
   OR c.summary ILIKE '%delivery%'
  )
ORDER BY c.created_at DESC;
```

---

## 8. High-value customers (churn risk > 0.7) resolved without escalation

These are the most valuable self-resolutions — at-risk customers whose issues were handled
without human escalation.

```sql
SELECT
    cp.customer_id,
    cp.churn_risk_score,
    c.id            AS conversation_id,
    c.platform,
    c.summary,
    c.created_at
FROM conversations c
JOIN customer_profile cp ON cp.customer_id = c.customer_id
WHERE c.escalated = false
  AND cp.churn_risk_score > 0.7
ORDER BY cp.churn_risk_score DESC, c.created_at DESC;
```

---

## 9. Non-escalated conversations where the KB chunk was close (well-covered topic)

A low embedding distance means the KB already had an answer — these are true self-service wins.
Replace `0.25` with your gap threshold from `17_kb_ugc_pipeline.py` (`GAP_DISTANCE_THRESHOLD`).

```sql
SELECT
    c.id                AS conversation_id,
    c.customer_id,
    c.summary,
    kc.title            AS closest_kb_article,
    kc.category,
    c.embedding <-> kc.embedding AS distance
FROM conversations c
CROSS JOIN LATERAL (
    SELECT title, category, embedding
    FROM kb_chunks
    ORDER BY embedding <-> c.summary_embedding
    LIMIT 1
) kc
WHERE c.escalated = false
  AND c.summary_embedding IS NOT NULL
  AND c.embedding <-> kc.embedding < 0.25
ORDER BY distance ASC
LIMIT 50;
```

---

## 10. Topic frequency for non-escalated conversations (keyword bucketing)

Groups by topic so you can see which KB categories carry the most self-service load.

```sql
SELECT
    CASE
        WHEN c.summary ILIKE '%cancel%'                        THEN 'cancellation'
        WHEN c.summary ILIKE '%refund%'                        THEN 'refund'
        WHEN c.summary ILIKE '%return%' OR
             c.summary ILIKE '%wrong item%'                    THEN 'returns'
        WHEN c.summary ILIKE '%ship%' OR
             c.summary ILIKE '%delivery%' OR
             c.summary ILIKE '%delay%'                         THEN 'shipping'
        WHEN c.summary ILIKE '%password%' OR
             c.summary ILIKE '%account%' OR
             c.summary ILIKE '%login%'                         THEN 'account'
        WHEN c.summary ILIKE '%payment%' OR
             c.summary ILIKE '%charge%' OR
             c.summary ILIKE '%billing%'                       THEN 'payment'
        ELSE 'other'
    END                    AS topic,
    COUNT(*)               AS resolved_count
FROM conversations c
WHERE c.escalated = false
GROUP BY topic
ORDER BY resolved_count DESC;
```

---

## 11. Non-escalated conversations with open support tickets (still in-flight)

Catches cases that were not escalated but have unresolved follow-up tickets —
a signal that the conversation "resolved" but the problem persisted.

```sql
SELECT
    c.id            AS conversation_id,
    c.customer_id,
    c.summary,
    st.id           AS ticket_id,
    st.subject,
    st.status       AS ticket_status,
    st.priority,
    st.created_at   AS ticket_created
FROM conversations c
JOIN support_tickets st ON st.customer_id = c.customer_id
WHERE c.escalated = false
  AND st.status NOT IN ('closed', 'resolved')
ORDER BY st.priority DESC, st.created_at DESC;
```

---

## 12. Weekly non-escalation rate trend (8 weeks)

Mirrors the sparkbar metric on the Escalations dashboard but shows the complement.

```sql
SELECT
    date_trunc('week', c.created_at)::date                        AS week,
    COUNT(*)                                                        AS total,
    SUM(CASE WHEN c.escalated = false THEN 1 ELSE 0 END)           AS resolved,
    ROUND(
        100.0 * SUM(CASE WHEN c.escalated = false THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                               AS resolution_pct
FROM conversations c
WHERE c.created_at >= now() - INTERVAL '8 weeks'
GROUP BY week
ORDER BY week ASC;
```

---

## 13. Fraud-flagged customers whose conversations still resolved without escalation

Per `policy_refunds_v1`: fraud-flagged accounts are never auto-approved.
These are edge cases worth reviewing — the agent resolved without needing escalation.

```sql
SELECT
    c.id            AS conversation_id,
    c.customer_id,
    cp.fraud_flag,
    c.summary,
    c.platform,
    c.created_at
FROM conversations c
JOIN customer_profile cp ON cp.customer_id = c.customer_id
WHERE c.escalated = false
  AND cp.fraud_flag = true
ORDER BY c.created_at DESC;
```

---

## 14. KB draft articles approved, with linked non-escalated conversations

Shows KB articles that came from escalation clusters and the non-escalated conversations
that are now covered by those articles.

```sql
SELECT
    kda.id              AS draft_id,
    kda.source_cluster_id,
    kda.escalation_count,
    kc.title            AS kb_title,
    kc.category,
    COUNT(c.id)         AS non_escalated_convs_covered
FROM kb_draft_articles kda
JOIN kb_chunks kc ON kc.id = kda.published_chunk_id
JOIN LATERAL (
    SELECT id
    FROM conversations c
    WHERE c.escalated = false
      AND c.summary_embedding IS NOT NULL
      AND c.summary_embedding <-> kc.embedding < 0.30
    LIMIT 500
) c ON true
WHERE kda.status = 'approved'
GROUP BY kda.id, kda.source_cluster_id, kda.escalation_count, kc.title, kc.category
ORDER BY non_escalated_convs_covered DESC;
```

---

## Notes on schema

| Table | Key columns used |
|---|---|
| `conversations` | `id`, `customer_id`, `escalated`, `platform`, `summary`, `summary_embedding`, `created_at` |
| `escalations` | `id`, `conversation_id`, `escalation_reason`, `created_at` |
| `customer_profile` | `customer_id`, `churn_risk_score`, `fraud_flag` |
| `orders` | `id`, `customer_id`, `status`, `total` |
| `refunds` | `id`, `order_id`, `amount`, `status` |
| `support_tickets` | `id`, `customer_id`, `subject`, `status`, `priority` |
| `kb_chunks` | `id`, `title`, `category`, `embedding` |
| `kb_draft_articles` | `id`, `source_cluster_id`, `status`, `published_chunk_id` |

KB docs referenced: `policy_cancellation_v1`, `policy_refunds_v1`, `policy_returns_v1`,
`policy_shipping_v1`, `policy_damaged_item_v1`, `policy_account_security_v1`,
`faq_payment_v1`, `faq_order_status_v1`, `faq_shipping_v1`, `faq_account_v1`.


