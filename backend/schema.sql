-- =============================================================================
-- Core Schema & Vector Layer
-- Verified against CockroachDB v25.2+ vector support (pgvector-compatible
-- VECTOR type, C-SPANN distributed vector indexes). Run this against a
-- CockroachDB Cloud cluster (Standard or Advanced -- vector indexing needs
-- v25.2+, check your cluster version first: SELECT version();)
-- =============================================================================

-- -----------------------------------------------------------------------
-- TABULAR CORE (from your generated dataset -- Phase 1)
-- -----------------------------------------------------------------------


CREATE TABLE IF NOT EXISTS customers (
    customer_id  UUID PRIMARY KEY,
    name         STRING,
    email        STRING,
    country      STRING,
    city         STRING,
    state        STRING,
    signup_date  DATE
);

CREATE TABLE IF NOT EXISTS products (
    product_id STRING PRIMARY KEY,
    name       STRING,
    category   STRING,
    price      DECIMAL(10,2),
    rating     FLOAT8,
    in_stock   BOOL DEFAULT true,
    image_url  STRING   -- NULL by default; UI falls back to a category icon tile.
                        -- Point this at a real S3/CDN URL once you have product photography.
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      UUID PRIMARY KEY,
    customer_id   UUID NOT NULL REFERENCES customers(customer_id),
    order_date    TIMESTAMPTZ NOT NULL,
    status        STRING NOT NULL,
    total_amount  DECIMAL(10,2),
    INDEX idx_orders_customer (customer_id, order_date DESC)
);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id UUID PRIMARY KEY,
    order_id      UUID NOT NULL REFERENCES orders(order_id),
    product_id    STRING NOT NULL REFERENCES products(product_id),
    unit_price    DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id      UUID PRIMARY KEY,
    order_id        UUID NOT NULL REFERENCES orders(order_id),
    method          STRING,
    amount          DECIMAL(10,2),
    status          STRING,
    billing_country STRING
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id       UUID PRIMARY KEY,
    order_id        UUID NOT NULL REFERENCES orders(order_id),
    customer_id     UUID NOT NULL REFERENCES customers(customer_id),
    reason          STRING,
    amount          DECIMAL(10,2),
    requested_date  TIMESTAMPTZ,
    status          STRING,
    INDEX idx_refunds_customer (customer_id)
);

CREATE TABLE IF NOT EXISTS fraud_flags (
    customer_id  UUID PRIMARY KEY REFERENCES customers(customer_id),
    fraud_score  FLOAT8,
    flag_reason  STRING,
    flagged_at   TIMESTAMPTZ
);

-- Raw behavioral signals live here. There is deliberately NO `segment`
-- column: the pipeline computes a multi-signal profile at request time
-- (customer_signals.py) instead of collapsing behavior into one label.
CREATE TABLE IF NOT EXISTS customer_profile (
    customer_id          UUID PRIMARY KEY REFERENCES customers(customer_id),
    tenure_days           INT,
    total_orders           INT,
    total_spent            DECIMAL(10,2),
    days_since_last_order  INT,
    churn_risk_score       FLOAT8,
    fraud_flag             BOOL DEFAULT false,
    fraud_score            FLOAT8,
    -- NOTE: no precomputed `segment` label. Routing/gating are derived at
    -- request time from these raw signals by customer_signals.py, so a
    -- customer is a VECTOR of behaviors, not one of four buckets.
    updated_at             TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id       UUID PRIMARY KEY,
    customer_id     UUID NOT NULL REFERENCES customers(customer_id),
    order_id        UUID REFERENCES orders(order_id),
    category        STRING,
    priority        STRING,
    channel         STRING,
    created_at      TIMESTAMPTZ,
    resolved_at     TIMESTAMPTZ,
    resolution_type STRING,
    INDEX idx_tickets_customer (customer_id, created_at DESC)
);

-- cohort-level aggregate used by the "new customer" and "churn-risk"
-- retrieval paths (population pattern, NOT a vector search -- see guide)
CREATE MATERIALIZED VIEW IF NOT EXISTS cohort_resolution_patterns AS
    SELECT
        t.category,
        CASE WHEN p.tenure_days < 30 THEN 'new'
             WHEN p.tenure_days < 180 THEN 'young'
             ELSE 'established' END AS tenure_bucket,
        t.resolution_type,
        count(*) AS frequency
    FROM support_tickets t
    JOIN customer_profile p ON p.customer_id = t.customer_id
    GROUP BY t.category, tenure_bucket, t.resolution_type;

-- -----------------------------------------------------------------------
-- VECTOR LAYER -- embeddings live NEXT TO the data they describe, not in
-- a separate store. 384 dims = all-MiniLM-L6-v2 (matches the embedding
-- model already used elsewhere in this project's NPS pipeline).
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kb_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id       STRING NOT NULL,
    version      INT NOT NULL DEFAULT 1,
    title        STRING,
    category     STRING,
    content      STRING NOT NULL,
    embedding    VECTOR(384) NOT NULL,
    updated_at   TIMESTAMPTZ DEFAULT now(),
    VECTOR INDEX (embedding)   -- no prefix column: KB is small & shared across all customers
);

-- conversation-level memory: rolling summary embedding, partitioned by
-- customer_id (prefix column) so per-customer nearest-neighbor search
-- scales independently of total corpus size -- this is the concrete
-- payoff of C-SPANN's prefix-column partitioning for a multi-tenant
-- support system.
CREATE TABLE IF NOT EXISTS conversations (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id        UUID NOT NULL REFERENCES customers(customer_id),
    ticket_id          UUID REFERENCES support_tickets(ticket_id),
    status             STRING NOT NULL DEFAULT 'open',
    summary            STRING,
    summary_embedding  VECTOR(384),
    escalated          BOOL DEFAULT false,
    created_at         TIMESTAMPTZ DEFAULT now(),
    VECTOR INDEX (customer_id, summary_embedding)
);

-- -----------------------------------------------------------------------
-- CASE STATE -- synthesized memory the LLM actually reads (append-only,
-- one row per turn). Separate from tool_calls, which is the raw audit
-- trail. See guide §3 for why these must stay two different objects.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS case_state (
    conversation_id           UUID NOT NULL REFERENCES conversations(id),
    turn_number               INT NOT NULL,
    state                     JSONB NOT NULL,
    built_from_tool_call_ids  UUID[],
    created_at                TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (conversation_id, turn_number)
);

-- Turn-by-turn transcript -- what "old chats" actually reads from. Distinct
-- from case_state (synthesized memory the LLM reads) and tool_calls (audit
-- trail) -- this is display-oriented, storing the literal text exchanged.
CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    role            STRING NOT NULL,   -- 'customer' | 'agent' | 'system'
    content         STRING NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    INDEX idx_messages_conv (conversation_id, created_at)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    agent_name      STRING NOT NULL,
    tool_name       STRING NOT NULL,
    input           JSONB,
    output          JSONB,
    latency_ms      INT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    INDEX idx_tool_calls_conv (conversation_id, created_at)
);


CREATE TABLE IF NOT EXISTS escalations (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  UUID NOT NULL REFERENCES conversations(id),
    reason           STRING NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id       UUID NOT NULL REFERENCES conversations(id),
    retrieval_correctness BOOL,
    response_accuracy     BOOL,
    grammar_language_ok   BOOL,
    coherence_to_context  BOOL,
    relevance_to_request  BOOL,
    judge_reasoning       STRING,
    judge_model           STRING,
    is_simulated          BOOL DEFAULT false,
    created_at            TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key         STRING PRIMARY KEY,
    result      JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);

--1/08/26

CREATE TABLE IF NOT EXISTS judge_calibration_runs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dimension     STRING NOT NULL,
    precision     FLOAT8, recall FLOAT8, f1 FLOAT8,
    sample_size   INT,
    human_labels  JSONB,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Verdict cache -- see 11_guardrail.py. Keyed by a signature of
-- (autonomy ceiling, category, cited KB chunks, failure reasons), not response
-- text, so repeated failure patterns skip a real tier-2 LLM call.
CREATE TABLE IF NOT EXISTS guardrail_verdict_cache (
    signature   STRING PRIMARY KEY,
    verdict     JSONB NOT NULL,
    hit_count   INT DEFAULT 1,
    created_at  TIMESTAMPTZ DEFAULT now()
);


-- "Memory Palace" pattern (Netra, Gemini 3 Hackathon 3rd place) -- durable,
-- human/agent-taggable facts about a customer that outlast any single
-- conversation, surfaced into case_state alongside signals/history.
CREATE TABLE IF NOT EXISTS customer_notes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES customers(customer_id),
    note        STRING NOT NULL,
    added_by    STRING,   -- 'agent' | 'human_reviewer'
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    seq         INT PRIMARY KEY,
    prev_hash   STRING NOT NULL,
    record_hash STRING NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

