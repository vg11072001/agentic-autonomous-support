# Autonomous customer support with persistent agent memory on CockroachDB

Aurora Support is an autonomous customer-support system designed to resolve customer issues using persistent memory, retrieval, customer history, similar cases, guardrails, versioned skills, and continuous evaluation.

Instead of treating every support request as an isolated LLM conversation, Aurora maintains a persistent memory layer in CockroachDB that allows the agent to reason over:

Customer and transactional history
Previous conversation turns
Knowledge-base content
Similar resolved cases
Synthesized case state
Versioned resolution skills
Guardrail decisions
Agent decisions and outcomes
Human escalation and review records
Tamper-evident audit history

The system is designed around a simple principle:

The model generates the response, but memory, retrieval, policies, guardrails, and evaluation determine what the model is allowed to know and do.

## 1. Which CockroachDB tools we used, and what the agent actually did with them

The rules require **≥2**. We use **3**, each on the critical path — not just
initialized.

**1. Distributed Vector Indexing (primary).**
Embeddings live in `VECTOR(384)` columns *next to the rows they describe* —
`kb_chunks.embedding` and `conversations.summary_embedding` — not in a
separate Qdrant/Pinecone. Two index shapes prove this is a real design, not
a toy:
- `kb_chunks`: `VECTOR INDEX (embedding)` — no prefix (KB is small, shared).
- `conversations`: `VECTOR INDEX (customer_id, summary_embedding)` — the
  prefix column gives C-SPANN a **separate k-means tree per customer**, so a
  per-customer history search stays fast regardless of total corpus size.
The Retrieval Agent (`08_retrieval_agent.py`) runs `<->` distance search;
the tier-1 guardrail (`11_guardrail.py`) runs a groundedness distance check
in **SQL, with no model call** — the cheap gate that keeps latency/cost down.

**2. CockroachDB Cloud Managed MCP Server (dev-time + ops).**
Endpoint `https://cockroachlabs.cloud/mcp`, connected read-only from Claude
Code / Cursor while writing the schema and queries. Concretely it caught
un-partitioned vector scans and index/access-pattern mismatches before they
shipped. Full audit logging + read-only mode is exactly the "safe by
default" posture we wanted for an agent touching a production DB. (Config
snippet lives in `SUBMISSION_GUIDE.md` § Deploy; keep it read-only for the
judge's cluster.)

**3. Agent Skills Repo (`cockroachlabs/cockroachdb-skills`).**
Pulled via `npx skills add cockroachlabs/cockroachdb-skills` — used at dev
time to encode CockroachDB query/schema/performance/security best practice
into the build agent. Model-agnostic, so it composed with our Bedrock path.

*(ccloud CLI is wired for the Ops/SRE health job — `ccloud backup list` — but
we count the three above as the meaningful integrations.)*

## 2. Which AWS services we used, and how

The rules require **≥1**. We use **3**:

- **Amazon Bedrock (Claude)** — all agent reasoning goes through
  `bedrock_client.py`: the Resolution agent, the tier-2 guardrail evaluator
  (adversarial critique), the 5-dimension Judge, the customer Simulator, and
  the KB draft-article writer. Circuit-breaker + retry wrapped; degrades to a
  deterministic, clearly-labelled offline path if Bedrock is unreachable.
- **AWS Lambda** — `22_aws_lambda_kb_embedder.py`, an S3-triggered function
  that embeds new KB policy docs into CockroachDB automatically.
- **Amazon S3** — durable source-of-truth store for KB policy documents; the
  Lambda reads `s3:ObjectCreated` events from the KB bucket.

## 3. Architecture diagram

See `ARCHITECTURE.md` / `ui/console.html` (the memory panel *is* the live
diagram). Flow: `planner → retrieval → case_state → resolution → guardrail →
{escalate | end}`, LangGraph `StateGraph`, checkpointed into CockroachDB via
`crdb_checkpointer.py`.


<a href="assests/architecture_aurora_agent.pdf">
  <img src="assests/architecture_aurora_agent.png" alt="Aurora Support Documentation" width="800">
</a>

**[Open full documentation (PDF) →](assests/architecture_aurora_agent.pdf)**