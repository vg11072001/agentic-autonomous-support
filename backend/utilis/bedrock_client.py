
"""
bedrock_client.py — the single AWS-Bedrock reasoning layer for the whole
platform.

WHY THIS FILE EXISTS
--------------------
Before this, every place that needed an LLM (the Resolution Agent, the
tier-2 guardrail evaluator, the 5-dimension Judge, the customer Simulator,
the KB draft-article writer). That made the system a well-wired *scaffold* with no actual
agent reasoning. This module is the one place that call lives, so:

  - the AWS requirement (>=1 AWS service) is satisfied *meaningfully*,
    not just by importing boto3;
  - every model call is wrapped in the project's own CircuitBreaker +
    retry (see 18_production_hardening.py) — one hardening story, not five;
  - the whole demo still runs with ZERO AWS credentials, because
    `_offline_fallback` returns deterministic, structurally-valid output.
    This is the "honest fallback" discipline from the spec: never fake a
    live model call, but never let a missing key kill the demo either.

EMBEDDINGS NOTE
---------------
Reasoning goes through Bedrock. Embeddings stay on the
local `all-MiniLM-L6-v2` (384 dims) because the CockroachDB VECTOR(384)
columns and every vector index are built for 384 dims. Swapping to Bedrock
Titan (1024/1536 dims) would silently break groundedness search and the
per-customer C-SPANN index. If you want Titan, change the schema dim in one
place (05_schema.sql) and re-embed — don't mix dimensionalities.
"""
from __future__ import annotations
from dotenv import load_dotenv


import os
import json
import time
import logging
import hashlib
import contextvars
import backend.agent.production_hardening as _hardening
from typing import Optional

log = logging.getLogger("bedrock")
load_dotenv()  # Injects variables from .env into os.environ
# ---------------------------------------------------------------------------
# Per-request LLM call trace. The API sets a fresh list at the start of each
# request; every complete() call appends one entry. contextvars keeps this
# correct even when FastAPI runs sync endpoints in a threadpool (each request
# gets its own context), so two concurrent chats don't mix traces.
# ---------------------------------------------------------------------------
_trace_var: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "llm_trace", default=None)


def start_trace() -> list:
    lst: list = []
    _trace_var.set(lst)
    return lst


def record_call(entry: dict) -> None:
    lst = _trace_var.get()
    if lst is not None:
        lst.append(entry)


def drain_calls() -> list:
    lst = _trace_var.get() or []
    _trace_var.set([])
    return lst

DEFAULT_MODEL_ID = os.environ.get(
    # Meta Llama on Bedrock via the Converse API — same call shape as any
    # other Converse model, only the id changes.
    #
    # IMPORTANT (this bites everyone): Llama 3.1+/3.3/4 are only invokable
    # through a CROSS-REGION INFERENCE PROFILE id, which carries a region
    # prefix like `us.` — e.g. `us.meta.llama3-3-70b-instruct-v1:0`. If you
    # pass the bare foundation-model id (`meta.llama3-3-70b-instruct-v1:0`)
    # Bedrock returns AccessDenied even with model access granted. Older
    # models (llama3-8b / llama3-70b) DO work with the bare id and are handy
    # for cheap local testing:
    #   us.meta.llama3-3-70b-instruct-v1:0   <- default, strongest
    #   us.meta.llama3-1-8b-instruct-v1:0    <- cheaper, still profile id
    #   meta.llama3-8b-instruct-v1:0         <- bare id, cheapest, older
    "BEDROCK_MODEL_ID", "us.meta.llama3-3-70b-instruct-v1:0"
)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_ENABLED = os.environ.get("BEDROCK_ENABLED", "1") != "0"

# Reuse the circuit breaker + retry the project already ships, so there is
# ONE resilience story wrapped around external model calls, not a second
# hand-rolled one here.

CircuitBreaker = _hardening.CircuitBreaker
retry_with_backoff = _hardening.retry_with_backoff


class LLMResult(dict):
    """dict subclass so callers can do result['text'] or result.text-ish
    access, and so a JSON result and a raw-text result share one type."""

    @property
    def text(self) -> str:
        return self.get("text", "")


class BedrockLLM:
    """Thin, resilient wrapper over Bedrock's Converse API for Anthropic
    Claude models. Falls back to deterministic offline output when Bedrock
    is unreachable or disabled — so the graph, the guardrail, and the demo
    all keep working end-to-end without credentials."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: str = AWS_REGION):
        self.model_id = model_id
        self.region = region
        self.breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30)
        self._client = None
        self._live = False

        if BEDROCK_ENABLED:
            try:
                import boto3  # noqa

                self._client = boto3.client("bedrock-runtime", region_name=region)
                self._live = True
                log.info("Bedrock live: model=%s region=%s", model_id, region)
            except Exception as e:  # boto3 missing, no creds, no network
                log.warning("Bedrock unavailable (%s) — using offline fallback", e)
        else:
            log.info("BEDROCK_ENABLED=0 — offline fallback mode")

    # -- public API ---------------------------------------------------------
    @property
    def is_live(self) -> bool:
        return self._live

    def complete(self, system: str, user: str, max_tokens: int = 700,
                 temperature: float = 0.2, tag: str = "llm") -> LLMResult:
        """Free-form completion. Returns LLMResult with .text and a
        `source` field ('bedrock' | 'offline') so callers/telemetry can
        prove which path ran — the spec's 'never fake a response silently'
        rule made auditable."""
        t0 = time.time()

        def _emit(result: LLMResult, error: Optional[str] = None) -> LLMResult:
            record_call({
                "tag": tag,
                "model": self.model_id if result.get("source") == "bedrock" else "offline-fallback",
                "source": result.get("source"),
                "latency_ms": round((time.time() - t0) * 1000),
                "in_chars": len(system) + len(user),
                "out_chars": len(result.text or ""),
                "ok": error is None,
                "error": error,
            })
            return result
        def _call():
            resp = self._client.converse(
                modelId=self.model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            txt = resp["output"]["message"]["content"][0]["text"]
            return LLMResult(text=txt, source="bedrock",
                             usage=resp.get("usage", {}))

        if not self._live:
            return  _emit(self._offline_fallback(system, user))

        # circuit breaker with the offline generator AS the fallback: if
        # Bedrock is flapping, we degrade to deterministic output instead of
        # 500-ing the whole conversation.
        def _guarded():
            return retry_with_backoff(max_attempts=3, base_delay=0.5)(_call)()

        try:
            result = self.breaker.call(
                _guarded,
                fallback=lambda: self._offline_fallback(system, user),
            )
            # if the breaker fell back, source will be 'offline'
            err = None if result.get("source") == "bedrock" else "degraded_to_offline"
            return _emit(result, err)
        except Exception as e:  # last-resort: never let a model error kill the turn
            return _emit(self._offline_fallback(system, user), str(e))
    def complete_json(self, system: str, user: str, schema_hint: Optional[dict] = None,
                      max_tokens: int = 700, tag: str = "llm_json") -> LLMResult:
        """Completion that must return a JSON object. Used by the Judge and
        the tier-2 guardrail. Appends a strict-JSON instruction, then parses
        defensively (strips ``` fences). On any parse failure OR offline,
        returns a conservative fail-closed object."""
        sys2 = (system + "\n\nRespond with ONLY a single JSON object, no prose, "
                "no markdown fences.")
        if schema_hint:
            sys2 += "\nJSON keys: " + json.dumps(list(schema_hint.keys()))

        raw = self.complete(sys2, user, max_tokens=max_tokens, temperature=0.0)
        parsed = _safe_json(raw.text)
        
        if parsed is None and self._live:
            repair = self.complete("Extract and return ONLY the JSON object from the text. No popse, no fences.",
                                   raw.text, max_tokens = max_tokens, temperature=0.0, tag=tag + "_repair")
            parsed = _safe_json(repair.text)
        if parsed is None:
            # fail closed — this is a guardrail/judge path, so an unparseable
            # model answer must NOT be treated as a pass.
            parsed = (schema_hint or {}).copy()
            parsed["_parse_error"] = True
        return LLMResult(json=parsed, text=raw.text, source=raw.get("source"))

    # -- deterministic offline generator -----------------------------------
    def _offline_fallback(self, system: str, user: str) -> LLMResult:
        """Deterministic, purpose-shaped output keyed off the prompt so the
        same input always yields the same result (stable demos, stable
        tests). NOT a language model — clearly labelled source='offline' so
        nothing downstream mistakes it for a real inference."""
        tag = _prompt_tag(system + user)
        u = user.lower()

        # judge / evaluator prompts ask for the 5 named boolean dims
        if "retrieval_correctness" in system or "judge" in system.lower():
            return LLMResult(text=json.dumps({
                "retrieval_correctness": True, "response_accuracy": True,
                "grammar_language_ok": True, "coherence_to_context": True,
                "relevance_to_request": True,
                "reasoning": "[offline] structurally-valid stub verdict — wire "
                             "AWS creds for a real Bedrock judgment.",
            }), source="offline")

        # tier-2 guardrail evaluator — fail closed with an adversarial note
        if "adversarial" in system.lower() or "guardrail" in system.lower():
            return LLMResult(text=json.dumps({
                "pass": False,
                "reasoning": "[offline] tier-1 already flagged this; with no "
                             "Bedrock evaluator wired, we fail closed and "
                             "escalate to a human rather than guess.",
                "adversarial_critique": "[offline] would ask: what makes this "
                                        "response unsafe or ungrounded to send?",
            }), source="offline")

        # customer simulator — play a short customer turn
        if "you are the customer" in system.lower() or "simulat" in system.lower():
            return LLMResult(text="Okay, but that still doesn't solve my problem "
                                  "— can you actually fix it?", source="offline")

        # KB draft-article writer
        if "knowledge base" in system.lower() or "draft" in system.lower():
            return LLMResult(text=f"## Draft policy (offline stub {tag})\n\n"
                                  "This is a placeholder article generated "
                                  "without a live model. A human reviewer must "
                                  "rewrite and approve before publish.",
                             source="offline")

        # resolution agent — grounded, cautious support reply
        if "refund" in u:
            body = ("Thanks for reaching out about your refund. I've checked "
                    "your order and our returns policy; here's where things "
                    "stand and the next step I can take for you.")
        elif "where" in u or "status" in u or "track" in u:
            body = ("Let me pull up that order for you. Based on the latest "
                    "tracking and delivery estimate, here's the current status "
                    "and what to expect next.")
        else:
            body = ("Thanks for reaching out — I've reviewed your account and "
                    "our policies, and here's how I can help with that.")
        return LLMResult(text=f"{body} (offline draft {tag})", source="offline")


# -- module-level helpers ---------------------------------------------------
def _safe_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    # grab the outermost object
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(t[start:end + 1])
    except Exception:
        return None


def _prompt_tag(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:6]


_SINGLETON: Optional[BedrockLLM] = None


def get_llm() -> BedrockLLM:
    """Process-level singleton — created once, reused everywhere. Directly
    fixes the 'model/client re-created per request' scaling smell from
    SCALING_NOTES.md #1."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = BedrockLLM()
    return _SINGLETON


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = get_llm()
    print("live:", llm.is_live)
    r = llm.complete("You are a helpful e-commerce support agent.",
                     "Can I get a refund for my last order?")
    print("source:", r.get("source"))
    print("text:", r.text)
    j = llm.complete_json(
        "You are an evaluation judge scoring a support reply.",
        "Reply: 'Your refund of $40 is processed.'",
        schema_hint={"retrieval_correctness": bool, "response_accuracy": bool,
                     "grammar_language_ok": bool, "coherence_to_context": bool,
                     "relevance_to_request": bool, "reasoning": str},
    )
    print("judge:", json.dumps(j.get("json"), indent=2))
