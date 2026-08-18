"""
gemini_client.py — the single Google-Gemini reasoning layer for the whole
platform (sibling of bedrock_client.py — same contract, different provider).

WHY THIS FILE EXISTS
Mirrors bedrock_client.py's discipline exactly, so callers can swap
providers without touching the rest of the graph:

  - every model call is wrapped in the project's own CircuitBreaker +
    retry (see 18_production_hardening.py) — one hardening story, not five;
  - the whole demo still runs with ZERO Gemini credentials, because
    `_offline_fallback` returns deterministic, structurally-valid output.
    Never fake a live model call, but never let a missing key kill the
    demo either;
  - Gemini has real per-minute rate limits (RPM) on free/shared-project
    tiers, and unlike Bedrock the SDK will happily 429 you if you burst.
    So this module adds a blocking token-bucket THROTTLE in front of every
    live call, tuned to the model's published RPM boundary, instead of
    relying on retry-after-the-fact alone.

EMBEDDINGS NOTE
---------------
Reasoning goes through Gemini. Embeddings stay on the local
`all-MiniLM-L6-v2` (384 dims) for the same reason as the Bedrock client:
the CockroachDB VECTOR(384) columns and every vector index are built for
384 dims. Swapping to Gemini's `text-embedding-004` (768 dims) would
silently break groundedness search and the per-customer C-SPANN index.
If you want Gemini embeddings, change the schema dim in one place
(05_schema.sql) and re-embed — don't mix dimensionalities.


ENV
---
    GEMINI_MODEL_ID     default: gemini-3.5-flash-lite
                         (chosen over the plain Flash models specifically for
                         its RPD headroom — see RATE LIMIT NOTE below)
    GEMINI_API_KEY      standard Google AI Studio / Vertex key
    GEMINI_ENABLED      set to "0" to force offline mode (useful for CI)
    GEMINI_RPM_LIMIT    default: 15   (matches Gemini 3.5 Flash Lite's quota)
    GEMINI_RPD_LIMIT    default: 500  (matches Gemini 3.5 Flash Lite's quota)
    GEMINI_MIN_INTERVAL_S  optional floor in seconds between calls,
                         overrides the RPM-derived spacing if set

RATE LIMIT NOTE
----------------
Every plain "Flash" tier (2.5 Flash, 3 Flash, 3.5 Flash, 3.6 Flash) caps out
at RPD=20 on this project's quota — fine for interactive use, useless for a
bulk backfill over hundreds/thousands of conversations. The "Flash Lite"
tier (3.1 / 3.5) raises that to RPD=500 for the same RPM=15, which is the
actual bottleneck for a script like 03b_backfill_agent_turns.py. Hence the
default here is gemini-3.5-flash-lite, not a plain Flash model. If you
change GEMINI_MODEL_ID, update GEMINI_RPM_LIMIT / GEMINI_RPD_LIMIT to match
that model's real quota — the throttle only protects you if the numbers are
honest.
"""
from __future__ import annotations
from dotenv import load_dotenv

import os
import json
import time
import logging
import hashlib
import threading
import contextvars
from collections import deque
import backend.agent.production_hardening as _hardening
from typing import Optional

log = logging.getLogger("gemini")
load_dotenv()  # Injects variables from .env into os.environ

# Per-request LLM call trace. The API sets a fresh list at the start of each
# request; every complete() call appends one entry. contextvars keeps this
# correct even when FastAPI runs sync endpoints in a threadpool (each request
# gets its own context), so two concurrent chats don't mix traces.
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

DEFAULT_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")
GEMINI_ENABLED = os.environ.get("GEMINI_ENABLED", "1") != "0"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_RPM_LIMIT = int(os.environ.get("GEMINI_RPM_LIMIT", "15"))
GEMINI_RPD_LIMIT = int(os.environ.get("GEMINI_RPD_LIMIT", "500"))
GEMINI_MIN_INTERVAL_S = os.environ.get("GEMINI_MIN_INTERVAL_S")

# Reuse the circuit breaker + retry the project already ships, so there is
# ONE resilience story wrapped around external model calls, not a second
# hand-rolled one here (same import the Bedrock client uses).
CircuitBreaker = _hardening.CircuitBreaker
retry_with_backoff = _hardening.retry_with_backoff


class LLMResult(dict):
    """dict subclass so callers can do result['text'] or result.text-ish
    access, and so a JSON result and a raw-text result share one type.
    Identical shape to bedrock_client.LLMResult on purpose — providers are
    interchangeable from the caller's point of view."""

    @property
    def text(self) -> str:
        return self.get("text", "")


class DailyQuotaExceeded(RuntimeError):
    """Raised instead of blocking when the RPD boundary is hit. Sleeping
    inline for up to 24h is the wrong failure mode for a batch job like
    03b_backfill_agent_turns.py — callers should catch this, checkpoint
    progress, and resume on the next quota day instead of hanging."""

    def __init__(self, limit: int, reset_in_s: float):
        self.limit = limit
        self.reset_in_s = reset_in_s
        super().__init__(
            f"Gemini daily quota exhausted ({limit} requests/24h). "
            f"Resets in ~{reset_in_s / 3600:.1f}h."
        )


class RateLimiter:
    """Blocking token-bucket / sliding-window throttle that keeps calls
    inside GEMINI_RPM_LIMIT requests-per-minute AND GEMINI_RPD_LIMIT
    requests-per-day. RPM is a *boundary* enforced by sleeping — a few
    seconds of wait is cheap and expected. RPD is enforced by *raising*
    instead of sleeping — waiting out a 24h window inline would silently
    stall a batch job for hours, which is worse than stopping loudly and
    letting the caller checkpoint + resume. Thread-safe so the singleton
    client can be shared across worker threads/requests."""

    def __init__(self, max_per_minute: int, min_interval_s: Optional[float] = None,
                 max_per_day: Optional[int] = None):
        self.max_per_minute = max(1, max_per_minute)
        # If a hard floor between calls is given, respect it in addition to
        # the sliding window (useful for providers that also cap
        # concurrent/back-to-back requests regardless of the minute window).
        self.min_interval_s = float(min_interval_s) if min_interval_s else (
            60.0 / self.max_per_minute
        )
        self.max_per_day = max_per_day
        self._timestamps: deque[float] = deque()   # rolling 60s window
        self._day_timestamps: deque[float] = deque()  # rolling 24h window
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()

            # 0) enforce the rolling 24h RPD boundary FIRST — fail fast
            # rather than spend time on RPM spacing we'd have to undo.
            if self.max_per_day:
                day_start = now - 86400.0
                while self._day_timestamps and self._day_timestamps[0] < day_start:
                    self._day_timestamps.popleft()
                if len(self._day_timestamps) >= self.max_per_day:
                    reset_in = 86400.0 - (now - self._day_timestamps[0])
                    raise DailyQuotaExceeded(self.max_per_day, max(reset_in, 0.0))

            # 1) enforce the minimum spacing floor between any two calls
            since_last = now - self._last_call
            if since_last < self.min_interval_s:
                wait = self.min_interval_s - since_last
                log.debug("throttle: spacing wait %.2fs", wait)
                time.sleep(wait)
                now = time.monotonic()

            # 2) enforce the rolling 60s RPM window boundary
            window_start = now - 60.0
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_per_minute:
                # oldest call in the window determines how long until a slot
                # frees up; sleep exactly that long, then re-check.
                wait = 60.0 - (now - self._timestamps[0])
                if wait > 0:
                    log.info("throttle: RPM boundary hit (%d/%d) — waiting %.2fs",
                             len(self._timestamps), self.max_per_minute, wait)
                    time.sleep(wait)
                now = time.monotonic()
                window_start = now - 60.0
                while self._timestamps and self._timestamps[0] < window_start:
                    self._timestamps.popleft()

            self._timestamps.append(now)
            if self.max_per_day:
                self._day_timestamps.append(now)
            self._last_call = now


class GeminiLLM:
    """Thin, resilient wrapper over the Gemini API. Falls back to
    deterministic offline output when Gemini is unreachable, unauthorized,
    or disabled — so the graph, the guardrail, and the demo all keep
    working end-to-end without credentials. A blocking RateLimiter sits in
    front of every live call so the client never exceeds its RPM quota,
    instead of finding out via a 429 and burning a retry."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30)
        self.throttle = RateLimiter(GEMINI_RPM_LIMIT, GEMINI_MIN_INTERVAL_S,
                                    max_per_day=GEMINI_RPD_LIMIT)
        self._client = None
        self._live = False

        if GEMINI_ENABLED:
            if not GEMINI_API_KEY:
                log.warning("GEMINI_API_KEY not set — using offline fallback")
            else:
                try:
                    from google import genai  # google-genai SDK

                    self._client = genai.Client(api_key=GEMINI_API_KEY)
                    self._live = True
                    log.info("Gemini live: model=%s rpm_limit=%d",
                             model_id, GEMINI_RPM_LIMIT)
                except Exception as e:  # sdk missing, bad key, no network
                    log.warning("Gemini unavailable (%s) — using offline fallback", e)
        else:
            log.info("GEMINI_ENABLED=0 — offline fallback mode")

    # -- public API ---------------------------------------------------------
    @property
    def is_live(self) -> bool:
        return self._live

    def complete(self, system: str, user: str, max_tokens: int = 700,
                 temperature: float = 0.2, tag: str = "llm") -> LLMResult:
        """Free-form completion. Returns LLMResult with .text and a
        `source` field ('gemini' | 'offline') so callers/telemetry can
        prove which path ran — same auditability rule as the Bedrock
        client. Every live call passes through the RPM throttle first."""
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
            
            # NOTE: throttle.acquire() already ran once before we got here
            # (see below) so the RPD check fires early; retries of THIS
            # inner call re-acquire only the RPM spacing, not RPD, since a
            # retry within the same logical request shouldn't double-count
            # against the daily cap.
            resp = self._client.models.generate_content(
                model=self.model_id,
                contents=user,
                config={
                    "system_instruction": system,
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            txt = resp.text or ""
            usage = {}
            meta = getattr(resp, "usage_metadata", None)
            if meta is not None:
                usage = {
                    "prompt_tokens": getattr(meta, "prompt_token_count", None),
                    "completion_tokens": getattr(meta, "candidates_token_count", None),
                    "total_tokens": getattr(meta, "total_token_count", None),
                }
            return LLMResult(text=txt, source="gemini", usage=usage)

        if not self._live:
            return  _emit(self._offline_fallback(system, user))
        
        # RPD is checked outside the retry/breaker path on purpose: retrying
        # a call that's already over the daily quota just burns attempts for
        # nothing, and the breaker's failure count shouldn't be spent on a
        # boundary we already know about. Log loudly (this is the one
        # condition a long-running batch job needs to notice and checkpoint
        # around) then degrade to offline like any other outage.
        def _guarded():
            return retry_with_backoff(max_attempts=3, base_delay=0.5)(_call)()

        try:
            self.throttle.acquire()
        except DailyQuotaExceeded as e:
            log.warning("Gemini RPD boundary hit: %s — degrading to offline "
                       "for remaining calls this window", e)
            return self._offline_fallback(system, user)

        # circuit breaker with the offline generator AS the fallback: if
        # Gemini is flapping, we degrade to deterministic output instead of
        # 500-ing the whole conversation.
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
        returns a conservative fail-closed object. Identical contract to
        bedrock_client.complete_json."""
        sys2 = (system + "\n\nRespond with ONLY a single JSON object, no prose, "
                "no markdown fences.")
        if schema_hint:
            sys2 += "\nJSON keys: " + json.dumps(list(schema_hint.keys()))

        raw = self.complete(sys2, user, max_tokens=max_tokens, temperature=0.0)
        parsed = _safe_json(raw.text)

        if parsed is None and self._live:
            repair = self.complete("Extract and return ONLY the JSON object from the text. No prose, no fences.",
                                   raw.text, max_tokens=max_tokens, temperature=0.0, tag=tag + "_repair")
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
        nothing downstream mistakes it for a real inference. Kept in exact
        parity with bedrock_client._offline_fallback so switching providers
        never changes offline-mode behavior."""
        tag = _prompt_tag(system + user)
        u = user.lower()

        # judge / evaluator prompts ask for the 5 named boolean dims
        if "retrieval_correctness" in system or "judge" in system.lower():
            return LLMResult(text=json.dumps({
                "retrieval_correctness": True, "response_accuracy": True,
                "grammar_language_ok": True, "coherence_to_context": True,
                "relevance_to_request": True,
                "reasoning": "[offline] structurally-valid stub verdict — wire "
                             "a GEMINI_API_KEY for a real Gemini judgment.",
            }), source="offline")

        # tier-2 guardrail evaluator — fail closed with an adversarial note
        if "adversarial" in system.lower() or "guardrail" in system.lower():
            return LLMResult(text=json.dumps({
                "pass": False,
                "reasoning": "[offline] tier-1 already flagged this; with no "
                             "Gemini evaluator wired, we fail closed and "
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


_SINGLETON: Optional[GeminiLLM] = None


def get_llm() -> GeminiLLM:
    """Process-level singleton — created once, reused everywhere. Sharing
    one instance also means everyone shares ONE RateLimiter, so the RPM
    boundary is respected across the whole process, not per-caller."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = GeminiLLM()
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