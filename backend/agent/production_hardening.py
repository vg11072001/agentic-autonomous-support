"""
Production hardening primitives
"""
import os
import sys
import json
import time
import logging
import functools
import subprocess
from enum import Enum
from datetime import datetime, timezone

try:
    import psycopg
except ImportError:
    sys.exit("Run: pip install 'psycopg[binary]' --break-system-packages")

CRDB_URL = os.environ.get("CRDB_URL")

# Circuit breaker -- wrap every Bedrock/external-model call with this.
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

    def call(self, fn, *args, fallback=None, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.opened_at > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                if fallback is not None:
                    return fallback(*args, **kwargs)
                raise CircuitBreakerOpen("circuit open, call skipped")

        try:
            result = fn(*args, **kwargs)
        except Exception as e:  # noqa: F841
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.time()
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise
        else:
            self.failures = 0
            self.state = CircuitState.CLOSED
            return result


# ---------------------------------------------------------------------------
# Retry with exponential backoff -- for transient errors, NOT for guardrail
# failures (those should escalate, never blindly retry -- see 11_guardrail.py)
# ---------------------------------------------------------------------------
def retry_with_backoff(max_attempts=3, base_delay=0.5):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(base_delay * (2 ** attempt))
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Idempotency -- every state-changing tool call (refund, account change)
# checks this BEFORE acting, so a retried request never double-executes.
# ---------------------------------------------------------------------------
def idempotency_check(conn, key):
    with conn.cursor() as cur:
        cur.execute("SELECT result FROM idempotency_keys WHERE key = %s", (key,))
        row = cur.fetchone()
    return row[0] if row else None


def idempotency_store(conn, key, result):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO idempotency_keys (key, result) VALUES (%s, %s)
               ON CONFLICT (key) DO NOTHING""",
            (key, json.dumps(result)),
        )
    conn.commit()


def idempotent(conn_getter, key_fn):
    """Decorator: conn_getter() -> psycopg connection, key_fn(*args, **kwargs) -> str"""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            conn = conn_getter()
            key = key_fn(*args, **kwargs)
            cached = idempotency_check(conn, key)
            if cached is not None:
                return cached
            result = fn(*args, **kwargs)
            idempotency_store(conn, key, result)
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Structured logging -- JSON lines, ready to ship to CloudWatch
# ---------------------------------------------------------------------------
def setup_structured_logger(name="support_platform"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
            }
            if hasattr(record, "extra_fields"):
                payload.update(record.extra_fields)
            return json.dumps(payload)

    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    return logger


# ---------------------------------------------------------------------------
# Health check -- CockroachDB reachability + (optional) ccloud CLI cluster
# status, for an Ops/SRE Agent scheduled job.
# ---------------------------------------------------------------------------
def check_crdb_health(conn_string):
    try:
        with psycopg.connect(conn_string, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"crdb": "healthy"}
    except Exception as e:
        return {"crdb": "unhealthy", "error": str(e)}


def check_last_backup_age(cluster_id):
    """Requires ccloud CLI installed + authenticated with a service account
    scoped to backup/monitoring only (not app data)."""
    try:
        out = subprocess.run(
            ["ccloud", "backup", "list", "--cluster", cluster_id, "-o", "json"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return {"backup_check": "failed", "stderr": out.stderr}
        backups = json.loads(out.stdout)
        return {"backup_check": "ok", "latest": backups[0] if backups else None}
    except FileNotFoundError:
        return {"backup_check": "skipped", "reason": "ccloud CLI not installed"}
    except Exception as e:
        return {"backup_check": "error", "error": str(e)}


if __name__ == "__main__":
    logger = setup_structured_logger()
    logger.info("Running circuit breaker self-test")

    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=1)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("simulated transient failure")
        return "ok"

    for i in range(4):
        try:
            r = breaker.call(flaky, fallback=lambda: "fallback_response")
            print(f"attempt {i}: state={breaker.state.value} result={r}")
        except CircuitBreakerOpen:
            print(f"attempt {i}: state={breaker.state.value} result=BLOCKED (no fallback)")

    if CRDB_URL:
        print(json.dumps(check_crdb_health(CRDB_URL), indent=2))
    else:
        print("CRDB_URL not set -- skipping live health check")
