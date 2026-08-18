"""
crdb_checkpointer.py

A CockroachDB-safe LangGraph checkpointer.

WHY THIS EXISTS
----------------
langgraph-checkpoint-postgres's PostgresSaver relies on JSONB
set-returning functions (e.g. jsonb_each_text) used as a data source in
a FROM clause, as part of its blob-separation / delta-channel-reconstruction
optimizations. CockroachDB is Postgres-wire-compatible but does not fully
support SRFs-in-FROM the way Postgres does, so setup()/get_tuple() blow up
with:

    psycopg.errors.UndefinedTable: no data source matches prefix:
    jsonb_each_text in this context

Tracked upstream: https://github.com/cockroachdb/cockroach/issues/150532
(open, C-bug, A-sql-pgcompat -- no fix timeline as of this writing).

This module re-implements the BaseCheckpointSaver contract from scratch,
against a deliberately simple two-table schema, using only plain
INSERT/SELECT/UPDATE -- no JSONB functions in FROM position, no
correlated subqueries, no lateral joins.

SERIALIZATION: checkpoint / metadata / write values are NOT stored as
naive JSON. Real graph state routinely contains non-JSON-native types
(uuid.UUID, datetime, LangChain message objects, Pydantic models, ...).
Every value is round-tripped through self.serde (LangGraph's
JsonPlusSerializer by default) via dumps_typed()/loads_typed() -- on
BOTH the write path (put/put_writes) and the read path
(get_tuple/list) -- and stored as BYTEA with a companion *_type column.
Do not swap these back to json.dumps()/json.loads() as a
"simplification"; that reintroduces
`TypeError: Object of type UUID is not JSON serializable` on write, or
hands the graph raw undecoded bytes on read.

CONCURRENCY: uses a psycopg_pool.ConnectionPool, not a single connection
behind a lock. A single shared connection would serialize every
checkpoint read/write across all concurrent requests in an API server --
fine for the one-shot CLI orchestrator, a real bottleneck under
concurrent load. Requires: pip install "psycopg[binary,pool]"
--break-system-packages

LIFECYCLE: two ways to construct this, matching two different callers:
  - Short-lived script (one process, one invoke, then exit):
        with CRDBCheckpointSaver.from_conn_string(CRDB_URL) as cp:
            cp.setup()
            graph.invoke(..., config=...)
    `from_conn_string` is a @contextmanager -- it yields the saver
    inside the `with` block and closes the pool on exit. Calling it
    without `with` gives you a _GeneratorContextManager object, not a
    saver (that's the AttributeError you hit) -- so it CANNOT be used
    for a long-running server's module-level singleton.

  - Long-running process (API server, holds the checkpointer for the
    life of the process, across many requests):
        _checkpointer = CRDBCheckpointSaver.open(CRDB_URL)   # at startup
        _checkpointer.setup()
        ...
        _checkpointer.close()                                # at shutdown
    `open()`/`close()` are plain methods, not a context manager --
    the pool stays alive until you explicitly close it (e.g. in a
    FastAPI lifespan handler or atexit).

Tradeoff vs. the real PostgresSaver: no blob de-duplication across
checkpoints (each checkpoint stores its full channel_values inline), and
list()-with-metadata-filter decodes rows in Python rather than filtering
in SQL (since metadata is now an opaque serialized blob, not JSONB).
For most workloads (short-lived support conversations, not
million-message threads) both are a non-issue. 
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence
from collections.abc import Generator
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)

# ---------------------------------------------------------------------------
# Schema -- plain tables, no JSONB functions used as a FROM-clause source
# anywhere in this file.
# ---------------------------------------------------------------------------
SETUP_SQL = """
CREATE TABLE IF NOT EXISTS crdb_checkpoints (
    thread_id             TEXT NOT NULL,
    checkpoint_ns         TEXT NOT NULL DEFAULT '',
    checkpoint_id         TEXT NOT NULL,
    parent_checkpoint_id  TEXT,
    checkpoint_type       TEXT NOT NULL,
    checkpoint            BYTEA NOT NULL,
    metadata_type         TEXT NOT NULL,
    metadata              BYTEA NOT NULL,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS crdb_checkpoint_writes (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    idx           INT NOT NULL,
    channel       TEXT NOT NULL,
    value_type    TEXT NOT NULL,
    value         BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE INDEX IF NOT EXISTS crdb_checkpoints_thread_idx
    ON crdb_checkpoints (thread_id, checkpoint_ns, checkpoint_id DESC);
"""


class CRDBCheckpointSaver(BaseCheckpointSaver):
    """CockroachDB-compatible checkpoint saver, backed by a connection pool.

    Implements the same public contract as
    langgraph.checkpoint.postgres.PostgresSaver (get_tuple, list, put,
    put_writes, setup) but against the simplified schema documented at
    the top of this file.
    """

    def __init__(self, pool: ConnectionPool, *, serde=None) -> None:
        super().__init__(serde=serde)
        self.pool = pool

    @classmethod
    @contextmanager
    def from_conn_string(cls, conn_string: str, **pool_kwargs) -> Iterator["CRDBCheckpointSaver"]:
        """For short-lived scripts only -- pool is closed on `with` exit.
        Do NOT use this for a long-running server; use .open()/.close()."""
        with ConnectionPool(
            conn_string,
            kwargs={"autocommit": True, "row_factory": dict_row},
            **pool_kwargs,
        ) as pool:
            yield cls(pool)

    @classmethod
    def open(cls, conn_string: str, **pool_kwargs) -> "CRDBCheckpointSaver":
        """For long-running processes (API servers). Call .close() at
        shutdown. Unlike from_conn_string, this is NOT a context manager --
        it returns a live saver directly, so it's safe to assign to a
        module-level variable and reuse across requests."""
        pool = ConnectionPool(
            conn_string,
            kwargs={"autocommit": True, "row_factory": dict_row},
            **pool_kwargs,
        )
        pool.wait()  # block until at least one connection is ready
        return cls(pool)

    def close(self) -> None:
        self.pool.close()

    def setup(self) -> None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(SETUP_SQL)

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_id = config["configurable"].get("checkpoint_id")

        # Graph state routinely contains non-JSON-native types (UUID,
        # datetime, LangChain messages, ...) -- route through the real
        # serializer, not json.dumps. dumps_typed returns (type_str, bytes).
        checkpoint_type, checkpoint_bytes = self.serde.dumps_typed(checkpoint)
        metadata_type, metadata_bytes = self.serde.dumps_typed(dict(metadata))

        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO crdb_checkpoints
                    (thread_id, checkpoint_ns, checkpoint_id,
                     parent_checkpoint_id, checkpoint_type, checkpoint,
                     metadata_type, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
                DO UPDATE SET
                    checkpoint_type = EXCLUDED.checkpoint_type,
                    checkpoint = EXCLUDED.checkpoint,
                    metadata_type = EXCLUDED.metadata_type,
                    metadata = EXCLUDED.metadata
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    parent_id,
                    checkpoint_type,
                    checkpoint_bytes,
                    metadata_type,
                    metadata_bytes,
                ),
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        rows = []
        for idx, (channel, value) in enumerate(writes):
            value_type, value_bytes = self.serde.dumps_typed(value)
            rows.append(
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    idx,
                    channel,
                    value_type,
                    value_bytes,
                )
            )
        if not rows:
            return

        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO crdb_checkpoint_writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id,
                     idx, channel, value_type, value)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                DO NOTHING
                """,
                rows,
            )

    def _load_writes(self, cur, thread_id, checkpoint_ns, checkpoint_id):
        cur.execute(
            """
            SELECT task_id, channel, value_type, value
            FROM crdb_checkpoint_writes
            WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
            ORDER BY task_id, idx
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        )
        return [
            (w["task_id"], w["channel"], self.serde.loads_typed((w["value_type"], w["value"])))
            for w in cur.fetchall()
        ]

    def _row_to_tuple(self, row, pending_writes) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed((row["checkpoint_type"], row["checkpoint"]))
        metadata = self.serde.loads_typed((row["metadata_type"], row["metadata"]))

        parent_config = None
        if row["parent_checkpoint_id"]:
            parent_config = {
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["parent_checkpoint_id"],
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["checkpoint_id"],
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self.pool.connection() as conn, conn.cursor() as cur:
            if checkpoint_id:
                cur.execute(
                    """
                    SELECT * FROM crdb_checkpoints
                    WHERE thread_id = %s AND checkpoint_ns = %s
                      AND checkpoint_id = %s
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM crdb_checkpoints
                    WHERE thread_id = %s AND checkpoint_ns = %s
                    ORDER BY checkpoint_id DESC LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                )
            row = cur.fetchone()
            if row is None:
                return None

            pending_writes = self._load_writes(
                cur, thread_id, checkpoint_ns, row["checkpoint_id"]
            )

        return self._row_to_tuple(row, pending_writes)

    def list(
        self,
        config: Optional[dict],
        *,
        filter: Optional[dict] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        clauses = []
        params: list[Any] = []

        if config is not None:
            clauses.append("thread_id = %s")
            params.append(config["configurable"]["thread_id"])
            ns = config["configurable"].get("checkpoint_ns")
            if ns is not None:
                clauses.append("checkpoint_ns = %s")
                params.append(ns)

        if before is not None:
            clauses.append("checkpoint_id < %s")
            params.append(before["configurable"]["checkpoint_id"])

        # NOTE: metadata is an opaque serialized blob (not JSONB), so we
        # can't push `filter` down into SQL. Over-fetch on the indexed
        # columns above, then filter by decoded metadata in Python below.
        # Fine at support-chat thread-list scale; revisit if a single
        # thread's checkpoint count grows very large.
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        query = f"""
            SELECT * FROM crdb_checkpoints
            {where}
            ORDER BY checkpoint_id DESC
        """

        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            yielded = 0
            for row in rows:
                metadata = self.serde.loads_typed((row["metadata_type"], row["metadata"]))
                if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                    continue

                pending_writes = self._load_writes(
                    cur, row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"]
                )
                yield self._row_to_tuple(row, pending_writes)

                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def delete_thread(self, thread_id: str) -> None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM crdb_checkpoint_writes WHERE thread_id = %s", (thread_id,)
            )
            cur.execute(
                "DELETE FROM crdb_checkpoints WHERE thread_id = %s", (thread_id,)
            )