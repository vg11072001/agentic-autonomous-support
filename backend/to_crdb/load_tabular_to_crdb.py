"""
Load the Phase-1 tabular dataset into a real CockroachDB cluster.

Usage:
  export CRDB_URL="postgresql://user:pass@<cluster-host>:26257/defaultdb?sslmode=verify-full"
  pip install psycopg[binary] pandas --break-system-packages
  python3 06_load_tabular_to_crdb.py
"""
from dotenv import load_dotenv
import argparse
import os
import csv
import re
import sys
import time
import random
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import pandas as pd
import psycopg

load_dotenv()

def _with_query_param(url, key, value):
    parts = urlsplit(url)
    params = parse_qsl(parts.query, keep_blank_values=True)
    params = [(k, v) for k, v in params if k != key]
    params.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def _get_crdb_url():
    url = os.environ.get("CRDB_URL") or os.environ.get("COCKROACH_URL")
    if url:
        return url

    try:
        import subprocess
        url = subprocess.check_output(
            ["ccloud", "cluster", "sql", "autosupport", "--database", "defaultdb", "--connection-url"],
            text=True,
        ).strip()
        if url:
            return url
    except Exception:
        pass

    return None


CRDB_URL = _get_crdb_url()
if CRDB_URL and "sslrootcert=" not in CRDB_URL:
    cert_path = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else "system"
    CRDB_URL = _with_query_param(CRDB_URL, "sslrootcert", cert_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "1_transactional_data")
SCHEMA_PATH = os.path.join(BASE_DIR, "05_schema.sql")

# order matters: respects FK dependencies (customers -> orders -> items/payments/refunds -> profile/tickets)
LOAD_ORDER = [
    ("customers.csv", "customers",
     ["customer_id", "name", "email", "country", "city", "state", "signup_date"]),
    ("products.csv", "products",
     ["product_id", "name", "category", "price", "rating", "in_stock", "image_url"]),
    ("orders.csv", "orders",
     ["order_id", "customer_id", "order_date", "status", "total_amount"]),
    ("order_items.csv", "order_items",
     ["order_item_id", "order_id", "product_id", "unit_price"]),
    ("payments.csv", "payments",
     ["payment_id", "order_id", "method", "amount", "status", "billing_country"]),
    ("refunds.csv", "refunds",
     ["refund_id", "order_id", "customer_id", "reason", "amount", "requested_date", "status"]),
    ("fraud_flags.csv", "fraud_flags",
     ["customer_id", "fraud_score", "flag_reason", "flagged_at"]),
    ("customer_profile.csv", "customer_profile",
     ["customer_id", "tenure_days", "total_orders", "total_spent", "days_since_last_order",
      "churn_risk_score", "fraud_flag", "fraud_score"]),
    ("support_tickets.csv", "support_tickets",
     ["ticket_id", "customer_id", "order_id", "category", "priority", "channel",
      "created_at", "resolved_at", "resolution_type"]),
]


def with_retries(conn, fn, label, max_attempts=7):
    """
    Run fn(conn) with automatic client-side retry on CockroachDB
    SerializationFailure (SQLSTATE 40001). fn should perform its own
    work inside a `with conn.transaction():` block so failures roll
    back automatically before we retry.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(conn)
        except psycopg.errors.SerializationFailure:
            # conn.transaction() already rolled back automatically here;
            # no manual conn.rollback() needed (and calling it again can
            # raise "no transaction in progress").
            if attempt == max_attempts:
                raise
            backoff = min(0.5 * (2 ** (attempt - 1)), 15) + random.uniform(0, 0.5)
            print(f"  retrying {label} after serialization failure "
                  f"(attempt {attempt}/{max_attempts}, backoff {backoff:.1f}s)")
            time.sleep(backoff)


def wait_for_schema_jobs(conn, timeout_s=60, poll_interval_s=1.0):
    """
    CREATE TABLE / ALTER TABLE (indexes, FKs) kick off async jobs
    (backfills, FK validation, stats collection) that keep touching
    the affected key spans after the DDL statement itself returns.
    Writing into those spans immediately afterward causes repeated
    SerializationFailure on the exact same key. Poll SHOW JOBS
    (the supported interface -- crdb_internal is restricted on
    CockroachDB Cloud) until those jobs finish, or until timeout.
    """
    conn.autocommit = True
    try:
        deadline = time.time() + timeout_s
        printed = False
        while time.time() < deadline:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM [SHOW JOBS] "
                    "WHERE status IN ('running', 'pending', 'pause-requested')"
                )
                (pending,) = cur.fetchone()
            if pending == 0:
                if printed:
                    print("  schema-change jobs finished.")
                return
            if not printed:
                print(f"  waiting for {pending} background schema job(s) to finish...")
                printed = True
            time.sleep(poll_interval_s)
        print("  warning: timed out waiting for schema jobs; proceeding anyway.")
    finally:
        conn.autocommit = False


# def run_schema(conn, reset=False):
#     with open(SCHEMA_PATH) as f:
#         sql = f.read()

#     if reset:
#         def _drop(conn):
#             with conn.transaction():
#                 with conn.cursor() as cur:
#                     for obj_name in [
#                         "cohort_resolution_patterns",
#                         "case_state",
#                         "conversations",
#                         "escalations",
#                         "evaluation_results",
#                         "idempotency_keys",
#                         "kb_chunks",
#                         "tool_calls",
#                         "workflow_checkpoints",
#                         "support_tickets",
#                         "guardrail_verdict_cache",
#                         "customer_profile",
#                         "fraud_flags",
#                         "refunds",
#                         "payments",
#                         "order_items",
#                         "orders",
#                         "products",
#                         "customers",
#                     ]:
#                         if obj_name == "cohort_resolution_patterns":
#                             cur.execute("DROP MATERIALIZED VIEW IF EXISTS cohort_resolution_patterns")
#                         else:
#                             cur.execute(f"DROP TABLE IF EXISTS {obj_name} CASCADE")

#         with_retries(conn, _drop, "schema reset (drop)")

#     def _create(conn):
#         with conn.transaction():
#             with conn.cursor() as cur:
#                 cur.execute(sql)

#     with_retries(conn, _create, "schema apply (create)")
#     print("Schema applied.")

#     # Let index/FK backfill + stats jobs finish before we hammer the
#     # same tables with inserts -- this is what was causing the repeated
#     # SerializationFailure on order_items right after schema apply.
#     wait_for_schema_jobs(conn)


TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\"?([a-zA-Z_][\w]*)\"?",
    re.IGNORECASE,
)
VIEW_RE = re.compile(
    r"CREATE\s+MATERIALIZED\s+VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?\"?([a-zA-Z_][\w]*)\"?",
    re.IGNORECASE,
)


def _extract_schema_objects(sql):
    """Pull table and materialized view names directly out of the schema SQL,
    so the drop list can never drift out of sync with what CREATE actually defines."""
    tables = TABLE_RE.findall(sql)
    views = VIEW_RE.findall(sql)
    # views are matched by both patterns' loose forms only if named "TABLE" too,
    # but since VIEW_RE requires MATERIALIZED VIEW specifically, no overlap in practice.
    return tables, views


def run_schema(conn, reset=False):
    with open(SCHEMA_PATH) as f:
        sql = f.read()

    if reset:
        tables, views = _extract_schema_objects(sql)

        def _drop(conn):
            with conn.transaction(), conn.cursor() as cur:
                for view in views:
                    cur.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view}")
                # CASCADE covers FK dependency order, so extraction order
                # (or reversing it) doesn't need to be exact.
                for table in reversed(tables):
                    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

        with_retries(conn, _drop, "schema reset (drop)")

    def _create(conn):
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(sql)

    with_retries(conn, _create, "schema apply (create)")
    print("Schema applied.")

    # Let index/FK backfill + stats jobs finish before we hammer the
    # same tables with inserts -- this is what was causing the repeated
    # SerializationFailure on order_items right after schema apply.
    wait_for_schema_jobs(conn)


BOOL_COLUMNS = {"fraud_flag", "in_stock"}


def resolve_targets(selected_files=None, selected_tables=None):
    selected_files = {f.lower() for f in (selected_files or [])}
    selected_tables = {t.lower() for t in (selected_tables or [])}

    if not selected_files and not selected_tables:
        return list(LOAD_ORDER)

    matches = []
    for filename, table, columns1 in LOAD_ORDER:
        name = filename.lower()
        table_name = table.lower()
        columns = pd.read_csv(os.path.join(DATA_DIR, filename), nrows=0).columns.tolist()
        if selected_files and name in selected_files:
            matches.append((filename, table, columns))
        elif selected_tables and table_name in selected_tables:
            matches.append((filename, table, columns))

    if not matches:
        unknown = sorted(selected_files | selected_tables)
        raise ValueError(f"No matching upload targets found for: {', '.join(unknown)}")

    return matches


def load_csv(conn, filename, table, columns):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"  skip {filename} (not found)")
        return

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = []
            for c in columns:
                v = r[c]
                if v == "":
                    v = None
                elif c in BOOL_COLUMNS:
                    v = str(v).strip().lower() in ("true", "1", "t")
                row.append(v)
            rows.append(tuple(row))

    if not rows:
        return

    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    BATCH = 200

    # IMPORTANT: each batch is its own short-lived transaction, retried
    # independently. Previously all batches for a file shared ONE
    # transaction with a 10s sleep between batches -- that held a single
    # transaction open for minutes, accumulating more written keys and
    # making a serialization conflict far more likely. Under SERIALIZABLE,
    # smaller and shorter transactions are what you want, not sleeps
    # inside an open one.
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]

        def _insert(conn, chunk=chunk):
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(sql, chunk)

        with_retries(conn, _insert, f"{table} [{i}:{i + len(chunk)}]")

    print(f"  loaded {len(rows)} rows -> {table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load tabular CSV data into CockroachDB")
    parser.add_argument("--file", dest="files", nargs="+", help="CSV file name(s) to upload, e.g. order_items.csv")
    parser.add_argument("--table", dest="tables", nargs="+", help="Table name(s) to upload, e.g. order_items")
    parser.add_argument("--reset-schema", action="store_true", help="Drop and recreate the schema before loading")
    parser.add_argument("--skip-schema", action="store_true", help="Skip DDL entirely and only load data")
    args = parser.parse_args()

    if not CRDB_URL:
        sys.exit("Set CRDB_URL env var to your CockroachDB Cloud connection string first.")

    try:
        targets = resolve_targets(args.files, args.tables)
    except ValueError as exc:
        sys.exit(str(exc))

    with psycopg.connect(CRDB_URL) as conn:
        if not args.skip_schema:
            run_schema(conn, reset=args.reset_schema)
        else:
            print("Skipping schema step (--skip-schema).")

        print("\nLoading tabular data:")
        for filename, table, cols in targets:
            load_csv(conn, filename, table, cols)

    print("\nDone. Verify with:")
    print("  SELECT segment, count(*) FROM customer_profile GROUP BY segment;")