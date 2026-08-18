"""
Phase 1 — Tabular / transactional data.

Public data hooks:
  - olist_customers_dataset.csv, olist_orders_dataset.csv, olist_order_payments_dataset.csv
    -> Kaggle: olistbr/brazilian-ecommerce
  - ieee_fraud_transaction.csv
    -> Kaggle: ieee-fraud-detection
"""
import os
import json
import random
from datetime import datetime, timedelta
import csv
from faker import Faker
import uuid
random.seed(42)
fake = Faker()
Faker.seed(42)

OUT = "backend/data/1_transactional_data"
PUBLIC = "backend/data/public_data"
os.makedirs(OUT, exist_ok=True)

N_CUSTOMERS = 2000
N_ORDERS = 6000
NOW = datetime(2026, 7, 17)

CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports", "Toys", "Books"]
TICKET_CATEGORIES = ["ORDER_STATUS", "REFUND", "SHIPPING_DELAY", "DAMAGED_ITEM",
                      "ACCOUNT", "PAYMENT_ISSUE", "FRAUD_REVIEW", "CANCELLATION"]


def load_public_or_none(name):
    path = os.path.join(PUBLIC, name)
    return path if os.path.exists(path) else None


def gen_customers():
    src = load_public_or_none("olist_customers_dataset.csv")
    customers = []
    if src:
        import pandas as pd
        df = pd.read_csv(src)
        df = df.drop_duplicates(subset="customer_unique_id").head(N_CUSTOMERS)
        print(f"Loaded {len(df)} unique customers from Olist public dataset")
        for _, row in df.iterrows():
            tenure_days = int(random.paretovariate(1.2) * 20)
            tenure_days = min(tenure_days, 1500)
            signup = NOW - timedelta(days=tenure_days)
            customers.append({
                "customer_id": str(row["customer_unique_id"]),
                "name": fake.name(),  # Olist ships anonymized data, no real names -- synthesize
                "email": fake.email(),
                "country": "BR",
                "city": row.get("customer_city", fake.city()),
                "state": row.get("customer_state", ""),
                "signup_date": signup.date().isoformat(),
            })
    else:
        print("No public_data/olist_customers_dataset.csv found -- using synthetic customers")
        for _ in range(N_CUSTOMERS):
            tenure_days = int(random.paretovariate(1.2) * 20)
            tenure_days = min(tenure_days, 1500)
            signup = NOW - timedelta(days=tenure_days)
            customers.append({
                "customer_id": str(uuid.uuid4()),
                "name": fake.name(),
                "email": fake.email(),
                "country": random.choices(["US", "IN", "GB", "DE", "BR"], weights=[50, 20, 10, 10, 10])[0],
                "city": fake.city(),
                "state": "",
                "signup_date": signup.date().isoformat(),
            })
    return customers


def load_olist_payment_value_distribution():
    src = load_public_or_none("olist_order_payments_dataset.csv")
    if not src:
        return None
    import pandas as pd
    df = pd.read_csv(src)
    values = [v for v in df["payment_value"].tolist() if 1 <= v <= 2000]
    print(f"Loaded {len(values)} real order values from Olist payments dataset")
    return values


def sample_order_total(chosen_products, olist_value_pool):
    """Blend real Olist order-value spread with the synthetic cart total,
    rather than either pure-synthetic or pure-Olist (which has no product
    linkage to our synthetic catalog)."""
    base = sum(p["price"] for p in chosen_products)
    if olist_value_pool:
        real_sample = random.choice(olist_value_pool)
        noise_factor = real_sample / (sum(olist_value_pool) / len(olist_value_pool))
        return round(base * min(max(noise_factor, 0.5), 2.5), 2)
    return round(base, 2)

def gen_products(n=300):
    products = []
    for _ in range(n):
        cat = random.choice(CATEGORIES)
        products.append({
            "product_id": str(uuid.uuid4()),
            "name": fake.catch_phrase(),
            "category": cat,
            "price": round(random.uniform(8, 450), 2),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "in_stock": random.random() > 0.08,  # ~92% in stock, matches real retail patterns
            "image_url": None,  # NULL -- UI falls back to category icon tile until real photos exist
        })
    return products


def gen_orders_and_related(customers, products, olist_value_pool=None):
    orders, items, payments, refunds = [], [], [], []
    cust_ids = [c["customer_id"] for c in customers]
    cust_signup = {c["customer_id"]: datetime.fromisoformat(c["signup_date"]) for c in customers}
    # order frequency per customer is uneven -> some "power users", many one-timers
    weights = [random.paretovariate(1.5) for _ in cust_ids]

    for _ in range(N_ORDERS):
        cid = random.choices(cust_ids, weights=weights, k=1)[0]
        earliest = cust_signup[cid]
        order_date = earliest + timedelta(days=random.randint(0, max(1, (NOW - earliest).days)))
        oid = str(uuid.uuid4())
        n_items = random.randint(1, 4)
        chosen = random.sample(products, n_items)
        total = sample_order_total(chosen, olist_value_pool)

        status = random.choices(
            ["delivered", "shipped", "cancelled", "returned", "delayed"],
            weights=[70, 10, 6, 8, 6],
        )[0]

        orders.append({
            "order_id": oid, "customer_id": cid, "order_date": order_date.isoformat(),
            "status": status, "total_amount": total,
        })
        for p in chosen:
            items.append({
                "order_item_id": str(uuid.uuid4()), "order_id": oid,
                "product_id": p["product_id"], "unit_price": p["price"],
            })

        pay_status = "failed" if random.random() < 0.03 else "completed"
        payments.append({
            "payment_id": str(uuid.uuid4()), "order_id": oid,
            "method": random.choice(["credit_card", "debit_card", "paypal", "wallet"]),
            "amount": total, "status": pay_status,
            "billing_country": random.choice(["US", "IN", "GB", "DE", "BR", "NG"]),
        })

        if status == "returned" or random.random() < 0.04:
            refunds.append({
                "refund_id": str(uuid.uuid4()), "order_id": oid, "customer_id": cid,
                "reason": random.choice(["damaged", "wrong_item", "not_as_described",
                                          "changed_mind", "never_arrived"]),
                "amount": total,
                "requested_date": (order_date + timedelta(days=random.randint(1, 20))).isoformat(),
                "status": random.choices(["approved", "pending", "denied"], weights=[70, 20, 10])[0],
            })
    return orders, items, payments, refunds

def compute_fraud_flags(customers, orders, payments, refunds):
    orders_by_cust = {}
    for o in orders:
        orders_by_cust.setdefault(o["customer_id"], []).append(o)

    refunds_by_cust = {}
    for r in refunds:
        refunds_by_cust.setdefault(r["customer_id"], []).append(r)

    cust_signup = {c["customer_id"]: datetime.fromisoformat(c["signup_date"]) for c in customers}

    fraud_rows = []
    for cid, custs_orders in orders_by_cust.items():
        signup = cust_signup[cid]
        early_order_burst = sum(
            1 for o in custs_orders
            if (datetime.fromisoformat(o["order_date"]) - signup).days < 2
        )
        refund_count = len(refunds_by_cust.get(cid, []))
        refund_value = sum(r["amount"] for r in refunds_by_cust.get(cid, []))
        high_value_orders = sum(1 for o in custs_orders if o["total_amount"] > 300)

        score = 0.0
        score += min(early_order_burst * 0.18, 0.5)      # signup -> rapid high orders
        score += min(refund_count * 0.12, 0.4)
        score += min(refund_value / 3000, 0.3)
        score += 0.15 if high_value_orders >= 3 else 0
        score += random.uniform(-0.05, 0.05)              # small noise
        score = max(0.0, min(1.0, score))

        if score > 0.55:
            fraud_rows.append({
                "customer_id": cid,
                "fraud_score": round(score, 3),
                "flag_reason": "order_velocity_post_signup" if early_order_burst >= 2 else "refund_abuse_pattern",
                "flagged_at": NOW.isoformat(),
            })
    return fraud_rows

def compute_customer_profile(customers, orders, fraud_rows):
    fraud_by_cust = {f["customer_id"]: f for f in fraud_rows}
    orders_by_cust = {}
    for o in orders:
        orders_by_cust.setdefault(o["customer_id"], []).append(o)

    profiles = []
    for c in customers:
        cid = c["customer_id"]
        signup = datetime.fromisoformat(c["signup_date"])
        tenure_days = (NOW - signup).days
        custs_orders = sorted(orders_by_cust.get(cid, []), key=lambda o: o["order_date"])
        total_orders = len(custs_orders)
        total_spent = round(sum(o["total_amount"] for o in custs_orders), 2)

        if custs_orders:
            last_order = datetime.fromisoformat(custs_orders[-1]["order_date"])
            days_since_last_order = (NOW - last_order).days
        else:
            days_since_last_order = tenure_days
        recency_ratio = days_since_last_order / max(tenure_days, 1)
        frequency_penalty = 1 / (1 + total_orders)
        churn_risk = min(1.0, 0.6 * recency_ratio + 0.4 * frequency_penalty)
        churn_risk = round(max(0.0, churn_risk + random.uniform(-0.05, 0.05)), 3)

        fraud_flag = cid in fraud_by_cust
        fraud_score = fraud_by_cust.get(cid, {}).get("fraud_score", 0.0)
        profiles.append({
            "customer_id": cid, "tenure_days": tenure_days, "total_orders": total_orders,
            "total_spent": total_spent, "days_since_last_order": days_since_last_order,
            "churn_risk_score": churn_risk, "fraud_flag": fraud_flag,
            "fraud_score": fraud_score,
        })
    return profiles


def _gen_behavior(p):
    """GENERATION-ONLY coarse behavior, derived from raw signals purely to
    vary synthetic ticket volume/tone. NOT stored, NOT used for routing —
    the production path never collapses a customer to one of these."""
    if p.get("fraud_flag"):
        return "fraud"
    if (p.get("churn_risk_score", 0) or 0) > 0.7 and (p.get("total_orders", 0) or 0) >= 1:
        return "churn_risk"
    if (p.get("tenure_days", 0) or 0) < 30 or (p.get("total_orders", 0) or 0) <= 1:
        return "new"
    return "existing"

def gen_support_tickets(customer_profiles, orders, refunds):
    orders_by_cust = {}
    for o in orders:
        orders_by_cust.setdefault(o["customer_id"], []).append(o)
    refunds_by_order = {r["order_id"]: r for r in refunds}

    tickets = []
    for p in customer_profiles:
        cid = p["customer_id"]
        custs_orders = orders_by_cust.get(cid, [])
        # ticket rate varies by segment -- fraud/churn customers contact support more
        behavior = _gen_behavior(p)   # generation-only; not stored
        base_rate = {"new": 0.15, "existing": 0.20, "churn_risk": 0.45, "fraud": 0.6}[behavior]
        for o in custs_orders:
            if random.random() > base_rate:
                continue
            if o["order_id"] in refunds_by_order:
                category = "REFUND"
            elif o["status"] == "delayed":
                category = "SHIPPING_DELAY"
            elif behavior == "fraud" and random.random() < 0.25:
                category = "FRAUD_REVIEW"
            else:
                category = random.choice([c for c in TICKET_CATEGORIES
                                          if c != "FRAUD_REVIEW"])

            created = datetime.fromisoformat(o["order_date"]) + timedelta(days=random.randint(0, 15))
            resolved = created + timedelta(hours=random.randint(1, 72))
            tickets.append({
                "ticket_id": str(uuid.uuid4()), "customer_id": cid, "order_id": o["order_id"],
                "category": category,
                "priority": "high" if category == "FRAUD_REVIEW" else random.choice(["low", "medium", "high"]),
                "channel": random.choice(["chat", "email", "app"]),
                "created_at": created.isoformat(),
                "resolved_at": resolved.isoformat(),
                "resolution_type": random.choice(
                    ["auto_resolved", "escalated_to_human", "refund_issued", "info_provided"]
                ) if category != "FRAUD_REVIEW" else "escalated_to_human",
                "gen_behavior": behavior,
            })
    return tickets


def write_csv(rows, name):
    if not rows:
        return
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):>6} rows -> {path}")


if __name__ == "__main__":
    customers = gen_customers()
    products = gen_products()
    olist_value_pool = load_olist_payment_value_distribution()
    orders, items, payments, refunds = gen_orders_and_related(customers, products, olist_value_pool)
    fraud_rows = compute_fraud_flags(customers, orders, payments, refunds)
    profiles = compute_customer_profile(customers, orders, fraud_rows)
    tickets = gen_support_tickets(profiles, orders, refunds)

    write_csv(customers, "customers.csv")
    write_csv(products, "products.csv")
    write_csv(orders, "orders.csv")
    write_csv(items, "order_items.csv")
    write_csv(payments, "payments.csv")
    write_csv(refunds, "refunds.csv")
    write_csv(fraud_rows, "fraud_flags.csv")
    write_csv(profiles, "customer_profile.csv")
    write_csv(tickets, "support_tickets.csv")

    seg_counts = {}
    for p in profiles:
        b = _gen_behavior(p)
        seg_counts[b] = seg_counts.get(b, 0) + 1
    print("\nBehavior distribution (generation-only, not stored):", json.dumps(seg_counts, indent=2))


