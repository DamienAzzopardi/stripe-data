import os
import json
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import stripe

# -----------------------
# Config
# -----------------------
CURRENCY = "eur"
TEST_PM = os.getenv("STRIPE_TEST_PAYMENT_METHOD", "pm_card_visa")

PLAN_PRICES_EUR_CENTS = {
    "Basic": 999,
    "Standard": 1999,
    "Premium": 2599,
    "Exec": 4999,
}

PLAN_WEIGHTS = {
    "Basic": 0.55,
    "Standard": 0.25,
    "Premium": 0.15,
    "Exec": 0.05,
}

# New customers per day: random between these bounds
NEW_CUSTOMERS_MIN = int(os.getenv("NEW_CUSTOMERS_MIN", "100"))
NEW_CUSTOMERS_MAX = int(os.getenv("NEW_CUSTOMERS_MAX", "500"))

# Churn knobs (daily probability applied to active subs)
DAILY_CHURN_RATE = float(os.getenv("DAILY_CHURN_RATE", "0.003"))  # ~0.3% / day ≈ ~8-9% / month
CHURN_MODE = os.getenv("CHURN_MODE", "period_end")  # period_end | immediate

# Optional plan change (off by default)
ENABLE_PLAN_CHANGES = os.getenv("ENABLE_PLAN_CHANGES", "false").lower() == "true"
DAILY_PLAN_CHANGE_RATE = float(os.getenv("DAILY_PLAN_CHANGE_RATE", "0.001"))

# Safety (avoid huge API bursts)
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_BETWEEN_CALLS_SEC", "0.02"))

STATE_DIR = Path(os.getenv("STATE_DIR", "state"))
STATE_FILE = STATE_DIR / "world.json"

# -----------------------
# Helpers
# -----------------------
def utc_today_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return {
            "schema_version": 1,
            "created_at": utc_today_date_str(),
            "plans": {},             # plan_name -> {"product_id":..., "price_id":...}
            "customers": {},         # customer_id -> {"created_date":..., "plan":..., "subscription_id":..., "status":...}
            "subscriptions": {},     # subscription_id -> {"customer_id":..., "plan":..., "status":..., "created_date":...}
            "last_run_date": None,
        }
    return json.loads(STATE_FILE.read_text())

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))

def init_stripe():
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]

def pick_plan() -> str:
    plans = list(PLAN_WEIGHTS.keys())
    weights = list(PLAN_WEIGHTS.values())
    return random.choices(plans, weights=weights, k=1)[0]

def find_product_by_name(name: str):
    # Robust: list+filter (search may not be enabled)
    products = stripe.Product.list(active=True, limit=100).data
    for p in products:
        if p.get("name") == name:
            return p
    return None

def find_monthly_price_for_product(product_id: str, unit_amount: int):
    prices = stripe.Price.list(product=product_id, active=True, limit=100).data
    for p in prices:
        if (
            p.get("unit_amount") == unit_amount
            and p.get("currency") == CURRENCY
            and p.get("recurring")
            and p["recurring"].get("interval") == "month"
        ):
            return p
    return None

def ensure_catalog(state: dict) -> dict:
    """
    Ensures 4 products + 4 monthly EUR prices exist.
    Stores IDs in state["plans"][plan].
    """
    for plan, amount in PLAN_PRICES_EUR_CENTS.items():
        product_name = f"DAH Music – {plan}"
        product = find_product_by_name(product_name)
        if not product:
            product = stripe.Product.create(
                name=product_name,
                description=f"{plan} monthly subscription",
                metadata={"domain": "music_streaming", "dah_plan": plan},
            )

        price = find_monthly_price_for_product(product.id, amount)
        if not price:
            price = stripe.Price.create(
                product=product.id,
                unit_amount=amount,
                currency=CURRENCY,
                recurring={"interval": "month"},
                metadata={"domain": "music_streaming", "dah_plan": plan},
            )

        state["plans"][plan] = {"product_id": product.id, "price_id": price.id, "amount": amount}

    return state

def attach_default_payment_method(customer_id: str):
    """
    IMPORTANT:
    A PaymentMethod (pm_...) cannot be attached to multiple customers.
    So we must create a fresh test PaymentMethod per customer, then attach it.
    """
    pm = stripe.PaymentMethod.create(
        type="card",
        card={"token": "tok_visa"},  # test token, creates a new reusable PM
    )
    stripe.PaymentMethod.attach(pm.id, customer=customer_id)
    stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": pm.id})

def create_customer(i: int, run_date: str) -> str:
    # Use run_date in email for easier debugging
    c = stripe.Customer.create(
        email=f"dah_{run_date}_{int(time.time())}_{i}@example.com",
        name=f"DAH User {run_date} #{i}",
        metadata={"domain": "music_streaming", "generator": "daily", "cohort_date": run_date},
    )
    return c.id

def create_subscription(customer_id: str, price_id: str, plan: str, run_date: str) -> str:
    s = stripe.Subscription.create(
        customer=customer_id,
        items=[{"price": price_id}],
        collection_method="charge_automatically",
        payment_behavior="allow_incomplete",
        metadata={"domain": "music_streaming", "dah_plan": plan, "generator": "daily", "start_date": run_date},
    )
    return s.id

def maybe_churn_subscription(subscription_id: str) -> bool:
    if random.random() >= DAILY_CHURN_RATE:
        return False

    if CHURN_MODE == "immediate":
        stripe.Subscription.delete(subscription_id)
    else:
        stripe.Subscription.modify(subscription_id, cancel_at_period_end=True)

    return True

def maybe_change_plan(subscription_id: str, current_plan: str, plans_state: dict) -> str | None:
    if not ENABLE_PLAN_CHANGES:
        return None
    if random.random() >= DAILY_PLAN_CHANGE_RATE:
        return None

    # Choose a different plan
    other_plans = [p for p in plans_state.keys() if p != current_plan]
    if not other_plans:
        return None
    new_plan = random.choice(other_plans)
    new_price_id = plans_state[new_plan]["price_id"]

    # Update the subscription item to the new price
    sub = stripe.Subscription.retrieve(subscription_id)
    item_id = sub["items"]["data"][0]["id"]

    stripe.Subscription.modify(
        subscription_id,
        items=[{"id": item_id, "price": new_price_id}],
        proration_behavior="create_prorations",
        metadata={"domain": "music_streaming", "dah_plan": new_plan, "generator": "daily"},
    )
    return new_plan

def list_active_music_subscriptions(limit: int = 100):
    # Safer than Search; list & filter by metadata
    subs = stripe.Subscription.list(status="active", limit=limit).data
    return [s for s in subs if (s.metadata or {}).get("domain") == "music_streaming"]

# -----------------------
# Main daily run
# -----------------------
def main():
    init_stripe()
    state = load_state()
    run_date = utc_today_date_str()

    # 1) Ensure catalog
    state = ensure_catalog(state)

    # 2) Create new customers + subscriptions
    n_new = random.randint(NEW_CUSTOMERS_MIN, NEW_CUSTOMERS_MAX)
    print(f"[{run_date}] Creating {n_new} new customers")

    for i in range(n_new):
        plan = pick_plan()
        price_id = state["plans"][plan]["price_id"]

        cid = create_customer(i, run_date)
        attach_default_payment_method(cid)
        sid = create_subscription(cid, price_id, plan, run_date)

        state["customers"][cid] = {
            "created_date": run_date,
            "plan": plan,
            "subscription_id": sid,
            "status": "active",
        }
        state["subscriptions"][sid] = {
            "customer_id": cid,
            "plan": plan,
            "status": "active",
            "created_date": run_date,
        }

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    # 3) Mutate existing subscriptions (churn + optional plan changes)
    # We use Stripe as source of truth for "active", but keep state consistent too.
    existing_active = list_active_music_subscriptions(limit=100)  # keep modest to avoid API spam
    churned = 0
    changed = 0

    for s in existing_active:
        sid = s.id
        current_plan = (s.metadata or {}).get("dah_plan") or state["subscriptions"].get(sid, {}).get("plan")

        # Plan change first (optional), then churn
        new_plan = maybe_change_plan(sid, current_plan, state["plans"])
        if new_plan:
            changed += 1
            if sid in state["subscriptions"]:
                state["subscriptions"][sid]["plan"] = new_plan
            # Also reflect on customer
            cid = state["subscriptions"].get(sid, {}).get("customer_id")
            if cid and cid in state["customers"]:
                state["customers"][cid]["plan"] = new_plan

        did_churn = maybe_churn_subscription(sid)
        if did_churn:
            churned += 1
            # Update state
            if sid in state["subscriptions"]:
                state["subscriptions"][sid]["status"] = "canceled"
            cid = state["subscriptions"].get(sid, {}).get("customer_id")
            if cid and cid in state["customers"]:
                state["customers"][cid]["status"] = "canceled"

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    state["last_run_date"] = run_date
    save_state(state)

    print(f"[{run_date}] Done.")
    print(f"  New customers: {n_new}")
    print(f"  Evaluated existing active subs: {len(existing_active)}")
    print(f"  Churn events today: {churned} (mode={CHURN_MODE}, rate={DAILY_CHURN_RATE})")
    print(f"  Plan changes today: {changed} (enabled={ENABLE_PLAN_CHANGES})")
    print(f"  State saved to {STATE_FILE}")


if __name__ == "__main__":
    main()
