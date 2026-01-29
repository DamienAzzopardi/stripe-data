import os
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import stripe

# -----------------------
# Config / Business
# -----------------------
STATE_DIR = Path(os.getenv("STATE_DIR", "state"))
STATE_FILE = STATE_DIR / "world.json"

# New customers per day
NEW_CUSTOMERS_MIN = int(os.getenv("NEW_CUSTOMERS_MIN", "100"))
NEW_CUSTOMERS_MAX = int(os.getenv("NEW_CUSTOMERS_MAX", "500"))

# Failure rate is randomized each run between 2% and 12%
FAIL_RATE_MIN = float(os.getenv("FAIL_RATE_MIN", "0.02"))
FAIL_RATE_MAX = float(os.getenv("FAIL_RATE_MAX", "0.12"))

# Geographic split
EU_SHARE = float(os.getenv("EU_SHARE", "0.80"))  # 80% EU / EUR
US_SHARE = 1.0 - EU_SHARE

# Churn knobs for subscriptions (applies to active subscriptions)
DAILY_CHURN_RATE = float(os.getenv("DAILY_CHURN_RATE", "0.003"))
CHURN_MODE = os.getenv("CHURN_MODE", "period_end")  # period_end | immediate

# Optional plan change
ENABLE_PLAN_CHANGES = os.getenv("ENABLE_PLAN_CHANGES", "false").lower() == "true"
DAILY_PLAN_CHANGE_RATE = float(os.getenv("DAILY_PLAN_CHANGE_RATE", "0.001"))

# API pacing
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_BETWEEN_CALLS_SEC", "0.03"))

# Plan definitions
# Monthly plans available in BOTH currencies (EUR and USD) for geo split
MONTHLY_PLANS = {
    "Basic":  {"eur": 999,  "usd": 1099},
    "Standard": {"eur": 1999, "usd": 2199},
    "Premium": {"eur": 2599, "usd": 2899},
    "Exec": {"eur": 4999, "usd": 5499},
}

# Weekly plan ONLY in EUR for speed (EU-only), per your request
WEEKLY_PLAN = {"Weekly": {"eur": 299}}

# Weights for plan assignment (monthly)
PLAN_WEIGHTS = {
    "Basic": 0.55,
    "Standard": 0.25,
    "Premium": 0.15,
    "Exec": 0.05,
}

# Mix of test card tokens for "successful" payment methods
SUCCESS_CARD_TOKENS = [
    ("visa", "tok_visa"),
    ("mastercard", "tok_mastercard"),
    # a few extras if you want variety
    ("amex", "tok_amex"),
    ("discover", "tok_discover"),
]

# Deterministic failing test tokens (Stripe test mode)
# We'll use these when we decide that a customer should fail on first attempt.
FAIL_CARD_TOKENS = [
    ("declined", "tok_chargeDeclined"),
    ("insufficient_funds", "tok_chargeDeclinedInsufficientFunds"),
]

# -----------------------
# Helpers
# -----------------------
def utc_today_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y_%m_%d")

def utc_today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def init_stripe():
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]

def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return {
            "schema_version": 2,
            "created_at": utc_today_iso(),
            "plans": {},         # key: plan_key -> {"product_id":..., "price_id":...}
            "customers": {},     # customer_id -> {...}
            "subscriptions": {}, # subscription_id -> {...}
            "last_run_date": None,
        }
    return json.loads(STATE_FILE.read_text())

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))

def find_product_by_name(name: str):
    products = stripe.Product.list(active=True, limit=100).data
    for p in products:
        if p.get("name") == name:
            return p
    return None

def find_price_for_product(product_id: str, unit_amount: int, currency: str, interval: str):
    prices = stripe.Price.list(product=product_id, active=True, limit=100).data
    for p in prices:
        if (
            p.get("unit_amount") == unit_amount
            and p.get("currency") == currency
            and p.get("recurring")
            and p["recurring"].get("interval") == interval
        ):
            return p
    return None

def ensure_catalog(state: dict) -> dict:
    """
    Ensure:
      - monthly plans exist in EUR and USD
      - weekly plan exists in EUR
    We store prices under keys like:
      - monthly_basic_eur
      - monthly_basic_usd
      - weekly_weekly_eur
    """
    # Monthly (EUR+USD)
    for plan, amounts in MONTHLY_PLANS.items():
        product_name = f"DAH Music – {plan} (Monthly)"
        product = find_product_by_name(product_name)
        if not product:
            product = stripe.Product.create(
                name=product_name,
                description=f"{plan} monthly subscription",
                metadata={"domain": "music_streaming", "dah_plan": plan, "cadence": "monthly"},
            )

        for currency, amount in amounts.items():
            price = find_price_for_product(product.id, amount, currency, interval="month")
            if not price:
                price = stripe.Price.create(
                    product=product.id,
                    unit_amount=amount,
                    currency=currency,
                    recurring={"interval": "month"},
                    metadata={"domain": "music_streaming", "dah_plan": plan, "cadence": "monthly"},
                )
            key = f"monthly_{plan.lower()}_{currency}"
            state["plans"][key] = {"product_id": product.id, "price_id": price.id, "amount": amount, "currency": currency, "interval": "month"}

    # Weekly (EUR only)
    for plan, amounts in WEEKLY_PLAN.items():
        product_name = f"DAH Music – {plan} (Weekly)"
        product = find_product_by_name(product_name)
        if not product:
            product = stripe.Product.create(
                name=product_name,
                description=f"{plan} weekly subscription",
                metadata={"domain": "music_streaming", "dah_plan": plan, "cadence": "weekly"},
            )

        amount = amounts["eur"]
        price = find_price_for_product(product.id, amount, "eur", interval="week")
        if not price:
            price = stripe.Price.create(
                product=product.id,
                unit_amount=amount,
                currency="eur",
                recurring={"interval": "week"},
                metadata={"domain": "music_streaming", "dah_plan": plan, "cadence": "weekly"},
            )
        key = "weekly_weekly_eur"
        state["plans"][key] = {"product_id": product.id, "price_id": price.id, "amount": amount, "currency": "eur", "interval": "week"}

    return state

def pick_geo() -> tuple[str, str]:
    """
    Returns (region, currency).
    """
    if random.random() < EU_SHARE:
        return ("EU", "eur")
    return ("US", "usd")

def pick_plan_for_customer(region: str) -> tuple[str, str]:
    """
    Returns (cadence, plan_name).
    Weekly is available only for EU/EUR and is injected to speed up renewals.
    We'll allocate a small share of EU customers to weekly plan.
    """
    if region == "EU":
        # 12% weekly among EU customers to accelerate visible renewals (tunable)
        if random.random() < 0.12:
            return ("weekly", "Weekly")

    # monthly plans
    plans = list(PLAN_WEIGHTS.keys())
    weights = list(PLAN_WEIGHTS.values())
    return ("monthly", random.choices(plans, weights=weights, k=1)[0])

def pick_card_token(should_fail: bool) -> tuple[str, str]:
    """
    Returns (label, token).
    """
    if should_fail:
        return random.choice(FAIL_CARD_TOKENS)
    return random.choice(SUCCESS_CARD_TOKENS)

def create_customer(run_date_yyyy_mm_dd: str, n: int, region: str, currency: str) -> str:
    """
    Enforces naming pattern:
      dah_user_yyyy_mm_dd_#N
    """
    name = f"dah_user_{run_date_yyyy_mm_dd}_#{n}"
    email = f"{name}@example.com"

    c = stripe.Customer.create(
        email=email,
        name=name,
        address={"country": "FR"} if region == "EU" else {"country": "US"},
        metadata={
            "domain": "music_streaming",
            "generator": "daily",
            "cohort_date": run_date_yyyy_mm_dd,
            "region": region,
            "currency": currency,
            "user_handle": name,
        },
    )
    return c.id

def attach_default_payment_method(customer_id: str, card_token: str, card_brand_label: str) -> str:
    """
    Creates a fresh PaymentMethod per customer (required).
    Uses Stripe test tokens (tok_visa, tok_mastercard, etc.).
    Returns payment_method_id.
    """
    pm = stripe.PaymentMethod.create(
        type="card",
        card={"token": card_token},
        metadata={"generator": "daily", "card_brand": card_brand_label, "domain": "music_streaming"},
    )
    stripe.PaymentMethod.attach(pm.id, customer=customer_id)
    stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": pm.id})
    return pm.id

def price_key_for(cadence: str, plan: str, currency: str) -> str:
    if cadence == "weekly":
        return "weekly_weekly_eur"
    return f"monthly_{plan.lower()}_{currency}"

def create_subscription(customer_id: str, price_id: str, cadence: str, plan: str, currency: str, region: str, run_date: str) -> str:
    s = stripe.Subscription.create(
        customer=customer_id,
        items=[{"price": price_id}],
        collection_method="charge_automatically",
        payment_behavior="allow_incomplete",
        metadata={
            "domain": "music_streaming",
            "dah_plan": plan,
            "cadence": cadence,
            "region": region,
            "currency": currency,
            "generator": "daily",
            "start_date": run_date,
        },
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

def maybe_change_plan(subscription_id: str, current_plan: str, current_currency: str, plans_state: dict) -> str | None:
    if not ENABLE_PLAN_CHANGES:
        return None
    if random.random() >= DAILY_PLAN_CHANGE_RATE:
        return None

    # Only change among monthly plans (keep weekly simple)
    if current_plan == "Weekly":
        return None

    other_plans = [p for p in PLAN_WEIGHTS.keys() if p != current_plan]
    if not other_plans:
        return None

    new_plan = random.choice(other_plans)
    new_key = f"monthly_{new_plan.lower()}_{current_currency}"
    if new_key not in plans_state:
        return None

    new_price_id = plans_state[new_key]["price_id"]

    sub = stripe.Subscription.retrieve(subscription_id)
    item_id = sub["items"]["data"][0]["id"]

    stripe.Subscription.modify(
        subscription_id,
        items=[{"id": item_id, "price": new_price_id}],
        proration_behavior="create_prorations",
        metadata={**(sub.metadata or {}), "dah_plan": new_plan},
    )
    return new_plan

def list_active_music_subscriptions(limit: int = 100):
    subs = stripe.Subscription.list(status="active", limit=limit).data
    return [s for s in subs if (s.metadata or {}).get("domain") == "music_streaming"]

def pay_latest_invoice_for_subscription(subscription_id: str, should_fail: bool) -> bool:
    """
    Forces an explicit payment attempt so we can control success/failure rates.
    We:
      - fetch latest invoice
      - if there's a PaymentIntent, confirm it
    Returns True if payment succeeded, False otherwise.
    """
    sub = stripe.Subscription.retrieve(subscription_id, expand=["latest_invoice.payment_intent"])
    inv = sub.get("latest_invoice")
    if not inv:
        return True  # nothing to pay yet

    pi = (inv.get("payment_intent") or {})
    pi_id = pi.get("id")
    if not pi_id:
        # Sometimes no payment_intent is created (depending on settings).
        return True

    try:
        # Confirm again is safe; if already succeeded, Stripe will just return succeeded.
        # For failing cards, Stripe should leave it in a failed/requires_payment_method status.
        stripe.PaymentIntent.confirm(
            pi_id,
            # prevent redirect issues
            automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
        )
        pi2 = stripe.PaymentIntent.retrieve(pi_id)
        return pi2.status == "succeeded"
    except stripe.error.StripeError:
        return False

# -----------------------
# Main
# -----------------------
def main():
    init_stripe()
    state = load_state()
    run_date = utc_today_iso()
    run_date_key = utc_today_date_str()

    # Randomize today's failure target between 2% and 12%
    failure_rate_today = random.uniform(FAIL_RATE_MIN, FAIL_RATE_MAX)

    # 1) Ensure catalog exists
    state = ensure_catalog(state)

    # 2) New customers
    n_new = random.randint(NEW_CUSTOMERS_MIN, NEW_CUSTOMERS_MAX)
    print(f"[{run_date}] Creating {n_new} new customers | target failure rate ~{failure_rate_today:.2%}")

    successes = 0
    failures = 0
    created_subs = 0

    for n in range(1, n_new + 1):
        region, currency = pick_geo()
        cadence, plan = pick_plan_for_customer(region)

        # weekly is EU/EUR only
        if cadence == "weekly":
            currency = "eur"
            region = "EU"

        # decide if this customer will fail initial payment attempt
        should_fail = random.random() < failure_rate_today

        card_label, card_token = pick_card_token(should_fail)

        # Create customer with required naming pattern
        cid = create_customer(run_date_key, n, region, currency)

        # Create a fresh payment method per customer (random card type)
        pm_id = attach_default_payment_method(cid, card_token, card_label)

        # Determine price
        pkey = price_key_for(cadence, plan, currency)
        price_id = state["plans"][pkey]["price_id"]

        # Create subscription
        sid = create_subscription(cid, price_id, cadence, plan, currency, region, run_date)
        created_subs += 1

        # Force a payment attempt and record success/failure (best-effort)
        paid_ok = pay_latest_invoice_for_subscription(sid, should_fail=should_fail)
        if paid_ok:
            successes += 1
        else:
            failures += 1

        # Save state for consistency/debug
        state["customers"][cid] = {
            "created_date": run_date,
            "region": region,
            "currency": currency,
            "plan": plan,
            "cadence": cadence,
            "subscription_id": sid,
            "payment_method_id": pm_id,
            "initial_payment_expected_to_fail": should_fail,
            "status": "active",
        }
        state["subscriptions"][sid] = {
            "customer_id": cid,
            "plan": plan,
            "cadence": cadence,
            "region": region,
            "currency": currency,
            "status": "active",
            "created_date": run_date,
        }

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    # 3) Mutate existing active subs (churn + optional plan change)
    existing_active = list_active_music_subscriptions(limit=100)
    churned = 0
    changed = 0

    for s in existing_active:
        sid = s.id
        meta = s.metadata or {}
        current_plan = meta.get("dah_plan") or state["subscriptions"].get(sid, {}).get("plan")
        current_currency = meta.get("currency") or state["subscriptions"].get(sid, {}).get("currency") or "eur"

        new_plan = maybe_change_plan(sid, current_plan, current_currency, state["plans"])
        if new_plan:
            changed += 1
            if sid in state["subscriptions"]:
                state["subscriptions"][sid]["plan"] = new_plan
            cid = state["subscriptions"].get(sid, {}).get("customer_id")
            if cid and cid in state["customers"]:
                state["customers"][cid]["plan"] = new_plan

        if maybe_churn_subscription(sid):
            churned += 1
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
    print(f"  New subscriptions: {created_subs}")
    print(f"  Payment success: {successes} | failures: {failures} | observed success rate: {(successes / max(1, successes + failures)):.2%}")
    print(f"  Existing active evaluated: {len(existing_active)} | churn events: {churned} | plan changes: {changed}")
    print(f"  Geo split target: EU {EU_SHARE:.0%} / US {US_SHARE:.0%} (actual stored in customer metadata)")
    print(f"  State saved to {STATE_FILE}")

if __name__ == "__main__":
    main()
