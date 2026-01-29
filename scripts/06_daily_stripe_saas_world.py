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

NEW_CUSTOMERS_MIN = int(os.getenv("NEW_CUSTOMERS_MIN", "100"))
NEW_CUSTOMERS_MAX = int(os.getenv("NEW_CUSTOMERS_MAX", "500"))

# Failure rate randomized each run between 2% and 12%
FAIL_RATE_MIN = float(os.getenv("FAIL_RATE_MIN", "0.02"))
FAIL_RATE_MAX = float(os.getenv("FAIL_RATE_MAX", "0.12"))

# Geographic split
EU_SHARE = float(os.getenv("EU_SHARE", "0.80"))  # 80% EU / EUR
US_SHARE = 1.0 - EU_SHARE

# Churn knobs for existing subscriptions
DAILY_CHURN_RATE = float(os.getenv("DAILY_CHURN_RATE", "0.003"))
CHURN_MODE = os.getenv("CHURN_MODE", "period_end")  # period_end | immediate

# Optional plan change
ENABLE_PLAN_CHANGES = os.getenv("ENABLE_PLAN_CHANGES", "false").lower() == "true"
DAILY_PLAN_CHANGE_RATE = float(os.getenv("DAILY_PLAN_CHANGE_RATE", "0.001"))

# API pacing
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_BETWEEN_CALLS_SEC", "0.05"))
MAX_EXISTING_ACTIVE_TO_MUTATE = int(os.getenv("MAX_EXISTING_ACTIVE_TO_MUTATE", "200"))

# Monthly plans available in BOTH currencies (EUR and USD)
MONTHLY_PLANS = {
    "Basic": {"eur": 999, "usd": 1099},
    "Standard": {"eur": 1999, "usd": 2199},
    "Premium": {"eur": 2599, "usd": 2899},
    "Exec": {"eur": 4999, "usd": 5499},
}

# Weekly plan ONLY in EUR (to see renewals faster)
WEEKLY_PLAN = {"Weekly": {"eur": 299}}

# Weights for monthly plan assignment
PLAN_WEIGHTS = {
    "Basic": 0.55,
    "Standard": 0.25,
    "Premium": 0.15,
    "Exec": 0.05,
}

# -----------------------
# Payment method mix (clean Stripe-recommended tokens)
# -----------------------
# NOTE: last4 will be fixed per token (Stripe limitation).
# This is the clean approach and does not require raw card data APIs.
SUCCESS_PM_TOKENS = [
    ("visa", "tok_visa"),
    ("mastercard", "tok_mastercard"),
    ("amex", "tok_amex"),
    ("discover", "tok_discover"),
    ("jcb", "tok_jcb"),
]

# Weight the mix (mostly Visa + Mastercard)
PM_WEIGHTS = {
    "visa": 0.50,
    "mastercard": 0.40,
    "amex": 0.05,
    "discover": 0.03,
    "jcb": 0.02,
}

# -----------------------
# Helpers
# -----------------------
def utc_today_date_str() -> str:
    # For customer naming: yyyy_mm_dd
    return datetime.now(timezone.utc).strftime("%Y_%m_%d")

def utc_today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def init_stripe():
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]

def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return {
            "schema_version": 5,
            "created_at": utc_today_iso(),
            "plans": {},
            "customers": {},
            "subscriptions": {},
            "last_run_date": None,
        }
    return json.loads(STATE_FILE.read_text())

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))

def stripe_sleep():
    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

def with_retries(fn, *, max_attempts=5, base_sleep=0.5):
    attempt = 0
    while True:
        try:
            return fn()
        except (stripe.error.RateLimitError, stripe.error.APIConnectionError):
            attempt += 1
            if attempt >= max_attempts:
                raise
            time.sleep(base_sleep * (2 ** (attempt - 1)))

def find_product_by_name(name: str):
    products = with_retries(lambda: stripe.Product.list(active=True, limit=100)).data
    for p in products:
        if p.get("name") == name:
            return p
    return None

def find_price_for_product(product_id: str, unit_amount: int, currency: str, interval: str):
    prices = with_retries(lambda: stripe.Price.list(product=product_id, active=True, limit=100)).data
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
    # Monthly plans (EUR+USD)
    for plan, amounts in MONTHLY_PLANS.items():
        product_name = f"DAH Music – {plan} (Monthly)"
        product = find_product_by_name(product_name)
        if not product:
            product = with_retries(lambda: stripe.Product.create(
                name=product_name,
                description=f"{plan} monthly subscription",
                metadata={"domain": "music_streaming", "dah_plan": plan, "cadence": "monthly"},
            ))

        for currency, amount in amounts.items():
            price = find_price_for_product(product.id, amount, currency, interval="month")
            if not price:
                price = with_retries(lambda: stripe.Price.create(
                    product=product.id,
                    unit_amount=amount,
                    currency=currency,
                    recurring={"interval": "month"},
                    metadata={"domain": "music_streaming", "dah_plan": plan, "cadence": "monthly"},
                ))
            key = f"monthly_{plan.lower()}_{currency}"
            state["plans"][key] = {"product_id": product.id, "price_id": price.id, "amount": amount, "currency": currency, "interval": "month"}

    # Weekly plan (EUR only)
    for plan, amounts in WEEKLY_PLAN.items():
        product_name = f"DAH Music – {plan} (Weekly)"
        product = find_product_by_name(product_name)
        if not product:
            product = with_retries(lambda: stripe.Product.create(
                name=product_name,
                description=f"{plan} weekly subscription",
                metadata={"domain": "music_streaming", "dah_plan": plan, "cadence": "weekly"},
            ))

        amount = amounts["eur"]
        price = find_price_for_product(product.id, amount, "eur", interval="week")
        if not price:
            price = with_retries(lambda: stripe.Price.create(
                product=product.id,
                unit_amount=amount,
                currency="eur",
                recurring={"interval": "week"},
                metadata={"domain": "music_streaming", "dah_plan": plan, "cadence": "weekly"},
            ))
        key = "weekly_weekly_eur"
        state["plans"][key] = {"product_id": product.id, "price_id": price.id, "amount": amount, "currency": "eur", "interval": "week"}

    return state

def pick_geo() -> tuple[str, str]:
    return ("EU", "eur") if random.random() < EU_SHARE else ("US", "usd")

def pick_plan_for_customer(region: str) -> tuple[str, str]:
    # Speed: a portion of EU customers get weekly
    if region == "EU" and random.random() < 0.12:
        return ("weekly", "Weekly")

    plans = list(PLAN_WEIGHTS.keys())
    weights = list(PLAN_WEIGHTS.values())
    return ("monthly", random.choices(plans, weights=weights, k=1)[0])

def price_key_for(cadence: str, plan: str, currency: str) -> str:
    if cadence == "weekly":
        return "weekly_weekly_eur"
    return f"monthly_{plan.lower()}_{currency}"

def weighted_pick_payment_token() -> tuple[str, str]:
    brands = list(PM_WEIGHTS.keys())
    weights = [PM_WEIGHTS[b] for b in brands]
    brand = random.choices(brands, weights=weights, k=1)[0]
    # Find a token for that brand
    for b, tok in SUCCESS_PM_TOKENS:
        if b == brand:
            return (b, tok)
    # Fallback
    return random.choice(SUCCESS_PM_TOKENS)

def create_customer(run_date_yyyy_mm_dd: str, n: int, region: str, currency: str) -> str:
    # Pattern required: dah_user_yyyy_mm_dd_#N
    name = f"dah_user_{run_date_yyyy_mm_dd}_#{n}"
    email = f"{name}@example.com"
    address = {"country": "FR"} if region == "EU" else {"country": "US"}

    c = with_retries(lambda: stripe.Customer.create(
        email=email,
        name=name,
        address=address,
        metadata={
            "domain": "music_streaming",
            "generator": "daily",
            "cohort_date": run_date_yyyy_mm_dd,
            "region": region,
            "currency": currency,
        },
    ))
    return c.id

def attach_default_payment_method(customer_id: str) -> tuple[str, str]:
    """
    Clean Stripe approach:
      - create a new PaymentMethod from a test token
      - attach it to the customer
      - set as default
    Returns (payment_method_id, brand_label)

    NOTE: last4 will be token-specific (Stripe limitation).
    """
    brand_label, token = weighted_pick_payment_token()

    pm = with_retries(lambda: stripe.PaymentMethod.create(
        type="card",
        card={"token": token},
        metadata={"generator": "daily", "domain": "music_streaming", "brand": brand_label},
    ))

    with_retries(lambda: stripe.PaymentMethod.attach(pm.id, customer=customer_id))
    with_retries(lambda: stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": pm.id}))
    return (pm.id, brand_label)

def create_subscription(customer_id: str, price_id: str, cadence: str, plan: str, currency: str, region: str, run_date: str) -> str:
    s = with_retries(lambda: stripe.Subscription.create(
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
    ))
    return s.id

def pay_latest_invoice_for_subscription(subscription_id: str) -> tuple[bool, str]:
    """
    Best-effort attempt to confirm the latest invoice's PaymentIntent.
    Returns (ok, status).
    Never crashes the run.
    """
    try:
        sub = with_retries(lambda: stripe.Subscription.retrieve(subscription_id, expand=["latest_invoice.payment_intent"]))
        inv = sub.get("latest_invoice")
        if not inv:
            return (True, "no_invoice")

        pi = (inv.get("payment_intent") or {})
        pi_id = pi.get("id")
        if not pi_id:
            return (True, "no_payment_intent")

        with_retries(lambda: stripe.PaymentIntent.confirm(
            pi_id,
            automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
        ))
        pi2 = with_retries(lambda: stripe.PaymentIntent.retrieve(pi_id))
        return (pi2.status == "succeeded", pi2.status)

    except (stripe.error.CardError, stripe.error.StripeError) as e:
        return (False, getattr(e, "code", "error"))

def maybe_churn_subscription(subscription_id: str) -> bool:
    if random.random() >= DAILY_CHURN_RATE:
        return False
    if CHURN_MODE == "immediate":
        with_retries(lambda: stripe.Subscription.delete(subscription_id))
    else:
        with_retries(lambda: stripe.Subscription.modify(subscription_id, cancel_at_period_end=True))
    return True

def maybe_change_plan(subscription_id: str, current_plan: str, current_currency: str, plans_state: dict) -> str | None:
    if not ENABLE_PLAN_CHANGES:
        return None
    if random.random() >= DAILY_PLAN_CHANGE_RATE:
        return None
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
    sub = with_retries(lambda: stripe.Subscription.retrieve(subscription_id))
    item_id = sub["items"]["data"][0]["id"]

    with_retries(lambda: stripe.Subscription.modify(
        subscription_id,
        items=[{"id": item_id, "price": new_price_id}],
        proration_behavior="create_prorations",
        metadata={**(sub.metadata or {}), "dah_plan": new_plan},
    ))
    return new_plan

def list_active_music_subscriptions(limit: int):
    subs = with_retries(lambda: stripe.Subscription.list(status="active", limit=limit)).data
    return [s for s in subs if (s.metadata or {}).get("domain") == "music_streaming"]

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

    state = ensure_catalog(state)

    n_new = random.randint(NEW_CUSTOMERS_MIN, NEW_CUSTOMERS_MAX)
    print(f"[{run_date}] Creating {n_new} new customers | target failure rate ~{failure_rate_today:.2%}")

    successes = 0
    failures = 0
    created_subs = 0
    eu_count = 0
    us_count = 0

    for n in range(1, n_new + 1):
        region, currency = pick_geo()
        if region == "EU":
            eu_count += 1
        else:
            us_count += 1

        cadence, plan = pick_plan_for_customer(region)
        if cadence == "weekly":
            region = "EU"
            currency = "eur"

        should_fail = random.random() < failure_rate_today

        cid = create_customer(run_date_key, n, region, currency)

        # Failure strategy (clean + robust):
        # - If should_fail: do NOT attach a default payment method
        #   -> invoice/payment intent will require a PM => unpaid/failed attempt.
        # - Else: attach a normal payment method (brand mix).
        pm_id = None
        pm_brand = None
        if not should_fail:
            pm_id, pm_brand = attach_default_payment_method(cid)

        pkey = price_key_for(cadence, plan, currency)
        price_id = state["plans"][pkey]["price_id"]

        sid = create_subscription(cid, price_id, cadence, plan, currency, region, run_date)
        created_subs += 1

        ok, status = pay_latest_invoice_for_subscription(sid)
        if ok:
            successes += 1
        else:
            failures += 1

        state["customers"][cid] = {
            "created_date": run_date,
            "region": region,
            "currency": currency,
            "plan": plan,
            "cadence": cadence,
            "subscription_id": sid,
            "payment_method_id": pm_id,
            "payment_method_brand": pm_brand,
            "initial_payment_target_fail": should_fail,
            "initial_payment_result": "success" if ok else "fail",
            "initial_payment_status": status,
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

        stripe_sleep()

    # Mutate existing subscriptions (churn + optional plan changes)
    existing_active = list_active_music_subscriptions(limit=MAX_EXISTING_ACTIVE_TO_MUTATE)
    churned = 0
    changed = 0

    for s in existing_active:
        sid = s.id
        meta = s.metadata or {}
        current_plan = meta.get("dah_plan") or state["subscriptions"].get(sid, {}).get("plan") or "Basic"
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

        stripe_sleep()

    state["last_run_date"] = run_date
    save_state(state)

    observed_rate = successes / max(1, (successes + failures))
    print(f"[{run_date}] Done.")
    print(f"  New customers: {n_new}")
    print(f"  New subscriptions: {created_subs}")
    print(f"  Payments succeeded: {successes} | failed: {failures} | observed success rate: {observed_rate:.2%}")
    print(f"  Geo split actual today: EU {eu_count} ({eu_count/max(1,n_new):.1%}) / US {us_count} ({us_count/max(1,n_new):.1%})")
    print(f"  Existing active evaluated: {len(existing_active)} | churn events: {churned} | plan changes: {changed}")
    print(f"  State saved to {STATE_FILE}")

if __name__ == "__main__":
    main()
