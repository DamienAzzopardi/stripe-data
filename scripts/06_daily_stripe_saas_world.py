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
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_BETWEEN_CALLS_SEC", "0.04"))
MAX_EXISTING_ACTIVE_TO_MUTATE = int(os.getenv("MAX_EXISTING_ACTIVE_TO_MUTATE", "200"))

# Plan definitions
# Monthly plans available in BOTH currencies (EUR and USD) for geo split
MONTHLY_PLANS = {
    "Basic": {"eur": 999, "usd": 1099},
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

# -----------------------
# Test card pools (numbers -> random last4)
# -----------------------
# SUCCESS cards (mix of brands) — these are standard Stripe test numbers.
SUCCESS_CARDS = [
    ("visa", "4242424242424242"),
    ("visa", "4012888888881881"),
    ("visa_debit", "4000056655665556"),
    ("mastercard", "5555555555554444"),
    ("mastercard", "5105105105105100"),
    ("amex", "378282246310005"),
    ("discover", "6011111111111117"),
    ("jcb", "3530111333300000"),
]

# FAIL cards — used to produce a failed payment attempt (declined/insufficient funds)
# These are also standard Stripe test numbers.
FAIL_CARDS = [
    ("declined", "4000000000000002"),
    ("insufficient_funds", "4000000000009995"),
]

# -----------------------
# Helpers
# -----------------------
def utc_today_date_str() -> str:
    # used in customer naming pattern
    return datetime.now(timezone.utc).strftime("%Y_%m_%d")

def utc_today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def init_stripe():
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]

def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return {
            "schema_version": 3,
            "created_at": utc_today_iso(),
            "plans": {},         # key: plan_key -> {"product_id":..., "price_id":...}
            "customers": {},     # customer_id -> {...}
            "subscriptions": {}, # subscription_id -> {...}
            "last_run_date": None,
        }
    return json.loads(STATE_FILE.read_text())

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))

def stripe_sleep():
    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

def with_retries(fn, *, max_attempts=5, base_sleep=0.5):
    """
    Simple retry wrapper for transient Stripe errors / rate limits.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except stripe.error.RateLimitError:
            attempt += 1
            if attempt >= max_attempts:
                raise
            time.sleep(base_sleep * (2 ** (attempt - 1)))
        except stripe.error.APIConnectionError:
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
    """
    Ensure:
      - monthly plans exist in EUR and USD
      - weekly plan exists in EUR
    Stored price keys:
      - monthly_basic_eur
      - monthly_basic_usd
      - weekly_weekly_eur
    """
    # Monthly (EUR+USD)
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
            state["plans"][key] = {
                "product_id": product.id,
                "price_id": price.id,
                "amount": amount,
                "currency": currency,
                "interval": "month",
            }

    # Weekly (EUR only)
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
        state["plans"][key] = {
            "product_id": product.id,
            "price_id": price.id,
            "amount": amount,
            "currency": "eur",
            "interval": "week",
        }

    return state

def pick_geo() -> tuple[str, str]:
    # 80% EU/EUR, 20% US/USD
    if random.random() < EU_SHARE:
        return ("EU", "eur")
    return ("US", "usd")

def pick_plan_for_customer(region: str) -> tuple[str, str]:
    """
    Returns (cadence, plan_name).
    Weekly is EU-only and injected to speed up renewals.
    """
    if region == "EU":
        # 12% weekly among EU customers (tunable)
        if random.random() < 0.12:
            return ("weekly", "Weekly")

    plans = list(PLAN_WEIGHTS.keys())
    weights = list(PLAN_WEIGHTS.values())
    return ("monthly", random.choices(plans, weights=weights, k=1)[0])

def price_key_for(cadence: str, plan: str, currency: str) -> str:
    if cadence == "weekly":
        return "weekly_weekly_eur"
    return f"monthly_{plan.lower()}_{currency}"

def create_customer(run_date_yyyy_mm_dd: str, n: int, region: str, currency: str) -> str:
    """
    Customer name pattern required:
      dah_user_yyyy_mm_dd_#N
    """
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
            "user_handle": name,
        },
    ))
    return c.id

def create_card_token(card_number: str) -> str:
    """
    Create a Token from a test card number so we get random last4 (realistic variety).
    Using secret key is OK in test mode for server-side token creation.
    """
    exp_month = random.randint(1, 12)
    exp_year = random.randint(2028, 2033)
    cvc = str(random.randint(100, 999))

    tok = with_retries(lambda: stripe.Token.create(
        card={
            "number": card_number,
            "exp_month": exp_month,
            "exp_year": exp_year,
            "cvc": cvc,
        }
    ))
    return tok.id

def attach_default_payment_method(customer_id: str, *, should_fail: bool) -> tuple[str, str, str]:
    """
    Creates a fresh PaymentMethod per customer (required) with randomized last4.
    We do NOT let attach failures crash the run.

    Returns: (payment_method_id, card_brand_label, card_number)
    """
    if should_fail:
        card_brand_label, card_number = random.choice(FAIL_CARDS)
    else:
        card_brand_label, card_number = random.choice(SUCCESS_CARDS)

    def _create_attach_set_default(number: str, label: str) -> tuple[str, str, str]:
        token_id = create_card_token(number)
        pm = with_retries(lambda: stripe.PaymentMethod.create(
            type="card",
            card={"token": token_id},
            metadata={"generator": "daily", "card_brand": label, "domain": "music_streaming"},
        ))
        # Attach
        with_retries(lambda: stripe.PaymentMethod.attach(pm.id, customer=customer_id))
        # Set as default
        with_retries(lambda: stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": pm.id}))
        return (pm.id, label, number)

    try:
        return _create_attach_set_default(card_number, card_brand_label)
    except stripe.error.CardError:
        # Some failing cards can error earlier than intended; fallback to a successful card
        fb_label, fb_number = random.choice(SUCCESS_CARDS)
        return _create_attach_set_default(fb_number, fb_label)

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
    Best-effort payment attempt:
    - Retrieve subscription with latest_invoice.payment_intent expanded
    - Confirm PI (disallow redirects)
    - Return (success, status)
    Never crashes the run; failures are returned as False.
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
        # Record as failed payment attempt, don't kill the run.
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

def list_active_music_subscriptions(limit: int = 200):
    subs = with_retries(lambda: stripe.Subscription.list(status="active", limit=limit)).data
    return [s for s in subs if (s.metadata or {}).get("domain") == "music_streaming"]

# -----------------------
# Main
# -----------------------
def main():
    init_stripe()
    state = load_state()
    run_date = utc_today_iso()
    run_date_key = utc_today_date_str()  # yyyy_mm_dd

    failure_rate_today = random.uniform(FAIL_RATE_MIN, FAIL_RATE_MAX)

    # 1) Ensure catalog exists
    state = ensure_catalog(state)

    # 2) Create new customers + subscriptions
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

        # weekly is EU/EUR only
        if cadence == "weekly":
            region = "EU"
            currency = "eur"

        # Decide failure for initial payment attempt
        should_fail = random.random() < failure_rate_today

        # Create customer with required naming
        cid = create_customer(run_date_key, n, region, currency)

        # Attach a randomized payment method (random last4), never crashes the run
        pm_id, card_label, card_number = attach_default_payment_method(cid, should_fail=should_fail)

        # Determine price
        pkey = price_key_for(cadence, plan, currency)
        price_id = state["plans"][pkey]["price_id"]

        # Create subscription
        sid = create_subscription(cid, price_id, cadence, plan, currency, region, run_date)
        created_subs += 1

        # Force a payment attempt so we get measurable success/failure today
        ok, status = pay_latest_invoice_for_subscription(sid)
        if ok:
            successes += 1
        else:
            failures += 1

        # Save state (debug-friendly)
        state["customers"][cid] = {
            "created_date": run_date,
            "region": region,
            "currency": currency,
            "plan": plan,
            "cadence": cadence,
            "subscription_id": sid,
            "payment_method_id": pm_id,
            "card_label": card_label,
            "card_number": card_number,
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

    # 3) Mutate existing active subs (churn + optional plan changes)
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
