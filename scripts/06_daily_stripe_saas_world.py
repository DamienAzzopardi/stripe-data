from __future__ import annotations

#!/usr/bin/env python3
"""
scripts/06_daily_stripe_saas_world.py

Daily generator for a B2C Fitness subscription app in Stripe (test mode).

Each run:
1) Advances a Stripe Test Clock by +1 day (sandbox time progression)
2) Creates 100-150 new customers
3) Assigns plan based on DAILY randomized plan-share within ranges
4) Creates subscriptions (weekly/monthly/yearly) with card payment methods
5) Writes a local run log JSON (for CI artifacts)

Required env vars:
- STRIPE_API_KEY
- STRIPE_TEST_CLOCK_ID (optional; if missing, script creates one)
- PRICE_WEEKLY_EUR, PRICE_MONTHLY_EUR, PRICE_YEARLY_EUR
- PRICE_WEEKLY_USD, PRICE_MONTHLY_USD, PRICE_YEARLY_USD

Optional:
- RUN_DATE_UTC (YYYY-MM-DD) to force a deterministic run date
- DRY_RUN (true/false)
"""

import os
import json
import uuid
import random
import datetime as dt
from typing import Dict, Tuple

import stripe


# ----------------------------
# Configuration
# ----------------------------

PLAN_RANGES = {
    "weekly": (0.35, 0.55),
    "monthly": (0.25, 0.40),
    "yearly": (0.10, 0.25),
}

EU_COUNTRIES = ["FR", "ES", "IT"]
US_COUNTRIES = ["US"]

TEST_CARD_NUMBERS = {
    "visa": "4242424242424242",
    "mastercard": "5555555555554444",
    "amex": "378282246310005",
    "discover": "6011111111111117",
    "jcb": "3530111333300000",
}

CARD_BRANDS = list(TEST_CARD_NUMBERS.keys())

DEFAULT_CARD_EXP_MONTH = 12
DEFAULT_CARD_EXP_YEAR = 2034


# ----------------------------
# Helpers
# ----------------------------

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y")


def utc_today_date() -> dt.date:
    forced = os.getenv("RUN_DATE_UTC")
    if forced:
        return dt.date.fromisoformat(forced)
    return dt.datetime.now(dt.timezone.utc).date()


def iso_now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def pick_country() -> str:
    # Simple market mix: 70% EU, 30% US
    if random.random() < 0.70:
        return random.choice(EU_COUNTRIES)
    return random.choice(US_COUNTRIES)


def currency_for_country(country: str) -> str:
    return "eur" if country in EU_COUNTRIES else "usd"


def price_id_for(plan: str, currency: str) -> str:
    key = f"PRICE_{plan.upper()}_{currency.upper()}"
    pid = os.getenv(key)
    if not pid:
        raise RuntimeError(f"Missing env var {key} (Stripe Price ID)")
    return pid


def generate_daily_plan_weights() -> Dict[str, float]:
    """
    Always produce valid weights within ranges that sum to 1.0.

    Strategy:
    1) Sample yearly within its range
    2) The remaining mass (1 - yearly) must be split between weekly & monthly
       while staying within their ranges.
    """
    y_lo, y_hi = PLAN_RANGES["yearly"]
    w_lo, w_hi = PLAN_RANGES["weekly"]
    m_lo, m_hi = PLAN_RANGES["monthly"]

    for _ in range(10_000):
        y = random.uniform(y_lo, y_hi)
        remaining = 1.0 - y

        # Weekly must be within [w_lo, w_hi]
        # Monthly = remaining - weekly must be within [m_lo, m_hi]
        w_min = max(w_lo, remaining - m_hi)
        w_max = min(w_hi, remaining - m_lo)

        if w_min <= w_max:
            w = random.uniform(w_min, w_max)
            m = remaining - w

            # Final sanity guard
            if (w_lo <= w <= w_hi) and (m_lo <= m <= m_hi) and (y_lo <= y <= y_hi):
                total = w + m + y
                return {"weekly": w / total, "monthly": m / total, "yearly": y / total}

    raise RuntimeError("Could not generate valid plan weights within ranges.")


def weighted_choice(weights: Dict[str, float]) -> str:
    r = random.random()
    cum = 0.0
    for k, w in weights.items():
        cum += w
        if r <= cum:
            return k
    return list(weights.keys())[-1]


def get_or_create_test_clock(run_date: dt.date):
    """
    IMPORTANT FIX:
    Use stripe.test_helpers.TestClock (not stripe.TestClock).
    """
    clock_id = os.getenv("STRIPE_TEST_CLOCK_ID")
    frozen_time = int(
        dt.datetime.combine(run_date, dt.time(0, 0), tzinfo=dt.timezone.utc).timestamp()
    )

    if clock_id:
        return stripe.test_helpers.TestClock.retrieve(clock_id)

    clock = stripe.test_helpers.TestClock.create(
        name="fitness_app_daily_simulation",
        frozen_time=frozen_time,
    )
    print(f"[INFO] Created STRIPE_TEST_CLOCK_ID={clock.id} (store this in GitHub Secrets)")
    return clock


def advance_test_clock_by_one_day(clock, dry_run: bool):
    if dry_run:
        print("[DRY_RUN] Would advance test clock by +1 day.")
        return clock

    current = stripe.test_helpers.TestClock.retrieve(clock.id)
    new_time = current.frozen_time + 86400
    updated = stripe.test_helpers.TestClock.advance(clock.id, frozen_time=new_time)
    return updated


def create_payment_method(card_brand: str, dry_run: bool) -> Tuple[str, Dict]:
    if dry_run:
        return "pm_dry_run", {"brand": card_brand}

    number = TEST_CARD_NUMBERS[card_brand]
    pm = stripe.PaymentMethod.create(
        type="card",
        card={
            "number": number,
            "exp_month": DEFAULT_CARD_EXP_MONTH,
            "exp_year": DEFAULT_CARD_EXP_YEAR,
            "cvc": "1234" if card_brand == "amex" else "123",
        },
    )
    return pm.id, {"brand": card_brand}


def create_customer(country: str, currency: str, payment_method_id: str, dry_run: bool) -> str:
    if dry_run:
        return f"cus_dry_{uuid.uuid4().hex[:10]}"

    customer = stripe.Customer.create(
        metadata={
            "simulator": "fitness_app_daily",
            "country": country,
            "currency": currency,
        }
    )

    stripe.PaymentMethod.attach(payment_method_id, customer=customer.id)
    stripe.Customer.modify(
        customer.id,
        invoice_settings={"default_payment_method": payment_method_id},
    )
    return customer.id


def create_subscription(customer_id: str, price_id: str, test_clock_id: str, plan: str, run_date: dt.date, dry_run: bool) -> str:
    if dry_run:
        return f"sub_dry_{uuid.uuid4().hex[:10]}"

    sub = stripe.Subscription.create(
        customer=customer_id,
        items=[{"price": price_id}],
        payment_behavior="default_incomplete",
        test_clock=test_clock_id,
        metadata={
            "simulator": "fitness_app_daily",
            "plan": plan,
            "run_date_utc": run_date.isoformat(),
        },
        expand=["latest_invoice.payment_intent"],
    )

    # Confirm payment intent to mark invoice paid (test mode usually succeeds)
    if sub.latest_invoice and sub.latest_invoice.payment_intent:
        stripe.PaymentIntent.confirm(sub.latest_invoice.payment_intent.id)

    return sub.id


def main():
    stripe.api_key = os.getenv("STRIPE_API_KEY")
    if not stripe.api_key:
        raise RuntimeError("Missing STRIPE_API_KEY")

    dry_run = env_bool("DRY_RUN", False)
    run_date = utc_today_date()

    # Optional debug to confirm SDK features in CI
    print("[INFO] stripe version:", getattr(stripe, "__version__", "unknown"))
    print("[INFO] stripe has test_helpers:", hasattr(stripe, "test_helpers"))

    # 1) Get or create test clock
    clock = get_or_create_test_clock(run_date)

    # 2) Advance clock by +1 day each run
    clock = advance_test_clock_by_one_day(clock, dry_run=dry_run)

    # 3) Determine today's new user count
    new_users = random.randint(100, 150)

    # 4) Pick daily plan weights within ranges
    plan_weights = generate_daily_plan_weights()

    results = {
        "run_date_utc": run_date.isoformat(),
        "run_ts_utc": iso_now_utc(),
        "test_clock_id": clock.id,
        "test_clock_frozen_time": clock.frozen_time,
        "new_users": new_users,
        "plan_weights": plan_weights,
        "created": {"customers": [], "subscriptions": []},
        "counts_by_plan": {"weekly": 0, "monthly": 0, "yearly": 0},
        "counts_by_currency": {"eur": 0, "usd": 0},
        "dry_run": dry_run,
    }

    print(f"[INFO] Run date (UTC): {run_date}")
    print(f"[INFO] New users today: {new_users}")
    print(f"[INFO] Plan weights today: {plan_weights}")
    print(f"[INFO] Test clock: {clock.id} frozen_time={clock.frozen_time}")

    for i in range(new_users):
        country = pick_country()
        currency = currency_for_country(country)
        plan = weighted_choice(plan_weights)

        card_brand = random.choice(CARD_BRANDS)
        pm_id, _ = create_payment_method(card_brand, dry_run=dry_run)

        customer_id = create_customer(country, currency, pm_id, dry_run=dry_run)
        price_id = price_id_for(plan, currency)

        sub_id = create_subscription(customer_id, price_id, clock.id, plan, run_date, dry_run=dry_run)

        results["created"]["customers"].append(
            {"customer_id": customer_id, "country": country, "currency": currency, "card_brand": card_brand}
        )
        results["created"]["subscriptions"].append(
            {"subscription_id": sub_id, "customer_id": customer_id, "plan": plan, "currency": currency, "price_id": price_id}
        )
        results["counts_by_plan"][plan] += 1
        results["counts_by_currency"][currency] += 1

        if (i + 1) % 25 == 0:
            print(f"[INFO] Created {i+1}/{new_users}")

    # Write run artifact
    os.makedirs("run_artifacts", exist_ok=True)
    out_path = f"run_artifacts/run_{run_date.isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Wrote run artifact: {out_path}")

    # Sanity check: actual plan shares (allow +-5pp due to sampling noise)
    actual = {k: results["counts_by_plan"][k] / new_users for k in results["counts_by_plan"]}
    print(f"[INFO] Actual plan shares: {actual}")

    tolerance = 0.05
    for plan, (lo, hi) in PLAN_RANGES.items():
        if not (lo - tolerance <= actual[plan] <= hi + tolerance):
            raise RuntimeError(
                f"Actual share for {plan} out of bounds (with tolerance). "
                f"actual={actual[plan]:.3f}, expected_range={lo}-{hi}"
            )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
