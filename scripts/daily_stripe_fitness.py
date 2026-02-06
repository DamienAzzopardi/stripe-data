#!/usr/bin/env python3
"""
Daily Stripe data generator (test-mode friendly, spec-driven).

Creates per-day:
- Catalog (products/prices) if missing
- New customers
- Subscriptions (monthly/yearly) with trials
- Optional add-on subscription (coaching)
- One-off purchases (PaymentIntents)
- Coupons / promo codes usage (best-effort)
- Churn (cancel some existing subs)
- Simulated failures (some customers without payment method)
- Refunds for a fraction of one-off purchases

Idempotency:
- Uses idempotency keys derived from (run_date, entity index) for safe re-runs.
- Stores minimal local state in .state/ (committed or not, your choice).

Requirements:
- STRIPE_API_KEY env var (GitHub Secret)
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import stripe
import yaml


STATE_DIR = ".state"
DEFAULT_SPEC_PATH = "specs/fitness_app.yml"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def stable_int_seed(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_state(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def idempotency_key(prefix: str, run_date: str, idx: int) -> str:
    return f"{prefix}:{run_date}:{idx}"


def gaussian_int(rng: random.Random, mean: float, std: float, lo: int, hi: int) -> int:
    x = rng.gauss(mean, std)
    return int(round(clamp(x, lo, hi)))


@dataclass
class CatalogIds:
    products: Dict[str, str]              # key -> product_id
    prices: Dict[Tuple[str, str], str]    # (product_key, price_key) -> price_id
    coupons: Dict[str, str]               # coupon_key -> coupon_id


def stripe_set_key() -> None:
    key = os.getenv("STRIPE_API_KEY")
    if not key:
        raise RuntimeError("Missing STRIPE_API_KEY")
    stripe.api_key = key


def upsert_catalog(spec: Dict[str, Any], run_date: str) -> CatalogIds:
    """
    Create products/prices/coupons if they do not exist.
    We look up by 'metadata.key' where possible.
    """
    currency = spec["app"]["currency"]

    # --- Products & Prices
    products_by_key: Dict[str, str] = {}
    prices_by_key: Dict[Tuple[str, str], str] = {}

    for p in spec["catalog"]["products"]:
        pkey = p["key"]
        pname = p["name"]

        # Find product by metadata.key
        existing = stripe.Product.search(
            query=f"metadata['key']:'{pkey}'",
            limit=1,
        )
        if existing.data:
            product_id = existing.data[0].id
        else:
            product = stripe.Product.create(
                name=pname,
                metadata={"key": pkey, "generator": "dah_fitness", "created_for": run_date},
                idempotency_key=f"product:{pkey}:{run_date}",
            )
            product_id = product.id

        products_by_key[pkey] = product_id

        # Prices
        for pr in p["prices"]:
            pr_key = pr["key"]
            nickname = pr.get("nickname", f"{pname} {pr_key}")
            unit_amount = int(pr["unit_amount"])
            interval = pr.get("interval")  # month/year for recurring
            # Find price by metadata keys
            existing_price = stripe.Price.search(
                query=f"metadata['product_key']:'{pkey}' AND metadata['price_key']:'{pr_key}'",
                limit=1,
            )
            if existing_price.data:
                price_id = existing_price.data[0].id
            else:
                params: Dict[str, Any] = {
                    "product": product_id,
                    "currency": currency,
                    "unit_amount": unit_amount,
                    "nickname": nickname,
                    "metadata": {
                        "product_key": pkey,
                        "price_key": pr_key,
                        "generator": "dah_fitness",
                        "created_for": run_date,
                    },
                }
                if interval:
                    params["recurring"] = {"interval": interval}
                price = stripe.Price.create(
                    **params,
                    idempotency_key=f"price:{pkey}:{pr_key}:{run_date}",
                )
                price_id = price.id

            prices_by_key[(pkey, pr_key)] = price_id

    # --- Coupons
    coupons_by_key: Dict[str, str] = {}
    for c in spec["catalog"].get("coupons", []):
        ckey = c["key"]
        # Stripe coupon can't be searched by metadata via search in some accounts,
        # so we list and match by metadata (best-effort).
        coupon_id = None
        for cup in stripe.Coupon.list(limit=100).auto_paging_iter():
            if cup.metadata and cup.metadata.get("key") == ckey:
                coupon_id = cup.id
                break

        if not coupon_id:
            params = {
                "metadata": {"key": ckey, "generator": "dah_fitness", "created_for": run_date},
                "duration": c["duration"],
            }
            if "percent_off" in c:
                params["percent_off"] = c["percent_off"]
            if "amount_off" in c:
                params["amount_off"] = c["amount_off"]
                params["currency"] = spec["app"]["currency"]
            if c.get("duration_in_months") is not None:
                params["duration_in_months"] = int(c["duration_in_months"])

            new_coupon = stripe.Coupon.create(**params, idempotency_key=f"coupon:{ckey}:{run_date}")
            coupon_id = new_coupon.id

        coupons_by_key[ckey] = coupon_id

    return CatalogIds(products=products_by_key, prices=prices_by_key, coupons=coupons_by_key)


def choose_weighted(rng: random.Random, weights: Dict[str, float]) -> str:
    total = sum(weights.values())
    r = rng.random() * total
    acc = 0.0
    for k, w in weights.items():
        acc += w
        if r <= acc:
            return k
    return next(iter(weights.keys()))


def maybe_pick_coupon(rng: random.Random, spec: Dict[str, Any], catalog: CatalogIds) -> Optional[str]:
    gen = spec["data_generation"]
    if rng.random() > float(gen.get("coupon_usage_rate", 0.0)):
        return None
    dist = gen.get("coupon_distribution", {})
    if not dist:
        return None
    chosen = choose_weighted(rng, dist)
    return catalog.coupons.get(chosen)


def create_customer(rng: random.Random, run_date: str, idx: int) -> stripe.Customer:
    # Use deterministic-ish fake identity
    email = f"dah.fitness.user.{run_date.replace('-', '')}.{idx}@example.com"
    customer = stripe.Customer.create(
        email=email,
        description="DAH Fitness synthetic customer",
        metadata={"generator": "dah_fitness", "run_date": run_date, "daily_idx": str(idx)},
        idempotency_key=idempotency_key("customer", run_date, idx),
    )
    return customer


def maybe_attach_test_payment_method(rng: random.Random, customer_id: str, failure_rate: float, run_date: str, idx: int) -> bool:
    """
    To simulate failures: some customers intentionally do NOT get a default payment method.
    When they get invoiced / charged later, it may fail depending on your Stripe settings.
    """
    if rng.random() < failure_rate:
        return False

    # A common Stripe test card payment method:
    pm = stripe.PaymentMethod.create(
        type="card",
        card={"token": "tok_visa"},
        idempotency_key=idempotency_key("pm_create", run_date, idx),
    )
    stripe.PaymentMethod.attach(
        pm.id,
        customer=customer_id,
        idempotency_key=idempotency_key("pm_attach", run_date, idx),
    )
    stripe.Customer.modify(
        customer_id,
        invoice_settings={"default_payment_method": pm.id},
        idempotency_key=idempotency_key("pm_set_default", run_date, idx),
    )
    return True


def create_subscription(customer_id: str, price_id: str, trial_days: int, coupon_id: Optional[str], run_date: str, idx: int) -> stripe.Subscription:
    params: Dict[str, Any] = {
        "customer": customer_id,
        "items": [{"price": price_id}],
        "metadata": {"generator": "dah_fitness", "run_date": run_date, "daily_idx": str(idx)},
    }

    if trial_days and trial_days > 0:
        params["trial_period_days"] = int(trial_days)

    # ✅ Stripe now prefers discounts=[{coupon: ...}] over coupon=...
    if coupon_id:
        params["discounts"] = [{"coupon": coupon_id}]

    sub = stripe.Subscription.create(
        **params,
        idempotency_key=idempotency_key("sub", run_date, idx),
    )
    return sub


def create_one_off_purchase(rng: random.Random, spec: Dict[str, Any], customer_id: str, coupon_id: Optional[str], run_date: str, idx: int) -> Optional[str]:
    """
    Create a PaymentIntent. Stripe does not apply coupons to PaymentIntents directly;
    so we just record coupon usage in metadata for analytics teaching.
    """
    basket = spec["data_generation"]["one_off_basket"]
    amount = rng.randint(int(basket["min_amount"]), int(basket["max_amount"]))
    currency = spec["app"]["currency"]

    pi = stripe.PaymentIntent.create(
        amount=amount,
        currency=currency,
        customer=customer_id,
        payment_method="pm_card_visa",   # uses a shared test PM
        confirm=True,
        metadata={
            "generator": "dah_fitness",
            "run_date": run_date,
            "daily_idx": str(idx),
            "purchase_type": "one_off",
            "coupon_used": "true" if coupon_id else "false",
        },
        idempotency_key=idempotency_key("pi", run_date, idx),
    )
    return pi.id


def maybe_refund_payment_intent(rng: random.Random, refund_rate: float, payment_intent_id: str, run_date: str, idx: int) -> Optional[str]:
    if rng.random() > refund_rate:
        return None
    refund = stripe.Refund.create(
        payment_intent=payment_intent_id,
        metadata={"generator": "dah_fitness", "run_date": run_date, "daily_idx": str(idx)},
        idempotency_key=idempotency_key("refund", run_date, idx),
    )
    return refund.id


def cancel_some_subscriptions(rng: random.Random, spec: Dict[str, Any], run_date: str) -> int:
    """
    Cancel a fraction of active subs.
    We approximate 'daily churn' by sampling across active subscriptions we can fetch.
    """
    churn_cfg = spec["data_generation"].get("daily_churn_rate", {})
    monthly_rate = float(churn_cfg.get("monthly", 0.0))
    yearly_rate = float(churn_cfg.get("yearly", 0.0))

    cancelled = 0
    # Pull a limited number of active subscriptions and sample.
    # (Enough for daily sim; increase limit if you need scale.)
    subs = stripe.Subscription.list(status="active", limit=100)
    for s in subs.auto_paging_iter():
        # Identify interval from first item price (best-effort)
        interval = None
        try:
            interval = s["items"]["data"][0]["price"]["recurring"]["interval"]
        except Exception:
            interval = None

        p = monthly_rate if interval == "month" else yearly_rate if interval == "year" else 0.0
        if p > 0 and rng.random() < p:
            stripe.Subscription.modify(
                s.id,
                cancel_at_period_end=True,
                metadata={**(s.metadata or {}), "cancel_marked_run_date": run_date},
                idempotency_key=f"cancel:{run_date}:{s.id}",
            )
            cancelled += 1

        # Keep it bounded (avoid canceling too many if you have thousands)
        if cancelled >= 200:
            break

    return cancelled


def main() -> None:
    stripe_set_key()

    spec_path = os.getenv("FITNESS_SPEC_PATH", DEFAULT_SPEC_PATH)
    spec = load_yaml(spec_path)

    # Run date: default to "today" in UTC unless provided
    run_date = os.getenv("RUN_DATE")
    if not run_date:
        run_date = dt.date.today().isoformat()

    ensure_dir(STATE_DIR)
    state_path = os.path.join(STATE_DIR, "fitness_state.json")
    state = load_state(state_path, default={"runs": []})

    # Deterministic seeding makes daily runs reproducible
    rng = random.Random()
    if spec.get("advanced", {}).get("deterministic_seed", False):
        rng.seed(stable_int_seed(f"dah_fitness:{run_date}"))
    else:
        rng.seed()

    # Upsert catalog
    catalog = upsert_catalog(spec, run_date)

    gen = spec["data_generation"]
    n_new = gaussian_int(
        rng,
        mean=float(gen["new_customers_per_day"]["mean"]),
        std=float(gen["new_customers_per_day"]["std"]),
        lo=int(gen["new_customers_per_day"]["min"]),
        hi=int(gen["new_customers_per_day"]["max"]),
    )

    # Plan mix mapping from spec keys to price_ids
    plan_mix = gen["plan_mix"]
    membership_monthly_price = catalog.prices[("membership", "monthly")]
    membership_yearly_price = catalog.prices[("membership", "yearly")]
    coaching_monthly_price = catalog.prices[("coaching", "monthly")]

    # Trial days from spec
    trial_days_monthly = next(p for p in spec["catalog"]["products"] if p["key"] == "membership")["prices"][0].get("trial_days", 0)
    trial_days_yearly = next(p for p in spec["catalog"]["products"] if p["key"] == "membership")["prices"][1].get("trial_days", 0)

    failure_rate = float(gen.get("payment_failure_rate", 0.0))
    coaching_attach = float(gen.get("coaching_attach_rate", 0.0))

    one_off_n = gaussian_int(
        rng,
        mean=float(gen["one_off_purchases_per_day"]["mean"]),
        std=float(gen["one_off_purchases_per_day"]["std"]),
        lo=int(gen["one_off_purchases_per_day"]["min"]),
        hi=int(gen["one_off_purchases_per_day"]["max"]),
    )

    created_customers = 0
    created_subs = 0
    created_addons = 0
    created_oneoffs = 0
    created_refunds = 0

    # Create new customers + subscriptions
    for i in range(n_new):
        c = create_customer(rng, run_date, i)
        created_customers += 1

        # Attach PM (or not) to simulate failures
        maybe_attach_test_payment_method(rng, c.id, failure_rate, run_date, i)

        # Choose plan
        choice = choose_weighted(
            rng,
            {
                "membership_monthly": float(plan_mix.get("membership_monthly", 0.0)),
                "membership_yearly": float(plan_mix.get("membership_yearly", 0.0)),
            },
        )
        if choice == "membership_yearly":
            price_id = membership_yearly_price
            trial_days = int(trial_days_yearly)
        else:
            price_id = membership_monthly_price
            trial_days = int(trial_days_monthly)

        coupon_id = maybe_pick_coupon(rng, spec, catalog)

        sub = create_subscription(c.id, price_id, trial_days, coupon_id, run_date, i)
        created_subs += 1

        # Optional coaching add-on as separate subscription
        if rng.random() < coaching_attach:
            add_sub = create_subscription(c.id, coaching_monthly_price, 0, None, run_date, 100000 + i)
            created_addons += 1

    # One-off purchases: sample across recent customers for the day
    # We reuse today's created customers by searching metadata.run_date = run_date
    todays_customers = stripe.Customer.search(
        query=f"metadata['run_date']:'{run_date}' AND metadata['generator']:'dah_fitness'",
        limit=min(100, max(10, n_new)),
    ).data

    for j in range(one_off_n):
        if not todays_customers:
            break
        cust = rng.choice(todays_customers)
        coupon_id = maybe_pick_coupon(rng, spec, catalog)
        pi_id = create_one_off_purchase(rng, spec, cust.id, coupon_id, run_date, 200000 + j)
        if pi_id:
            created_oneoffs += 1
            refund_id = maybe_refund_payment_intent(rng, float(gen.get("refund_rate", 0.0)), pi_id, run_date, 300000 + j)
            if refund_id:
                created_refunds += 1

    # Churn: cancel some existing subscriptions
    churned = cancel_some_subscriptions(rng, spec, run_date)

    run_summary = {
        "run_date": run_date,
        "created_customers": created_customers,
        "created_subscriptions": created_subs,
        "created_addon_subscriptions": created_addons,
        "created_one_off_purchases": created_oneoffs,
        "created_refunds": created_refunds,
        "marked_cancellations": churned,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
    }

    state["runs"].append(run_summary)
    save_state(state_path, state)

    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
