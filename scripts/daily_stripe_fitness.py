#!/usr/bin/env python3
"""
Daily Stripe data generator (test-mode, spec-driven).

Design principles (important for DAH):
- No Stripe idempotency keys for synthetic data creation
- Business assumptions live in YAML, not in code
- Re-runs are allowed and safe
- Metadata is the source of truth for analytics attribution
"""

from __future__ import annotations

import os
import sys
import json
import random
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import stripe
import yaml


STATE_DIR = ".state"
DEFAULT_SPEC_PATH = "specs/fitness_app.yml"


# -------------------------
# Generic helpers
# -------------------------

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


def gaussian_int(rng: random.Random, mean: float, std: float, lo: int, hi: int) -> int:
    x = rng.gauss(mean, std)
    return int(round(clamp(x, lo, hi)))


def choose_weighted_key(rng: random.Random, weights: Dict[str, float]) -> str:
    total = sum(weights.values())
    r = rng.random() * total
    acc = 0.0
    for k, w in weights.items():
        acc += w
        if r <= acc:
            return k
    return next(iter(weights.keys()))


# -------------------------
# Stripe setup
# -------------------------

def stripe_set_key() -> None:
    key = os.getenv("STRIPE_API_KEY")
    if not key:
        raise RuntimeError("Missing STRIPE_API_KEY")
    stripe.api_key = key


# -------------------------
# Catalog handling
# -------------------------

@dataclass
class CatalogIds:
    products: Dict[str, str]
    prices: Dict[Tuple[str, str], str]
    coupons: Dict[str, str]


def upsert_catalog(spec: Dict[str, Any], run_date: str) -> CatalogIds:
    currency = spec["app"]["currency"]

    products_by_key: Dict[str, str] = {}
    prices_by_key: Dict[Tuple[str, str], str] = {}

    for p in spec["catalog"]["products"]:
        pkey = p["key"]

        existing = stripe.Product.search(
            query=f"metadata['key']:'{pkey}'",
            limit=1,
        )

        if existing.data:
            product_id = existing.data[0].id
        else:
            product = stripe.Product.create(
                name=p["name"],
                metadata={
                    "key": pkey,
                    "generator": "dah_fitness",
                    "created_for": run_date,
                },
            )
            product_id = product.id

        products_by_key[pkey] = product_id

        for pr in p["prices"]:
            pr_key = pr["key"]

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
                    "unit_amount": int(pr["unit_amount"]),
                    "nickname": pr.get("nickname"),
                    "metadata": {
                        "product_key": pkey,
                        "price_key": pr_key,
                        "generator": "dah_fitness",
                        "created_for": run_date,
                    },
                }

                if pr.get("interval"):
                    params["recurring"] = {"interval": pr["interval"]}

                price = stripe.Price.create(**params)
                price_id = price.id

            prices_by_key[(pkey, pr_key)] = price_id

    coupons_by_key: Dict[str, str] = {}
    for c in spec["catalog"].get("coupons", []):
        ckey = c["key"]
        coupon_id = None

        for cup in stripe.Coupon.list(limit=100).auto_paging_iter():
            if cup.metadata and cup.metadata.get("key") == ckey:
                coupon_id = cup.id
                break

        if not coupon_id:
            params = {
                "duration": c["duration"],
                "metadata": {
                    "key": ckey,
                    "generator": "dah_fitness",
                    "created_for": run_date,
                },
            }
            if "percent_off" in c:
                params["percent_off"] = c["percent_off"]
            if c.get("duration_in_months"):
                params["duration_in_months"] = int(c["duration_in_months"])

            coupon = stripe.Coupon.create(**params)
            coupon_id = coupon.id

        coupons_by_key[ckey] = coupon_id

    return CatalogIds(products_by_key, prices_by_key, coupons_by_key)


# -------------------------
# Core creation logic
# -------------------------

def create_customer(run_date: str, idx: int) -> stripe.Customer:
    return stripe.Customer.create(
        email=f"dah.fitness.user.{run_date.replace('-', '')}.{idx}@example.com",
        metadata={
            "generator": "dah_fitness",
            "run_date": run_date,
            "daily_idx": str(idx),
        },
    )


def maybe_attach_default_card(
    rng: random.Random,
    customer_id: str,
    failure_rate: float,
    card_cfg: Dict[str, Any],
    run_date: str,
    idx: int,
) -> Optional[str]:
    if rng.random() < failure_rate:
        return None

    card_network = choose_weighted_key(rng, card_cfg["distribution"])
    token = card_cfg["stripe_test_tokens"][card_network]

    pm = stripe.PaymentMethod.create(
        type="card",
        card={"token": token},
        metadata={
            "generator": "dah_fitness",
            "run_date": run_date,
            "daily_idx": str(idx),
            "card_network": card_network,
        },
    )

    stripe.PaymentMethod.attach(pm.id, customer=customer_id)
    stripe.Customer.modify(
        customer_id,
        invoice_settings={"default_payment_method": pm.id},
    )

    return card_network


def create_subscription(
    customer_id: str,
    price_id: str,
    trial_days: int,
    coupon_id: Optional[str],
    run_date: str,
    idx: int,
) -> None:
    params: Dict[str, Any] = {
        "customer": customer_id,
        "items": [{"price": price_id}],
        "metadata": {
            "generator": "dah_fitness",
            "run_date": run_date,
            "daily_idx": str(idx),
        },
    }

    if trial_days > 0:
        params["trial_period_days"] = trial_days

    if coupon_id:
        params["discounts"] = [{"coupon": coupon_id}]

    stripe.Subscription.create(**params)


def create_one_off_purchase(
    rng: random.Random,
    spec: Dict[str, Any],
    customer_id: str,
    run_date: str,
    idx: int,
) -> Optional[str]:
    basket = spec["data_generation"]["one_off_basket"]
    card_cfg = spec["payment_methods"]["cards"]

    amount = rng.randint(int(basket["min_amount"]), int(basket["max_amount"]))
    currency = spec["app"]["currency"]

    card_network = choose_weighted_key(rng, card_cfg["distribution"])
    token = card_cfg["stripe_test_tokens"][card_network]

    pm = stripe.PaymentMethod.create(
        type="card",
        card={"token": token},
        metadata={
            "generator": "dah_fitness",
            "run_date": run_date,
            "daily_idx": str(idx),
            "card_network": card_network,
        },
    )

    stripe.PaymentMethod.attach(pm.id, customer=customer_id)

    pi = stripe.PaymentIntent.create(
        amount=amount,
        currency=currency,
        customer=customer_id,
        payment_method=pm.id,
        confirm=True,
        automatic_payment_methods={
            "enabled": True,
            "allow_redirects": "never",
        },
        metadata={
            "generator": "dah_fitness",
            "run_date": run_date,
            "daily_idx": str(idx),
            "purchase_type": "one_off",
            "card_network": card_network,
        },
    )

    return pi.id


# -------------------------
# Main
# -------------------------

def main() -> None:
    stripe_set_key()

    spec = load_yaml(os.getenv("FITNESS_SPEC_PATH", DEFAULT_SPEC_PATH))
    run_date = os.getenv("RUN_DATE") or dt.date.today().isoformat()

    rng = random.Random()
    if spec.get("advanced", {}).get("deterministic_seed", False):
        rng.seed(stable_int_seed(f"dah_fitness:{run_date}"))

    catalog = upsert_catalog(spec, run_date)
    gen = spec["data_generation"]
    card_cfg = spec["payment_methods"]["cards"]

    n_new = gaussian_int(
        rng,
        gen["new_customers_per_day"]["mean"],
        gen["new_customers_per_day"]["std"],
        gen["new_customers_per_day"]["min"],
        gen["new_customers_per_day"]["max"],
    )

    membership_prices = catalog.prices
    plan_mix = gen["plan_mix"]

    created_customers = 0

    for i in range(n_new):
        cust = create_customer(run_date, i)
        created_customers += 1

        maybe_attach_default_card(
            rng,
            cust.id,
            gen["payment_failure_rate"],
            card_cfg,
            run_date,
            i,
        )

        plan = choose_weighted_key(rng, plan_mix)
        price_id = membership_prices[
            ("membership", "yearly" if plan.endswith("yearly") else "monthly")
        ]

        trial_days = 14 if plan.endswith("yearly") else 7
        coupon_id = None

        create_subscription(
            cust.id,
            price_id,
            trial_days,
            coupon_id,
            run_date,
            i,
        )

        if rng.random() < gen["coaching_attach_rate"]:
            create_subscription(
                cust.id,
                catalog.prices[("coaching", "monthly")],
                0,
                None,
                run_date,
                100_000 + i,
            )

    print(
        json.dumps(
            {
                "run_date": run_date,
                "created_customers": created_customers,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
