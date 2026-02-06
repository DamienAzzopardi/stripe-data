#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import random
import hashlib
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import stripe
import yaml


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def weighted_choice(rng: random.Random, items: List[Tuple[Any, float]]) -> Any:
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for v, w in items:
        acc += w
        if acc >= r:
            return v
    return items[-1][0]


def sku_key(
    project: str,
    category: str,
    brand: str,
    model: str,
    grade: str,
    storage: Optional[str],
) -> str:
    s = storage or "na"
    raw = f"{project}|{category}|{brand}|{model}|{grade}|{s}"
    return raw.lower().replace(" ", "_")


# --------------------------------------------------
# Stripe catalog helpers (idempotent)
# --------------------------------------------------

def stripe_search_one_product_by_sku(sku: str) -> Optional[str]:
    res = stripe.Product.search(query=f"metadata['sku']:'{sku}'", limit=1)
    if res.data:
        return res.data[0].id
    return None


def stripe_search_one_price(
    product_id: str,
    sku: str,
    currency: str,
    unit_amount: int,
) -> Optional[str]:
    q = (
        f"metadata['sku']:'{sku}' AND "
        f"metadata['currency']:'{currency}' AND "
        f"metadata['unit_amount']:'{unit_amount}'"
    )
    res = stripe.Price.search(query=q, limit=1)
    if res.data:
        return res.data[0].id
    return None


def upsert_sku_as_product_price(
    project: str,
    category: str,
    brand: str,
    model: str,
    grade: str,
    storage: Optional[str],
    currency: str,
    unit_amount: int,
) -> Tuple[str, str]:
    sku = sku_key(project, category, brand, model, grade, storage)
    name = f"{brand} {model} – {grade}" + (f" – {storage}" if storage else "")

    product_id = stripe_search_one_product_by_sku(sku)
    if not product_id:
        product = stripe.Product.create(
            name=name,
            metadata={
                "project": project,
                "sku": sku,
                "category": category,
                "brand": brand,
                "model": model,
                "grade": grade,
                "storage": storage or "",
            },
        )
        product_id = product.id

    price_id = stripe_search_one_price(product_id, sku, currency, unit_amount)
    if not price_id:
        price = stripe.Price.create(
            product=product_id,
            currency=currency,
            unit_amount=unit_amount,
            metadata={
                "project": project,
                "sku": sku,
                "currency": currency,
                "unit_amount": str(unit_amount),
            },
        )
        price_id = price.id

    return product_id, price_id


def flatten_catalog(project: str, catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a flat list of SKUs with weights.
    """
    out: List[Dict[str, Any]] = []
    for category, families in catalog["catalog"].items():
        for fam in families:
            weight = float(fam.get("popularity_weight", 1.0))
            brand = fam["brand"]
            model = fam["model"]
            for v in fam["variants"]:
                out.append(
                    {
                        "category": category,
                        "brand": brand,
                        "model": model,
                        "grade": v["grade"],
                        "storage": v.get("storage"),
                        "price": int(v["price"]),
                        "family_weight": weight,
                    }
                )
    return out


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    # --- Stripe auth
    key = os.getenv("STRIPE_API_KEY")
    if not key:
        raise RuntimeError("Missing STRIPE_API_KEY")
    stripe.api_key = key

    # --- Load specs
    spec = load_yaml(os.getenv("REBOOT_SPEC_PATH", "specs/reboot_tech.yml"))
    project = spec["app"]["project_name"]
    currency = spec["app"]["currency"]

    catalog = load_yaml(spec["catalog"]["file"])

    # --- RNG
    run_date = dt.date.today().isoformat()
    seed = (
        stable_seed(f"{project}:{run_date}")
        if spec.get("advanced", {}).get("deterministic_seed", False)
        else None
    )
    rng = random.Random(seed)

    # --- Catalog
    skus = flatten_catalog(project, catalog)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for s in skus:
        by_cat.setdefault(s["category"], []).append(s)

    # Upsert Stripe catalog
    for s in skus:
        upsert_sku_as_product_price(
            project=project,
            category=s["category"],
            brand=s["brand"],
            model=s["model"],
            grade=s["grade"],
            storage=s["storage"],
            currency=currency,
            unit_amount=s["price"],
        )

    # --- Countries
    country_items = [(c["code"], float(c["weight"])) for c in spec["markets"]["countries"]]

    # --- Cards (SUCCESS tokens only here)
    cards_cfg = spec["payments"]["cards"]
    success_tokens = list(cards_cfg["tokens"].values())

    # --- Failure simulation
    failure_rate = float(cards_cfg.get("failure_rate", 0.0))
    failure_tokens = cards_cfg.get("failure_tokens", {}) or {}
    failure_types = list(failure_tokens.keys())

    # --- Order parameters
    n_orders = rng.randint(
        int(spec["orders"]["per_day"]["min"]),
        int(spec["orders"]["per_day"]["max"]),
    )

    main_cat_items = [(k, float(v)) for k, v in spec["orders"]["main_item_category_mix"].items()]
    refund_rate = float(spec["refunds"]["refund_rate"])
    refund_reasons = [(k, float(v)) for k, v in spec["refunds"]["reasons"].items()]

    accessory_attach_rate = float(spec["orders"]["accessory_attach_rate"])
    acc_min = int(spec["orders"]["accessories_per_order"]["min"])
    acc_max = int(spec["orders"]["accessories_per_order"]["max"])

    items_min = int(spec["orders"]["items_per_order"]["min"])
    items_max = int(spec["orders"]["items_per_order"]["max"])

    stats = {
        "payment_attempts": 0,
        "orders_succeeded": 0,
        "orders_failed": 0,
        "refunds": 0,
    }

    # --------------------------------------------------
    # Orders loop
    # --------------------------------------------------

    for i in range(n_orders):
        country = weighted_choice(rng, country_items)

        customer = stripe.Customer.create(
            email=f"{project}.{run_date.replace('-', '')}.{i}@example.com",
            address={"country": country},
            metadata={"project": project, "run_date": run_date, "country": country},
        )

        # Always attach a VALID card
        success_token = rng.choice(success_tokens)
        pm = stripe.PaymentMethod.create(type="card", card={"token": success_token})
        stripe.PaymentMethod.attach(pm.id, customer=customer.id)
        stripe.Customer.modify(
            customer.id,
            invoice_settings={"default_payment_method": pm.id},
        )

        # Decide failure at confirmation time
        use_failure = bool(failure_types) and (rng.random() < failure_rate)
        failure_type = rng.choice(failure_types) if use_failure else "success"
        failure_token = failure_tokens.get(failure_type)

        # Choose items
        main_cat = weighted_choice(rng, main_cat_items)
        main_sku = weighted_choice(
            rng,
            [(s, s["family_weight"]) for s in by_cat[main_cat]],
        )
        line_items = [main_sku]

        if rng.random() < accessory_attach_rate:
            for _ in range(rng.randint(acc_min, acc_max)):
                acc = weighted_choice(
                    rng,
                    [(s, s["family_weight"]) for s in by_cat.get("accessories", [])],
                )
                line_items.append(acc)

        while len(line_items) < items_min:
            acc = weighted_choice(
                rng,
                [(s, s["family_weight"]) for s in by_cat.get("accessories", [])],
            )
            line_items.append(acc)

        line_items = line_items[:items_max]

        amount = sum(int(s["price"]) for s in line_items)
        sku_keys = [
            sku_key(project, s["category"], s["brand"], s["model"], s["grade"], s["storage"])
            for s in line_items
        ]

        stats["payment_attempts"] += 1

        # Create PI without confirmation
        pi = stripe.PaymentIntent.create(
            customer=customer.id,
            amount=amount,
            currency=currency,
            payment_method=pm.id,
            confirm=False,
            automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
            metadata={
                "project": project,
                "run_date": run_date,
                "country": country,
                "item_count": str(len(line_items)),
                "sku_keys": ",".join(sku_keys[:30]),
                "attempt_expected": "failed" if use_failure else "succeeded",
                "failure_type": failure_type,
            },
        )

        try:
            if use_failure:
                stripe.PaymentIntent.confirm(
                    pi.id,
                    payment_method_data={
                        "type": "card",
                        "card": {"token": failure_token},
                    },
                )
            else:
                stripe.PaymentIntent.confirm(pi.id)

            stats["orders_succeeded"] += 1

            if rng.random() < refund_rate:
                reason = weighted_choice(rng, refund_reasons)
                stripe.Refund.create(
                    payment_intent=pi.id,
                    metadata={"project": project, "run_date": run_date, "reason": reason},
                )
                stats["refunds"] += 1

        except stripe.error.CardError:
            stats["orders_failed"] += 1
            continue

    print(json.dumps({"project": project, "run_date": run_date, **stats}, indent=2))


if __name__ == "__main__":
    main()
