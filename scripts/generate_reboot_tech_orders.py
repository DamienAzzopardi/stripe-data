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


def sku_key(project: str, category: str, brand: str, model: str, grade: str, storage: Optional[str]) -> str:
    s = storage or "na"
    raw = f"{project}|{category}|{brand}|{model}|{grade}|{s}"
    return raw.lower().replace(" ", "_")


def stripe_search_one_product_by_sku(sku: str) -> Optional[str]:
    # Product Search supports metadata queries. If Search isn't enabled, you can replace with list+scan.
    res = stripe.Product.search(query=f"metadata['sku']:'{sku}'", limit=1)
    if res.data:
        return res.data[0].id
    return None


def stripe_search_one_price(product_id: str, sku: str, currency: str, unit_amount: int) -> Optional[str]:
    # Price search by metadata
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
    Returns list of SKUs:
      {category, brand, model, grade, storage, price, family_weight}
    """
    out: List[Dict[str, Any]] = []
    for category, families in catalog["catalog"].items():
        for fam in families:
            w = float(fam.get("popularity_weight", 1.0))
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
                        "family_weight": w,
                    }
                )
    return out


def main() -> None:
    key = os.getenv("STRIPE_API_KEY")
    if not key:
        raise RuntimeError("Missing STRIPE_API_KEY")
    stripe.api_key = key

    spec = load_yaml(os.getenv("REBOOT_SPEC_PATH", "specs/reboot_tech.yml"))
    project = spec["app"]["project_name"]
    currency = spec["app"]["currency"]

    catalog_path = spec["catalog"]["file"]
    catalog = load_yaml(catalog_path)

    # deterministic per day if enabled
    tz_date = dt.date.today().isoformat()
    seed = stable_seed(f"{project}:{tz_date}") if spec.get("advanced", {}).get("deterministic_seed", False) else None
    rng = random.Random(seed)

    # Flatten catalog to SKUs and build weighted pools by category
    skus = flatten_catalog(project, catalog)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for s in skus:
        by_cat.setdefault(s["category"], []).append(s)

    # Upsert catalog (idempotent)
    sku_to_price: Dict[str, str] = {}
    for s in skus:
        sku = sku_key(project, s["category"], s["brand"], s["model"], s["grade"], s["storage"])
        _, price_id = upsert_sku_as_product_price(
            project=project,
            category=s["category"],
            brand=s["brand"],
            model=s["model"],
            grade=s["grade"],
            storage=s["storage"],
            currency=currency,
            unit_amount=s["price"],
        )
        sku_to_price[sku] = price_id

    # Countries
    countries = spec["markets"]["countries"]
    country_items = [(c["code"], float(c["weight"])) for c in countries]

    # Cards
    card_tokens = list(spec["payments"]["cards"]["tokens"].values())

    # Orders per day
    n_orders = rng.randint(int(spec["orders"]["per_day"]["min"]), int(spec["orders"]["per_day"]["max"]))

    # Main item category mix
    main_mix = spec["orders"]["main_item_category_mix"]
    main_cat_items = [(k, float(v)) for k, v in main_mix.items()]

    refund_rate = float(spec["refunds"]["refund_rate"])
    refund_reasons = [(k, float(v)) for k, v in spec["refunds"]["reasons"].items()]

    accessory_attach_rate = float(spec["orders"]["accessory_attach_rate"])
    acc_min = int(spec["orders"]["accessories_per_order"]["min"])
    acc_max = int(spec["orders"]["accessories_per_order"]["max"])

    items_min = int(spec["orders"]["items_per_order"]["min"])
    items_max = int(spec["orders"]["items_per_order"]["max"])

    created = {"orders": 0, "refunds": 0}

    for i in range(n_orders):
        country = weighted_choice(rng, country_items)

        customer = stripe.Customer.create(
            email=f"{project}.{tz_date.replace('-','')}.{i}@example.com",
            address={"country": country},
            metadata={"project": project, "run_date": tz_date, "country": country},
        )

        token = rng.choice(card_tokens)
        pm = stripe.PaymentMethod.create(type="card", card={"token": token})
        stripe.PaymentMethod.attach(pm.id, customer=customer.id)
        stripe.Customer.modify(customer.id, invoice_settings={"default_payment_method": pm.id})

        # choose main item category (phones/laptops/tablets/audio)
        main_cat = weighted_choice(rng, main_cat_items)
        pool = by_cat[main_cat]

        # weighted sku sampling: family_weight dominates
        main_sku = weighted_choice(rng, [(s, float(s["family_weight"])) for s in pool])

        line_items = [main_sku]

        # optionally attach accessories
        if rng.random() < accessory_attach_rate:
            n_acc = rng.randint(acc_min, acc_max)
            acc_pool = by_cat.get("accessories", [])
            if acc_pool:
                for _ in range(n_acc):
                    acc_sku = weighted_choice(rng, [(s, float(s["family_weight"])) for s in acc_pool])
                    line_items.append(acc_sku)

        # ensure items_per_order bounds overall
        while len(line_items) < items_min:
            # add accessory if needed
            acc_pool = by_cat.get("accessories", [])
            if not acc_pool:
                break
            acc_sku = weighted_choice(rng, [(s, float(s["family_weight"])) for s in acc_pool])
            line_items.append(acc_sku)

        if len(line_items) > items_max:
            line_items = line_items[:items_max]

        amount = 0
        sku_list = []
        categories = []
        for s in line_items:
            sku = sku_key(project, s["category"], s["brand"], s["model"], s["grade"], s["storage"])
            sku_list.append(sku)
            categories.append(s["category"])
            amount += int(s["price"])

        pi = stripe.PaymentIntent.create(
            customer=customer.id,
            amount=amount,
            currency=currency,
            payment_method=pm.id,
            confirm=True,
            automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
            metadata={
                "project": project,
                "run_date": tz_date,
                "country": country,
                "item_count": str(len(line_items)),
                "sku_keys": ",".join(sku_list[:30]),  # keep metadata small
                "categories": ",".join(categories[:30]),
            },
        )

        created["orders"] += 1

        # refund some orders
        if rng.random() < refund_rate:
            reason = weighted_choice(rng, refund_reasons)
            stripe.Refund.create(
                payment_intent=pi.id,
                metadata={"project": project, "run_date": tz_date, "reason": reason},
            )
            created["refunds"] += 1

    print(json.dumps({"project": project, "run_date": tz_date, **created}, indent=2))


if __name__ == "__main__":
    main()
