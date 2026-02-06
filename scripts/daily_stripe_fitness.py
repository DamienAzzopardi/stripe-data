#!/usr/bin/env python3
"""
DAH Fitness App — Daily Stripe data generator
Implements the business model specs from the attached PDF.

Key behaviors implemented from specs  [oai_citation:11‡e9840e76-2ffd-40cc-9538-bf89a19c0cb7_Fitness_App_Business_Model_Specs_(B2C).pdf](sediment://file_00000000ba00720a8f3e3c679360ff75):
- Markets: FR/ES/IT billed in EUR; US billed in USD
- Card networks: Visa/Mastercard/Amex/Discover/JCB using Stripe test tokens (randomly assigned)
- Daily new customers: random integer between 100 and 150
- At signup: assign exactly 1 plan using a DAILY randomized distribution within:
  Weekly 35–55%, Monthly 25–40%, Yearly 10–25% (must sum to 100%)
- Each customer can have at most one active subscription at any time (enforced by using 1 Stripe Subscription per customer)
- Decision points occur at billing period end. At each decision point, evaluate strictly:
  1) Upgrade (if eligible)
  2) Churn
  3) Renewal (do nothing; Stripe will renew)
- Upgrades (no downgrades):
  Weekly -> Monthly: after >=1 completed weekly cycle; 20% chance at each weekly decision point
  Monthly -> Yearly: after >=2 completed monthly cycles; 12% chance at each eligible monthly decision point
- Churn probabilities at decision points:
  Weekly 25%, Monthly 8%, Yearly 15%
- Invoicing realism:
  Stripe subscription invoices are generated at start and at renewal.
  On upgrade we switch price with proration_behavior='none' (no proration as per spec simplification).

Notes:
- We attach a default card payment method to each new customer.
- We store plan + cycles_completed in Stripe Subscription metadata so daily runs can apply eligibility.
"""

from __future__ import annotations

import os
import sys
import json
import random
import hashlib
import datetime as dt
from typing import Any, Dict, Optional, Tuple

import stripe
import yaml
from zoneinfo import ZoneInfo


DEFAULT_SPEC_PATH = "specs/fitness_app.yml"
STATE_DIR = ".state"
STATE_PATH = os.path.join(STATE_DIR, "fitness_state.json")


# -------------------------
# Utilities
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"runs": []}


def save_state(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stable_int_seed(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def choose_weighted_key(rng: random.Random, weights: Dict[str, float]) -> str:
    total = sum(weights.values())
    r = rng.random() * total
    acc = 0.0
    for k, w in weights.items():
        acc += w
        if r <= acc:
            return k
    return next(iter(weights.keys()))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def random_plan_distribution_within_ranges(
    rng: random.Random,
    ranges: Dict[str, Dict[str, float]],
    max_tries: int = 5000,
) -> Dict[str, float]:
    """
    Sample a distribution within per-plan min/max bounds that sums to 1.0.
    Ranges are inclusive floats.
    """
    keys = list(ranges.keys())

    for _ in range(max_tries):
        vals = {}
        # sample weekly and monthly, then derive yearly to sum 1.0
        # works because we have exactly 3 plans in spec
        w = rng.uniform(ranges["weekly"]["min"], ranges["weekly"]["max"])
        m = rng.uniform(ranges["monthly"]["min"], ranges["monthly"]["max"])
        y = 1.0 - (w + m)

        if y < ranges["yearly"]["min"] or y > ranges["yearly"]["max"]:
            continue

        vals["weekly"] = w
        vals["monthly"] = m
        vals["yearly"] = y

        # Normalize small float drift
        s = sum(vals.values())
        vals = {k: v / s for k, v in vals.items()}

        # Safety check
        ok = True
        for k in keys:
            if vals[k] < ranges[k]["min"] - 1e-9 or vals[k] > ranges[k]["max"] + 1e-9:
                ok = False
                break
        if ok:
            return vals

    raise RuntimeError("Could not sample a valid plan distribution within ranges.")


def utc_now_ts() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def ts_to_local_date(ts: int, tz: ZoneInfo) -> dt.date:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).astimezone(tz).date()


# -------------------------
# Stripe setup
# -------------------------

def stripe_set_key() -> None:
    key = os.getenv("STRIPE_API_KEY")
    if not key:
        raise RuntimeError("Missing STRIPE_API_KEY")
    stripe.api_key = key


# -------------------------
# Stripe catalog (products/prices)
# -------------------------

def upsert_product(product_key: str, product_name: str) -> str:
    existing = stripe.Product.search(query=f"metadata['key']:'{product_key}'", limit=1)
    if existing.data:
        return existing.data[0].id
    p = stripe.Product.create(
        name=product_name,
        metadata={"key": product_key, "generator": "dah_fitness"},
    )
    return p.id


def upsert_price(
    product_id: str,
    plan_key: str,
    currency: str,
    unit_amount: int,
    interval: str,
    interval_count: int,
) -> str:
    # Uniqueness by metadata (plan_key + currency)
    q = (
        f"metadata['plan_key']:'{plan_key}' AND "
        f"metadata['currency']:'{currency}' AND "
        f"metadata['generator']:'dah_fitness'"
    )
    existing = stripe.Price.search(query=q, limit=1)
    if existing.data:
        return existing.data[0].id

    price = stripe.Price.create(
        product=product_id,
        currency=currency,
        unit_amount=unit_amount,
        recurring={"interval": interval, "interval_count": interval_count},
        nickname=f"{plan_key}_{currency}",
        metadata={
            "generator": "dah_fitness",
            "plan_key": plan_key,
            "currency": currency,
        },
    )
    return price.id


def build_price_map(spec: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    """
    Returns mapping: (plan_key, currency) -> price_id
    """
    product_key = spec["stripe"]["product"]["key"]
    product_name = spec["stripe"]["product"]["name"]
    product_id = upsert_product(product_key, product_name)

    price_map: Dict[Tuple[str, str], str] = {}
    for plan_key, plan_cfg in spec["plans"].items():
        interval = plan_cfg["interval"]
        interval_count = int(plan_cfg["interval_count"])

        eur_amount = int(plan_cfg["eur_unit_amount"])
        usd_amount = int(plan_cfg["usd_unit_amount"])

        price_map[(plan_key, "eur")] = upsert_price(
            product_id=product_id,
            plan_key=plan_key,
            currency="eur",
            unit_amount=eur_amount,
            interval=interval,
            interval_count=interval_count,
        )
        price_map[(plan_key, "usd")] = upsert_price(
            product_id=product_id,
            plan_key=plan_key,
            currency="usd",
            unit_amount=usd_amount,
            interval=interval,
            interval_count=interval_count,
        )

    return price_map


# -------------------------
# Customer + payment method
# -------------------------

def create_customer(
    run_date: str,
    idx: int,
    country_code: str,
    currency: str,
) -> stripe.Customer:
    email = f"dah.fitness.{country_code.lower()}.{run_date.replace('-', '')}.{idx}@example.com"
    return stripe.Customer.create(
        email=email,
        description="DAH Fitness synthetic customer",
        address={"country": country_code},
        metadata={
            "generator": "dah_fitness",
            "run_date": run_date,
            "daily_idx": str(idx),
            "country": country_code,
            "currency": currency,
        },
    )


def attach_default_card_from_spec(
    rng: random.Random,
    customer_id: str,
    card_cfg: Dict[str, Any],
    run_date: str,
    idx: int,
) -> str:
    network = choose_weighted_key(rng, card_cfg["distribution"])
    token = card_cfg["stripe_test_tokens"][network]

    pm = stripe.PaymentMethod.create(
        type="card",
        card={"token": token},
        metadata={
            "generator": "dah_fitness",
            "run_date": run_date,
            "daily_idx": str(idx),
            "card_network": network,
        },
    )
    stripe.PaymentMethod.attach(pm.id, customer=customer_id)
    stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": pm.id})
    return network


# -------------------------
# Subscriptions + decision points
# -------------------------

def create_subscription(
    customer_id: str,
    plan_key: str,
    currency: str,
    price_id: str,
    run_date: str,
    idx: int,
) -> stripe.Subscription:
    # Exactly one active subscription per customer (we create only one; upgrades mutate it)
    sub = stripe.Subscription.create(
        customer=customer_id,
        items=[{"price": price_id}],
        proration_behavior="none",
        metadata={
            "generator": "dah_fitness",
            "run_date": run_date,
            "daily_idx": str(idx),
            "plan_key": plan_key,
            "currency": currency,
            "cycles_completed": "0",
        },
    )
    return sub


def parse_int(s: Optional[str], default: int = 0) -> int:
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def maybe_upgrade_subscription(
    rng: random.Random,
    spec: Dict[str, Any],
    sub: stripe.Subscription,
    price_map: Dict[Tuple[str, str], str],
    prospective_cycles_completed: int,
) -> Optional[str]:
    """
    Returns new_plan_key if upgraded, else None.
    Upgrade is evaluated BEFORE churn at decision points.  [oai_citation:12‡e9840e76-2ffd-40cc-9538-bf89a19c0cb7_Fitness_App_Business_Model_Specs_(B2C).pdf](sediment://file_00000000ba00720a8f3e3c679360ff75)
    """
    plan_key = (sub.metadata or {}).get("plan_key")
    currency = (sub.metadata or {}).get("currency", "eur")

    upgrades = spec["upgrades"]

    # Weekly -> Monthly
    if plan_key == upgrades["weekly_to_monthly"]["from"]:
        eligible_after = int(upgrades["weekly_to_monthly"]["eligible_after_completed_cycles"])
        p = float(upgrades["weekly_to_monthly"]["probability"])
        if prospective_cycles_completed >= eligible_after and rng.random() < p:
            new_plan = upgrades["weekly_to_monthly"]["to"]
            new_price_id = price_map[(new_plan, currency)]
            stripe.Subscription.modify(
                sub.id,
                items=[{"id": sub["items"]["data"][0]["id"], "price": new_price_id}],
                proration_behavior="none",
                billing_cycle_anchor="now",
                metadata={
                    **(sub.metadata or {}),
                    "plan_key": new_plan,
                    "cycles_completed": "0",
                    "upgraded_from": plan_key,
                },
            )
            return new_plan

    # Monthly -> Yearly
    if plan_key == upgrades["monthly_to_yearly"]["from"]:
        eligible_after = int(upgrades["monthly_to_yearly"]["eligible_after_completed_cycles"])
        p = float(upgrades["monthly_to_yearly"]["probability"])
        if prospective_cycles_completed >= eligible_after and rng.random() < p:
            new_plan = upgrades["monthly_to_yearly"]["to"]
            new_price_id = price_map[(new_plan, currency)]
            stripe.Subscription.modify(
                sub.id,
                items=[{"id": sub["items"]["data"][0]["id"], "price": new_price_id}],
                proration_behavior="none",
                billing_cycle_anchor="now",
                metadata={
                    **(sub.metadata or {}),
                    "plan_key": new_plan,
                    "cycles_completed": "0",
                    "upgraded_from": plan_key,
                },
            )
            return new_plan

    return None


def maybe_churn_subscription(
    rng: random.Random,
    spec: Dict[str, Any],
    sub: stripe.Subscription,
    prospective_cycles_completed: int,
) -> bool:
    """
    Churn evaluated only at decision points (after upgrade attempt).  [oai_citation:13‡e9840e76-2ffd-40cc-9538-bf89a19c0cb7_Fitness_App_Business_Model_Specs_(B2C).pdf](sediment://file_00000000ba00720a8f3e3c679360ff75)
    """
    plan_key = (sub.metadata or {}).get("plan_key")
    churn_p = float(spec["plans"][plan_key]["churn_probability"])

    if rng.random() < churn_p:
        # Cancel immediately at decision point: no future invoices
        stripe.Subscription.cancel(
            sub.id,
            invoice_now=False,
            prorate=False,
        )
        return True

    # Not churned => renew (Stripe handles invoice + new period)
    # We still increment cycles_completed in metadata to track eligibility.
    stripe.Subscription.modify(
        sub.id,
        metadata={**(sub.metadata or {}), "cycles_completed": str(prospective_cycles_completed)},
    )
    return False


def process_decision_points_for_today(
    rng: random.Random,
    spec: Dict[str, Any],
    run_date: str,
    tz: ZoneInfo,
    price_map: Dict[Tuple[str, str], str],
) -> Dict[str, int]:
    """
    Process decision points for subscriptions whose current period ends on run_date (local tz).
    Event priority: Upgrade -> Churn -> Renewal.  [oai_citation:14‡e9840e76-2ffd-40cc-9538-bf89a19c0cb7_Fitness_App_Business_Model_Specs_(B2C).pdf](sediment://file_00000000ba00720a8f3e3c679360ff75)
    """
    counters = {
        "decision_points_processed": 0,
        "upgrades": 0,
        "churns": 0,
        "renewals": 0,
    }

    # Stripe lists are paginated; keep it bounded but auto_page.
    subs = stripe.Subscription.list(status="active", limit=100)

    for sub in subs.auto_paging_iter():
        md = sub.metadata or {}
        if md.get("generator") != "dah_fitness":
            continue

        end_ts = int(sub.current_period_end)
        end_date_local = ts_to_local_date(end_ts, tz)

        if end_date_local.isoformat() != run_date:
            continue

        counters["decision_points_processed"] += 1

        cycles_completed = parse_int(md.get("cycles_completed"), default=0)
        prospective = cycles_completed + 1  # you just completed a period at this decision point

        # 1) Upgrade (if eligible)
        upgraded_to = maybe_upgrade_subscription(rng, spec, sub, price_map, prospective)
        if upgraded_to:
            counters["upgrades"] += 1
            continue

        # 2) Churn (else)
        churned = maybe_churn_subscription(rng, spec, sub, prospective)
        if churned:
            counters["churns"] += 1
            continue

        # 3) Renewal (do nothing, Stripe will renew)
        counters["renewals"] += 1

    return counters


# -------------------------
# Main daily run
# -------------------------

def main() -> None:
    stripe_set_key()

    spec_path = os.getenv("FITNESS_SPEC_PATH", DEFAULT_SPEC_PATH)
    spec = load_yaml(spec_path)

    tz = ZoneInfo(spec["app"]["timezone"])

    # RUN_DATE defaults to today's date in the app timezone.
    run_date = os.getenv("RUN_DATE")
    if not run_date:
        run_date = dt.datetime.now(tz).date().isoformat()

    rng = random.Random()
    if spec.get("advanced", {}).get("deterministic_seed", False):
        rng.seed(stable_int_seed(f"dah_fitness:{run_date}"))

    price_map = build_price_map(spec)

    # 0) Process decision points (existing subs ending today)
    dp_counters = process_decision_points_for_today(
        rng=rng,
        spec=spec,
        run_date=run_date,
        tz=tz,
        price_map=price_map,
    )

    # 1) Daily new customers
    new_min = int(spec["daily_generation"]["new_customers_per_day"]["min"])
    new_max = int(spec["daily_generation"]["new_customers_per_day"]["max"])
    n_new = rng.randint(new_min, new_max)

    # 2) Daily randomized plan distribution within ranges
    ranges = spec["daily_generation"]["plan_allocation_ranges"]
    daily_dist = random_plan_distribution_within_ranges(rng, ranges)

    # 3) Prepare market/country distribution
    countries = spec["markets"]["countries"]
    country_weights = {c["code"]: float(c["weight"]) for c in countries}
    country_currency = {c["code"]: c["currency"] for c in countries}

    # Determine per-plan target counts from daily distribution
    # (adjust rounding to keep total == n_new)
    raw_counts = {k: int(round(v * n_new)) for k, v in daily_dist.items()}
    # Fix rounding drift
    drift = n_new - sum(raw_counts.values())
    # Distribute drift to the largest fractions
    if drift != 0:
        # sort by share desc
        order = sorted(daily_dist.items(), key=lambda kv: kv[1], reverse=True)
        idx = 0
        while drift != 0 and idx < 1000:
            plan = order[idx % len(order)][0]
            raw_counts[plan] += 1 if drift > 0 else -1
            drift += -1 if drift > 0 else 1
            idx += 1

    # Build a shuffled plan assignment list
    plan_assignments = []
    for plan, cnt in raw_counts.items():
        plan_assignments.extend([plan] * max(0, cnt))
    rng.shuffle(plan_assignments)
    plan_assignments = plan_assignments[:n_new]  # safety

    card_cfg = spec["payment_methods"]["cards"]

    created_customers = 0
    created_subscriptions = 0
    created_by_plan = {"weekly": 0, "monthly": 0, "yearly": 0}
    created_by_country = {c["code"]: 0 for c in countries}

    for i in range(n_new):
        country = choose_weighted_key(rng, country_weights)
        currency = country_currency[country]

        cust = create_customer(
            run_date=run_date,
            idx=i,
            country_code=country,
            currency=currency,
        )
        created_customers += 1
        created_by_country[country] += 1

        # Attach default card (random network)
        attach_default_card_from_spec(rng, cust.id, card_cfg, run_date, i)

        # Assign exactly one subscription plan
        plan_key = plan_assignments[i] if i < len(plan_assignments) else choose_weighted_key(rng, daily_dist)
        price_id = price_map[(plan_key, currency)]
        _sub = create_subscription(cust.id, plan_key, currency, price_id, run_date, i)

        created_subscriptions += 1
        created_by_plan[plan_key] += 1

    # Persist a lightweight run log
    ensure_dir(STATE_DIR)
    state = load_state(STATE_PATH)

    summary = {
        "run_date": run_date,
        "new_customers_created": created_customers,
        "subscriptions_created": created_subscriptions,
        "created_by_plan": created_by_plan,
        "created_by_country": created_by_country,
        "daily_plan_distribution": daily_dist,
        "decision_points": dp_counters,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
    }

    state["runs"].append(summary)
    save_state(STATE_PATH, state)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
