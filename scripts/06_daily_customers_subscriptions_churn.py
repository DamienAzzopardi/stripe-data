import os
import random
import stripe
from _common import init_stripe, load_json, now_ts

TEST_PM = os.getenv("STRIPE_TEST_PAYMENT_METHOD", "pm_card_visa")

def attach_default_payment_method(customer_id: str):
    stripe.PaymentMethod.attach(TEST_PM, customer=customer_id)
    stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": TEST_PM})

def create_customer(i: int) -> str:
    c = stripe.Customer.create(
        email=f"dah_user_{now_ts()}_{i}@example.com",
        name=f"DAH User {i}",
        metadata={"domain": "music_streaming", "generator": "github-actions", "cohort": "daily"},
    )
    return c.id

def create_subscription(customer_id: str, price_id: str, plan: str):
    stripe.Subscription.create(
        customer=customer_id,
        items=[{"price": price_id}],
        collection_method="charge_automatically",
        payment_behavior="allow_incomplete",
        metadata={"domain": "music_streaming", "dah_plan": plan, "generator": "github-actions"},
    )

def pick_plan() -> str:
    # tune whenever you want
    weights = {
        "Basic": 0.55,
        "Standard": 0.25,
        "Premium": 0.15,
        "Exec": 0.05,
    }
    plans = list(weights.keys())
    w = list(weights.values())
    return random.choices(plans, weights=w, k=1)[0]

def cancel_some_active_subscriptions(n_cancel: int, cancel_mode: str):
    """
    cancel_mode:
      - "period_end": cancel_at_period_end=True
      - "immediate": cancel immediately
    """
    if n_cancel <= 0:
        print("Churn disabled (0 cancellations).")
        return

    # Fetch active subs created by our generator/domain
    # Stripe Search API supports queries; if Search isn't enabled for your account,
    # fall back to listing and filtering in code.
    candidates = []

    try:
        res = stripe.Subscription.search(
            query="status:'active' AND metadata['domain']:'music_streaming'",
            limit=100
        )
        candidates = res.data
    except Exception:
        # fallback: list and filter
        res = stripe.Subscription.list(status="active", limit=100)
        candidates = [s for s in res.data if (s.metadata or {}).get("domain") == "music_streaming"]

    if not candidates:
        print("No active subscriptions found to churn.")
        return

    random.shuffle(candidates)
    to_cancel = candidates[: min(n_cancel, len(candidates))]

    for s in to_cancel:
        if cancel_mode == "immediate":
            stripe.Subscription.delete(s.id)
        else:
            stripe.Subscription.modify(s.id, cancel_at_period_end=True)

    print(f"Churned {len(to_cancel)} subscriptions (mode={cancel_mode}).")

def main():
    init_stripe()

    prices = load_json("price_ids.json", None)
    if not prices:
        raise RuntimeError("Missing price_ids.json. Run 00_create_catalog.py first in the workflow.")

    n_new = int(os.getenv("N_NEW_CUSTOMERS", "10"))
    n_cancel = int(os.getenv("N_CHURN_EVENTS", "0"))
    cancel_mode = os.getenv("CHURN_MODE", "period_end")  # period_end | immediate

    new_customers = []
    for i in range(n_new):
        cid = create_customer(i)
        attach_default_payment_method(cid)

        plan = pick_plan()
        create_subscription(cid, prices[plan], plan)

        new_customers.append(cid)

    print(f"Created {len(new_customers)} customers + subscriptions.")
    cancel_some_active_subscriptions(n_cancel=n_cancel, cancel_mode=cancel_mode)

if __name__ == "__main__":
    main()
