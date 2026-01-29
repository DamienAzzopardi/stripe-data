import random
import stripe
from _common import init_stripe, load_json

def main():
    init_stripe()

    prices = load_json("price_ids.json", None)
    if not prices:
        raise RuntimeError("Run 00_create_catalog.py first")

    data = load_json("customer_ids.json", None)
    if not data:
        raise RuntimeError("Run 02_create_customers.py first")

    # Feel free to tune this distribution later
    plan_weights = {
        "Basic": 0.55,
        "Standard": 0.25,
        "Premium": 0.15,
        "Exec": 0.05,
    }

    plans = list(plan_weights.keys())
    weights = list(plan_weights.values())

    for cid in data["customer_ids"]:
        plan = random.choices(plans, weights=weights, k=1)[0]
        price_id = prices[plan]

        stripe.Subscription.create(
            customer=cid,
            items=[{"price": price_id}],
            collection_method="charge_automatically",
            payment_behavior="allow_incomplete",
            metadata={"dah_plan": plan, "domain": "music_streaming"},
        )

    print(f"Created subscriptions for {len(data['customer_ids'])} customers.")

if __name__ == "__main__":
    main()
