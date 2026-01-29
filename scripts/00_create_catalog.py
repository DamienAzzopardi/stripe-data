import stripe
from _common import init_stripe, load_json, save_json

# Business configuration
PLANS = {
    "Basic": 999,     # €9.99
    "Standard": 1999, # €19.99
    "Premium": 2599,  # €25.99
    "Exec": 4999,     # €49.99
}

CURRENCY = "eur"


def find_product_by_name(name: str):
    """
    Stripe Search API is not always enabled for Products.
    Listing + filtering is the most robust approach.
    """
    products = stripe.Product.list(active=True, limit=100).data
    for p in products:
        if p.get("name") == name:
            return p
    return None


def find_price_for_product(product_id: str, unit_amount: int, currency: str):
    """
    Price.search does NOT support unit_amount.
    We must list prices for the product and filter client-side.
    """
    prices = stripe.Price.list(product=product_id, active=True, limit=100).data
    for p in prices:
        if (
            p.get("unit_amount") == unit_amount
            and p.get("currency") == currency
            and p.get("recurring")
            and p["recurring"].get("interval") == "month"
        ):
            return p
    return None


def main():
    init_stripe()

    # Load existing state (if any)
    state = load_json("price_ids.json", {})

    for plan_name, amount in PLANS.items():
        product_name = f"DAH Music – {plan_name}"

        # 1️⃣ Product
        product = find_product_by_name(product_name)
        if not product:
            product = stripe.Product.create(
                name=product_name,
                description=f"{plan_name} monthly subscription",
                metadata={
                    "dah_plan": plan_name,
                    "domain": "music_streaming",
                    "billing_model": "subscription",
                },
            )
            print(f"Created product: {product.name}")
        else:
            print(f"Found product: {product.name}")

        # 2️⃣ Monthly Price (EUR)
        price = find_price_for_product(product.id, amount, CURRENCY)
        if not price:
            price = stripe.Price.create(
                product=product.id,
                unit_amount=amount,
                currency=CURRENCY,
                recurring={"interval": "month"},
                metadata={
                    "dah_plan": plan_name,
                    "domain": "music_streaming",
                },
            )
            print(f"Created price for {plan_name}: {amount} {CURRENCY}")
        else:
            print(f"Found price for {plan_name}: {amount} {CURRENCY}")

        # Persist price ID for downstream scripts
        state[plan_name] = price.id

    save_json("price_ids.json", state)
    print("Saved price IDs to state/price_ids.json")


if __name__ == "__main__":
    main()
