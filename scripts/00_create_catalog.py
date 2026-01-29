import stripe
from _common import init_stripe, load_json, save_json

PLANS = {
  "Basic":    999,   # cents
  "Standard": 1999,
  "Premium":  2599,
  "Exec":     4999,
}
CURRENCY = "eur"

def find_product_by_name(name: str):
    # Stripe Search API for Products
    res = stripe.Product.search(query=f"name:'{name}' AND active:'true'", limit=1)
    return res.data[0] if res.data else None

def find_price_for_product(product_id: str, unit_amount: int):
    res = stripe.Price.search(
        query=f"product:'{product_id}' AND active:'true' AND unit_amount:{unit_amount} AND currency:'{CURRENCY}'",
        limit=1
    )
    return res.data[0] if res.data else None

def main():
    init_stripe()
    state = load_json("price_ids.json", {})

    for plan_name, amount in PLANS.items():
        product_name = f"DAH Music - {plan_name}"

        product = find_product_by_name(product_name)
        if not product:
            product = stripe.Product.create(
                name=product_name,
                metadata={"dah_plan": plan_name, "domain": "music_streaming"},
            )

        price = find_price_for_product(product.id, amount)
        if not price:
            price = stripe.Price.create(
                product=product.id,
                unit_amount=amount,
                currency=CURRENCY,
                recurring={"interval": "month"},
                metadata={"dah_plan": plan_name},
            )

        state[plan_name] = price.id
        print(f"{plan_name}: price={price.id} product={product.id}")

    save_json("price_ids.json", state)
    print("Saved stripe/state/price_ids.json")

if __name__ == "__main__":
    main()
