import stripe
from _common import init_stripe, load_json

TEST_PM = "pm_card_visa"  # Stripe test PM

def main():
    init_stripe()

    data = load_json("customer_ids.json", None)
    if not data:
        raise RuntimeError("Run 02_create_customers.py first")

    customer_ids = data["customer_ids"]

    for cid in customer_ids:
        # Attach test payment method
        stripe.PaymentMethod.attach(TEST_PM, customer=cid)

        # Set it as default for invoice payments
        stripe.Customer.modify(
            cid,
            invoice_settings={"default_payment_method": TEST_PM},
        )

    print(f"Attached default payment method to {len(customer_ids)} customers.")

if __name__ == "__main__":
    main()
