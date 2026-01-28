import os
import random
import stripe

# Env vars set in GitHub Actions Secrets
stripe.api_key = os.environ["STRIPE_SECRET_KEY"]  # sk_test_...

CURRENCY = os.getenv("CURRENCY", "usd")
N_TX = int(os.getenv("N_TRANSACTIONS", "100"))

# Stripe test PaymentMethod for a successful card payment:
# pm_card_visa is a standard test PaymentMethod ID used in Stripe test mode.
TEST_PAYMENT_METHOD = os.getenv("STRIPE_TEST_PAYMENT_METHOD", "pm_card_visa")

def create_one_payment() -> str:
    """
    Creates and confirms a PaymentIntent in Stripe test mode.
    This results in a successful payment (a "transaction") you can sync downstream.
    """
    amount_cents = random.choice([499, 999, 1499, 2999, 4999])  # vary amounts

    intent = stripe.PaymentIntent.create(
        amount=amount_cents,
        currency=CURRENCY,
        payment_method=TEST_PAYMENT_METHOD,
        confirm=True,  # create + confirm in one call
        description="dbt course synthetic transaction",
        metadata={
            "generator": "github-actions",
            "course": "dbt",
        },
    )
    return intent.id

def main():
    created = []
    for _ in range(N_TX):
        created.append(create_one_payment())

    print(f"Created {len(created)} payment_intents")
    # Print a few IDs to help debugging
    print("Sample:", created[:5])

if __name__ == "__main__":
    main()
