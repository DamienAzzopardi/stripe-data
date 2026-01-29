import random
import stripe
from _common import init_stripe, load_json, save_json, now_ts

def main():
    init_stripe()

    clock = load_json("test_clock.json", None)
    if not clock:
        raise RuntimeError("Run 01_create_test_clock.py first")

    n = int(random.choice([50, 80, 120]))  # you can control this
    customers = []

    for i in range(n):
        c = stripe.Customer.create(
            email=f"dah_user_{now_ts()}_{i}@example.com",
            name=f"DAH User {i}",
            test_clock=clock["test_clock_id"],
            metadata={"domain": "music_streaming"},
        )
        customers.append(c.id)

    save_json("customer_ids.json", {"customer_ids": customers})
    print(f"Created {len(customers)} customers (under test_clock).")

if __name__ == "__main__":
    main()
