import time
import stripe
from _common import init_stripe, load_json

SECONDS_IN_DAY = 86400

def main():
    init_stripe()

    clock_data = load_json("test_clock.json", None)
    if not clock_data:
        raise RuntimeError("Run 01_create_test_clock.py first")

    test_clock_id = clock_data["test_clock_id"]

    # Advance ~6 months, in 1-month steps (30 days)
    step_days = 30
    steps = 6

    # Start from current frozen time in state
    frozen_time = clock_data["frozen_time"]

    for i in range(steps):
        frozen_time += step_days * SECONDS_IN_DAY
        stripe.test_helpers.TestClock.advance(
            test_clock_id,
            frozen_time=frozen_time,
        )

        # Stripe advances asynchronously; poll until Ready
        while True:
            tc = stripe.test_helpers.TestClock.retrieve(test_clock_id)
            if tc.status == "ready":
                break
            time.sleep(1)

        print(f"Advanced to step {i+1}/{steps} (frozen_time={frozen_time})")

    print("Done. You should now have ~6 months of renewal activity.")

if __name__ == "__main__":
    main()
