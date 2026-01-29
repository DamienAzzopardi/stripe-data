import stripe
from _common import init_stripe, now_ts, save_json

SECONDS_IN_DAY = 86400

def main():
    init_stripe()

    frozen_time = now_ts() - 180 * SECONDS_IN_DAY  # ~6 months ago
    clock = stripe.test_helpers.TestClock.create(
        frozen_time=frozen_time,
        name="DAH Music Streaming - 6mo backfill",
    )

    save_json("test_clock.json", {"test_clock_id": clock.id, "frozen_time": frozen_time})
    print(f"Created test_clock: {clock.id}")

if __name__ == "__main__":
    main()
