import json
import os
import time
from pathlib import Path
import stripe

def init_stripe():
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]

def state_dir() -> Path:
    d = Path(os.getenv("DAH_STATE_DIR", "stripe/state"))
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_json(name: str, default):
    p = state_dir() / name
    if not p.exists():
        return default
    return json.loads(p.read_text())

def save_json(name: str, data):
    p = state_dir() / name
    p.write_text(json.dumps(data, indent=2, sort_keys=True))

def now_ts() -> int:
    return int(time.time())
