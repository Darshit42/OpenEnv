"""OpenEnv SRE — Validation inference script (ROOT).

Calls: POST /reset → POST /step (×3) → GET /state
Prints: [START] ... [STEP] ... [END]

Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
"""

import os
import sys
import time
import json
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")


def main():
    base_url = API_BASE_URL.rstrip("/")

    print("[START]")
    try:
        # 1. Reset (empty body — validator requirement)
        res_reset = requests.post(f"{base_url}/reset", json={}, timeout=30)
        res_reset.raise_for_status()
        reset_data = res_reset.json()

        # 2. Step (multiple times)
        for i in range(3):
            print("[STEP]")
            res_step = requests.post(f"{base_url}/step", json={
                "action_type": "query_counterfactual",
                "service_id": "api-gateway"
            }, timeout=30)
            res_step.raise_for_status()
            step_data = res_step.json()

        # 3. State
        res_state = requests.get(f"{base_url}/state", timeout=30)
        res_state.raise_for_status()

        print("[END]")
    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    time.sleep(1)
    main()
