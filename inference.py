"""
OpenEnv SRE — LLM-guided baseline inference script (ROOT).

Runs tasks sequentially using an LLM-guided policy.
Hits local FastAPI environment and uses LiteLLM proxy for the agent.

Required environment variables:
  API_BASE_URL      LiteLLM proxy endpoint
  API_KEY           LiteLLM API key
  MODEL_NAME        Model identifier
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("openenv.inference")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

# Connect directly to the local FastAPI app that should be running
ENV_API_URL = "http://127.0.0.1:7860"

def run_episode(task_id: int, seed: int) -> float:
    # Need to load agent here to avoid import issues if not in path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
    from openenv.agent import HybridSREAgent
    
    agent = HybridSREAgent(use_llm=True, model_name=MODEL_NAME)
    cf_called = False

    logger.info(f"Connecting to environment at {ENV_API_URL}/reset (task={task_id})")
    try:
        r = requests.post(f"{ENV_API_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observation", {})
    except Exception as e:
        logger.error(f"Failed to reset environment: {e}")
        return 0.0

    print("[START]", flush=True)

    step = 0
    done = False
    episode_score = 0.0
    max_steps = {1: 30, 2: 45, 3: 60}.get(task_id, 30)

    while not done and step < max_steps:
        step += 1

        # Agent decision
        try:
            action_dict = agent.decide_action(obs, step, cf_called)
            if not action_dict:
                action_dict = {"action_type": "declare_resolution", "service_id": None, "parameters": None, "reasoning": "Fallback"}
        except Exception as e:
            logger.error(f"Agent error: {e}")
            action_dict = {"action_type": "declare_resolution", "service_id": None, "parameters": None, "reasoning": "Error fallback"}

        if action_dict.get("action_type") == "query_counterfactual":
            cf_called = True

        step_payload = {
            "step": step,
            "root_cause_prediction": obs.get("root_cause_prediction", ""),
            "top_anomaly": max(obs.get("anomaly_scores", {}).items(), key=lambda x: x[1], default=("?", 0)),
            "drift_alert": obs.get("drift_alert", False),
            "action": action_dict,
        }
        print(f"[STEP] {json.dumps(step_payload)}", flush=True)

        req_action = {
            "action_type": action_dict["action_type"],
            "service_id": action_dict.get("service_id"),
            "parameters": action_dict.get("parameters")
        }

        # Step Environment
        try:
            r = requests.post(f"{ENV_API_URL}/step", json=req_action, timeout=30)
            r.raise_for_status()
            result = r.json()
            obs = result.get("observation", {})
            done = result.get("done", False)
            info = result.get("info", {})
            if done:
                episode_score = info.get("episode_score", 0.0)
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            break

    # Call /state as explicitly required by the validator checks
    try:
        st_req = requests.get(f"{ENV_API_URL}/state", timeout=30)
        st_req.raise_for_status()
    except Exception as e:
        logger.warning(f"Warning: /state check failed: {e}")

    return episode_score


def main() -> None:
    import numpy as np
    np.random.seed(RANDOM_SEED)

    task_weights = {1: 0.20, 2: 0.35, 3: 0.45}
    total_weighted_score = 0.0
    start_time = time.time()

    for task_id in [1, 2, 3]:
        logger.info("Starting Task %d (seed=%d)", task_id, RANDOM_SEED)
        task_score = run_episode(task_id, RANDOM_SEED)
        weighted = task_score * task_weights[task_id]
        total_weighted_score += weighted
        logger.info("Task %d score: %.4f (weighted: %.4f)", task_id, task_score, weighted)

    elapsed = time.time() - start_time
    final = round(total_weighted_score, 4)
    print(f"[END] {json.dumps({'final_score': final, 'elapsed_seconds': round(elapsed, 1), 'seed': RANDOM_SEED})}", flush=True)
    logger.info("All tasks complete. Final score: %.4f in %.1fs", final, elapsed)


if __name__ == "__main__":
    # Give the backend server a moment to init if started concurrently
    time.sleep(2)
    main()
