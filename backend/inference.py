"""
OpenEnv SRE — LLM-guided baseline inference script.

Runs all three tasks sequentially using an LLM-guided policy.
The LLM selects the next action based on the full serialised Observation
(including SHAP top-5, causal DAG, counterfactual results, and forecast).

STDOUT format:
  [START]          — at episode start
  [STEP] {json}    — before each action (observation + chosen action)
  [END] {score}    — at episode close with final normalised score

Required environment variables:
  API_BASE_URL      LLM inference endpoint
  MODEL_NAME        Model identifier for the OpenAI-compatible client
  HF_TOKEN          Hugging Face API token
  RANDOM_SEED       Integer seed for reproducibility

Usage:
  python inference.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("openenv.inference")

# ─────────────────────────────────────────────────────────────────────────────
# Environment Variables
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))

# Whether to use local environment directly or call the HTTP API
USE_LOCAL_ENV: bool = os.environ.get("USE_LOCAL_ENV", "true").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────────────────────────────────────

try:
    from openai import OpenAI
    _llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "na")
    _LLM_OK = True
except ImportError:
    _LLM_OK = False
    _llm_client = None
    logger.warning("openai package not installed; using heuristic fallback policy")
except TypeError as e:
    # Handles httpx incompatibility: "Client.__init__() got unexpected keyword 'proxies'"
    _LLM_OK = False
    _llm_client = None
    logger.warning("OpenAI client init failed (likely httpx version conflict: %s); using heuristic fallback", e)
except Exception as e:
    _LLM_OK = False
    _llm_client = None
    logger.warning("OpenAI client init failed (%s); using heuristic fallback", e)


SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) AI agent operating
within the OpenEnv Causal SRE Intelligence environment.

At each step you receive a JSON observation containing:
  - metrics: per-service MetricSnapshot (cpu, memory_rss, latency_p50/p95/p99, error_rate, pool_saturation)
  - alerts: list of triggered threshold alerts (some may be red herrings)
  - anomaly_scores: per-service anomaly score [0,1]
  - causal_dag: directed acyclic graph of causal relationships between services {child: [parents]}
  - root_cause_prediction: ML pipeline's best guess
  - root_cause_probabilities: full class probability distribution
  - shap_top5: top-5 SHAP features driving the prediction [(feature_name, shap_value), ...]
  - forecast: {forecast_t5, forecast_t15, drift_alert, confidence_band_width}
  - counterfactual_result: (if you queried counterfactual) predicted outcome of your chosen action

Decision rules:
1. Always check causal_dag first — a service with no causal parents in the DAG is likely the root cause.
2. Prefer services that appear as causal ancestors over those with merely high anomaly scores.
3. Before executing any restart_service or scale_service, call query_counterfactual first.
4. If counterfactual_result.harm_flag is True, DO NOT proceed with that action.
5. Silence red-herring alerts (alerts on services not in the causal path) before acting.
6. Once you have identified and fixed the root cause, call declare_resolution.

Respond only with a valid JSON action:
{
  "action_type": "<one of: restart_service | scale_service | run_diagnostic | silence_alert | query_counterfactual | escalate_incident | declare_resolution>",
  "service_id": "<service name or null>",
  "parameters": <null or object>
}"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Policy / Heuristic Fallback
# ─────────────────────────────────────────────────────────────────────────────

# (Removed old inline functions in favour of openenv.agent)

# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env_or_url: Any, task_id: int, seed: int, use_local: bool) -> float:
    from openenv.agent import HybridSREAgent
    agent = HybridSREAgent(use_llm=True, model_name=MODEL_NAME)
    
    """Run a single episode. Returns the normalised episode score [0, 1]."""
    cf_called = False

    if use_local:
        # Direct Python API
        obs_obj = env_or_url.reset(task_id=task_id, seed=seed)
        obs = obs_obj.model_dump()
    else:
        import requests
        r = requests.post(f"{env_or_url}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
        r.raise_for_status()
        obs = r.json()["observation"]

    print("[START]", flush=True)

    step = 0
    done = False
    episode_score = 0.0
    max_steps = {1: 30, 2: 45, 3: 60}[task_id]

    while not done and step < max_steps:
        step += 1

        # Choose action
        try:
            action_dict = agent.decide_action(obs, step, cf_called)
        except Exception:
            # Absolute worst-case fallback
            action_dict = {"action_type": "declare_resolution", "service_id": None, "parameters": None}

        if action_dict.get("action_type") == "query_counterfactual":
            cf_called = True

        # Log the step
        step_payload = {
            "step": step,
            "root_cause_prediction": obs.get("root_cause_prediction", ""),
            "top_anomaly": max(obs.get("anomaly_scores", {}).items(), key=lambda x: x[1], default=("?", 0)),
            "drift_alert": obs.get("drift_alert", False),
            "action": action_dict,
        }
        print(f"[STEP] {json.dumps(step_payload)}", flush=True)

        # Execute action
        if use_local:
            from openenv.models import Action, ActionType
            action = Action(
                action_type=ActionType(action_dict["action_type"]),
                service_id=action_dict.get("service_id"),
                parameters=action_dict.get("parameters"),
            )
            obs_obj, reward_obj, done, info = env_or_url.step(action)
            obs = obs_obj.model_dump()
            done_flag = done
            episode_score = info.get("episode_score", 0.0)
        else:
            import requests
            r = requests.post(f"{env_or_url}/step", json=action_dict, timeout=30)
            r.raise_for_status()
            result = r.json()
            obs = result["observation"]
            done_flag = result["done"]
            info = result.get("info", {})
            if done_flag:
                episode_score = info.get("episode_score", 0.0)
            done = done_flag

    return episode_score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import numpy as np
    np.random.seed(RANDOM_SEED)

    if USE_LOCAL_ENV:
        sys.path.insert(0, os.path.dirname(__file__))
        from openenv.environment import OpenEnvSRE
        env_or_url = OpenEnvSRE()
    else:
        env_or_url = API_BASE_URL

    task_weights = {1: 0.20, 2: 0.35, 3: 0.45}
    total_weighted_score = 0.0
    start_time = time.time()

    for task_id in [1, 2, 3]:
        logger.info("Starting Task %d (seed=%d)", task_id, RANDOM_SEED)
        task_score = run_episode(env_or_url, task_id, RANDOM_SEED, USE_LOCAL_ENV)
        weighted = task_score * task_weights[task_id]
        total_weighted_score += weighted
        logger.info("Task %d score: %.4f (weighted: %.4f)", task_id, task_score, weighted)

    elapsed = time.time() - start_time
    final = round(total_weighted_score, 4)
    print(f"[END] {json.dumps({'final_score': final, 'elapsed_seconds': round(elapsed, 1), 'seed': RANDOM_SEED})}", flush=True)
    logger.info("All tasks complete. Final score: %.4f in %.1fs", final, elapsed)


if __name__ == "__main__":
    main()
