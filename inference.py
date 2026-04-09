"""
OpenEnv SRE — Validation inference script.

Runs all three tasks sequentially using a hybrid SRE agent policy.
Uses the local Python environment directly (no HTTP dependency) for reliability.

STDOUT format:
  [START]          — at episode start
  [STEP] {json}    — before each action (observation + chosen action)
  [END] {score}    — at episode close with final normalised score

Required environment variables:
  API_BASE_URL      LLM inference endpoint (for optional LLM-guided policy)
  MODEL_NAME        Model identifier for the OpenAI-compatible client
  HF_TOKEN          Hugging Face API token
  RANDOM_SEED       Integer seed for reproducibility
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict

# ── Ensure backend is importable ──────────────────────────────────────────────
# This script runs from /app in Docker; backend/ is at /app/backend
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("openenv.inference")

# ── Environment Variables ─────────────────────────────────────────────────────
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")
RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env: Any, task_id: int, seed: int) -> float:
    """Run a single episode locally. Returns the normalised episode score [0, 1]."""
    try:
        from openenv.agent import HybridSREAgent
        agent = HybridSREAgent(use_llm=False, model_name=MODEL_NAME)
    except Exception as e:
        logger.warning("Could not init HybridSREAgent (%s); using inline fallback", e)
        agent = None

    cf_called = False

    try:
        obs_obj = env.reset(task_id=task_id, seed=seed)
        obs = obs_obj.model_dump()
    except Exception as e:
        logger.error("reset() failed: %s", e)
        return 0.0

    print("[START]", flush=True)

    step = 0
    done = False
    episode_score = 0.0
    max_steps = {1: 30, 2: 45, 3: 60}.get(task_id, 30)

    while not done and step < max_steps:
        step += 1

        # ── Choose action ────────────────────────────────────────────────────
        try:
            if agent is not None:
                action_dict = agent.decide_action(obs, step, cf_called)
            else:
                action_dict = _fallback_action(obs, step, cf_called)
        except Exception:
            action_dict = {
                "action_type": "declare_resolution",
                "service_id": None,
                "parameters": None,
            }

        if action_dict.get("action_type") == "query_counterfactual":
            cf_called = True

        # ── Log the step ─────────────────────────────────────────────────────
        try:
            anomaly_scores = obs.get("anomaly_scores", {})
            top_anomaly = max(
                anomaly_scores.items(), key=lambda x: x[1], default=("?", 0)
            )
            step_payload = {
                "step": step,
                "root_cause_prediction": obs.get("root_cause_prediction", ""),
                "top_anomaly": top_anomaly,
                "drift_alert": obs.get("drift_alert", False),
                "action": action_dict,
            }
            print(f"[STEP] {json.dumps(step_payload)}", flush=True)
        except Exception:
            print(f"[STEP] {json.dumps({'step': step, 'action': action_dict})}", flush=True)

        # ── Execute action ───────────────────────────────────────────────────
        try:
            from openenv.models import Action, ActionType

            action = Action(
                action_type=ActionType(action_dict["action_type"]),
                service_id=action_dict.get("service_id"),
                parameters=action_dict.get("parameters"),
            )
            obs_obj, reward_obj, done, info = env.step(action)
            obs = obs_obj.model_dump()
            if done:
                episode_score = info.get("episode_score", 0.0)
        except Exception as e:
            logger.error("step() failed at step %d: %s", step, e)
            done = True
            episode_score = 0.0

    return episode_score


def _fallback_action(obs: Dict[str, Any], step: int, cf_called: bool) -> Dict[str, Any]:
    """Minimal heuristic fallback if HybridSREAgent can't be imported."""
    services = obs.get("services", [])
    anomaly_scores = obs.get("anomaly_scores", {})

    target = max(services, key=lambda s: anomaly_scores.get(s, 0)) if services else None

    if not cf_called and target:
        return {
            "action_type": "query_counterfactual",
            "service_id": target,
            "parameters": None,
        }
    if step >= 15:
        return {
            "action_type": "declare_resolution",
            "service_id": None,
            "parameters": None,
        }
    if target:
        return {
            "action_type": "restart_service",
            "service_id": target,
            "parameters": None,
        }
    return {
        "action_type": "declare_resolution",
        "service_id": None,
        "parameters": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import numpy as np
        np.random.seed(RANDOM_SEED)
    except Exception:
        pass

    # Import environment locally — no HTTP dependency
    try:
        from openenv.environment import OpenEnvSRE
        env = OpenEnvSRE()
    except Exception as e:
        logger.error("Failed to create OpenEnvSRE: %s\n%s", e, traceback.format_exc())
        print("[START]", flush=True)
        print("[END] " + json.dumps({"final_score": 0.0, "error": str(e)}), flush=True)
        sys.exit(1)

    task_weights = {1: 0.20, 2: 0.35, 3: 0.45}
    total_weighted_score = 0.0
    start_time = time.time()

    for task_id in [1, 2, 3]:
        logger.info("Starting Task %d (seed=%d)", task_id, RANDOM_SEED)
        try:
            task_score = run_episode(env, task_id, RANDOM_SEED)
        except Exception as e:
            logger.error("Task %d crashed: %s\n%s", task_id, e, traceback.format_exc())
            task_score = 0.0
        weighted = task_score * task_weights[task_id]
        total_weighted_score += weighted
        logger.info("Task %d score: %.4f (weighted: %.4f)", task_id, task_score, weighted)

    elapsed = time.time() - start_time
    final = round(total_weighted_score, 4)
    print(
        f"[END] {json.dumps({'final_score': final, 'elapsed_seconds': round(elapsed, 1), 'seed': RANDOM_SEED})}",
        flush=True,
    )
    logger.info("All tasks complete. Final score: %.4f in %.1fs", final, elapsed)


if __name__ == "__main__":
    main()
