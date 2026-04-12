"""
OpenEnv SRE — LLM-guided baseline inference script (ROOT).

Self-contained isolated script for evaluation environments.
No local imports. No external HTTP libraries (only urllib & openai).
"""

import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("openenv.inference")

ENV_API_URL = "http://127.0.0.1:7860"

def make_request(url: str, method: str = "GET", payload: dict = None) -> dict:
    headers = {"Content-Type": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        logger.error(f"HTTP request failed: {e}")
        raise


class IsolatedAgent:
    def __init__(self):
        try:
            from openai import OpenAI
            base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
            api_key = os.environ.get("API_KEY", "na")
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model_name = os.environ.get("MODEL_NAME", "gpt-4o")
            self.use_llm = True
        except ImportError:
            self.use_llm = False
            logger.warning("openai package not found, using pure heuristic fallback.")

    def decide_action(self, obs: dict, step: int, cf_called: bool) -> dict:
        if self.use_llm:
            try:
                obs_compact = {k: v for k, v in obs.items() if k != "raw_logs"}
                sys_prompt = (
                    "You are an SRE AI operating in OpenEnv. You must output JSON with 'action_type', "
                    "'service_id', 'parameters', and 'reasoning'.\n"
                    "Rules:\n"
                    "1. Trust the causal_dag: Services with NO parents (empty list) are root causes.\n"
                    "2. Ignore anomaly scores if a service is downstream of another degraded service.\n"
                    "3. query_counterfactual prior to restart_service.\n"
                    "4. If cf_result.harm_flag is true, do not restart it."
                )
                
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": json.dumps(obs_compact)}
                    ],
                    temperature=0.1
                )
                raw = resp.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                
                action_dict = json.loads(raw)
                if "reasoning" not in action_dict:
                    action_dict["reasoning"] = "LLM decided."
                return action_dict
            except Exception as e:
                logger.warning(f"LLM proxy call failed: {e}")

        # Heuristic fallback
        services = obs.get("services", [])
        anomaly_scores = obs.get("anomaly_scores", {})
        target = max(services, key=lambda s: anomaly_scores.get(s, 0)) if services else None
        
        if step >= 20:
            return {"action_type": "declare_resolution", "service_id": None, "parameters": None, "reasoning": "Timeout fallback"}
        if not cf_called and target:
            return {"action_type": "query_counterfactual", "service_id": target, "parameters": None, "reasoning": "Fallback cf"}
        if target:
            return {"action_type": "restart_service", "service_id": target, "parameters": None, "reasoning": "Fallback restart"}
            
        return {"action_type": "declare_resolution", "service_id": None, "parameters": None, "reasoning": "Default resolution"}


def run_episode(task_id: int, seed: int, agent: IsolatedAgent) -> float:
    cf_called = False

    logger.info(f"Connecting to environment at {ENV_API_URL}/reset (task={task_id})")
    try:
        result = make_request(f"{ENV_API_URL}/reset", method="POST", payload={"task_id": task_id, "seed": seed})
        obs = result.get("observation", {})
    except Exception as e:
        logger.error(f"Failed to reset environment: {e}")
        return 0.01

    print("[START]", flush=True)

    step = 0
    done = False
    episode_score = 0.01
    max_steps = {1: 30, 2: 45, 3: 60}.get(task_id, 30)

    while not done and step < max_steps:
        step += 1

        action_dict = agent.decide_action(obs, step, cf_called)

        if action_dict.get("action_type") == "query_counterfactual":
            cf_called = True

        step_payload = {
            "step": step,
            "root_cause_prediction": obs.get("root_cause_prediction", ""),
            "top_anomaly": max(obs.get("anomaly_scores", {}).items(), key=lambda x: x[1], default=("?", 0)),
            "action": action_dict,
        }
        print(f"[STEP] {json.dumps(step_payload)}", flush=True)

        req_action = {
            "action_type": action_dict.get("action_type", "declare_resolution"),
            "service_id": action_dict.get("service_id"),
            "parameters": action_dict.get("parameters")
        }

        try:
            result = make_request(f"{ENV_API_URL}/step", method="POST", payload=req_action)
            obs = result.get("observation", {})
            done = result.get("done", False)
            info = result.get("info", {})
            if done:
                raw_score = info.get("episode_score", 0.01)
                # Ensure strictly between 0 and 1 (exclusive)
                episode_score = float(max(0.01, min(0.99, raw_score)))
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            break

    # If episode ended without done signal, clamp fallback score
    episode_score = float(max(0.01, min(0.99, episode_score)))

    try:
        make_request(f"{ENV_API_URL}/state", method="GET")
    except Exception as e:
        logger.warning(f"Warning: /state check failed: {e}")

    return episode_score


def main() -> None:
    random_seed = int(os.environ.get("RANDOM_SEED", "42"))
    random.seed(random_seed)

    agent = IsolatedAgent()
    task_weights = {1: 0.20, 2: 0.35, 3: 0.45}
    total_weighted_score = 0.0
    task_scores: dict = {}
    start_time = time.time()

    for task_id in [1, 2, 3]:
        logger.info(f"Starting Task {task_id} (seed={random_seed})")
        task_score = run_episode(task_id, random_seed, agent)
        # Ensure task_score strictly complies with (0, 1) exclusive bounds
        task_score = round(float(max(0.01, min(0.99, task_score))), 4)
        task_scores[task_id] = task_score
        weighted = task_score * task_weights[task_id]
        total_weighted_score += weighted
        logger.info(f"Task {task_id} score: {task_score:.4f} (weighted: {weighted:.4f})")
        # Emit per-task score line so evaluator can parse individual task scores
        print(f"[TASK] {json.dumps({'task_id': task_id, 'score': task_score, 'weighted': round(weighted, 4)})}", flush=True)

    elapsed = time.time() - start_time
    final = round(float(max(0.01, min(0.99, total_weighted_score))), 4)
    print(
        f"[END] {json.dumps({'final_score': final, 'task_scores': task_scores, 'elapsed_seconds': round(elapsed, 1), 'seed': random_seed})}",
        flush=True,
    )
    logger.info(f"All tasks complete. Final score: {final:.4f} in {elapsed:.1f}s")


if __name__ == "__main__":
    time.sleep(2)
    main()
