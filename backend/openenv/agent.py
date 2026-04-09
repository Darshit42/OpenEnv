"""
Agent logic for OpenEnv SRE. Provides a generic Hybrid agent with LLM and Heuristic capabilities.
"""
from typing import Dict, Any, Optional
import os
import json
import logging

logger = logging.getLogger("openenv.agent")

class HybridSREAgent:
    def __init__(self, use_llm: bool = True, model_name: str = "gpt-4o"):
        self.use_llm = use_llm
        self.model_name = model_name
        self._llm_client = self._init_llm()

    def _init_llm(self):
        if not self.use_llm:
            return None
        try:
            from openai import OpenAI
            base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
            api_key = os.environ.get("API_KEY", "na")
            # Special case for local deepseek/llama: we just pass a fake api key
            return OpenAI(base_url=base_url, api_key=api_key)
        except Exception as e:
            logger.warning("Agent LLM initialization failed: %s", e)
            return None

    def decide_action(self, obs: Dict[str, Any], step: int, cf_called: bool) -> Dict[str, Any]:
        """Entry point. First attempts LLM, falls back to heuristic."""
        if self._llm_client:
            action = self._llm_policy(obs)
            if action:
                return action
        
        # Fallback
        return self._heuristic_policy(obs, step, cf_called)

    def _llm_policy(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        try:
            resp = self._llm_client.chat.completions.create(
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
                action_dict["reasoning"] = "LLM decided based on causal relations."
            return action_dict
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None

    def _heuristic_policy(self, obs: Dict[str, Any], step: int, cf_called: bool) -> Dict[str, Any]:
        services = obs.get("services", [])
        alerts = obs.get("alerts", [])
        anomaly_scores = obs.get("anomaly_scores", {})
        causal_dag = obs.get("causal_dag", {})
        cf_result = obs.get("counterfactual_result")

        if not hasattr(self, "dead_ends"):
            self.dead_ends = set()

        # 1. Silence red herring alerts
        for alert in alerts:
            if alert.get("is_red_herring") and not alert.get("silenced"):
                return {
                    "action_type": "silence_alert",
                    "service_id": None,
                    "parameters": {"alert_id": alert["alert_id"]},
                    "reasoning": f"Alert {alert['alert_id']} is a known red-herring, silencing."
                }

        # If counterfactual result from previous step showed harm, add to dead ends!
        if cf_result and cf_result.get("harm_flag"):
            self.dead_ends.add(cf_result.get("service_id"))

        # Sort anomalous nodes by score descending
        anomalous = sorted(
            [s for s in services if anomaly_scores.get(s, 0) > 0.3], 
            key=lambda s: anomaly_scores.get(s, 0), 
            reverse=True
        )
        
        # Determine the target, avoiding dead ends
        candidates = [s for s in anomalous if s not in self.dead_ends]
        
        target = candidates[0] if candidates else (max(services, key=lambda s: anomaly_scores.get(s, 0)) if services else None)

        # 2. Counterfactual
        if not cf_called and target and target not in self.dead_ends:
            return {
                "action_type": "query_counterfactual",
                "service_id": target,
                "parameters": None,
                "reasoning": f"Querying counterfactual on {target} (identified root) before committing."
            }
        
        # 4. Resolve or restart
        if step >= 20: # Fast resolution logic for later steps if nothing explicit
            return {
                "action_type": "declare_resolution",
                "service_id": None,
                "parameters": None,
                "reasoning": "Sufficient diagnostic steps taken, declaring resolution."
            }
            
        if target:
            return {
                "action_type": "restart_service",
                "service_id": target,
                "parameters": None,
                "reasoning": f"Restarting {target} to clear anomalous state."
            }

        return {
            "action_type": "declare_resolution", 
            "service_id": None, 
            "parameters": None,
            "reasoning": "Defaulting to resolution."
        }
