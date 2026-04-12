"""
OpenEnvSRE — Main RL environment class.

Implements the OpenEnv step / reset / state API with full typed Pydantic models.
Runs the five-stage ML pipeline on every step() call.

Reward function:
  R(t) = α·correctness + β·speed_bonus + γ·partial_credit + δ·counterfactual_bonus − ε·harm_penalty
  α=0.50, β=0.20, γ=0.15, δ=0.10, ε∈[0.05, 0.40]
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from openenv.graders import ActionRecord, GradingResult, create_grader
from openenv.models import (
    Action,
    ActionType,
    Alert,
    Observation,
    Reward,
    RewardBreakdown,
)
from openenv.pipeline import SREPipeline
from openenv.scenarios import BaseScenario, create_scenario

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Reward coefficients (PRD §4.5)
# ─────────────────────────────────────────────────────────────────────────────

ALPHA = 0.50   # correctness
BETA = 0.20    # speed bonus
GAMMA = 0.15   # partial credit
DELTA = 0.10   # counterfactual bonus
EPSILON_MIN = 0.05   # harm penalty — wrong service
EPSILON_MAX = 0.40   # harm penalty — lethal service (cache/database)

TASK_MAX_STEPS = {1: 30, 2: 45, 3: 60}
TASK_OPTIMAL_STEPS = {1: 15, 2: 25, 3: 40}


class OpenEnvSRE:
    """
    Production-grade RL environment for SRE incident diagnosis.

    Usage:
        env = OpenEnvSRE()
        obs = env.reset(task_id=1, seed=42)
        action = Action(action_type=ActionType.query_counterfactual, service_id="api-server")
        obs, reward, done, info = env.step(action)
    """

    def __init__(self) -> None:
        self._pipeline = SREPipeline()
        self._scenario: Optional[BaseScenario] = None
        self._task_id: int = 1
        self._seed: int = 42
        self._step_count: int = 0
        self._done: bool = False

        # Episode state
        self._metric_history: Dict[str, List[Dict[str, float]]] = {}
        self._log_buffer: List[str] = []
        self._action_trajectory: List[ActionRecord] = []
        self._silenced_alerts: List[str] = []
        self._causal_dag: Dict[str, List[str]] = {}
        self._counterfactual_called: bool = False
        self._lethal_actions: List[str] = []

        # Cumulative reward tracking
        self._cumulative_reward: float = 0.0
        self._step_rewards: List[float] = []

        # Last observation (for state())
        self._last_obs: Optional[Observation] = None

    # ─────────────────────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, task_id: int = 1, seed: Optional[int] = None) -> Observation:
        """
        Start a new episode. Generates a new scenario and runs the PC algorithm.
        Returns the initial Observation.
        """
        if task_id not in TASK_MAX_STEPS:
            raise ValueError(f"Invalid task_id={task_id}. Valid: {list(TASK_MAX_STEPS.keys())}")

        self._task_id = task_id
        self._seed = seed if seed is not None else 42
        np.random.seed(self._seed)

        # Instantiate the scenario
        self._scenario = create_scenario(task_id, seed=self._seed)
        self._step_count = 0
        self._done = False
        self._action_trajectory = []
        self._silenced_alerts = []
        self._counterfactual_called = False
        self._lethal_actions = []
        self._cumulative_reward = 0.0
        self._step_rewards = []
        self._log_buffer = []

        services = self._scenario.all_services

        # Pre-warm metric history with a synthetic "pre-episode" buffer for the PC algorithm
        pre_steps = 20
        self._metric_history = {svc: [] for svc in services}
        for pre_step in range(pre_steps):
            metrics_at_step = self._scenario.get_metrics(pre_step)
            for svc in services:
                snap = metrics_at_step.get(svc)
                if snap:
                    self._metric_history[svc].append(snap.model_dump())

        # Stage 3: Run PC algorithm on pre-episode buffer
        self._causal_dag = self._pipeline.reset_causal(self._metric_history, services)
        logger.info("Episode reset (task=%d, seed=%d). PC DAG: %s", task_id, self._seed, self._causal_dag)

        # Get initial observation at step 0
        obs = self._build_observation(step=0)
        self._last_obs = obs
        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Returns:
            obs       — typed Observation (all 5 pipeline stage outputs)
            reward    — shaped Reward with per-term breakdown
            done      — True if episode terminated
            info      — metadata dict (score, scenario info, grader notes)
        """
        if self._scenario is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        services = self._scenario.all_services

        # ── Record action ─────────────────────────────────────────────────────
        record = ActionRecord(
            step=self._step_count,
            action_type=action.action_type.value,
            service_id=action.service_id,
            parameters=action.parameters,
        )
        self._action_trajectory.append(record)

        # ── Handle silence_alert ──────────────────────────────────────────────
        if action.action_type == ActionType.silence_alert:
            alert_id = (action.parameters or {}).get("alert_id", action.service_id or "")
            if alert_id:
                self._silenced_alerts.append(alert_id)

        # ── Track counterfactual use ──────────────────────────────────────────
        if action.action_type == ActionType.query_counterfactual:
            self._counterfactual_called = True

        # ── Track lethal actions ──────────────────────────────────────────────
        if (
            action.action_type in (ActionType.restart_service, ActionType.scale_service)
            and action.service_id in self._scenario.lethal_services
        ):
            self._lethal_actions.append(action.service_id or "")

        # ── Advance scenario: get new metrics/logs/alerts ─────────────────────
        metrics = self._scenario.get_metrics(self._step_count)
        logs = self._scenario.get_logs(self._step_count)
        alerts = self._scenario.get_alerts(self._step_count)

        # Append to metric history
        for svc in services:
            snap = metrics.get(svc)
            if snap:
                self._metric_history[svc].append(snap.model_dump())

        # Append log messages to buffer
        for log_entry in logs:
            self._log_buffer.append(log_entry.message)

        # ── Run pipeline ──────────────────────────────────────────────────
        # Extract simulated_action for counterfactual queries
        simulated_action = None
        if action.action_type == ActionType.query_counterfactual and action.parameters:
            simulated_action = action.parameters.get("simulated_action", "restart_service")

        pipeline_out = self._pipeline.run(
            metrics=metrics,
            metric_history=self._metric_history,
            log_messages=[le.message for le in logs],
            services=services,
            action_type=action.action_type.value,
            action_service=action.service_id,
            scenario_truth=self._scenario.ground_truth_root_cause,
            scenario_service=self._scenario.ground_truth_service,
            lethal_services=self._scenario.lethal_services,
            simulated_action=simulated_action,
        )

        # ── Build observation ─────────────────────────────────────────────────
        obs = self._build_observation(
            step=self._step_count,
            metrics=metrics,
            logs=logs,
            alerts=alerts,
            pipeline_out=pipeline_out,
        )
        self._last_obs = obs

        # ── Termination check ─────────────────────────────────────────────────
        max_steps = TASK_MAX_STEPS[self._task_id]
        done_by_action = action.action_type in (
            ActionType.declare_resolution, ActionType.escalate_incident
        )
        done_by_steps = self._step_count >= max_steps
        self._done = done_by_action or done_by_steps

        # ── Compute per-step shaped reward ───────────────────────────────────
        reward = self._compute_step_reward(action, pipeline_out, self._done)
        record.reward_at_step = reward.total
        self._cumulative_reward += reward.total
        self._step_rewards.append(reward.total)

        # ── Final grading if episode done ─────────────────────────────────────
        info: Dict[str, Any] = {
            "step": self._step_count,
            "task_id": self._task_id,
            "scenario_type": self._scenario.ground_truth_root_cause,
            "ground_truth_service": self._scenario.ground_truth_service,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "causal_dag": self._causal_dag,
        }

        if self._done:
            grading = self._grade_episode()
            info["episode_score"] = grading.score
            info["grader_breakdown"] = grading.breakdown
            info["grader_notes"] = grading.notes
            info["action_count"] = len(self._action_trajectory)
            logger.info(
                "Episode complete (task=%d, seed=%d, steps=%d, score=%.3f)",
                self._task_id, self._seed, self._step_count, grading.score,
            )

        return obs, reward, self._done, info

    # ─────────────────────────────────────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        """
        Return the full serialisable internal state for monitoring/debugging.
        Exposed via GET /state.
        """
        if self._scenario is None:
            return {"status": "not_started"}

        return {
            "task_id": self._task_id,
            "seed": self._seed,
            "step": self._step_count,
            "done": self._done,
            "scenario_type": self._scenario.ground_truth_root_cause,
            "ground_truth_service": self._scenario.ground_truth_service,
            "services": self._scenario.all_services,
            "causal_dag": self._causal_dag,
            "metric_history": {
                svc: hist[-10:]  # Return last 10 snapshots per service
                for svc, hist in self._metric_history.items()
            },
            "action_trajectory": [
                {
                    "step": r.step,
                    "action_type": r.action_type,
                    "service_id": r.service_id,
                    "parameters": r.parameters,
                    "reward": r.reward_at_step,
                }
                for r in self._action_trajectory
            ],
            "silenced_alerts": self._silenced_alerts,
            "counterfactual_called": self._counterfactual_called,
            "lethal_actions": self._lethal_actions,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "last_observation": self._last_obs.model_dump() if self._last_obs else None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        step: int,
        metrics: Optional[Dict] = None,
        logs: Optional[List] = None,
        alerts: Optional[List] = None,
        pipeline_out: Optional[Dict] = None,
    ) -> Observation:
        """Assemble typed Observation from raw scenario outputs and pipeline results."""
        if self._scenario is None:
            return Observation(step_number=step)

        services = self._scenario.all_services

        if metrics is None:
            metrics = self._scenario.get_metrics(step)
        if logs is None:
            logs = self._scenario.get_logs(step)
        if alerts is None:
            alerts = self._scenario.get_alerts(step)

        # Apply silenced flags to alerts
        for alert in alerts:
            if alert.alert_id in self._silenced_alerts:
                alert.silenced = True

        pipeline_out = pipeline_out or {}

        return Observation(
            raw_logs=logs,
            metrics=metrics,
            alerts=alerts,
            anomaly_scores=pipeline_out.get("anomaly_scores", {svc: 0.0 for svc in services}),
            anomaly_flags=pipeline_out.get("anomaly_flags", {svc: False for svc in services}),
            causal_dag=self._causal_dag,
            causal_effects=pipeline_out.get("causal_effects", {}),
            counterfactual_result=pipeline_out.get("counterfactual_result"),
            root_cause_prediction=pipeline_out.get("root_cause_prediction", ""),
            root_cause_probabilities=pipeline_out.get("root_cause_probabilities", {}),
            shap_top5=pipeline_out.get("shap_top5", []),
            forecast=pipeline_out.get("forecast"),
            drift_alert=pipeline_out.get("drift_alert", False),
            step_number=step,
            task_id=self._task_id,
            services=services,
            silenced_alerts=self._silenced_alerts,
        )

    def _compute_step_reward(
        self,
        action: Action,
        pipeline_out: Dict,
        done: bool,
    ) -> Reward:
        """
        Five-term shaped reward function (PRD §4.5).
        R(t) = α·correctness + β·speed_bonus + γ·partial_credit + δ·counterfactual_bonus − ε·harm_penalty
        """
        if self._scenario is None:
            return Reward(total=0.0, breakdown=RewardBreakdown())

        bd = RewardBreakdown()
        scenario = self._scenario

        # ── α: Correctness ─────────────────────────────────────────────────────
        predicted = pipeline_out.get("root_cause_prediction", "")
        correct_root_cause = predicted == scenario.ground_truth_root_cause
        correct_service = action.service_id == scenario.ground_truth_service
        correct_action_type = action.action_type in (
            ActionType.restart_service, ActionType.scale_service, ActionType.run_diagnostic
        )
        if correct_root_cause and correct_service and correct_action_type:
            bd.correctness = ALPHA * 1.0
        elif correct_root_cause and not correct_service:
            bd.correctness = ALPHA * 0.5
        elif correct_service and correct_action_type:
            bd.correctness = ALPHA * 0.6
        elif done and action.action_type == ActionType.declare_resolution:
            # ANTI-EXPLOIT: Only award correctness if agent actually fixed the issue.
            # Requires a prior restart/scale/diagnostic on the ground-truth service.
            prior_fix = any(
                r.action_type in ("restart_service", "scale_service", "run_diagnostic")
                and r.service_id == scenario.ground_truth_service
                for r in self._action_trajectory
            )
            if prior_fix and correct_root_cause:
                bd.correctness = ALPHA * 0.3
            else:
                bd.correctness = 0.0
        else:
            bd.correctness = 0.0

        # ── β: Speed bonus (only on terminal step) ────────────────────────────
        if done:
            max_s = TASK_MAX_STEPS[self._task_id]
            opt_s = TASK_OPTIMAL_STEPS[self._task_id]
            _EPS = 1e-4
            if self._step_count <= opt_s:
                speed = 1.0 - _EPS
            elif self._step_count >= max_s:
                speed = _EPS
            else:
                speed = 1.0 - (self._step_count - opt_s) / (max_s - opt_s)
            bd.speed_bonus = BETA * speed
        else:
            bd.speed_bonus = 0.0

        # ── γ: Partial credit ─────────────────────────────────────────────────
        if correct_service and not correct_action_type:
            bd.partial_credit = GAMMA * 0.5
        elif correct_root_cause and not correct_service:
            bd.partial_credit = GAMMA * 0.4
        else:
            bd.partial_credit = 0.0

        # ── δ: Counterfactual bonus ───────────────────────────────────────────
        if action.action_type == ActionType.query_counterfactual:
            # Check if next action would be optimal (graded retrospectively at episode end)
            bd.counterfactual_bonus = DELTA * 0.5  # Partial now; full bonus in grader
        elif self._counterfactual_called and correct_service and correct_action_type:
            # Agent used CF and then acted correctly
            bd.counterfactual_bonus = DELTA * 1.0
        else:
            bd.counterfactual_bonus = 0.0

        # ── ε: Harm penalty ───────────────────────────────────────────────────
        if action.action_type in (ActionType.restart_service, ActionType.scale_service):
            if action.service_id in scenario.lethal_services:
                bd.harm_penalty = EPSILON_MAX  # lethal service
            elif action.service_id and action.service_id != scenario.ground_truth_service:
                bd.harm_penalty = EPSILON_MIN  # wrong service
            else:
                bd.harm_penalty = 0.0
        else:
            bd.harm_penalty = 0.0

        total = (
            bd.correctness
            + bd.speed_bonus
            + bd.partial_credit
            + bd.counterfactual_bonus
            - bd.harm_penalty
        )
        # Clip to reasonable range (harm can drive negative)
        total = max(-0.99, min(0.99, total))

        return Reward(total=round(total, 4), breakdown=bd)

    def _grade_episode(self) -> GradingResult:
        """Run the deterministic task grader on the completed action trajectory."""
        grader = create_grader(self._task_id)
        result = grader.grade(
            action_trajectory=self._action_trajectory,
            scenario_truth=self._scenario.ground_truth_root_cause,
            scenario_service=self._scenario.ground_truth_service,
            total_steps=self._step_count,
            silenced_alerts=self._silenced_alerts,
            counterfactual_called=self._counterfactual_called,
            lethal_actions_taken=self._lethal_actions,
        )
        # Belt-and-suspenders: ensure score is strictly open even if grader is misconfigured
        result.score = max(0.02, min(0.98, result.score))
        return result
