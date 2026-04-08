"""
Deterministic graders for all three OpenEnv SRE tasks.

Graders evaluate the complete action trajectory at episode end.
Identical action sequences always produce identical scores (seed-independent).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionRecord:
    step: int
    action_type: str
    service_id: Optional[str]
    parameters: Optional[Dict[str, Any]]
    reward_at_step: float = 0.0


@dataclass
class GradingResult:
    score: float                          # Normalised [0.0, 1.0]
    breakdown: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Base Grader
# ─────────────────────────────────────────────────────────────────────────────

class BaseGrader:
    task_id: int = 0
    max_steps: int = 30
    score_weight: float = 1.0

    def grade(
        self,
        action_trajectory: List[ActionRecord],
        scenario_truth: str,
        scenario_service: str,
        total_steps: int,
        silenced_alerts: List[str],
        counterfactual_called: bool,
        lethal_actions_taken: List[str],
    ) -> GradingResult:
        raise NotImplementedError

    @staticmethod
    def _speed_bonus(step: int, optimal_step: int, max_step: int) -> float:
        """Linear decay from 1.0 at optimal_step to 0.0 at max_step."""
        if step <= optimal_step:
            return 1.0
        if step >= max_step:
            return 0.0
        return 1.0 - (step - optimal_step) / (max_step - optimal_step)

    @staticmethod
    def _find_action(trajectory: List[ActionRecord], action_type: str) -> Optional[ActionRecord]:
        for r in trajectory:
            if r.action_type == action_type:
                return r
        return None

    @staticmethod
    def _find_action_on_service(
        trajectory: List[ActionRecord], action_type: str, service_id: str
    ) -> Optional[ActionRecord]:
        for r in trajectory:
            if r.action_type == action_type and r.service_id == service_id:
                return r
        return None

    @staticmethod
    def _resolution_step(trajectory: List[ActionRecord]) -> Optional[int]:
        for r in trajectory:
            if r.action_type == "declare_resolution":
                return r.step
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: Memory Leak Grader
# ─────────────────────────────────────────────────────────────────────────────

class EasyGrader(BaseGrader):
    """
    Grading criteria (Task 1 — Easy):
    +0.40  Root cause identified as memory_leak within first 10 steps
           (proxy: agent restarted or ran diagnostic on api-server before step 10)
    +0.30  Correct remediation: restart_service on api-server
    +0.20  Speed bonus: resolution declared within 15 steps (linear decay to step 30)
    +0.10  No harmful actions taken (no database restart)
    +0.05  query_counterfactual called before restart (bonus)
    -0.15  Incorrect service restarted (per occurrence, capped at -0.30)
    -0.40  Database restarted
    Max: 1.05 (capped to 1.0)
    """

    task_id = 1
    max_steps = 30
    score_weight = 0.20

    def grade(
        self,
        action_trajectory: List[ActionRecord],
        scenario_truth: str,
        scenario_service: str,
        total_steps: int,
        silenced_alerts: List[str],
        counterfactual_called: bool,
        lethal_actions_taken: List[str],
    ) -> GradingResult:
        score = 0.0
        breakdown: Dict[str, float] = {}
        notes: List[str] = []

        correct_service = "api-server"
        lethal_service = "database"

        # ── 1. Root cause identification (+0.40) ─────────────────────────────
        # Proxy: agent acted on api-server (restart or diagnostic) within step 10
        early_action = self._find_action_on_service(action_trajectory[:10], "restart_service", correct_service)
        early_diag = self._find_action_on_service(action_trajectory[:10], "run_diagnostic", correct_service)
        identified_early = early_action is not None or early_diag is not None
        rc_score = 0.40 if identified_early else 0.20  # partial if identified later
        # Check if identified at all (even after step 10)
        any_action = self._find_action_on_service(action_trajectory, "restart_service", correct_service)
        if not identified_early and any_action:
            rc_score = 0.20
        elif not any_action:
            rc_score = 0.0
        score += rc_score
        breakdown["root_cause_identification"] = rc_score
        notes.append(f"Root cause ID: {rc_score:.2f}")

        # ── 2. Correct remediation (+0.30) ────────────────────────────────────
        correct_restart = self._find_action_on_service(action_trajectory, "restart_service", correct_service)
        remediation_score = 0.30 if correct_restart else 0.0
        score += remediation_score
        breakdown["correct_remediation"] = remediation_score
        notes.append(f"Remediation: {remediation_score:.2f}")

        # ── 3. Speed bonus (+0.20) ────────────────────────────────────────────
        res_step = self._resolution_step(action_trajectory)
        if res_step is not None:
            speed = self._speed_bonus(res_step, optimal_step=15, max_step=30)
            speed_score = 0.20 * speed
        else:
            speed_score = 0.0
        score += speed_score
        breakdown["speed_bonus"] = speed_score
        notes.append(f"Speed bonus: {speed_score:.2f} (resolution at step {res_step})")

        # ── 4. No harmful actions (+0.10) ─────────────────────────────────────
        db_restart = self._find_action_on_service(action_trajectory, "restart_service", lethal_service)
        if db_restart is None:
            harmless_score = 0.10
        else:
            harmless_score = 0.0
        score += harmless_score
        breakdown["no_harmful_actions"] = harmless_score
        notes.append(f"No-harm: {harmless_score:.2f}")

        # ── 5. Counterfactual bonus (+0.05) ──────────────────────────────────
        cf_bonus = 0.05 if counterfactual_called else 0.0
        score += cf_bonus
        breakdown["counterfactual_bonus"] = cf_bonus
        notes.append(f"Counterfactual bonus: {cf_bonus:.2f}")

        # ── Penalties ────────────────────────────────────────────────────────
        wrong_restarts = [
            r for r in action_trajectory
            if r.action_type == "restart_service" and r.service_id not in (correct_service, None)
            and r.service_id != lethal_service
        ]
        wrong_penalty = min(0.30, len(wrong_restarts) * 0.15)
        score -= wrong_penalty
        breakdown["wrong_service_penalty"] = -wrong_penalty

        db_penalty = 0.40 if db_restart else 0.0
        score -= db_penalty
        breakdown["lethal_action_penalty"] = -db_penalty
        if db_penalty:
            notes.append("PENALTY: Database restarted (-0.40)")

        final_score = round(max(0.0, min(1.0, score)), 4)
        return GradingResult(score=final_score, breakdown=breakdown, notes=notes)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Medium: Dependency Failure Grader
# ─────────────────────────────────────────────────────────────────────────────

class MediumGrader(BaseGrader):
    """
    Grading criteria (Task 2 — Medium):
    +0.35  Correct root cause: dependency_timeout on data-layer
    +0.25  Correct remediation on data-layer (restart or scale)
    +0.10  Red-herring alert (background-worker) silenced without action
    +0.10  query_counterfactual used before acting on data-layer
    +0.10  Speed bonus: resolution within 25 steps (linear decay to 45)
    +0.15  Partial credit: correct service identified, suboptimal action order
    -0.20  api-gateway restart (per occurrence)
    Max: 1.05 (capped to 1.0)
    """

    task_id = 2
    max_steps = 45
    score_weight = 0.35

    def grade(
        self,
        action_trajectory: List[ActionRecord],
        scenario_truth: str,
        scenario_service: str,
        total_steps: int,
        silenced_alerts: List[str],
        counterfactual_called: bool,
        lethal_actions_taken: List[str],
    ) -> GradingResult:
        score = 0.0
        breakdown: Dict[str, float] = {}
        notes: List[str] = []

        correct_service = "data-layer"
        trap_service = "api-gateway"
        red_herring_alert_svc = "background-worker"

        # Check if any action was taken on correct service
        correct_restart = self._find_action_on_service(action_trajectory, "restart_service", correct_service)
        correct_scale = self._find_action_on_service(action_trajectory, "scale_service", correct_service)
        correct_action = correct_restart or correct_scale
        trap_restart = self._find_action_on_service(action_trajectory, "restart_service", trap_service)

        # ── 1. Root cause identification (+0.35) ─────────────────────────────
        if correct_action:
            rc_score = 0.35
        else:
            rc_score = 0.0
        score += rc_score
        breakdown["root_cause_identification"] = rc_score
        notes.append(f"Root cause (data-layer): {rc_score:.2f}")

        # ── 2. Correct remediation (+0.25) ────────────────────────────────────
        remediation_score = 0.25 if correct_action else 0.0
        score += remediation_score
        breakdown["correct_remediation"] = remediation_score
        notes.append(f"Remediation: {remediation_score:.2f}")

        # ── 3. Red-herring silenced without acting (+0.10) ───────────────────
        rh_silenced = any("background-worker" in s for s in silenced_alerts)
        rh_not_acted = self._find_action_on_service(action_trajectory, "restart_service", red_herring_alert_svc) is None
        rh_score = 0.10 if (rh_silenced and rh_not_acted) else 0.0
        score += rh_score
        breakdown["red_herring_handled"] = rh_score
        notes.append(f"Red-herring handling: {rh_score:.2f}")

        # ── 4. Counterfactual used before data-layer action (+0.10) ──────────
        cf_before_action = False
        if correct_action and counterfactual_called:
            cf_step = next((r.step for r in action_trajectory if r.action_type == "query_counterfactual"), 999)
            action_step = correct_action.step
            cf_before_action = cf_step < action_step
        cf_score = 0.10 if cf_before_action else 0.0
        score += cf_score
        breakdown["counterfactual_bonus"] = cf_score
        notes.append(f"Counterfactual bonus: {cf_score:.2f}")

        # ── 5. Speed bonus (+0.10) ────────────────────────────────────────────
        res_step = self._resolution_step(action_trajectory)
        if res_step is not None:
            speed = self._speed_bonus(res_step, optimal_step=25, max_step=45)
            speed_score = 0.10 * speed
        else:
            speed_score = 0.0
        score += speed_score
        breakdown["speed_bonus"] = speed_score
        notes.append(f"Speed bonus: {speed_score:.2f}")

        # ── 6. Partial credit (+0.15) ─────────────────────────────────────────
        # Correct service identified but suboptimal order (e.g., acted on GW first)
        if correct_action and trap_restart and correct_action.step > trap_restart.step:
            partial = 0.15
        else:
            partial = 0.0
        # Ensure we don't double-count with remediation
        if remediation_score == 0 and correct_action:
            partial = 0.15  # Identified but slightly wrong sequence
        score += partial
        breakdown["partial_credit"] = partial
        notes.append(f"Partial credit: {partial:.2f}")

        # ── Penalties ────────────────────────────────────────────────────────
        gw_restarts = [
            r for r in action_trajectory
            if r.action_type == "restart_service" and r.service_id == trap_service
        ]
        gw_penalty = min(0.40, len(gw_restarts) * 0.20)
        score -= gw_penalty
        breakdown["wrong_service_penalty"] = -gw_penalty
        if gw_penalty:
            notes.append(f"PENALTY: api-gateway restart(s) (-{gw_penalty:.2f})")

        final_score = round(max(0.0, min(1.0, score)), 4)
        return GradingResult(score=final_score, breakdown=breakdown, notes=notes)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard: Config Drift Grader
# ─────────────────────────────────────────────────────────────────────────────

class HardGrader(BaseGrader):
    """
    Grading criteria (Task 3 — Hard):
    +0.15  Drift detected via drift_alert before threshold breach (step < 19)
    +0.10  Misleading alerts silenced without acting on those services
    +0.10  query_counterfactual called before any restart action
    +0.10  Cache restart correctly avoided
    +0.20  Correct root cause (config_drift on origin-service) identified
    +0.20  Correct and complete action sequence executed
    +0.15  Speed bonus: resolution within 40 steps (linear decay to 60)
    -0.40  Cache restarted (harm — applied even if later corrected)
    +0.10  Partial credit: correct root cause, non-harmful but suboptimal sequence
    Max: 1.10 (capped to 1.0)
    """

    task_id = 3
    max_steps = 60
    score_weight = 0.45

    RED_HERRING_SERVICES = {"background-monitor", "database"}

    def grade(
        self,
        action_trajectory: List[ActionRecord],
        scenario_truth: str,
        scenario_service: str,
        total_steps: int,
        silenced_alerts: List[str],
        counterfactual_called: bool,
        lethal_actions_taken: List[str],
    ) -> GradingResult:
        score = 0.0
        breakdown: Dict[str, float] = {}
        notes: List[str] = []

        correct_service = "origin-service"
        lethal_service = "cache"

        correct_diag = self._find_action_on_service(action_trajectory, "run_diagnostic", correct_service)
        correct_restart = self._find_action_on_service(action_trajectory, "restart_service", correct_service)
        # "patch config" represented as run_diagnostic with parameters
        patch_action = next(
            (r for r in action_trajectory
             if r.action_type == "run_diagnostic"
             and r.service_id == correct_service
             and r.parameters and "patch" in str(r.parameters).lower()),
            None,
        )
        cache_restart = self._find_action_on_service(action_trajectory, "restart_service", lethal_service)

        # ── 1. Drift detected early (+0.15) ──────────────────────────────────
        # Proxy: agent took any action before step 19 (threshold breach step)
        early_actions = [r for r in action_trajectory if r.step < 19 and r.action_type not in ("query_counterfactual", "silence_alert")]
        drift_score = 0.15 if early_actions else 0.0
        score += drift_score
        breakdown["early_drift_detection"] = drift_score
        notes.append(f"Early drift detection: {drift_score:.2f}")

        # ── 2. Misleading alerts silenced without acting (+0.10) ─────────────
        rh_silenced = any(
            any(rh_svc in s for rh_svc in self.RED_HERRING_SERVICES)
            for s in silenced_alerts
        )
        rh_not_acted = all(
            self._find_action_on_service(action_trajectory, "restart_service", svc) is None
            for svc in self.RED_HERRING_SERVICES
        )
        rh_score = 0.10 if (rh_silenced and rh_not_acted) else 0.0
        score += rh_score
        breakdown["red_herring_handled"] = rh_score
        notes.append(f"Red-herring handling: {rh_score:.2f}")

        # ── 3. Counterfactual called before any restart (+0.10) ───────────────
        first_restart_step = next(
            (r.step for r in action_trajectory if r.action_type == "restart_service"), 999
        )
        first_cf_step = next(
            (r.step for r in action_trajectory if r.action_type == "query_counterfactual"), 999
        )
        cf_score = 0.10 if (counterfactual_called and first_cf_step < first_restart_step) else 0.0
        score += cf_score
        breakdown["counterfactual_bonus"] = cf_score
        notes.append(f"Counterfactual pre-screen: {cf_score:.2f}")

        # ── 4. Cache restart avoided (+0.10) ─────────────────────────────────
        avoided_cache = 0.10 if cache_restart is None else 0.0
        score += avoided_cache
        breakdown["lethal_avoided"] = avoided_cache
        notes.append(f"Cache restart avoided: {avoided_cache:.2f}")

        # ── 5. Correct root cause (+0.20) ─────────────────────────────────────
        acted_on_origin = correct_diag or correct_restart or patch_action
        rc_score = 0.20 if acted_on_origin else 0.10 if draft_near_correct(action_trajectory) else 0.0
        score += rc_score
        breakdown["root_cause_identification"] = rc_score
        notes.append(f"Root cause (origin-service): {rc_score:.2f}")

        # ── 6. Complete correct sequence (+0.20) ─────────────────────────────
        # Ideal: silence misleading → cf query → act on origin → declare_resolution
        sequence_complete = (
            rh_silenced
            and counterfactual_called
            and acted_on_origin
            and cache_restart is None
            and self._resolution_step(action_trajectory) is not None
        )
        seq_score = 0.20 if sequence_complete else 0.0
        score += seq_score
        breakdown["complete_sequence"] = seq_score
        notes.append(f"Complete sequence: {seq_score:.2f}")

        # ── 7. Speed bonus (+0.15) ─────────────────────────────────────────────
        res_step = self._resolution_step(action_trajectory)
        if res_step is not None:
            speed = self._speed_bonus(res_step, optimal_step=40, max_step=60)
            speed_score = 0.15 * speed
        else:
            speed_score = 0.0
        score += speed_score
        breakdown["speed_bonus"] = speed_score
        notes.append(f"Speed bonus: {speed_score:.2f}")

        # ── 8. Partial credit (+0.10) ─────────────────────────────────────────
        if acted_on_origin and cache_restart is None and not sequence_complete:
            partial = 0.10
            score += partial
            breakdown["partial_credit"] = partial
            notes.append(f"Partial credit: {partial:.2f}")

        # ── Cache penalty (-0.40) ─────────────────────────────────────────────
        if cache_restart is not None:
            score -= 0.40
            breakdown["lethal_action_penalty"] = -0.40
            notes.append("SEVERE PENALTY: Cache restarted — write-loss cascade (-0.40)")

        final_score = round(max(0.0, min(1.0, score)), 4)
        return GradingResult(score=final_score, breakdown=breakdown, notes=notes)


def draft_near_correct(trajectory: List[ActionRecord]) -> bool:
    """Heuristic: agent took some action on a non-lethal service."""
    return any(
        r.action_type in ("run_diagnostic", "restart_service")
        and r.service_id not in ("cache", "database", "background-monitor", None)
        for r in trajectory
    )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

GRADER_REGISTRY: Dict[int, type] = {
    1: EasyGrader,
    2: MediumGrader,
    3: HardGrader,
}


def create_grader(task_id: int) -> BaseGrader:
    if task_id not in GRADER_REGISTRY:
        raise ValueError(f"Unknown task_id={task_id}. Valid: {list(GRADER_REGISTRY.keys())}")
    return GRADER_REGISTRY[task_id]()
