"""
All Pydantic data models for the OpenEnv SRE environment.
These are the typed observation, action, and reward contracts.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Primitives
# ─────────────────────────────────────────────────────────────────────────────

class MetricSnapshot(BaseModel):
    """Point-in-time metric snapshot for a single service."""
    cpu_utilization: float = Field(..., ge=0.0, le=100.0, description="CPU utilization %")
    memory_rss: float = Field(..., ge=0.0, description="Memory RSS in MB")
    latency_p50: float = Field(..., ge=0.0, description="p50 request latency ms")
    latency_p95: float = Field(..., ge=0.0, description="p95 request latency ms")
    latency_p99: float = Field(..., ge=0.0, description="p99 request latency ms")
    error_rate: float = Field(..., ge=0.0, description="Errors per second")
    connection_pool_saturation: float = Field(..., ge=0.0, le=1.0, description="Pool saturation [0,1]")


class Alert(BaseModel):
    """Threshold-triggered alert event."""
    alert_id: str
    source_service: str
    alert_type: str
    current_value: float
    threshold: float
    severity: str  # "info" | "warning" | "critical"
    is_red_herring: bool = False
    silenced: bool = False


class LogEntry(BaseModel):
    """Structured JSON log record."""
    timestamp: str
    service_id: str
    severity: str
    message: str
    trace_id: str


# ─────────────────────────────────────────────────────────────────────────────
# ML Pipeline Outputs
# ─────────────────────────────────────────────────────────────────────────────

class CounterfactualResult(BaseModel):
    """
    Output of Stage 3 counterfactual simulator.
    Returned when agent calls query_counterfactual(action, service_id).
    """
    action_type: str
    service_id: str
    # Predicted metric trajectories for t+1..t+5 per service
    predicted_metrics: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="service_id -> [error_rate at t+1, t+2, t+3, t+4, t+5]",
    )
    predicted_resolution_probability: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    harm_flag: bool = False
    harm_description: Optional[str] = None


class ForecastResult(BaseModel):
    """
    Output of Stage 5 temporal forecasting (LSTM + linear trend).
    Provides forward-looking metric projections.
    """
    forecast_t5: Dict[str, float] = Field(
        default_factory=dict,
        description="service_id -> predicted error_rate at t+5",
    )
    forecast_t15: Dict[str, float] = Field(
        default_factory=dict,
        description="service_id -> predicted error_rate at t+15",
    )
    confidence_band_width: Dict[str, float] = Field(
        default_factory=dict,
        description="service_id -> width of prediction interval",
    )
    drift_alert: bool = False
    lstm_reconstruction_error: Dict[str, float] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Action Space
# ─────────────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    restart_service = "restart_service"
    scale_service = "scale_service"
    run_diagnostic = "run_diagnostic"
    silence_alert = "silence_alert"
    query_counterfactual = "query_counterfactual"
    escalate_incident = "escalate_incident"
    declare_resolution = "declare_resolution"


class Action(BaseModel):
    """Agent action submitted to step()."""
    action_type: ActionType
    service_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    correctness: float = 0.0         # alpha term
    speed_bonus: float = 0.0         # beta term
    partial_credit: float = 0.0      # gamma term
    counterfactual_bonus: float = 0.0  # delta term
    harm_penalty: float = 0.0        # epsilon term (stored as positive, subtracted)


class Reward(BaseModel):
    total: float
    breakdown: RewardBreakdown


# ─────────────────────────────────────────────────────────────────────────────
# Observation — the full typed state returned by step() / reset()
# ─────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    Complete typed observation returned to the agent at every step.
    All five pipeline stage outputs are included.
    """
    # ── Raw telemetry ──────────────────────────────────────────────────────
    raw_logs: List[LogEntry] = Field(default_factory=list)
    metrics: Dict[str, MetricSnapshot] = Field(default_factory=dict)
    alerts: List[Alert] = Field(default_factory=list)

    # ── Stage 2: Anomaly detection ────────────────────────────────────────
    anomaly_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="service_id -> anomaly score in [0, 1]",
    )
    anomaly_flags: Dict[str, bool] = Field(default_factory=dict)

    # ── Stage 3: Causal DAG + counterfactual ─────────────────────────────
    causal_dag: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Adjacency list {service: [causal_parent_services]}",
    )
    causal_effects: Dict[str, float] = Field(
        default_factory=dict,
        description="Estimated causal effect magnitude per service",
    )
    counterfactual_result: Optional[CounterfactualResult] = None

    # ── Stage 4: Root cause classification + SHAP ─────────────────────────
    root_cause_prediction: str = ""
    root_cause_probabilities: Dict[str, float] = Field(default_factory=dict)
    shap_top5: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="Top-5 SHAP features: [(feature_name, shap_value), ...]",
    )

    # ── Stage 5: Temporal forecasting ────────────────────────────────────
    forecast: Optional[ForecastResult] = None
    drift_alert: bool = False

    # ── Episode metadata ──────────────────────────────────────────────────
    step_number: int = 0
    task_id: int = 1
    services: List[str] = Field(default_factory=list)
    silenced_alerts: List[str] = Field(default_factory=list)
