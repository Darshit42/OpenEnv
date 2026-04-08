"""
Request / Response schemas for the OpenEnv SRE API.
Thin wrappers over the core environment models.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.models import ActionType


# ─────────────────────────────────────────────────────────────────────────────
# Requests
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Task ID: 1=Easy, 2=Medium, 3=Hard",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. Defaults to RANDOM_SEED env var.",
    )

    def __init__(self, **data: Any) -> None:
        if data.get("seed") is None:
            env_seed = os.environ.get("RANDOM_SEED")
            if env_seed:
                data["seed"] = int(env_seed)
        super().__init__(**data)


class StepRequest(BaseModel):
    action_type: ActionType = Field(..., description="Action type from the ActionType enum")
    service_id: Optional[str] = Field(
        default=None,
        description="Target service identifier (required for most action types)",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional action parameters (e.g., replicas for scale_service)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action_type": "query_counterfactual",
                    "service_id": "api-server",
                    "parameters": None,
                },
                {
                    "action_type": "restart_service",
                    "service_id": "api-server",
                    "parameters": None,
                },
                {
                    "action_type": "silence_alert",
                    "service_id": None,
                    "parameters": {"alert_id": "alert-cpu-background-worker"},
                },
                {
                    "action_type": "declare_resolution",
                    "service_id": None,
                    "parameters": None,
                },
            ]
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Responses
# ─────────────────────────────────────────────────────────────────────────────

class ResetResponse(BaseModel):
    status: str = "reset successful"
    observation: Dict[str, Any]
    task_id: int
    seed: Optional[int]
    message: str


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str


class TaskInfo(BaseModel):
    id: int
    name: str
    description: str
    max_steps: int
    score_weight: float


class TasksResponse(BaseModel):
    tasks: List[Dict[str, Any]]


class AgentStepRequest(BaseModel):
    use_llm: bool = True
    model_name: str = "gpt-4o"
    

class AgentStepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
    action_taken: Dict[str, Any]


class LeaderboardEntry(BaseModel):
    run_id: str
    agent_name: str
    total_score: float
    task_id: int
    steps_taken: int
    timestamp: str


class LeaderboardResponse(BaseModel):
    entries: List[LeaderboardEntry]
