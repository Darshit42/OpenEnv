"""
OpenEnv SRE — FastAPI HTTP server.

Routes:
  POST /reset          Start a new episode
  POST /step           Submit an action, get observation + reward
  GET  /state          Full internal state (for dashboard/monitoring)
  GET  /health         Liveness probe
  GET  /tasks          List available tasks
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List, Dict, Any, Optional

# Allow imports from backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    HealthResponse, TasksResponse,
    AgentStepRequest, AgentStepResponse,
    LeaderboardEntry, LeaderboardResponse,
)
from openenv.environment import OpenEnvSRE
from openenv.models import Action
from openenv.agent import HybridSREAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("openenv.api")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv — Causal SRE Intelligence",
    description=(
        "Production-grade RL environment for SRE incident diagnosis. "
        "Integrates causal DAG discovery, counterfactual action pre-screening, "
        "and SHAP explainability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton environment — one per server process
_env = OpenEnvSRE()
_leaderboard: List[LeaderboardEntry] = []
# Cache agent state per run
cf_called_state = False


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness probe."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/tasks", response_model=TasksResponse, tags=["Tasks"])
async def list_tasks():
    """List all available tasks with descriptions and scoring weights."""
    return TasksResponse(tasks=[
        {
            "id": 1,
            "name": "memory_leak_basic",
            "description": "One service has a classic monotonic RSS memory leak. Clear signal.",
            "max_steps": 30,
            "score_weight": 0.20,
        },
        {
            "id": 2,
            "name": "cascading_failure",
            "description": (
                "data-layer timeouts cascade; api-gateway has higher anomaly score (trap). "
                "Drift from step 0, alert at step 18. One red-herring alert."
            ),
            "max_steps": 45,
            "score_weight": 0.35,
        },
        {
            "id": 3,
            "name": "latency_spike_chain",
            "description": (
                "Config drift across 5 services. Cache restart is lethal. "
                "Two misleading critical alerts. Use causal DAG + counterfactual simulator."
            ),
            "max_steps": 60,
            "score_weight": 0.45,
        },
    ])


@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
async def reset(request: Request):
    """
    Start a new episode for the specified task and seed.
    Runs the PC causal DAG algorithm and returns the initial Observation.
    """
    try:
        try:
            body = await request.json()
        except BaseException:
            body = {}
            
        global cf_called_state
        cf_called_state = False
        
        task_id = body.get("task_id", 1)
        seed = body.get("seed", 42)
        
        obs = _env.reset(task_id=task_id, seed=seed)
        return ResetResponse(
            observation=obs.model_dump(),
            reward=0.0,
            done=False,
            info={}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("reset() failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: Request):
    """
    Submit an action. Returns (Observation, Reward, done, info).
    """
    try:
        try:
            body = await request.json()
        except BaseException:
            body = {}

        action_type = body.get("action_type", "query_counterfactual")
        service_id = body.get("service_id", "api-gateway")
        parameters = body.get("parameters", None)

        action = Action(
            action_type=action_type,
            service_id=service_id,
            parameters=parameters,
        )
        global cf_called_state
        if action_type == "query_counterfactual":
            cf_called_state = True
            
        obs, reward, done, info = _env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=float(getattr(reward, "total_score", 0.0)),
            done=bool(done),
            info=dict(info) if info else {}
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("step() failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/agent-step", response_model=AgentStepResponse, tags=["Agent"])
async def agent_step(req: AgentStepRequest):
    """
    Auto-execute one step using the configured Agent policy.
    """
    try:
        global cf_called_state
        state_dict = _env.state()
        obs = state_dict.get("observation", {})
        step_idx = state_dict.get("step", 0)
        
        agent = HybridSREAgent(use_llm=req.use_llm, model_name=req.model_name)
        action_dict = agent.decide_action(obs, step_idx, cf_called_state)
        
        # Execute the action via internal _env call
        if action_dict.get("action_type") == "query_counterfactual":
            cf_called_state = True
            
        action = Action(
            action_type=action_dict["action_type"],
            service_id=action_dict.get("service_id"),
            parameters=action_dict.get("parameters"),
        )
        obs_res, reward_res, done, info = _env.step(action)
        return AgentStepResponse(
            observation=obs_res.model_dump(),
            reward=reward_res.model_dump(),
            done=done,
            info=info,
            action_taken=action_dict
        )
    except Exception as e:
        logger.exception("agent_step() failed")
        raise HTTPException(status_code=500, detail=f"Agent flow failed: {e}")

@app.get("/leaderboard", response_model=LeaderboardResponse, tags=["Leaderboard"])
async def get_leaderboard():
    """Fetch the leaderboard rankings."""
    sorted_lb = sorted(_leaderboard, key=lambda x: x.total_score, reverse=True)
    return LeaderboardResponse(entries=sorted_lb)

@app.post("/leaderboard", tags=["Leaderboard"])
async def post_leaderboard(entry: LeaderboardEntry):
    """Submit a score to the leaderboard."""
    _leaderboard.append(entry)
    return {"status": "ok"}



@app.get("/state", tags=["Environment"])
async def state():
    """
    Return the complete serialisable internal state of the environment.
    Includes current scenario type, causal DAG, metric histories, and action trajectory.
    """
    try:
        return _env.state()
    except Exception as e:
        logger.exception("state() failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
