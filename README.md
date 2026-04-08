# OpenEnv: Causal SRE Intelligence

OpenEnv is a production-grade, reinforcement learning (RL) simulation platform designed for advanced Site Reliability Engineering (SRE) research and autonomous incident response. It models complex microservice architectures, injects realistic anomalies, and evaluates agent performance (LLMs or classical algorithms) based on correctness, speed, and system harm.

![OpenEnv Architecture](https://via.placeholder.com/800x400?text=OpenEnv+Architecture)

## Core Features

- **Causal Discovery & Reasoning:** Uses the PC algorithm to discover causal graphs (DAGs) from telemetry data dynamically, allowing agents to trace cascading failures to their origin.
- **Counterfactual Engine:** Features a Structural Equation Modeling (SEM) engine where agents can ask "What if I restart `api-gateway`?" and safely preview the simulated outcome before committing to a destructive action.
- **Robust ML Pipeline:** Integrates multiple models for root cause classification and drift detection:
  - **XGBoost Classifier:** Predicts the exact failure mode (e.g., `memory_leak`, `config_drift`) using an extensive 485-feature contract.
  - **Isolation Forest:** Detects multidimensional anomalies in log frequencies and metric distributions.
  - **LSTM Autoencoder:** Identifies temporal drift and early warning signs in latency/cpu sequences.
  - **SHAP Integration:** Explains ML predictions by mapping importance scores directly to interpretable system state variables.
- **Realistic SRE Workflows:** Provides actions typical of a real SRE, including `restart_service`, `scale_service`, `run_diagnostic`, `silence_alert`, and `escalate_incident`.
- **Dynamic Reward System:** Scores agent performance holistically, rewarding rapid resolution and counterfactual exploration while heavily penalizing destructive or incorrect actions.

## Repository Structure

```
├── backend/
│   ├── api/            # FastAPI endpoints (reset, step, counterfactuals)
│   ├── models/         # Training scripts and compiled model weights (.pkl/.pt)
│   └── openenv/        # Core simulation, ML pipeline, and SEM causal engine
├── frontend/           # React + Vite dashboard
│   ├── src/components/ # Service health, Action dispatch, AI insights, Causal DAG visualizer
│   └── src/App.jsx     # Main state management and API integration
└── Dockerfile          # Multi-stage production Docker configurations
```

## Setup & Installation

### 1. Backend (Python/FastAPI)

Requires Python 3.11+. We recommend using a virtual environment (`pyenv` or `venv`).

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
> **Note:** If `port 8000` is already in use by another process, start it on a different port like `8001`: `uvicorn api.main:app --host 0.0.0.0 --port 8001`

### 2. Frontend (React/Vite)

Requires Node.js 18+.

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

> **Important:** If you started your backend on a custom port (e.g., `8001`), inject the environment variable before running the dev server:
> ```bash
> VITE_API_URL="http://localhost:8001" npm run dev
> ```

## How to Play / Train Agents

1. Open the React frontend in your browser.
2. Select a scenario difficulty (`Easy`, `Medium`, `Hard`) and press **Reset Episode**.
3. The environment will initialize and populate the **Service Health** and **Causal DAG** widgets.
4. **Step-by-Step Execution:** As time passes, alerts may fire. The AI Insights panel will display real-time SHAP analysis and suspected root causes.
5. **Take Actions:** Use the **Action Panel** to simulate counterfactuals or execute interventions.
6. **Resolve:** Once you identify the culprit, fix it and invoke `declare_resolution`. The reward engine will grade your response.

## Development & Testing

**Model Training:**
If you make changes to the `feature_contract.py`, you must retrain the base ML models:
```bash
source .venv/bin/activate
cd backend
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python models/train_models.py
```
*(The thread limits are required to prevent OpenMP SHM allocation crashes during parallel training.)*

---
*OpenEnv represents a major step forward in autonomous DevSecOps and SRE training simulation.*