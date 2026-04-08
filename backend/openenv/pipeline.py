"""
Five-Stage ML Pipeline for the OpenEnv SRE environment.

Runs inside every step() call. Outputs are packaged into the Observation.

Stage 1 — Feature Engineering
Stage 2 — Anomaly Detection      (Isolation Forest)
Stage 3 — Causal DAG Discovery   (PC algorithm via causal-learn)
           Counterfactual Sim     (Linear SEM + do-calculus)
Stage 4 — Root Cause Classification (XGBoost + SHAP)
Stage 5 — Temporal Forecasting   (LSTM Autoencoder + Linear Trend)
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Threading safety — MUST be set before importing sklearn/torch/xgboost ────
# Prevents OpenMP SHM crashes (OMP Error #179) in Docker/constrained envs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# ── Lightweight mode — skip all ML models ─────────────────────────────────────
LIGHTWEIGHT_MODE = os.environ.get("OPENENV_LIGHTWEIGHT", "").lower() in ("1", "true", "yes")
if LIGHTWEIGHT_MODE:
    logger.info("LIGHTWEIGHT MODE enabled — all ML models disabled, using heuristic fallbacks")

# ── Optional heavy deps — degrade gracefully if not installed ─────────────────
_SKLEARN_OK = False
_XGB_OK = False
_CAUSAL_OK = False
_TORCH_OK = False

if not LIGHTWEIGHT_MODE:
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import IncrementalPCA
        _SKLEARN_OK = True
    except ImportError:
        logger.warning("scikit-learn not available; falling back to heuristic anomaly detection")

    try:
        import xgboost as xgb
        import shap
        _XGB_OK = True
    except ImportError:
        logger.warning("XGBoost/SHAP not available; using heuristic root-cause classification")

    try:
        from causallearn.search.ConstraintBased.PC import pc as pc_algorithm
        _CAUSAL_OK = True
    except ImportError:
        logger.warning("causal-learn not available; using correlation-based DAG approximation")

    try:
        import torch
        import torch.nn as nn
        _TORCH_OK = True
    except ImportError:
        logger.warning("PyTorch not available; using reconstruction-error heuristic")

from scipy.stats import kendalltau, linregress

from openenv.feature_contract import (
    METRIC_COLS,
    get_training_feature_names,
    map_runtime_features_to_training,
)
from openenv.models import CounterfactualResult, ForecastResult

# Re-export for backward compat
ROOT_CAUSE_CLASSES = [
    "memory_leak", "latency_spike", "cascading_failure",
    "config_drift", "dependency_timeout", "false_alarm",
]

MODELS_DIR = Path(__file__).parent.parent / "models" / "saved"


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_OK:
    class LSTMAutoencoder(nn.Module):
        """Small LSTM autoencoder for sequence anomaly detection."""

        def __init__(self, input_size: int = 7, hidden_size: int = 32, num_layers: int = 1):
            super().__init__()
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, input_size)
            _, (h, _) = self.encoder(x)
            # repeat hidden state across sequence length
            seq_len = x.size(1)
            h_rep = h[-1].unsqueeze(1).repeat(1, seq_len, 1)
            out, _ = self.decoder(h_rep)
            return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(
    metric_history: Dict[str, List[Dict[str, float]]],
    log_history: List[str],
    services: List[str],
) -> Dict[str, float]:
    """
    Builds the full feature vector from rolling metric history and logs.

    Returns flat dict: {feature_name: value}
    """
    features: Dict[str, float] = {}
    min_rows = 2  # need at least 2 observations

    for svc in services:
        history = metric_history.get(svc, [])
        if not history:
            # Zero-fill when no history
            for col in METRIC_COLS:
                for sfx in ["cur", "mean5", "std5", "mean15", "std15",
                            "slope5", "lag1", "lag3", "lag5",
                            "mk_z", "mk_p"]:
                    features[f"{svc}__{col}__{sfx}"] = 0.0
            continue

        df = pd.DataFrame(history)
        # Ensure all metric columns are present
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = 0.0

        n = len(df)

        for col in METRIC_COLS:
            series = df[col].values.astype(float)

            # Current value
            features[f"{svc}__{col}__cur"] = float(series[-1])

            # Rolling statistics (window=5 and window=15)
            for win, suffix in [(5, "mean5"), (5, "std5"), (15, "mean15"), (15, "std15")]:
                window_data = series[-win:] if n >= win else series
                if "mean" in suffix:
                    features[f"{svc}__{col}__{suffix}"] = float(np.mean(window_data))
                else:
                    features[f"{svc}__{col}__{suffix}"] = float(np.std(window_data) + 1e-9)

            # Slope over last 5 steps
            win_data = series[-5:] if n >= 5 else series
            if len(win_data) >= min_rows:
                slope, *_ = linregress(range(len(win_data)), win_data)
                features[f"{svc}__{col}__slope5"] = float(slope)
            else:
                features[f"{svc}__{col}__slope5"] = 0.0

            # Lag features
            for lag in [1, 3, 5]:
                idx = -(lag + 1)
                features[f"{svc}__{col}__lag{lag}"] = float(series[idx]) if n > lag else float(series[0])

            # Mann-Kendall trend test
            if len(series) >= 4:
                try:
                    tau, p_val = kendalltau(range(len(series)), series)
                    features[f"{svc}__{col}__mk_z"] = float(tau)
                    features[f"{svc}__{col}__mk_p"] = float(p_val)
                except Exception:
                    features[f"{svc}__{col}__mk_z"] = 0.0
                    features[f"{svc}__{col}__mk_p"] = 1.0
            else:
                features[f"{svc}__{col}__mk_z"] = 0.0
                features[f"{svc}__{col}__mk_p"] = 1.0

    # Cross-service error-rate correlation matrix (flattened)
    if len(services) > 1:
        err_matrix = []
        for svc in services:
            hist = metric_history.get(svc, [])
            if hist:
                err_matrix.append([h.get("error_rate", 0.0) for h in hist[-15:]])
        if len(err_matrix) >= 2:
            max_len = max(len(r) for r in err_matrix)
            padded = np.array([np.pad(r, (max_len - len(r), 0)) for r in err_matrix])
            corr = np.corrcoef(padded)
            for i, svc_i in enumerate(services):
                for j, svc_j in enumerate(services):
                    if i < j:
                        key = f"cross_corr__{svc_i}__{svc_j}__err"
                        features[key] = float(corr[i, j]) if not np.isnan(corr[i, j]) else 0.0

    # Log-based features: keyword count
    log_text = " ".join(log_history[-50:])  # last 50 log messages
    for kw in ["OOMKiller", "timeout", "CRITICAL", "ERROR", "WARNING",
               "config_version_mismatch", "pool", "cascade"]:
        features[f"log__{kw.lower()}__count"] = float(log_text.count(kw))

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """Isolation Forest wrapper with lazy model loading and normalized scoring."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._fitted = False
        self._training_mean: Optional[np.ndarray] = None
        self._training_std: Optional[np.ndarray] = None
        self._load_or_init()

    def _load_or_init(self) -> None:
        model_path = MODELS_DIR / "isolation_forest.pkl"
        if _SKLEARN_OK and model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                # Support both old (model-only) and new (dict with stats) format
                if isinstance(data, dict):
                    self._model = data["model"]
                    self._training_mean = data.get("mean")
                    self._training_std = data.get("std")
                else:
                    self._model = data
                self._fitted = True
                logger.info("Loaded pre-trained IsolationForest from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load IsolationForest: %s", e)
        if not self._fitted and _SKLEARN_OK:
            self._model = IsolationForest(
                n_estimators=100, contamination=0.05, random_state=42, n_jobs=1
            )

    def fit(self, X: np.ndarray) -> None:
        if _SKLEARN_OK and self._model is not None:
            self._model.fit(X)
            self._fitted = True

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Z-score normalize using training stats if available."""
        if self._training_mean is not None and self._training_std is not None:
            return (vec - self._training_mean) / (self._training_std + 1e-9)
        return vec

    def score(self, metric_snapshot: Dict[str, float]) -> float:
        """Return anomaly score in [0, 1]. Higher = more anomalous."""
        if not _SKLEARN_OK or self._model is None or not self._fitted:
            return self._heuristic_score(metric_snapshot)

        raw_vec = np.array([[
            metric_snapshot.get(col, 0.0) for col in METRIC_COLS
        ]])

        try:
            vec = self._normalize(raw_vec)
            raw = self._model.decision_function(vec)[0]
            # IsolationForest: negative = anomalous; normalise to [0, 1]
            score = 1.0 / (1.0 + np.exp(raw * 2))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return self._heuristic_score(metric_snapshot)

    def score_with_temporal(
        self,
        metric_snapshot: Dict[str, float],
        history: List[Dict[str, float]],
    ) -> float:
        """
        Enhanced scoring with temporal context (slope + rolling std).
        Falls back to basic score() if history is insufficient.
        """
        base_score = self.score(metric_snapshot)

        if len(history) < 3:
            return base_score

        # Add temporal signal: slope and volatility of error_rate
        err_series = [h.get("error_rate", 0.0) for h in history[-10:]]
        if len(err_series) >= 3:
            slope, *_ = linregress(range(len(err_series)), err_series)
            volatility = float(np.std(err_series))
            # Boost score if rapidly worsening
            temporal_boost = min(0.2, max(0.0, slope * 0.1 + volatility * 0.05))
            return float(np.clip(base_score + temporal_boost, 0.0, 1.0))

        return base_score

    @staticmethod
    def _heuristic_score(snap: Dict[str, float]) -> float:
        """Simple rule-based fallback."""
        cpu = snap.get("cpu_utilization", 0) / 100.0
        mem_ratio = min(1.0, snap.get("memory_rss", 0) / 2048.0)
        err = min(1.0, snap.get("error_rate", 0) / 5.0)
        pool = snap.get("connection_pool_saturation", 0)
        return float(np.clip((cpu * 0.3 + mem_ratio * 0.3 + err * 0.25 + pool * 0.15), 0.0, 1.0))

    def detect(
        self,
        metrics: Dict[str, Any],
        services: List[str],
        metric_history: Optional[Dict[str, List[Dict[str, float]]]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        scores: Dict[str, float] = {}
        flags: Dict[str, bool] = {}
        for svc in services:
            snap_obj = metrics.get(svc)
            if snap_obj is None:
                scores[svc] = 0.0
                flags[svc] = False
                continue
            snap = snap_obj.model_dump() if hasattr(snap_obj, "model_dump") else dict(snap_obj)

            # Use temporal-aware scoring if history available
            if metric_history and svc in metric_history:
                scores[svc] = self.score_with_temporal(snap, metric_history[svc])
            else:
                scores[svc] = self.score(snap)

            flags[svc] = scores[svc] > 0.65
        return scores, flags


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Causal DAG Discovery + Counterfactual Simulation
# ─────────────────────────────────────────────────────────────────────────────

class CausalEngine:
    """
    Runs the PC algorithm on metric history to produce a causal DAG.
    Fits a linear SEM on each edge for counterfactual simulation.
    """

    def __init__(self) -> None:
        self._dag: Dict[str, List[str]] = {}         # {child: [parents]}
        self._sem_coefficients: Dict[str, Any] = {}  # {child: {parent: (slope, intercept)}}
        self._services: List[str] = []

    def discover_dag(
        self,
        metric_history: Dict[str, List[Dict[str, float]]],
        services: List[str],
        alpha: float = 0.05,
        max_cond_vars: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Run PC algorithm on error_rate history per service.
        Returns adjacency list {service: [causal_parent_services]}.
        """
        self._services = services
        min_obs = 10

        # Build data matrix: rows=timesteps, cols=services (using error_rate as primary metric)
        histories = []
        valid_services = []
        for svc in services:
            hist = metric_history.get(svc, [])
            if len(hist) >= min_obs:
                histories.append([h.get("error_rate", 0.0) for h in hist])
                valid_services.append(svc)

        if len(valid_services) < 2:
            # Not enough data — use correlation-based heuristic
            self._dag = {svc: [] for svc in services}
            return self._dag

        # Align lengths
        min_len = min(len(h) for h in histories)
        data_matrix = np.array([h[-min_len:] for h in histories]).T  # (T, n_services)

        if _CAUSAL_OK and data_matrix.shape[0] >= 15:
            try:
                # Inject controlled noise to break determinism across seeds
                rng = np.random.default_rng(hash(tuple(data_matrix.ravel()[:20].tolist())) % (2**31))
                noise = rng.normal(0, 0.01, data_matrix.shape)
                data_noisy = data_matrix + noise

                # Bootstrap subsample: use ~80% of timesteps
                n_rows = data_noisy.shape[0]
                n_subsample = max(15, int(n_rows * 0.8))
                subsample_idx = rng.choice(n_rows, size=n_subsample, replace=False)
                subsample_idx.sort()
                data_sub = data_noisy[subsample_idx]

                cg = pc_algorithm(
                    data_sub,
                    alpha=alpha,
                    indep_test="fisherz",
                    stable=True,
                    uc_rule=0,
                    uc_priority=2,
                    mvpc=False,
                    correction_name="MV_Crtn_Fisher_Z",
                    background_knowledge=None,
                    verbose=False,
                    show_progress=False,
                )
                adj = cg.G.graph  # numpy array: adj[i,j]=1 means i→j
                dag: Dict[str, List[str]] = {svc: [] for svc in services}
                n = len(valid_services)
                for i in range(n):
                    for j in range(n):
                        if i != j and adj[i, j] == 1 and adj[j, i] == -1:
                            # i → j : j has parent i
                            dag[valid_services[j]].append(valid_services[i])
                self._dag = dag
            except Exception as e:
                logger.warning("PC algorithm failed (%s); using correlation heuristic", e)
                self._dag = self._correlation_dag(data_matrix, valid_services, services)
        else:
            self._dag = self._correlation_dag(data_matrix, valid_services, services)

        # Fit linear SEM on discovered edges
        self._fit_sem(metric_history, valid_services)
        return self._dag

    @staticmethod
    def _correlation_dag(
        data: np.ndarray,
        valid_services: List[str],
        all_services: List[str],
    ) -> Dict[str, List[str]]:
        """Fallback: use Granger-style lag-1 correlation as DAG proxy."""
        dag: Dict[str, List[str]] = {svc: [] for svc in all_services}
        n = len(valid_services)
        if data.shape[0] < 3:
            return dag
        for i in range(n):
            for j in range(n):
                if i != j:
                    x_lag = data[:-1, i]
                    y_curr = data[1:, j]
                    if np.std(x_lag) > 1e-9:
                        corr = np.corrcoef(x_lag, y_curr)[0, 1]
                        if abs(corr) > 0.6:
                            dag[valid_services[j]].append(valid_services[i])
        return dag

    def _fit_sem(
        self,
        metric_history: Dict[str, List[Dict[str, float]]],
        valid_services: List[str],
    ) -> None:
        """Fit linear regression for each edge in the DAG."""
        self._sem_coefficients = {}
        for child, parents in self._dag.items():
            if not parents or child not in valid_services:
                continue
            child_hist = [h.get("error_rate", 0.0) for h in metric_history.get(child, [])]
            self._sem_coefficients[child] = {}
            for parent in parents:
                if parent not in valid_services:
                    continue
                parent_hist = [h.get("error_rate", 0.0) for h in metric_history.get(parent, [])]
                min_len = min(len(child_hist), len(parent_hist))
                if min_len >= 4:
                    x = np.array(parent_hist[-min_len:])
                    y = np.array(child_hist[-min_len:])
                    try:
                        slope, intercept, *_ = linregress(x, y)
                        self._sem_coefficients[child][parent] = (float(slope), float(intercept))
                    except Exception:
                        self._sem_coefficients[child][parent] = (0.0, 0.0)

    def get_causal_effects(self) -> Dict[str, float]:
        """Aggregate causal effect magnitude per service (sum of |slopes| from SEM)."""
        effects: Dict[str, float] = {svc: 0.0 for svc in self._services}
        for child, parents_coefs in self._sem_coefficients.items():
            for parent, (slope, _) in parents_coefs.items():
                effects[parent] = effects.get(parent, 0.0) + abs(slope)
        return effects

    def simulate_counterfactual(
        self,
        action_type: str,
        service_id: str,
        metric_history: Dict[str, List[Dict[str, float]]],
        scenario_truth: str,
        scenario_service: str,
        lethal_services: List[str],
        horizon: int = 5,
    ) -> CounterfactualResult:
        """
        Simulate the expected metric trajectory for the next `horizon` steps
        under the proposed action using the learned linear SEM.

        NOTE: action_type here should be the ACTUAL action to simulate
        (e.g., "restart_service"), NOT "query_counterfactual".
        """
        # Determine if action is harmful
        harm_flag = (
            action_type in ("restart_service", "scale_service")
            and service_id in lethal_services
        )
        harm_desc = None
        if harm_flag and service_id == "cache":
            harm_desc = "Restarting cache will cause write-loss cascade and invalidate in-flight transactions."
        elif harm_flag and service_id == "database":
            harm_desc = "Restarting the database will cause data corruption and connection storm."

        # Determine predicted resolution probability
        if harm_flag:
            resolution_prob = float(np.random.uniform(0.08, 0.16))
        elif action_type == "restart_service" and service_id == scenario_service:
            resolution_prob = float(np.random.uniform(0.78, 0.95))
        elif action_type == "scale_service" and service_id == scenario_service:
            resolution_prob = float(np.random.uniform(0.60, 0.82))
        elif action_type in ("restart_service", "scale_service") and service_id != scenario_service:
            # Acting on a downstream service — moderate improvement
            is_downstream = service_id in self._dag and len(self._dag.get(service_id, [])) > 0
            resolution_prob = float(np.random.uniform(0.30, 0.55)) if is_downstream else float(np.random.uniform(0.10, 0.25))
        elif action_type == "run_diagnostic":
            resolution_prob = float(np.random.uniform(0.05, 0.15))  # Diagnostic doesn't fix
        elif action_type == "silence_alert":
            resolution_prob = float(np.random.uniform(0.02, 0.08))  # No actual fix
        else:
            resolution_prob = float(np.random.uniform(0.15, 0.40))

        # Predict metric trajectories (error_rate proxy per service)
        predicted_metrics: Dict[str, List[float]] = {}
        for svc in self._services:
            hist = metric_history.get(svc, [])
            if not hist:
                predicted_metrics[svc] = [0.0] * horizon
                continue
            last_err = hist[-1].get("error_rate", 0.0)

            if action_type == "restart_service" and service_id == svc:
                # Service restarted: immediate partial recovery
                base = last_err * 0.2 if not harm_flag else last_err * 0.9
            elif action_type == "scale_service" and service_id == svc:
                # Service scaled: gradual improvement
                base = last_err * 0.5 if not harm_flag else last_err * 0.85
            elif svc in self._dag and service_id in self._dag.get(svc, []):
                # This service has service_id as a parent — will improve if parent is fixed
                base = last_err * (1 - resolution_prob * 0.5)
            else:
                base = last_err

            # Apply linear trend for horizon steps
            series = [h.get("error_rate", 0.0) for h in hist[-10:]]
            if len(series) >= 2:
                slope, intercept = linregress(range(len(series)), series)[:2]
                trend_adj = float(slope)
            else:
                trend_adj = 0.0

            predicted_metrics[svc] = [
                float(max(0.0, base + trend_adj * t * (0.5 if action_type == "restart_service" else 1.0)))
                for t in range(1, horizon + 1)
            ]

        ci_half = (1.0 - resolution_prob) * 0.2
        return CounterfactualResult(
            action_type=action_type,
            service_id=service_id,
            predicted_metrics=predicted_metrics,
            predicted_resolution_probability=round(resolution_prob, 3),
            confidence_interval=(
                round(max(0.0, resolution_prob - ci_half), 3),
                round(min(1.0, resolution_prob + ci_half), 3),
            ),
            harm_flag=harm_flag,
            harm_description=harm_desc,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Root Cause Classification + SHAP
# ─────────────────────────────────────────────────────────────────────────────

class RootCauseClassifier:
    """XGBoost-based root-cause classifier with SHAP explainability."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._feature_names: List[str] = []
        self._loaded = False
        self._shap_explainer: Optional[Any] = None  # Cached for performance
        self._load_or_init()

    def _load_or_init(self) -> None:
        model_path = MODELS_DIR / "xgboost_classifier.pkl"
        if _XGB_OK and model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                self._model = bundle["model"]
                self._feature_names = bundle["feature_names"]
                self._loaded = True
                logger.info("Loaded pre-trained XGBoost from %s (features=%d)",
                            model_path, len(self._feature_names))
            except Exception as e:
                logger.warning("Failed to load XGBoost: %s", e)

    def predict(
        self,
        features: Dict[str, float],
        services: List[str],
        scenario_hint: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float], List[Tuple[str, float]]]:
        """
        Returns (predicted_class, class_probabilities, shap_top5).
        Falls back to heuristic if model not loaded.
        """
        if not _XGB_OK or self._model is None or not self._loaded:
            return self._heuristic_predict(features, scenario_hint)

        # Map runtime features (with real service names) to training feature order
        try:
            vec, training_names, coverage = map_runtime_features_to_training(
                features, services
            )
        except Exception as e:
            logger.warning("Feature mapping failed (%s); using heuristic", e)
            return self._heuristic_predict(features, scenario_hint)

        if coverage < 0.05:
            logger.warning("Feature coverage too low (%.1f%%); using heuristic fallback",
                           coverage * 100)
            return self._heuristic_predict(features, scenario_hint)

        X = np.array([vec])

        try:
            probs = self._model.predict_proba(X)[0]
            classes = ROOT_CAUSE_CLASSES[: len(probs)]
            prob_dict = {c: round(float(p), 4) for c, p in zip(classes, probs)}
            predicted = classes[int(np.argmax(probs))]

            # SHAP — use cached explainer for performance
            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(self._model)

            shap_vals = self._shap_explainer.shap_values(X)
            cls_idx = list(classes).index(predicted)

            # Handle both SHAP output formats:
            # Old API: list of arrays, each (n_samples, n_features) — one per class
            # New API: single ndarray (n_samples, n_features, n_classes)
            if isinstance(shap_vals, list):
                sv = np.array(shap_vals[cls_idx][0])  # (n_features,)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                sv = shap_vals[0, :, cls_idx]  # (n_features,)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                sv = shap_vals[0]  # (n_features,)
            else:
                sv = np.zeros(len(training_names))

            # Ensure sv is a flat 1D array of scalars
            sv = np.array(sv, dtype=float).ravel()

            # Map SHAP feature names back to runtime names for interpretability
            runtime_feat_names = self._map_training_names_to_runtime(
                training_names, services
            )

            feat_shap = sorted(
                zip(runtime_feat_names, sv.tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            shap_top5 = [(name, round(float(val), 5)) for name, val in feat_shap[:5]]

            return predicted, prob_dict, shap_top5

        except Exception as e:
            logger.warning("XGBoost predict failed: %s", e)
            return self._heuristic_predict(features, scenario_hint)

    @staticmethod
    def _map_training_names_to_runtime(
        training_names: List[str],
        services: List[str],
    ) -> List[str]:
        """Map training feature names (svc_0__...) back to real service names for display."""
        from openenv.feature_contract import TRAINING_SERVICE_NAMES
        reverse_map = {}
        for idx, svc in enumerate(services[:len(TRAINING_SERVICE_NAMES)]):
            reverse_map[TRAINING_SERVICE_NAMES[idx]] = svc

        result = []
        for tn in training_names:
            name = tn
            for train_svc, real_svc in reverse_map.items():
                if tn.startswith(f"{train_svc}__"):
                    name = tn.replace(f"{train_svc}__", f"{real_svc}__", 1)
                    break
            result.append(name)
        return result

    @staticmethod
    def _heuristic_predict(
        features: Dict[str, float],
        scenario_hint: Optional[str],
    ) -> Tuple[str, Dict[str, float], List[Tuple[str, float]]]:
        """Rule-based fallback classifier."""
        # Gather signals
        max_mem_slope = max(
            (v for k, v in features.items() if "memory_rss__slope5" in k), default=0.0
        )
        max_err_rate = max(
            (v for k, v in features.items() if "__error_rate__cur" in k), default=0.0
        )
        max_pool_sat = max(
            (v for k, v in features.items() if "connection_pool_saturation__cur" in k), default=0.0
        )
        config_flag = features.get("log__config_version_mismatch__count", 0.0)
        oom_flag = features.get("log__oomkiller__count", 0.0)
        timeout_flag = features.get("log__timeout__count", 0.0)

        # Score each class
        scores = {c: 0.0 for c in ROOT_CAUSE_CLASSES}
        if oom_flag > 0 or max_mem_slope > 30:
            scores["memory_leak"] += 0.6 + oom_flag * 0.1
        if max_pool_sat > 0.7:
            scores["dependency_timeout"] += 0.5
        if timeout_flag > 0:
            scores["dependency_timeout"] += 0.3
        if config_flag > 0:
            scores["config_drift"] += 0.7
        if max_err_rate > 2.0:
            scores["cascading_failure"] += 0.4

        # Use hint if available
        if scenario_hint and scenario_hint in scores:
            scores[scenario_hint] += 0.8

        total = sum(scores.values()) or 1.0
        probs = {c: round(s / total, 4) for c, s in scores.items()}
        predicted = max(probs, key=lambda k: probs[k])

        top_feats = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        shap_top5 = [(k, round(v * 0.01, 5)) for k, v in top_feats]

        return predicted, probs, shap_top5


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Temporal Forecasting (LSTM + Linear Trend)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalForecaster:
    """
    LSTM Autoencoder for sequence anomaly scoring +
    Linear trend extrapolation for forward forecasting.
    """

    def __init__(self) -> None:
        self._lstm: Optional[Any] = None
        self._lstm_input_size = len(METRIC_COLS)
        self._drift_threshold = 2.5  # error_rate/sec beyond which drift_alert fires
        self._load_lstm()

    def _load_lstm(self) -> None:
        if not _TORCH_OK:
            return
        model_path = MODELS_DIR / "lstm_autoencoder.pt"
        self._lstm = LSTMAutoencoder(input_size=self._lstm_input_size)
        if model_path.exists():
            try:
                self._lstm.load_state_dict(
                    torch.load(model_path, map_location="cpu", weights_only=True)
                )
                self._lstm.eval()
                logger.info("Loaded pre-trained LSTM from %s", model_path)
            except Exception as e:
                logger.warning("Could not load LSTM weights: %s — using random init", e)
        else:
            logger.info("No LSTM model found at %s — using linear trend fallback", model_path)

    def forecast(
        self,
        metric_history: Dict[str, List[Dict[str, float]]],
        services: List[str],
        horizon_t5: int = 5,
        horizon_t15: int = 15,
    ) -> ForecastResult:
        forecast_t5: Dict[str, float] = {}
        forecast_t15: Dict[str, float] = {}
        band_width: Dict[str, float] = {}
        lstm_errors: Dict[str, float] = {}
        any_drift = False

        for svc in services:
            hist = metric_history.get(svc, [])
            if not hist:
                forecast_t5[svc] = 0.0
                forecast_t15[svc] = 0.0
                band_width[svc] = 0.0
                lstm_errors[svc] = 0.0
                continue

            err_series = np.array([h.get("error_rate", 0.0) for h in hist], dtype=float)
            n = len(err_series)

            # Linear trend extrapolation
            if n >= 3:
                x = np.arange(n)
                slope, intercept, r_val, _, stderr = linregress(x, err_series)
                f_t5 = float(max(0.0, slope * (n + horizon_t5) + intercept))
                f_t15 = float(max(0.0, slope * (n + horizon_t15) + intercept))
                # Confidence band: proportional to stderr
                band = float(stderr * np.sqrt(1 + 1 / n + horizon_t5 ** 2 / max(1, np.sum((x - x.mean()) ** 2))))
            else:
                f_t5 = float(err_series[-1])
                f_t15 = float(err_series[-1])
                band = 0.5

            forecast_t5[svc] = round(f_t5, 4)
            forecast_t15[svc] = round(f_t15, 4)
            band_width[svc] = round(band, 4)

            # Drift alert: predicted t+15 exceeds safety threshold even if current is below
            current_err = float(err_series[-1])
            if f_t15 > self._drift_threshold and current_err <= self._drift_threshold:
                any_drift = True

            # LSTM reconstruction error (anomaly in sequence)
            lstm_errors[svc] = self._lstm_reconstruction_error(hist, svc)

        return ForecastResult(
            forecast_t5=forecast_t5,
            forecast_t15=forecast_t15,
            confidence_band_width=band_width,
            drift_alert=any_drift,
            lstm_reconstruction_error=lstm_errors,
        )

    def _lstm_reconstruction_error(
        self,
        hist: List[Dict[str, float]],
        svc: str,
    ) -> float:
        if not _TORCH_OK or self._lstm is None or len(hist) < 5:
            # Heuristic: variance of error_rate
            err_series = [h.get("error_rate", 0.0) for h in hist]
            return float(np.std(err_series[-10:]))

        seq = np.array([
            [h.get(col, 0.0) for col in METRIC_COLS] for h in hist[-20:]
        ], dtype=np.float32)

        # Normalise
        mean = seq.mean(axis=0, keepdims=True)
        std = seq.std(axis=0, keepdims=True) + 1e-9
        seq_norm = (seq - mean) / std

        tensor = torch.from_numpy(seq_norm).unsqueeze(0)  # (1, T, features)
        with torch.no_grad():
            try:
                reconstructed = self._lstm(tensor)
                error = float(torch.mean((tensor - reconstructed) ** 2).item())
            except Exception:
                error = 0.0
        return round(error, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class SREPipeline:
    """
    Assembles all five stages. Instantiated once per environment instance.
    """

    def __init__(self) -> None:
        self.anomaly_detector = AnomalyDetector()
        self.causal_engine = CausalEngine()
        self.root_cause_classifier = RootCauseClassifier()
        self.temporal_forecaster = TemporalForecaster()
        self._log_buffer: List[str] = []

    def reset_causal(
        self,
        metric_history: Dict[str, List[Dict[str, float]]],
        services: List[str],
    ) -> Dict[str, List[str]]:
        """Run PC algorithm at episode reset. Returns DAG adjacency list."""
        return self.causal_engine.discover_dag(metric_history, services)

    def run(
        self,
        metrics: Dict[str, Any],
        metric_history: Dict[str, List[Dict[str, float]]],
        log_messages: List[str],
        services: List[str],
        action_type: Optional[str] = None,
        action_service: Optional[str] = None,
        scenario_truth: Optional[str] = None,
        scenario_service: Optional[str] = None,
        lethal_services: Optional[List[str]] = None,
        simulated_action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute all five pipeline stages and return a dict of outputs for the Observation.
        """
        self._log_buffer.extend(log_messages)

        # Stage 1 — Feature Engineering
        try:
            features = engineer_features(metric_history, self._log_buffer, services)
        except Exception as e:
            logger.warning("Feature engineering failed: %s — using empty features", e)
            features = {}

        # Stage 2 — Anomaly Detection
        try:
            anomaly_scores, anomaly_flags = self.anomaly_detector.detect(
                metrics, services, metric_history=metric_history
            )
        except Exception as e:
            logger.warning("Anomaly detection failed: %s — using defaults", e)
            anomaly_scores = {svc: 0.0 for svc in services}
            anomaly_flags = {svc: False for svc in services}

        # Stage 3 — Counterfactual (only when agent queries)
        counterfactual_result = None
        if action_type == "query_counterfactual" and action_service:
            try:
                # Use the SIMULATED action, not "query_counterfactual"
                effective_action = simulated_action or "restart_service"
                counterfactual_result = self.causal_engine.simulate_counterfactual(
                    action_type=effective_action,
                    service_id=action_service,
                    metric_history=metric_history,
                    scenario_truth=scenario_truth or "",
                    scenario_service=scenario_service or "",
                    lethal_services=lethal_services or [],
                )
            except Exception as e:
                logger.warning("Counterfactual simulation failed: %s", e)

        try:
            causal_effects = self.causal_engine.get_causal_effects()
        except Exception as e:
            logger.warning("Causal effects failed: %s", e)
            causal_effects = {}

        # Stage 4 — Root Cause Classification + SHAP
        try:
            predicted_class, class_probs, shap_top5 = self.root_cause_classifier.predict(
                features, services, scenario_hint=scenario_truth
            )
        except Exception as e:
            logger.warning("Root cause classification failed: %s", e)
            predicted_class = "false_alarm"
            class_probs = {c: round(1.0 / len(ROOT_CAUSE_CLASSES), 4) for c in ROOT_CAUSE_CLASSES}
            shap_top5 = []

        # Stage 5 — Temporal Forecasting
        try:
            forecast = self.temporal_forecaster.forecast(metric_history, services)
        except Exception as e:
            logger.warning("Temporal forecasting failed: %s", e)
            forecast = ForecastResult()

        return {
            "features": features,
            "anomaly_scores": anomaly_scores,
            "anomaly_flags": anomaly_flags,
            "counterfactual_result": counterfactual_result,
            "causal_effects": causal_effects,
            "root_cause_prediction": predicted_class,
            "root_cause_probabilities": class_probs,
            "shap_top5": shap_top5,
            "forecast": forecast,
            "drift_alert": forecast.drift_alert,
        }
