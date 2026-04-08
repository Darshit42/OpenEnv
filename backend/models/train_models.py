"""
Model training script — run once at Docker build time.

Trains and serialises three models:
  1. IsolationForest  — anomaly detection (normal-ops corpus)
  2. XGBoostClassifier — root-cause classification (labeled synthetic incidents)
  3. LSTM Autoencoder  — sequence anomaly scoring (normal metric sequences)

Feature schema is defined by openenv.feature_contract (single source of truth).

Usage:
    python models/train_models.py
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── Threading safety — MUST be set before any ML imports ─────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Allow backend imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("openenv.train")

from openenv.feature_contract import (
    METRIC_COLS,
    STAT_SUFFIXES,
    LOG_KEYWORDS,
    MAX_SERVICE_SLOTS,
    TRAINING_SERVICE_NAMES,
    get_training_feature_names,
    get_feature_count,
)

SAVE_DIR = Path(__file__).parent / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ROOT_CAUSE_CLASSES = [
    "memory_leak", "latency_spike", "cascading_failure",
    "config_drift", "dependency_timeout", "false_alarm",
]

N_NORMAL_SAMPLES = 50_000
N_INCIDENT_SAMPLES_PER_CLASS = 2_000
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_normal_ops(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate N synthetic normal-operations metric snapshots (7 features)."""
    return np.column_stack([
        rng.uniform(10, 35, n),          # cpu_utilization
        rng.uniform(200, 500, n),         # memory_rss
        rng.uniform(50, 150, n),          # latency_p50
        rng.uniform(100, 300, n),         # latency_p95
        rng.uniform(150, 400, n),         # latency_p99
        rng.uniform(0, 0.5, n),           # error_rate
        rng.uniform(0.05, 0.35, n),       # connection_pool_saturation
    ])


def generate_incident_features(
    root_cause: str,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate synthetic feature vectors matching the REAL feature contract.
    The feature schema is: for each of MAX_SERVICE_SLOTS services,
    11 stat-suffixes per 7 metrics = 77 features per service,
    plus cross-correlations and log keyword counts.
    """
    n_features = get_feature_count()
    feature_names = get_training_feature_names()

    # Start with small Gaussian noise (baseline)
    X = rng.normal(0, 0.5, (n, n_features))

    # Build feature name → index map for targeted signal injection
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    def inject(pattern: str, values: np.ndarray) -> None:
        """Inject values into all features matching the pattern substring."""
        for name, idx in name_to_idx.items():
            if pattern in name:
                X[:, idx] += values

    # Service 0 is usually the "affected" service in training scenarios
    svc0 = TRAINING_SERVICE_NAMES[0]
    svc1 = TRAINING_SERVICE_NAMES[1]

    if root_cause == "memory_leak":
        # Strong memory_rss signals on svc_0
        inject(f"{svc0}__memory_rss__cur", rng.uniform(800, 2000, n))
        inject(f"{svc0}__memory_rss__slope5", rng.uniform(30, 120, n))
        inject(f"{svc0}__memory_rss__mk_z", rng.uniform(0.5, 1.0, n))
        inject(f"{svc0}__memory_rss__mk_p", rng.uniform(0.0, 0.05, n))
        inject(f"{svc0}__memory_rss__mean5", rng.uniform(600, 1500, n))
        inject(f"{svc0}__cpu_utilization__cur", rng.uniform(40, 80, n))
        inject(f"{svc0}__cpu_utilization__slope5", rng.uniform(1, 5, n))
        inject("log__oomkiller__count", rng.uniform(1, 5, n))
        # Normal metrics on other services
        inject(f"{svc1}__memory_rss__cur", rng.uniform(200, 500, n))
        inject(f"{svc1}__error_rate__cur", rng.uniform(0, 0.3, n))

    elif root_cause == "latency_spike":
        inject(f"{svc0}__latency_p95__cur", rng.uniform(500, 3000, n))
        inject(f"{svc0}__latency_p99__cur", rng.uniform(1000, 5000, n))
        inject(f"{svc0}__latency_p95__slope5", rng.uniform(50, 500, n))
        inject(f"{svc0}__latency_p50__cur", rng.uniform(200, 1000, n))
        inject(f"{svc0}__latency_p95__mk_z", rng.uniform(0.4, 0.9, n))
        # Moderate error rate from timeouts
        inject(f"{svc0}__error_rate__cur", rng.uniform(0.5, 3.0, n))

    elif root_cause == "cascading_failure":
        # Multiple services affected
        for svc in TRAINING_SERVICE_NAMES[:3]:
            inject(f"{svc}__error_rate__cur", rng.uniform(1.0, 5.0, n))
            inject(f"{svc}__error_rate__slope5", rng.uniform(0.1, 0.5, n))
        inject(f"{svc0}__connection_pool_saturation__cur", rng.uniform(0.7, 1.0, n))
        inject(f"{svc1}__connection_pool_saturation__cur", rng.uniform(0.5, 0.9, n))
        # Cross-service correlation signal
        inject("cross_corr__", rng.uniform(0.6, 1.0, n))
        inject("log__timeout__count", rng.uniform(2, 8, n))
        inject("log__cascade__count", rng.uniform(1, 4, n))

    elif root_cause == "config_drift":
        inject(f"{svc0}__latency_p95__cur", rng.uniform(300, 1200, n))
        inject(f"{svc0}__latency_p95__slope5", rng.uniform(10, 80, n))
        inject(f"{svc0}__error_rate__cur", rng.uniform(0.5, 3.0, n))
        inject(f"{svc0}__error_rate__slope5", rng.uniform(0.05, 0.3, n))
        inject(f"{svc0}__error_rate__mk_z", rng.uniform(0.3, 0.9, n))
        inject("log__config_version_mismatch__count", rng.uniform(1, 5, n))
        # Throughput degradation but not memory
        inject(f"{svc0}__memory_rss__cur", rng.uniform(200, 600, n))

    elif root_cause == "dependency_timeout":
        inject(f"{svc0}__connection_pool_saturation__cur", rng.uniform(0.7, 1.0, n))
        inject(f"{svc0}__connection_pool_saturation__slope5", rng.uniform(0.02, 0.08, n))
        inject(f"{svc0}__latency_p99__cur", rng.uniform(800, 4000, n))
        inject(f"{svc0}__error_rate__cur", rng.uniform(0.3, 2.0, n))
        # Upstream gets hit too
        inject(f"{svc1}__latency_p95__cur", rng.uniform(300, 1500, n))
        inject(f"{svc1}__error_rate__cur", rng.uniform(0.5, 3.0, n))
        inject("log__timeout__count", rng.uniform(3, 10, n))
        inject("log__pool__count", rng.uniform(1, 5, n))
        inject("cross_corr__", rng.uniform(0.5, 0.9, n))

    elif root_cause == "false_alarm":
        # Normal-ish with some noise — should look benign
        inject(f"{svc0}__cpu_utilization__cur", rng.uniform(50, 85, n))
        inject(f"{svc0}__memory_rss__cur", rng.uniform(200, 500, n))
        inject(f"{svc0}__error_rate__cur", rng.uniform(0, 0.3, n))
        inject(f"{svc1}__cpu_utilization__cur", rng.uniform(20, 40, n))

    return X


def build_labeled_dataset(rng: np.random.Generator):
    """Build (X, y) for XGBoost training."""
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for idx, cls in enumerate(ROOT_CAUSE_CLASSES):
        X_cls = generate_incident_features(cls, N_INCIDENT_SAMPLES_PER_CLASS, rng)
        y_cls = np.full(N_INCIDENT_SAMPLES_PER_CLASS, idx, dtype=int)
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Train IsolationForest
# ─────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(rng: np.random.Generator) -> None:
    logger.info("Training IsolationForest on %d normal-ops samples…", N_NORMAL_SAMPLES)
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        logger.warning("scikit-learn not installed; skipping IsolationForest training")
        return

    X_normal = generate_normal_ops(N_NORMAL_SAMPLES, rng)

    # Compute normalization stats for runtime use
    training_mean = X_normal.mean(axis=0)
    training_std = X_normal.std(axis=0) + 1e-9

    model = IsolationForest(
        n_estimators=200, contamination=0.05, random_state=SEED,
        n_jobs=1  # Prevent OpenMP SHM crashes
    )
    model.fit(X_normal)

    # Save model + normalization stats as a dict
    bundle = {
        "model": model,
        "mean": training_mean,
        "std": training_std,
    }
    out_path = SAVE_DIR / "isolation_forest.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Saved IsolationForest → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Train XGBoostClassifier
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(rng: np.random.Generator) -> None:
    n_features = get_feature_count()
    feature_names = get_training_feature_names()
    logger.info(
        "Training XGBoostClassifier on %d samples (%d features)…",
        len(ROOT_CAUSE_CLASSES) * N_INCIDENT_SAMPLES_PER_CLASS,
        n_features,
    )
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed; skipping XGBoost training")
        return

    X, y = build_labeled_dataset(rng)
    assert X.shape[1] == n_features, (
        f"Feature count mismatch: generated {X.shape[1]}, contract expects {n_features}"
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=1,  # Prevent OpenMP SHM crashes
        num_class=len(ROOT_CAUSE_CLASSES),
        objective="multi:softprob",
    )
    model.fit(X, y, verbose=False)

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "classes": ROOT_CAUSE_CLASSES,
    }
    out_path = SAVE_DIR / "xgboost_classifier.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Saved XGBoostClassifier → %s  (features=%d)", out_path, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train LSTM Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

def train_lstm(rng: np.random.Generator) -> None:
    logger.info("Training LSTM Autoencoder…")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("PyTorch not installed; skipping LSTM training")
        return

    # Limit PyTorch threads for stability
    torch.set_num_threads(1)

    # Generate normal-ops sequences: (N, seq_len=20, features=7)
    N_SEQ = 5_000
    SEQ_LEN = 20

    sequences = []
    for _ in range(N_SEQ):
        seq = generate_normal_ops(SEQ_LEN, rng)
        sequences.append(seq)
    X_seq = np.array(sequences, dtype=np.float32)  # (N, 20, 7)

    # Normalise per-feature
    mean = X_seq.mean(axis=(0, 1), keepdims=True)
    std = X_seq.std(axis=(0, 1), keepdims=True) + 1e-9
    X_norm = (X_seq - mean) / std

    tensor = torch.from_numpy(X_norm)
    dataset = TensorDataset(tensor, tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Model
    class LSTMAEncoder(nn.Module):
        def __init__(self, input_size=7, hidden_size=32):
            super().__init__()
            self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

        def forward(self, x):
            _, (h, _) = self.encoder(x)
            h_rep = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
            out, _ = self.decoder(h_rep)
            return out

    model = LSTMAEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    EPOCHS = 20
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            # Gradient clipping to prevent training instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.info("  LSTM Epoch %d/%d  loss=%.5f", epoch + 1, EPOCHS, total_loss / len(loader))

    out_path = SAVE_DIR / "lstm_autoencoder.pt"
    torch.save(model.state_dict(), out_path)
    logger.info("Saved LSTM → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    train_isolation_forest(rng)
    train_xgboost(rng)
    train_lstm(rng)
    logger.info("All models trained and saved to %s", SAVE_DIR)
