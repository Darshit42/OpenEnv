"""
Feature Contract — Single source of truth for the ML pipeline feature schema.

This module defines the EXACT feature names, order, and dimensionality used by
both the training pipeline (train_models.py) and the runtime pipeline (pipeline.py).

Any change to feature engineering MUST be reflected here.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Constants — MUST match pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "cpu_utilization", "memory_rss", "latency_p50",
    "latency_p95", "latency_p99", "error_rate",
    "connection_pool_saturation",
]

STAT_SUFFIXES = [
    "cur", "mean5", "std5", "mean15", "std15",
    "slope5", "lag1", "lag3", "lag5", "mk_z", "mk_p",
]

LOG_KEYWORDS = [
    "oomkiller", "timeout", "critical", "error", "warning",
    "config_version_mismatch", "pool", "cascade",
]

# Maximum number of service slots for the model.
# Scenarios with fewer services zero-pad; scenarios with more are truncated.
MAX_SERVICE_SLOTS = 6

# Canonical service slot names used during training.
TRAINING_SERVICE_NAMES = [f"svc_{i}" for i in range(MAX_SERVICE_SLOTS)]


# ─────────────────────────────────────────────────────────────────────────────
# Feature Name Generation
# ─────────────────────────────────────────────────────────────────────────────

def get_metric_feature_names(services: List[str]) -> List[str]:
    """Generate ordered feature names for metric-based features."""
    names = []
    for svc in services:
        for col in METRIC_COLS:
            for sfx in STAT_SUFFIXES:
                names.append(f"{svc}__{col}__{sfx}")
    return names


def get_cross_corr_feature_names(services: List[str]) -> List[str]:
    """Generate ordered feature names for cross-service correlation features."""
    names = []
    for i, svc_i in enumerate(services):
        for j, svc_j in enumerate(services):
            if i < j:
                names.append(f"cross_corr__{svc_i}__{svc_j}__err")
    return names


def get_log_feature_names() -> List[str]:
    """Generate ordered feature names for log-based features."""
    return [f"log__{kw}__count" for kw in LOG_KEYWORDS]


def get_all_feature_names(services: List[str]) -> List[str]:
    """
    Complete ordered feature name list for a given service set.
    This is the canonical schema for XGBoost input.
    """
    names = []
    names.extend(get_metric_feature_names(services))
    names.extend(get_cross_corr_feature_names(services))
    names.extend(get_log_feature_names())
    return names


def get_training_feature_names() -> List[str]:
    """Feature names used during model training (fixed service slots)."""
    return get_all_feature_names(TRAINING_SERVICE_NAMES)


def get_feature_count() -> int:
    """Total number of features in the training schema."""
    return len(get_training_feature_names())


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Mapping
# ─────────────────────────────────────────────────────────────────────────────

def map_runtime_features_to_training(
    runtime_features: Dict[str, float],
    runtime_services: List[str],
) -> Tuple[List[float], List[str], float]:
    """
    Map runtime feature dict (with real service names) to training feature vector.

    Returns:
        (feature_vector, training_feature_names, coverage_ratio)

    coverage_ratio: fraction of training features that had a non-zero runtime match.
    """
    training_names = get_training_feature_names()
    n_training = len(training_names)

    # Build slot mapping: runtime service -> training slot
    slot_map: Dict[str, str] = {}
    for idx, svc in enumerate(runtime_services[:MAX_SERVICE_SLOTS]):
        slot_map[svc] = TRAINING_SERVICE_NAMES[idx]

    # Remap runtime feature names to training names
    remapped: Dict[str, float] = {}

    for rt_name, value in runtime_features.items():
        mapped_name = rt_name
        for real_svc, train_svc in slot_map.items():
            if rt_name.startswith(f"{real_svc}__"):
                mapped_name = rt_name.replace(f"{real_svc}__", f"{train_svc}__", 1)
                break
            elif f"__{real_svc}__" in rt_name:
                mapped_name = rt_name.replace(f"__{real_svc}__", f"__{train_svc}__", 1)
                # Handle cross-corr with two service names
                for real_svc2, train_svc2 in slot_map.items():
                    if f"__{real_svc2}__" in mapped_name and real_svc2 != real_svc:
                        mapped_name = mapped_name.replace(
                            f"__{real_svc2}__", f"__{train_svc2}__", 1
                        )
                break
        remapped[mapped_name] = value

    # Build vector in training order
    vec = []
    matched = 0
    for name in training_names:
        val = remapped.get(name, 0.0)
        vec.append(val)
        if val != 0.0:
            matched += 1

    coverage = matched / max(1, n_training)
    return vec, training_names, coverage


def validate_features(features: Dict[str, float], services: List[str]) -> List[str]:
    """
    Validate that a runtime feature dict contains expected feature names.
    Returns list of missing feature names (empty if all present).
    """
    expected = set(get_all_feature_names(services))
    actual = set(features.keys())
    missing = expected - actual
    return sorted(missing)
