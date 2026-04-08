"""
Procedural failure scenario generators for the OpenEnv SRE environment.

Each scenario generates metrics, logs, and alerts step-by-step, simulating
a distinct failure taxonomy. All randomness is seeded for reproducibility.
"""
from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from openenv.models import Alert, LogEntry, MetricSnapshot


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseScenario(ABC):
    """Abstract base for all failure scenarios."""

    ground_truth_root_cause: str = ""
    ground_truth_service: str = ""
    all_services: List[str] = []
    affected_services: List[str] = []

    # Harmful actions that should never be taken (for grader)
    lethal_services: List[str] = []

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self._start_time = datetime(2024, 1, 15, 9, 0, 0)
        self._current_step = 0
        self._setup()

    def _setup(self) -> None:
        """Override to initialise scenario-specific parameters."""
        pass

    def get_metrics(self, step: int) -> Dict[str, MetricSnapshot]:
        self._current_step = step
        return self._generate_metrics(step)

    def get_logs(self, step: int) -> List[LogEntry]:
        self._current_step = step
        return self._generate_logs(step)

    def get_alerts(self, step: int) -> List[Alert]:
        self._current_step = step
        return self._generate_alerts(step)

    @abstractmethod
    def _generate_metrics(self, step: int) -> Dict[str, MetricSnapshot]:
        ...

    @abstractmethod
    def _generate_logs(self, step: int) -> List[LogEntry]:
        ...

    @abstractmethod
    def _generate_alerts(self, step: int) -> List[Alert]:
        ...

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _ts(self, step: int) -> str:
        return (self._start_time + timedelta(minutes=step)).isoformat() + "Z"

    def _log(self, step: int, service: str, severity: str, message: str) -> LogEntry:
        return LogEntry(
            timestamp=self._ts(step),
            service_id=service,
            severity=severity,
            message=message,
            trace_id=str(uuid.uuid4()),
        )

    def _jitter(self, value: float, pct: float = 0.05) -> float:
        """Add ±pct% Gaussian noise."""
        return float(value + self.rng.normal(0, value * pct))

    def _normal_snapshot(
        self,
        cpu: float = 20.0,
        mem: float = 350.0,
        p50: float = 80.0,
        p95: float = 180.0,
        p99: float = 280.0,
        err: float = 0.1,
        pool: float = 0.2,
    ) -> MetricSnapshot:
        return MetricSnapshot(
            cpu_utilization=max(0.0, min(100.0, self._jitter(cpu, 0.1))),
            memory_rss=max(0.0, self._jitter(mem, 0.05)),
            latency_p50=max(1.0, self._jitter(p50, 0.08)),
            latency_p95=max(1.0, self._jitter(p95, 0.08)),
            latency_p99=max(1.0, self._jitter(p99, 0.08)),
            error_rate=max(0.0, self._jitter(err, 0.15)),
            connection_pool_saturation=max(0.0, min(1.0, self._jitter(pool, 0.1))),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Memory Leak (Easy)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryLeakScenario(BaseScenario):
    """
    Task 1 — Easy.
    Service 'api-server' has a monotonic RSS memory leak (~45 MB/step).
    Clear signal from step 5 onward. Causal DAG: api-server is a root node.
    No misleading alerts.
    """

    ground_truth_root_cause = "memory_leak"
    ground_truth_service = "api-server"

    def _setup(self) -> None:
        self.all_services = ["web-app", "api-server", "database"]
        self.affected_services = ["api-server"]
        self.lethal_services = ["database"]  # Do NOT restart the DB
        self._base_rss = 256.0  # MB at step 0
        self._rss_leak_rate = 45.0  # MB per step
        self._oom_ceiling = 2048.0  # 2 GB ceiling

    def _rss_at(self, step: int) -> float:
        return self._base_rss + step * self._rss_leak_rate + float(self.rng.normal(0, 4))

    def _generate_metrics(self, step: int) -> Dict[str, MetricSnapshot]:
        metrics: Dict[str, MetricSnapshot] = {}

        # web-app — healthy
        metrics["web-app"] = self._normal_snapshot(cpu=18, mem=310, p50=90, p95=170, p99=260, err=0.08)

        # api-server — leaking
        rss = self._rss_at(step)
        saturation_ratio = rss / self._oom_ceiling
        cpu = min(100.0, 20 + step * 1.8 + float(self.rng.normal(0, 2)))
        lat_factor = 1.0 + max(0.0, (step - 10) * 0.12)
        err = max(0.0, float(self.rng.uniform(0, 0.3)) + max(0.0, (step - 15) * 0.2))

        metrics["api-server"] = MetricSnapshot(
            cpu_utilization=cpu,
            memory_rss=rss,
            latency_p50=max(1.0, self._jitter(90 * lat_factor, 0.05)),
            latency_p95=max(1.0, self._jitter(180 * lat_factor, 0.05)),
            latency_p99=max(1.0, self._jitter(280 * lat_factor, 0.05)),
            error_rate=err,
            connection_pool_saturation=min(1.0, 0.2 + step * 0.025),
        )

        # database — mostly normal, slight pressure after step 15
        db_lat_delta = max(0.0, (step - 15) * 2.5)
        metrics["database"] = MetricSnapshot(
            cpu_utilization=self._jitter(28, 0.1),
            memory_rss=self._jitter(540, 0.04),
            latency_p50=max(1.0, self._jitter(12, 0.08)),
            latency_p95=max(1.0, 40 + db_lat_delta + float(self.rng.normal(0, 3))),
            latency_p99=max(1.0, 80 + db_lat_delta * 1.5 + float(self.rng.normal(0, 5))),
            error_rate=max(0.0, self._jitter(0.05, 0.2)),
            connection_pool_saturation=min(1.0, 0.28 + max(0.0, (step - 15) * 0.015)),
        )

        return metrics

    def _generate_logs(self, step: int) -> List[LogEntry]:
        logs: List[LogEntry] = []
        logs.append(self._log(step, "web-app", "INFO", "Request batch processed successfully"))
        logs.append(self._log(step, "api-server", "INFO", "Handling inbound request queue"))

        if step >= 8:
            rss = self._rss_at(step)
            logs.append(self._log(step, "api-server", "WARNING",
                f"High memory utilization detected: {rss:.0f} MB — monitoring closely"))

        if step >= 15:
            logs.append(self._log(step, "api-server", "ERROR",
                "OOMKiller: process memory approaching 75% of ceiling (1536 MB / 2048 MB)"))

        if step >= 22:
            logs.append(self._log(step, "api-server", "CRITICAL",
                "OOMKiller: process flagged for termination — memory RSS critical, GC loops detected"))

        return logs

    def _generate_alerts(self, step: int) -> List[Alert]:
        alerts: List[Alert] = []
        if step >= 5:
            rss = self._rss_at(step)
            alerts.append(Alert(
                alert_id=f"alert-mem-apiserver-{step}",
                source_service="api-server",
                alert_type="HIGH_MEMORY_RSS",
                current_value=round(rss, 1),
                threshold=1024.0,
                severity="warning" if step < 12 else "critical",
                is_red_herring=False,
            ))
        return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Dependency Timeout Cascade (Medium)
# ─────────────────────────────────────────────────────────────────────────────

class DependencyFailureScenario(BaseScenario):
    """
    Task 2 — Medium.
    'data-layer' is the root cause (dependency_timeout), but 'api-gateway'
    has a higher anomaly score — the naive-agent trap.
    Drift begins at step 0, alert fires at step 18.
    One red-herring high-severity CPU alert on 'background-worker'.
    """

    ground_truth_root_cause = "dependency_timeout"
    ground_truth_service = "data-layer"

    def _setup(self) -> None:
        self.all_services = ["api-gateway", "auth-service", "data-layer", "background-worker"]
        self.affected_services = ["data-layer", "api-gateway", "auth-service"]
        self.lethal_services = ["api-gateway"]  # Restarting GW makes things worse
        self._threshold_step = 14

    def _generate_metrics(self, step: int) -> Dict[str, MetricSnapshot]:
        metrics: Dict[str, MetricSnapshot] = {}

        # data-layer — root cause: connection pool filling up
        dl_factor = 1.0 + step * 0.09
        metrics["data-layer"] = MetricSnapshot(
            cpu_utilization=min(100.0, self._jitter(42 + step * 0.6, 0.05)),
            memory_rss=self._jitter(860 + step * 6, 0.03),
            latency_p50=max(1.0, self._jitter(60 * dl_factor, 0.06)),
            latency_p95=max(1.0, self._jitter(230 * dl_factor, 0.06)),
            latency_p99=max(1.0, self._jitter(600 * dl_factor, 0.07)),
            error_rate=max(0.0, step * 0.14 + float(self.rng.uniform(0, 0.25))),
            connection_pool_saturation=min(1.0, 0.28 + step * 0.038),
        )

        # api-gateway — downstream: degrades faster (higher anomaly score — the trap)
        gw_factor = 1.0 + step * 0.14
        metrics["api-gateway"] = MetricSnapshot(
            cpu_utilization=min(100.0, self._jitter(55 + step * 0.9, 0.05)),
            memory_rss=self._jitter(650 + step * 9, 0.04),
            latency_p50=max(1.0, self._jitter(110 * gw_factor, 0.07)),
            latency_p95=max(1.0, self._jitter(480 * gw_factor, 0.07)),
            latency_p99=max(1.0, self._jitter(900 * gw_factor, 0.08)),
            error_rate=max(0.0, step * 0.22 + float(self.rng.uniform(0, 0.4))),
            connection_pool_saturation=min(1.0, 0.38 + step * 0.046),
        )

        # auth-service — tertiary degradation
        auth_factor = 1.0 + step * 0.07
        metrics["auth-service"] = MetricSnapshot(
            cpu_utilization=min(100.0, self._jitter(35 + step * 0.45, 0.06)),
            memory_rss=self._jitter(440 + step * 3.5, 0.04),
            latency_p50=max(1.0, self._jitter(90 * auth_factor, 0.06)),
            latency_p95=max(1.0, self._jitter(280 * auth_factor, 0.07)),
            latency_p99=max(1.0, self._jitter(480 * auth_factor, 0.07)),
            error_rate=max(0.0, step * 0.09 + float(self.rng.uniform(0, 0.18))),
            connection_pool_saturation=min(1.0, 0.22 + step * 0.028),
        )

        # background-worker — completely healthy, high CPU (red herring)
        metrics["background-worker"] = MetricSnapshot(
            cpu_utilization=min(100.0, self._jitter(78, 0.06)),
            memory_rss=self._jitter(220, 0.04),
            latency_p50=max(1.0, self._jitter(15, 0.1)),
            latency_p95=max(1.0, self._jitter(55, 0.1)),
            latency_p99=max(1.0, self._jitter(110, 0.1)),
            error_rate=max(0.0, self._jitter(0.05, 0.2)),
            connection_pool_saturation=self._jitter(0.10, 0.12),
        )

        return metrics

    def _generate_logs(self, step: int) -> List[LogEntry]:
        logs: List[LogEntry] = []

        if step >= 3:
            sat = min(100, 28 + step * 4)
            logs.append(self._log(step, "data-layer", "WARNING",
                f"Connection pool at {sat}% saturation — watch for upstream impact"))

        if step >= 8:
            logs.append(self._log(step, "api-gateway", "WARNING",
                "Upstream data-layer response time exceeding SLO; applying retry backoff"))
            logs.append(self._log(step, "auth-service", "WARNING",
                "Slow responses from data-layer dependency; auth token validation delayed"))

        if step >= 15:
            logs.append(self._log(step, "data-layer", "ERROR",
                "Connection pool EXHAUSTED — requests queuing, new connections refused"))
            logs.append(self._log(step, "api-gateway", "ERROR",
                "Upstream dependency timeout: data-layer not responding within 5000ms"))

        return logs

    def _generate_alerts(self, step: int) -> List[Alert]:
        alerts: List[Alert] = []

        # Red herring: background-worker high CPU — fires intermittently from step 5
        if step >= 5 and step % 3 == 0:
            alerts.append(Alert(
                alert_id="alert-cpu-background-worker",
                source_service="background-worker",
                alert_type="HIGH_CPU_UTILIZATION",
                current_value=round(78 + float(self.rng.uniform(0, 8)), 1),
                threshold=80.0,
                severity="info",
                is_red_herring=True,
            ))

        # Real alert — fires late
        if step >= self._threshold_step:
            alerts.append(Alert(
                alert_id=f"alert-pool-datalayer-{step}",
                source_service="data-layer",
                alert_type="CONNECTION_POOL_EXHAUSTED",
                current_value=round(min(1.0, 0.28 + step * 0.038), 3),
                threshold=0.80,
                severity="critical",
                is_red_herring=False,
            ))

        return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Config Drift + Counterfactual Trap (Hard)
# ─────────────────────────────────────────────────────────────────────────────

class ConfigDriftScenario(BaseScenario):
    """
    Task 3 — Hard.
    Config version mismatch on 'origin-service' silently degrades throughput
    across 5 services for 18 steps before any alert fires.

    LETHAL TRAP: restarting 'cache' causes write-loss cascade (harm_flag=True,
    predicted_resolution_probability=0.12). Correct action: patch config on origin-service.

    Two misleading critical alerts on 'background-monitor' and 'database' (red herrings).
    """

    ground_truth_root_cause = "config_drift"
    ground_truth_service = "origin-service"

    def _setup(self) -> None:
        self.all_services = [
            "frontend", "api-gateway", "cache", "database",
            "origin-service", "background-monitor",
        ]
        self.affected_services = ["origin-service", "api-gateway", "cache", "frontend"]
        # cache restart → write-loss cascade; database restart → data corruption
        self.lethal_services = ["cache", "database"]
        self._threshold_step = 16
        self._config_mismatch_version = "2.1.4"
        self._expected_config_version = "2.1.7"

    def _generate_metrics(self, step: int) -> Dict[str, MetricSnapshot]:
        metrics: Dict[str, MetricSnapshot] = {}
        drift = min(1.0, step * 0.042)  # 0→1 over ~24 steps

        # origin-service — config drift root node
        metrics["origin-service"] = MetricSnapshot(
            cpu_utilization=min(100.0, self._jitter(28 + drift * 38, 0.05)),
            memory_rss=self._jitter(460 + drift * 220, 0.04),
            latency_p50=max(1.0, self._jitter(70 * (1 + drift * 3.2), 0.06)),
            latency_p95=max(1.0, self._jitter(200 * (1 + drift * 4.0), 0.06)),
            latency_p99=max(1.0, self._jitter(380 * (1 + drift * 5.0), 0.07)),
            error_rate=max(0.0, drift * 2.8 + float(self.rng.uniform(0, 0.25))),
            connection_pool_saturation=min(1.0, 0.18 + drift * 0.62),
        )

        # cache — downstream impact; high anomaly score (part of the trap)
        cache_impact = drift * 0.72
        metrics["cache"] = MetricSnapshot(
            cpu_utilization=min(100.0, self._jitter(38 + cache_impact * 28, 0.05)),
            memory_rss=self._jitter(2200 + cache_impact * 600, 0.03),
            latency_p50=max(1.0, self._jitter(8 * (1 + cache_impact * 2.2), 0.07)),
            latency_p95=max(1.0, self._jitter(35 * (1 + cache_impact * 3.0), 0.07)),
            latency_p99=max(1.0, self._jitter(75 * (1 + cache_impact * 3.8), 0.08)),
            error_rate=max(0.0, cache_impact * 1.6 + float(self.rng.uniform(0, 0.18))),
            connection_pool_saturation=min(1.0, 0.25 + cache_impact * 0.52),
        )

        # api-gateway — secondary degradation
        gw_impact = drift * 0.52
        metrics["api-gateway"] = MetricSnapshot(
            cpu_utilization=min(100.0, self._jitter(42 + gw_impact * 22, 0.05)),
            memory_rss=self._jitter(580 + gw_impact * 110, 0.04),
            latency_p50=max(1.0, self._jitter(110 * (1 + gw_impact * 2.0), 0.06)),
            latency_p95=max(1.0, self._jitter(380 * (1 + gw_impact * 2.5), 0.06)),
            latency_p99=max(1.0, self._jitter(720 * (1 + gw_impact * 3.0), 0.07)),
            error_rate=max(0.0, gw_impact * 1.0 + float(self.rng.uniform(0, 0.18))),
            connection_pool_saturation=min(1.0, 0.22 + gw_impact * 0.42),
        )

        # frontend — mildly affected (user-facing latency)
        fe_impact = drift * 0.35
        metrics["frontend"] = MetricSnapshot(
            cpu_utilization=min(100.0, self._jitter(18 + fe_impact * 14, 0.07)),
            memory_rss=self._jitter(245, 0.04),
            latency_p50=max(1.0, self._jitter(230 * (1 + fe_impact * 1.6), 0.07)),
            latency_p95=max(1.0, self._jitter(620 * (1 + fe_impact * 2.0), 0.07)),
            latency_p99=max(1.0, self._jitter(1100 * (1 + fe_impact * 2.4), 0.08)),
            error_rate=max(0.0, fe_impact * 0.5 + float(self.rng.uniform(0, 0.08))),
            connection_pool_saturation=self._jitter(0.14, 0.1),
        )

        # database — healthy (red herring alert will fire on this)
        metrics["database"] = MetricSnapshot(
            cpu_utilization=self._jitter(26, 0.08),
            memory_rss=self._jitter(1100, 0.03),
            latency_p50=max(1.0, self._jitter(12, 0.08)),
            latency_p95=max(1.0, self._jitter(42, 0.08)),
            latency_p99=max(1.0, self._jitter(90, 0.09)),
            error_rate=max(0.0, self._jitter(0.04, 0.2)),
            connection_pool_saturation=self._jitter(0.20, 0.08),
        )

        # background-monitor — completely healthy (red herring alert fires here)
        metrics["background-monitor"] = MetricSnapshot(
            cpu_utilization=self._jitter(14, 0.1),
            memory_rss=self._jitter(120, 0.05),
            latency_p50=max(1.0, self._jitter(8, 0.1)),
            latency_p95=max(1.0, self._jitter(22, 0.1)),
            latency_p99=max(1.0, self._jitter(45, 0.1)),
            error_rate=max(0.0, self._jitter(0.02, 0.2)),
            connection_pool_saturation=self._jitter(0.07, 0.1),
        )

        return metrics

    def _generate_logs(self, step: int) -> List[LogEntry]:
        logs: List[LogEntry] = []

        if step >= 2:
            logs.append(self._log(step, "origin-service", "DEBUG",
                f"Config version: {self._config_mismatch_version} "
                f"(expected: {self._expected_config_version}) — config_version_mismatch_flag=True"))

        if step >= 5:
            logs.append(self._log(step, "origin-service", "WARNING",
                f"Throughput degradation detected — possible config misalignment "
                f"(version {self._config_mismatch_version} vs expected {self._expected_config_version})"))

        if step >= 10:
            logs.append(self._log(step, "cache", "WARNING",
                "Cache invalidation storm increasing — upstream config mismatch causing key stampede"))

        if step >= 15:
            logs.append(self._log(step, "api-gateway", "ERROR",
                "origin-service SLO breach: p95 latency > 2000ms; downstream impact spreading"))

        return logs

    def _generate_alerts(self, step: int) -> List[Alert]:
        alerts: List[Alert] = []

        # Red herring #1: background-monitor disk I/O (warning, not critical)
        # Fires intermittently from step 6
        if step >= 6 and step % 4 == 0:
            alerts.append(Alert(
                alert_id="alert-disk-bgmon",
                source_service="background-monitor",
                alert_type="HIGH_DISK_IO",
                current_value=87.0,
                threshold=80.0,
                severity="warning",
                is_red_herring=True,
            ))

        # Red herring #2: database network saturation (warning, fires from step 10)
        if step >= 10 and step % 5 == 0:
            alerts.append(Alert(
                alert_id="alert-net-database",
                source_service="database",
                alert_type="NETWORK_SATURATION",
                current_value=79.5,
                threshold=75.0,
                severity="warning",
                is_red_herring=True,
            ))

        # Real alert — fires late at step 19
        if step >= self._threshold_step:
            drift = min(1.0, step * 0.042)
            alerts.append(Alert(
                alert_id=f"alert-config-origin-{step}",
                source_service="origin-service",
                alert_type="CONFIG_VERSION_MISMATCH",
                current_value=round(drift, 3),
                threshold=0.70,
                severity="critical",
                is_red_herring=False,
            ))

        return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_REGISTRY: Dict[int, type] = {
    1: MemoryLeakScenario,
    2: DependencyFailureScenario,
    3: ConfigDriftScenario,
}


def create_scenario(task_id: int, seed: int = 42) -> BaseScenario:
    """Instantiate the appropriate scenario for a given task ID."""
    if task_id not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown task_id={task_id}. Valid: {list(SCENARIO_REGISTRY.keys())}")
    return SCENARIO_REGISTRY[task_id](seed=seed)
