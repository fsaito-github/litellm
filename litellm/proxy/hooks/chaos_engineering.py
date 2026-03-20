"""
Chaos Engineering – fault injection for resilience testing of LLM proxy.

Provides:
    * ``ChaosExperiment``  – Pydantic model describing one experiment.
    * ``ChaosConfig``      – Pydantic configuration model.
    * ``ChaosEngine``      – manages experiments and decides when to inject faults.
"""

import random
import threading
import time
import traceback
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ChaosExperimentType(str, Enum):
    PROVIDER_FAILURE = "provider_failure"
    LATENCY_INJECTION = "latency_injection"
    RATE_LIMIT_SIM = "rate_limit_sim"
    BUDGET_EXHAUSTION = "budget_exhaustion"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChaosExperiment(BaseModel):
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: ChaosExperimentType
    target_model: str
    config: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 60.0
    is_active: bool = False


class ChaosConfig(BaseModel):
    enabled: bool = False
    max_concurrent_experiments: int = 5
    default_duration_seconds: float = 60.0
    safe_mode: bool = True


# ---------------------------------------------------------------------------
# Internal tracking
# ---------------------------------------------------------------------------


class _ExperimentState:
    """Mutable runtime state for an active experiment."""

    __slots__ = (
        "started_at",
        "requests_total",
        "requests_faulted",
        "rate_limit_counter",
    )

    def __init__(self) -> None:
        self.started_at: float = time.time()
        self.requests_total: int = 0
        self.requests_faulted: int = 0
        self.rate_limit_counter: int = 0


# ---------------------------------------------------------------------------
# ChaosEngine
# ---------------------------------------------------------------------------


class ChaosEngine:
    """Manages chaos experiments and fault injection logic."""

    def __init__(self, config: Optional[ChaosConfig] = None) -> None:
        self.config = config or ChaosConfig()
        self._experiments: Dict[str, ChaosExperiment] = {}
        self._states: Dict[str, _ExperimentState] = {}
        self._lock = threading.Lock()
        verbose_proxy_logger.info("ChaosEngine initialized (enabled=%s)", self.config.enabled)

    # -- experiment lifecycle -----------------------------------------------

    def create_experiment(self, experiment: ChaosExperiment) -> None:
        with self._lock:
            self._experiments[experiment.experiment_id] = experiment
            verbose_proxy_logger.debug(
                "chaos_engineering.py::create_experiment(): id=%s name=%s",
                experiment.experiment_id,
                experiment.name,
            )

    def start_experiment(self, experiment_id: str) -> None:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if exp is None:
                verbose_proxy_logger.warning(
                    "chaos_engineering.py::start_experiment(): unknown id=%s",
                    experiment_id,
                )
                return

            active_count = sum(
                1 for e in self._experiments.values() if e.is_active
            )
            if active_count >= self.config.max_concurrent_experiments:
                verbose_proxy_logger.warning(
                    "chaos_engineering.py::start_experiment(): max concurrent "
                    "experiments reached (%d)",
                    self.config.max_concurrent_experiments,
                )
                return

            exp.is_active = True
            self._states[experiment_id] = _ExperimentState()
            verbose_proxy_logger.info(
                "chaos_engineering.py::start_experiment(): started id=%s",
                experiment_id,
            )

    def stop_experiment(self, experiment_id: str) -> None:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if exp is None:
                verbose_proxy_logger.warning(
                    "chaos_engineering.py::stop_experiment(): unknown id=%s",
                    experiment_id,
                )
                return
            exp.is_active = False
            verbose_proxy_logger.info(
                "chaos_engineering.py::stop_experiment(): stopped id=%s",
                experiment_id,
            )

    # -- fault injection ----------------------------------------------------

    def should_inject_fault(self, model: str) -> Optional[Dict[str, Any]]:
        """Check active experiments and return a fault config if applicable."""
        if not self.config.enabled:
            return None

        with self._lock:
            now = time.time()
            for eid, exp in self._experiments.items():
                if not exp.is_active:
                    continue
                if exp.target_model != model and exp.target_model != "*":
                    continue

                state = self._states.get(eid)
                if state is None:
                    continue

                # Auto-expire experiments that exceeded their duration
                if now - state.started_at > exp.duration_seconds:
                    exp.is_active = False
                    verbose_proxy_logger.info(
                        "chaos_engineering.py::should_inject_fault(): "
                        "experiment %s expired after %.0fs",
                        eid,
                        exp.duration_seconds,
                    )
                    continue

                state.requests_total += 1
                fault = self._evaluate_fault(exp, state)
                if fault is not None:
                    state.requests_faulted += 1
                    return fault

        return None

    def _evaluate_fault(
        self, exp: ChaosExperiment, state: _ExperimentState
    ) -> Optional[Dict[str, Any]]:
        """Return a fault descriptor or ``None``."""
        try:
            if exp.type == ChaosExperimentType.PROVIDER_FAILURE:
                failure_pct = exp.config.get("failure_percentage", 50)
                if random.randint(1, 100) <= failure_pct:
                    return {
                        "type": "provider_failure",
                        "status_code": 500,
                        "message": f"Chaos: simulated provider failure (experiment={exp.name})",
                    }

            elif exp.type == ChaosExperimentType.LATENCY_INJECTION:
                delay_ms = exp.config.get("delay_ms", 2000)
                return {
                    "type": "latency_injection",
                    "delay_ms": delay_ms,
                    "message": f"Chaos: injecting {delay_ms}ms latency (experiment={exp.name})",
                }

            elif exp.type == ChaosExperimentType.RATE_LIMIT_SIM:
                max_requests = exp.config.get("max_requests", 10)
                state.rate_limit_counter += 1
                if state.rate_limit_counter > max_requests:
                    return {
                        "type": "rate_limit_sim",
                        "status_code": 429,
                        "message": (
                            f"Chaos: simulated rate limit after "
                            f"{max_requests} requests (experiment={exp.name})"
                        ),
                        "retry_after": exp.config.get("retry_after", 30),
                    }

            elif exp.type == ChaosExperimentType.BUDGET_EXHAUSTION:
                return {
                    "type": "budget_exhaustion",
                    "status_code": 402,
                    "message": f"Chaos: simulated budget exhaustion (experiment={exp.name})",
                }
        except Exception as e:
            verbose_proxy_logger.exception(
                "chaos_engineering.py::_evaluate_fault(): Exception occurred - %s",
                str(e),
            )

        return None

    # -- results ------------------------------------------------------------

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """Return aggregated results for an experiment."""
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if exp is None:
                return {"error": f"experiment {experiment_id} not found"}

            state = self._states.get(experiment_id)
            if state is None:
                return {
                    "experiment_id": experiment_id,
                    "name": exp.name,
                    "status": "not_started",
                }

            elapsed = time.time() - state.started_at
            return {
                "experiment_id": experiment_id,
                "name": exp.name,
                "type": exp.type.value,
                "target_model": exp.target_model,
                "is_active": exp.is_active,
                "elapsed_seconds": round(elapsed, 2),
                "duration_seconds": exp.duration_seconds,
                "requests_total": state.requests_total,
                "requests_faulted": state.requests_faulted,
                "fault_rate": (
                    round(state.requests_faulted / state.requests_total, 4)
                    if state.requests_total > 0
                    else 0.0
                ),
            }


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[ChaosEngine] = None
_engine_lock = threading.Lock()


def get_chaos_engine(config: Optional[ChaosConfig] = None) -> ChaosEngine:
    """Return (or lazily create) the module-level singleton."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = ChaosEngine(config=config)
        return _engine_instance
