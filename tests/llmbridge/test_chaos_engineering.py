"""Tests for litellm.proxy.hooks.chaos_engineering – pure-logic unit tests."""

import pytest

from litellm.proxy.hooks.chaos_engineering import (
    ChaosConfig,
    ChaosEngine,
    ChaosExperiment,
    ChaosExperimentType,
)


# ---------------------------------------------------------------------------
# ChaosExperiment model
# ---------------------------------------------------------------------------


class TestChaosExperiment:
    def test_defaults(self):
        exp = ChaosExperiment(
            name="test",
            type=ChaosExperimentType.PROVIDER_FAILURE,
            target_model="gpt-4",
        )
        assert exp.name == "test"
        assert exp.type == ChaosExperimentType.PROVIDER_FAILURE
        assert exp.target_model == "gpt-4"
        assert exp.config == {}
        assert exp.duration_seconds == 60.0
        assert exp.is_active is False
        assert exp.experiment_id  # auto-generated UUID

    def test_custom_config(self):
        exp = ChaosExperiment(
            name="my-exp",
            type=ChaosExperimentType.LATENCY_INJECTION,
            target_model="*",
            config={"delay_ms": 5000},
            duration_seconds=120.0,
        )
        assert exp.config["delay_ms"] == 5000
        assert exp.duration_seconds == 120.0

    def test_unique_ids(self):
        e1 = ChaosExperiment(name="a", type=ChaosExperimentType.PROVIDER_FAILURE, target_model="m")
        e2 = ChaosExperiment(name="b", type=ChaosExperimentType.PROVIDER_FAILURE, target_model="m")
        assert e1.experiment_id != e2.experiment_id


# ---------------------------------------------------------------------------
# ChaosConfig model
# ---------------------------------------------------------------------------


class TestChaosConfig:
    def test_defaults(self):
        cfg = ChaosConfig()
        assert cfg.enabled is False
        assert cfg.max_concurrent_experiments == 5
        assert cfg.default_duration_seconds == 60.0
        assert cfg.safe_mode is True

    def test_custom(self):
        cfg = ChaosConfig(enabled=True, max_concurrent_experiments=3)
        assert cfg.enabled is True
        assert cfg.max_concurrent_experiments == 3


# ---------------------------------------------------------------------------
# ChaosEngine lifecycle
# ---------------------------------------------------------------------------


class TestChaosEngineLifecycle:
    def _make_engine(self, **kwargs) -> ChaosEngine:
        return ChaosEngine(config=ChaosConfig(enabled=True, **kwargs))

    def _make_experiment(self, **kwargs) -> ChaosExperiment:
        defaults = dict(
            name="test-exp",
            type=ChaosExperimentType.PROVIDER_FAILURE,
            target_model="gpt-4",
        )
        defaults.update(kwargs)
        return ChaosExperiment(**defaults)

    def test_create_experiment(self):
        engine = self._make_engine()
        exp = self._make_experiment()
        engine.create_experiment(exp)
        assert exp.experiment_id in engine._experiments

    def test_start_stop_lifecycle(self):
        engine = self._make_engine()
        exp = self._make_experiment()
        engine.create_experiment(exp)

        assert exp.is_active is False
        engine.start_experiment(exp.experiment_id)
        assert exp.is_active is True

        engine.stop_experiment(exp.experiment_id)
        assert exp.is_active is False

    def test_start_unknown_experiment_does_nothing(self):
        engine = self._make_engine()
        engine.start_experiment("nonexistent")  # should not raise

    def test_stop_unknown_experiment_does_nothing(self):
        engine = self._make_engine()
        engine.stop_experiment("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# should_inject_fault
# ---------------------------------------------------------------------------


class TestShouldInjectFault:
    def _make_engine(self, **kwargs) -> ChaosEngine:
        return ChaosEngine(config=ChaosConfig(enabled=True, **kwargs))

    def test_returns_none_when_no_active_experiments(self):
        engine = self._make_engine()
        assert engine.should_inject_fault("gpt-4") is None

    def test_returns_none_when_disabled(self):
        engine = ChaosEngine(config=ChaosConfig(enabled=False))
        exp = ChaosExperiment(
            name="x",
            type=ChaosExperimentType.PROVIDER_FAILURE,
            target_model="gpt-4",
        )
        engine.create_experiment(exp)
        engine.start_experiment(exp.experiment_id)
        assert engine.should_inject_fault("gpt-4") is None

    def test_returns_config_when_active_targets_model(self):
        engine = self._make_engine()
        exp = ChaosExperiment(
            name="lat",
            type=ChaosExperimentType.LATENCY_INJECTION,
            target_model="gpt-4",
            config={"delay_ms": 1000},
        )
        engine.create_experiment(exp)
        engine.start_experiment(exp.experiment_id)
        fault = engine.should_inject_fault("gpt-4")
        assert fault is not None
        assert fault["type"] == "latency_injection"
        assert fault["delay_ms"] == 1000

    def test_wildcard_target_matches_any_model(self):
        engine = self._make_engine()
        exp = ChaosExperiment(
            name="lat",
            type=ChaosExperimentType.LATENCY_INJECTION,
            target_model="*",
            config={"delay_ms": 500},
        )
        engine.create_experiment(exp)
        engine.start_experiment(exp.experiment_id)
        fault = engine.should_inject_fault("claude-3-opus")
        assert fault is not None
        assert fault["type"] == "latency_injection"

    def test_no_match_for_different_model(self):
        engine = self._make_engine()
        exp = ChaosExperiment(
            name="lat",
            type=ChaosExperimentType.LATENCY_INJECTION,
            target_model="gpt-4",
        )
        engine.create_experiment(exp)
        engine.start_experiment(exp.experiment_id)
        assert engine.should_inject_fault("claude-3") is None


# ---------------------------------------------------------------------------
# Fault types
# ---------------------------------------------------------------------------


class TestFaultTypes:
    def _active_engine(self, exp_type, config=None, target="gpt-4"):
        engine = ChaosEngine(config=ChaosConfig(enabled=True))
        exp = ChaosExperiment(
            name="ft",
            type=exp_type,
            target_model=target,
            config=config or {},
        )
        engine.create_experiment(exp)
        engine.start_experiment(exp.experiment_id)
        return engine

    def test_provider_failure(self):
        engine = self._active_engine(
            ChaosExperimentType.PROVIDER_FAILURE,
            config={"failure_percentage": 100},
        )
        fault = engine.should_inject_fault("gpt-4")
        assert fault is not None
        assert fault["type"] == "provider_failure"
        assert fault["status_code"] == 500

    def test_provider_failure_zero_pct(self):
        engine = self._active_engine(
            ChaosExperimentType.PROVIDER_FAILURE,
            config={"failure_percentage": 0},
        )
        fault = engine.should_inject_fault("gpt-4")
        assert fault is None

    def test_latency_injection(self):
        engine = self._active_engine(
            ChaosExperimentType.LATENCY_INJECTION,
            config={"delay_ms": 3000},
        )
        fault = engine.should_inject_fault("gpt-4")
        assert fault is not None
        assert fault["type"] == "latency_injection"
        assert fault["delay_ms"] == 3000

    def test_rate_limit_sim(self):
        engine = self._active_engine(
            ChaosExperimentType.RATE_LIMIT_SIM,
            config={"max_requests": 2, "retry_after": 10},
        )
        # First 2 requests under limit → no fault
        assert engine.should_inject_fault("gpt-4") is None
        assert engine.should_inject_fault("gpt-4") is None
        # Third request triggers rate limit
        fault = engine.should_inject_fault("gpt-4")
        assert fault is not None
        assert fault["type"] == "rate_limit_sim"
        assert fault["status_code"] == 429
        assert fault["retry_after"] == 10

    def test_budget_exhaustion(self):
        engine = self._active_engine(ChaosExperimentType.BUDGET_EXHAUSTION)
        fault = engine.should_inject_fault("gpt-4")
        assert fault is not None
        assert fault["type"] == "budget_exhaustion"
        assert fault["status_code"] == 402


# ---------------------------------------------------------------------------
# Multiple experiments
# ---------------------------------------------------------------------------


class TestMultipleExperiments:
    def test_only_active_experiments_apply(self):
        engine = ChaosEngine(config=ChaosConfig(enabled=True))

        active = ChaosExperiment(
            name="active",
            type=ChaosExperimentType.LATENCY_INJECTION,
            target_model="gpt-4",
            config={"delay_ms": 100},
        )
        inactive = ChaosExperiment(
            name="inactive",
            type=ChaosExperimentType.BUDGET_EXHAUSTION,
            target_model="gpt-4",
        )

        engine.create_experiment(active)
        engine.create_experiment(inactive)
        engine.start_experiment(active.experiment_id)
        # inactive is never started

        fault = engine.should_inject_fault("gpt-4")
        assert fault is not None
        assert fault["type"] == "latency_injection"


# ---------------------------------------------------------------------------
# get_results
# ---------------------------------------------------------------------------


class TestGetResults:
    def test_returns_experiment_data(self):
        engine = ChaosEngine(config=ChaosConfig(enabled=True))
        exp = ChaosExperiment(
            name="res",
            type=ChaosExperimentType.LATENCY_INJECTION,
            target_model="gpt-4",
            config={"delay_ms": 100},
        )
        engine.create_experiment(exp)
        engine.start_experiment(exp.experiment_id)
        engine.should_inject_fault("gpt-4")  # 1 request

        results = engine.get_results(exp.experiment_id)
        assert results["experiment_id"] == exp.experiment_id
        assert results["name"] == "res"
        assert results["is_active"] is True
        assert results["requests_total"] == 1
        assert results["requests_faulted"] == 1

    def test_unknown_experiment(self):
        engine = ChaosEngine(config=ChaosConfig(enabled=True))
        results = engine.get_results("nope")
        assert "error" in results

    def test_not_started_experiment(self):
        engine = ChaosEngine(config=ChaosConfig(enabled=True))
        exp = ChaosExperiment(
            name="ns",
            type=ChaosExperimentType.LATENCY_INJECTION,
            target_model="gpt-4",
        )
        engine.create_experiment(exp)
        results = engine.get_results(exp.experiment_id)
        assert results["status"] == "not_started"
