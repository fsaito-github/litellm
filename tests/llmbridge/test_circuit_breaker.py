"""Unit tests for litellm.proxy.hooks.circuit_breaker module."""

import threading
import time

import pytest

from litellm.proxy.hooks.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
)


# -------------------------------------------------------------------
# CircuitState Enum
# -------------------------------------------------------------------


class TestCircuitState:
    def test_enum_values(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_enum_member_count(self):
        assert len(CircuitState) == 3


# -------------------------------------------------------------------
# CircuitBreaker — state transitions
# -------------------------------------------------------------------


class TestCircuitBreakerState:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_closed_to_open_after_failure_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_below_threshold_stays_closed(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_open_blocks_requests(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_open_to_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.05)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.05)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.05)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


# -------------------------------------------------------------------
# CircuitBreaker — counters and thresholds
# -------------------------------------------------------------------


class TestCircuitBreakerCounters:
    def test_record_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2
        cb.record_success()
        assert cb.failure_count == 0

    def test_custom_thresholds(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 5
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_allow_request_in_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True

    def test_allow_request_in_half_open_limited(self):
        cb = CircuitBreaker(
            failure_threshold=2, recovery_timeout=0.05, half_open_max_calls=2
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        assert cb.allow_request() is True   # 1st probe
        assert cb.allow_request() is True   # 2nd probe
        assert cb.allow_request() is False  # exceeded max

    def test_get_status_snapshot(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        status = cb.get_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 5
        assert status["recovery_timeout"] == 30.0
        assert status["half_open_max_calls"] == 3

    def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


# -------------------------------------------------------------------
# CircuitBreakerManager
# -------------------------------------------------------------------


class TestCircuitBreakerManager:
    def test_get_or_create_returns_same_instance(self):
        mgr = CircuitBreakerManager()
        b1 = mgr.get_or_create("gpt-4")
        b2 = mgr.get_or_create("gpt-4")
        assert b1 is b2

    def test_get_or_create_different_models(self):
        mgr = CircuitBreakerManager()
        b1 = mgr.get_or_create("gpt-4")
        b2 = mgr.get_or_create("gpt-3.5")
        assert b1 is not b2

    def test_get_status_returns_all_breakers(self):
        mgr = CircuitBreakerManager()
        mgr.get_or_create("model-a")
        mgr.get_or_create("model-b")
        status = mgr.get_status()
        assert "model-a" in status
        assert "model-b" in status
        assert status["model-a"]["state"] == "closed"

    def test_reset_single_breaker(self):
        mgr = CircuitBreakerManager(default_failure_threshold=2)
        b = mgr.get_or_create("model-x")
        b.record_failure()
        b.record_failure()
        assert b.state == CircuitState.OPEN
        mgr.reset("model-x")
        assert b.state == CircuitState.CLOSED

    def test_reset_all(self):
        mgr = CircuitBreakerManager(default_failure_threshold=2)
        b1 = mgr.get_or_create("m1")
        b2 = mgr.get_or_create("m2")
        b1.record_failure()
        b1.record_failure()
        b2.record_failure()
        b2.record_failure()
        assert b1.state == CircuitState.OPEN
        assert b2.state == CircuitState.OPEN
        mgr.reset_all()
        assert b1.state == CircuitState.CLOSED
        assert b2.state == CircuitState.CLOSED

    def test_reset_nonexistent_is_noop(self):
        mgr = CircuitBreakerManager()
        mgr.reset("nonexistent")  # should not raise


# -------------------------------------------------------------------
# Thread safety
# -------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_circuit_breaker_access(self):
        cb = CircuitBreaker(failure_threshold=100, recovery_timeout=0.01)
        errors: list = []

        def worker():
            try:
                for _ in range(100):
                    cb.allow_request()
                    cb.record_failure()
                    cb.record_success()
                    cb.get_status()
                    _ = cb.state
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_manager_access(self):
        mgr = CircuitBreakerManager()
        errors: list = []

        def worker(model_id: str):
            try:
                b = mgr.get_or_create(model_id)
                for _ in range(50):
                    b.record_failure()
                    b.record_success()
                    b.allow_request()
                    mgr.get_status()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"model-{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Thread safety errors: {errors}"
