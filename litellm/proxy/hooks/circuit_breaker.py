"""
Circuit Breaker Hook

Implements the circuit breaker pattern for LLM provider calls to prevent
cascading failures when a provider/model becomes unhealthy:

1. CLOSED  — requests flow normally; failures are counted
2. OPEN    — requests are blocked; the breaker waits for recovery_timeout
3. HALF_OPEN — a limited number of probe requests are allowed through

## Usage

    from litellm.proxy.hooks.circuit_breaker import circuit_breaker_manager

    breaker = circuit_breaker_manager.get_or_create("gpt-4")
    if breaker.allow_request():
        try:
            response = await call_provider(...)
            breaker.record_success()
        except Exception:
            breaker.record_failure()
    else:
        raise HTTPException(status_code=503, detail="Circuit breaker open")
"""

import threading
import time
from enum import Enum
from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger


class CircuitState(str, Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Thread-safe circuit breaker for a single provider/model.

    Transitions:
        CLOSED  →  OPEN       when failure_count >= failure_threshold
        OPEN    →  HALF_OPEN  when recovery_timeout has elapsed
        HALF_OPEN → CLOSED    when a success is recorded
        HALF_OPEN → OPEN      when a failure is recorded or max probe calls exceeded
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """
        Args:
            failure_threshold: Number of consecutive failures before opening.
            recovery_timeout: Seconds to wait in OPEN state before probing.
            half_open_max_calls: Max probe requests allowed in HALF_OPEN state.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._lock = threading.Lock()
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count_in_half_open: int = 0
        self._half_open_call_count: int = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state (may trigger OPEN→HALF_OPEN)."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    @property
    def last_failure_time(self) -> Optional[float]:
        with self._lock:
            return self._last_failure_time

    @property
    def success_count_in_half_open(self) -> int:
        with self._lock:
            return self._success_count_in_half_open

    def allow_request(self) -> bool:
        """
        Check whether a request should be allowed through.

        Returns:
            True if the request is permitted, False otherwise.
        """
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_call_count < self.half_open_max_calls:
                    self._half_open_call_count += 1
                    return True
                return False

            # OPEN
            return False

    def record_success(self) -> None:
        """Record a successful call. Resets the breaker when in HALF_OPEN."""
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count_in_half_open += 1
                verbose_proxy_logger.debug(
                    "CircuitBreaker: success in HALF_OPEN "
                    f"({self._success_count_in_half_open}/{self.half_open_max_calls})"
                )
                # A single success in half-open is enough to close
                self._transition(CircuitState.CLOSED)
                return

            # CLOSED — just reset failure count
            if self._failure_count > 0:
                verbose_proxy_logger.debug(
                    "CircuitBreaker: success in CLOSED, resetting failure count"
                )
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call. Opens the breaker when the threshold is reached."""
        with self._lock:
            self._maybe_transition_to_half_open()
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                verbose_proxy_logger.debug(
                    "CircuitBreaker: failure in HALF_OPEN, re-opening"
                )
                self._transition(CircuitState.OPEN)
                return

            # CLOSED
            self._failure_count += 1
            verbose_proxy_logger.debug(
                f"CircuitBreaker: failure count {self._failure_count}/{self.failure_threshold}"
            )
            if self._failure_count >= self.failure_threshold:
                verbose_proxy_logger.debug(
                    "CircuitBreaker: threshold reached, opening circuit"
                )
                self._transition(CircuitState.OPEN)

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        with self._lock:
            self._transition(CircuitState.CLOSED)

    def get_status(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the breaker state."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "half_open_max_calls": self.half_open_max_calls,
                "last_failure_time": self._last_failure_time,
                "success_count_in_half_open": self._success_count_in_half_open,
            }

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold self._lock)
    # ------------------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        if self._state != CircuitState.OPEN:
            return
        if self._last_failure_time is None:
            return
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self.recovery_timeout:
            verbose_proxy_logger.debug(
                f"CircuitBreaker: recovery timeout elapsed ({elapsed:.1f}s), "
                "transitioning OPEN → HALF_OPEN"
            )
            self._transition(CircuitState.HALF_OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state
        self._last_state_change_time = time.monotonic()

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count_in_half_open = 0
            self._half_open_call_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count_in_half_open = 0
            self._half_open_call_count = 0

        if old_state != new_state:
            verbose_proxy_logger.debug(
                f"CircuitBreaker: {old_state.value} → {new_state.value}"
            )


class CircuitBreakerManager:
    """
    Manages per-model circuit breakers.

    Thread-safe registry that lazily creates a ``CircuitBreaker`` for each
    model_id on first access.
    """

    def __init__(
        self,
        default_failure_threshold: int = 5,
        default_recovery_timeout: float = 30.0,
        default_half_open_max_calls: int = 3,
    ):
        self._lock = threading.Lock()
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_failure_threshold = default_failure_threshold
        self._default_recovery_timeout = default_recovery_timeout
        self._default_half_open_max_calls = default_half_open_max_calls

    def get_or_create(self, model_id: str) -> CircuitBreaker:
        """
        Return the circuit breaker for *model_id*, creating one if needed.

        Args:
            model_id: Unique identifier for the provider/model deployment.

        Returns:
            The ``CircuitBreaker`` instance for this model.
        """
        with self._lock:
            if model_id not in self._breakers:
                verbose_proxy_logger.debug(
                    f"CircuitBreakerManager: creating breaker for '{model_id}'"
                )
                self._breakers[model_id] = CircuitBreaker(
                    failure_threshold=self._default_failure_threshold,
                    recovery_timeout=self._default_recovery_timeout,
                    half_open_max_calls=self._default_half_open_max_calls,
                )
            return self._breakers[model_id]

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Return the status of every registered breaker.

        Returns:
            Mapping of model_id → breaker status dict.
        """
        with self._lock:
            return {
                model_id: breaker.get_status()
                for model_id, breaker in self._breakers.items()
            }

    def reset(self, model_id: str) -> None:
        """
        Reset the breaker for a single model back to CLOSED.

        Args:
            model_id: The model whose breaker should be reset.
        """
        with self._lock:
            breaker = self._breakers.get(model_id)
        if breaker is not None:
            breaker.reset()
            verbose_proxy_logger.debug(
                f"CircuitBreakerManager: reset breaker for '{model_id}'"
            )

    def reset_all(self) -> None:
        """Reset every registered breaker back to CLOSED."""
        with self._lock:
            breakers = list(self._breakers.values())
        for breaker in breakers:
            breaker.reset()
        verbose_proxy_logger.debug("CircuitBreakerManager: all breakers reset")


# Module-level singleton
circuit_breaker_manager = CircuitBreakerManager()
