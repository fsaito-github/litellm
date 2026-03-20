"""
Cost Router — Intelligent cost-aware routing for LiteLLM.

Routes queries to cheaper models when the request is simple, reserving
expensive models for complex tasks.  Inspired by RouteLLM's approach of
achieving ~85 % cost reduction on typical workloads.

Usage example (standalone)::

    from litellm.proxy.hooks.cost_router import CostRouter

    router = CostRouter(
        model_tiers={
            "simple":   ["gpt-4o-mini"],
            "moderate": ["gpt-4o"],
            "complex":  ["gpt-4o", "claude-sonnet-4-20250514"],
        },
    )
    model = router.route(
        messages=[{"role": "user", "content": "Hello!"}],
        default_model="gpt-4o",
    )
    print(model)   # -> "gpt-4o-mini"
    print(router.get_stats())

Usage example (with proxy config)::

    # In litellm proxy config YAML:
    litellm_settings:
      cost_router:
        enabled: true
        model_tiers:
          simple:   ["gpt-4o-mini"]
          moderate: ["gpt-4o"]
          complex:  ["gpt-4o", "claude-sonnet-4-20250514"]
        simple_token_threshold: 100
        complex_token_threshold: 500
"""

from __future__ import annotations

import random
import re
import threading
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

Complexity = Literal["simple", "moderate", "complex"]


class CostRouterConfig(BaseModel):
    """Pydantic model for the cost router configuration.

    Attributes:
        enabled: Whether cost-aware routing is active.
        model_tiers: Mapping of complexity tier to a list of model names.
            Example::

                {
                    "simple":   ["gpt-4o-mini"],
                    "moderate": ["gpt-4o"],
                    "complex":  ["gpt-4o", "claude-sonnet-4-20250514"],
                }

        simple_token_threshold: Messages with token count below this value
            are classified as *simple* (default ``100``).
        complex_token_threshold: Messages with token count above this value
            are classified as *complex* (default ``500``).
    """

    enabled: bool = False
    model_tiers: Dict[str, List[str]] = Field(default_factory=dict)
    simple_token_threshold: int = 100
    complex_token_threshold: int = 500


# ---------------------------------------------------------------------------
# Heuristic complexity signals
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_MATH_KEYWORDS = re.compile(
    r"\b(solve|equation|integral|derivative|proof|theorem|calculate|compute"
    r"|algorithm|matrix|vector|probability|regression|optimize)\b",
    re.IGNORECASE,
)
_REASONING_KEYWORDS = re.compile(
    r"\b(step[- ]by[- ]step|explain|compare|analyze|evaluate|reason"
    r"|pros?\s+and\s+cons?|trade[- ]?offs?|advantages?\s+and\s+disadvantages?)\b",
    re.IGNORECASE,
)
_SIMPLE_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|yes|no|ok|okay|sure|got it"
    r"|good morning|good afternoon|good evening|bye|goodbye"
    r"|what is your name|how are you|who are you)\b",
    re.IGNORECASE,
)


def _extract_text(messages: List[Dict[str, Any]]) -> str:
    """Concatenate all user / system message text into a single string."""
    parts: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Multi-modal messages: extract text parts only
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
    return "\n".join(parts)


def _count_tokens(text: str, messages: Optional[List[Dict[str, Any]]] = None) -> int:
    """Return an estimated token count.

    Tries ``litellm.token_counter`` first; falls back to ``word_count / 0.75``.
    """
    try:
        import litellm

        if messages is not None:
            return litellm.token_counter(messages=messages)  # type: ignore[arg-type]
        return litellm.token_counter(text=text)
    except Exception:
        # Fallback: rough approximation
        return int(len(text.split()) / 0.75)


def _estimate_model_cost(model: str) -> Optional[float]:
    """Return the sum of input + output cost-per-token for *model*, or ``None``."""
    try:
        import litellm

        info = litellm.model_cost.get(model)
        if info:
            return (
                info.get("input_cost_per_token", 0.0)
                + info.get("output_cost_per_token", 0.0)
            )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# QueryComplexityClassifier
# ---------------------------------------------------------------------------


class QueryComplexityClassifier:
    """Heuristic-based classifier that buckets a chat message list into
    ``"simple"``, ``"moderate"``, or ``"complex"``.

    The heuristics intentionally avoid ML dependencies so that the module
    can be used out-of-the-box.  The classifier can be swapped for an ML
    model later by subclassing and overriding :meth:`classify`.

    Args:
        simple_token_threshold: Upper token bound for *simple* (default 100).
        complex_token_threshold: Lower token bound for *complex* (default 500).
    """

    def __init__(
        self,
        simple_token_threshold: int = 100,
        complex_token_threshold: int = 500,
    ) -> None:
        self.simple_token_threshold = simple_token_threshold
        self.complex_token_threshold = complex_token_threshold

    # ------------------------------------------------------------------

    def classify(self, messages: List[Dict[str, Any]]) -> Complexity:
        """Classify the complexity of *messages*.

        Returns:
            ``"simple"``, ``"moderate"``, or ``"complex"``.
        """
        text = _extract_text(messages)
        token_count = _count_tokens(text, messages)

        score = self._score(messages, text, token_count)

        if score <= 1:
            return "simple"
        if score >= 4:
            return "complex"
        return "moderate"

    # ------------------------------------------------------------------

    def _score(
        self,
        messages: List[Dict[str, Any]],
        text: str,
        token_count: int,
    ) -> int:
        """Compute a 0–6 complexity score from independent heuristic signals."""
        score = 0

        # --- Token length ---
        if token_count > self.complex_token_threshold:
            score += 3
        elif token_count > self.simple_token_threshold:
            score += 1

        # --- Trivial / greeting patterns ---
        user_msgs = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        if user_msgs and _SIMPLE_PATTERNS.match(user_msgs[-1].strip()):
            return 0  # short-circuit: definitely simple

        # --- Number of conversation turns ---
        if len(user_msgs) > 3:
            score += 1

        # --- Code blocks ---
        code_blocks = _CODE_BLOCK_RE.findall(text)
        inline_code = _INLINE_CODE_RE.findall(text)
        if code_blocks:
            score += 2
        elif inline_code:
            score += 1

        # --- Math / logic keywords ---
        if _MATH_KEYWORDS.search(text):
            score += 1

        # --- Multi-step reasoning keywords ---
        if _REASONING_KEYWORDS.search(text):
            score += 1

        # --- Complex system prompt ---
        system_msgs = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "system" and isinstance(m.get("content"), str)
        ]
        if system_msgs:
            sys_tokens = _count_tokens("\n".join(system_msgs))
            if sys_tokens > self.complex_token_threshold:
                score += 2
            elif sys_tokens > self.simple_token_threshold:
                score += 1

        return score


# ---------------------------------------------------------------------------
# CostRouter
# ---------------------------------------------------------------------------


class CostRouter:
    """Intelligent cost-aware router.

    Routes each request to the cheapest model tier that can handle its
    complexity.

    Args:
        model_tiers: Maps complexity tier names (``"simple"``,
            ``"moderate"``, ``"complex"``) to lists of model names.
        thresholds: Optional dict with keys ``simple_token_threshold``
            and ``complex_token_threshold`` forwarded to the classifier.

    Example::

        router = CostRouter(
            model_tiers={
                "simple":   ["gpt-4o-mini"],
                "moderate": ["gpt-4o"],
                "complex":  ["gpt-4o", "claude-sonnet-4-20250514"],
            },
        )
        model = router.route(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            default_model="gpt-4o",
        )
    """

    def __init__(
        self,
        model_tiers: Dict[str, List[str]],
        thresholds: Optional[Dict[str, int]] = None,
    ) -> None:
        self.model_tiers = model_tiers
        thresholds = thresholds or {}
        self.classifier = QueryComplexityClassifier(
            simple_token_threshold=thresholds.get("simple_token_threshold", 100),
            complex_token_threshold=thresholds.get("complex_token_threshold", 500),
        )

        # --- Routing statistics (thread-safe) ---
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._requests_per_tier: Dict[str, int] = {
            "simple": 0,
            "moderate": 0,
            "complex": 0,
        }
        self._estimated_cost_savings: float = 0.0

        verbose_proxy_logger.info(
            "CostRouter initialised with tiers: %s", list(model_tiers.keys())
        )

    # ------------------------------------------------------------------

    def route(
        self,
        messages: List[Dict[str, Any]],
        default_model: str,
    ) -> str:
        """Select the best model for *messages* based on complexity.

        Args:
            messages: The chat messages (OpenAI format).
            default_model: Fallback model when the tier has no mapping.

        Returns:
            The model name to use.
        """
        tier: Complexity = self.classifier.classify(messages)
        candidates = self.model_tiers.get(tier)

        if not candidates:
            verbose_proxy_logger.debug(
                "CostRouter: no models configured for tier '%s', "
                "falling back to default_model='%s'",
                tier,
                default_model,
            )
            selected = default_model
        else:
            # Pick one at random (basic load-spread within a tier)
            selected = random.choice(candidates)

        # --- Update stats ---
        savings = self._estimate_savings(selected, default_model)
        with self._lock:
            self._total_requests += 1
            self._requests_per_tier[tier] = self._requests_per_tier.get(tier, 0) + 1
            self._estimated_cost_savings += savings

        verbose_proxy_logger.debug(
            "CostRouter: tier=%s, selected=%s (default=%s)",
            tier,
            selected,
            default_model,
        )
        return selected

    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of routing statistics.

        Returns:
            A dict with ``total_requests``, ``requests_per_tier``, and
            ``estimated_cost_savings``.

        Example return value::

            {
                "total_requests": 150,
                "requests_per_tier": {"simple": 90, "moderate": 40, "complex": 20},
                "estimated_cost_savings": 0.0042,
            }
        """
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "requests_per_tier": dict(self._requests_per_tier),
                "estimated_cost_savings": round(self._estimated_cost_savings, 6),
            }

    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_savings(selected_model: str, default_model: str) -> float:
        """Return a rough per-token cost delta between *default_model* and
        *selected_model*.  Positive means money saved."""
        default_cost = _estimate_model_cost(default_model)
        selected_cost = _estimate_model_cost(selected_model)
        if default_cost is not None and selected_cost is not None:
            return max(default_cost - selected_cost, 0.0)
        return 0.0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

cost_router_instance: Optional[CostRouter] = None


def initialize_cost_router(config: CostRouterConfig) -> Optional[CostRouter]:
    """Create (or replace) the module-level :data:`cost_router_instance`.

    Args:
        config: A :class:`CostRouterConfig` instance.

    Returns:
        The new :class:`CostRouter`, or ``None`` if ``config.enabled`` is
        ``False``.

    Example::

        from litellm.proxy.hooks.cost_router import (
            CostRouterConfig,
            initialize_cost_router,
            cost_router_instance,
        )

        cfg = CostRouterConfig(
            enabled=True,
            model_tiers={
                "simple":   ["gpt-4o-mini"],
                "moderate": ["gpt-4o"],
                "complex":  ["gpt-4o"],
            },
        )
        initialize_cost_router(cfg)
        # cost_router_instance is now set
    """
    global cost_router_instance

    if not config.enabled:
        verbose_proxy_logger.info("CostRouter is disabled by configuration.")
        cost_router_instance = None
        return None

    if not config.model_tiers:
        verbose_proxy_logger.warning(
            "CostRouter enabled but model_tiers is empty — routing will "
            "always fall back to the default model."
        )

    cost_router_instance = CostRouter(
        model_tiers=config.model_tiers,
        thresholds={
            "simple_token_threshold": config.simple_token_threshold,
            "complex_token_threshold": config.complex_token_threshold,
        },
    )
    verbose_proxy_logger.info("CostRouter singleton initialised successfully.")
    return cost_router_instance
