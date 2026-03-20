"""Unit tests for litellm.proxy.hooks.cost_router module."""

from unittest.mock import patch

import pytest

from litellm.proxy.hooks.cost_router import (
    CostRouter,
    CostRouterConfig,
    QueryComplexityClassifier,
)


# -------------------------------------------------------------------
# QueryComplexityClassifier
# -------------------------------------------------------------------


class TestQueryComplexityClassifier:
    @pytest.fixture
    def classifier(self):
        return QueryComplexityClassifier()

    def test_simple_query_greeting(self, classifier):
        """'Hello, how are you?' matches the greeting pattern → simple."""
        msgs = [{"role": "user", "content": "Hello, how are you?"}]
        assert classifier.classify(msgs) == "simple"

    def test_complex_query_with_code_block(self, classifier):
        """Code block (2) + math keyword (1) + reasoning keyword (1) = 4 → complex."""
        msgs = [
            {
                "role": "user",
                "content": (
                    "```python\n"
                    "def solve(x):\n"
                    "    return x ** 2\n"
                    "```\n"
                    "Please analyze this algorithm step by step"
                ),
            }
        ]
        with patch(
            "litellm.proxy.hooks.cost_router._count_tokens", return_value=50
        ):
            assert classifier.classify(msgs) == "complex"

    def test_moderate_query(self, classifier):
        """Inline code (1) + math keyword (1) = 2 → moderate."""
        msgs = [
            {
                "role": "user",
                "content": "Can you help me calculate the `factorial` of 10?",
            }
        ]
        with patch(
            "litellm.proxy.hooks.cost_router._count_tokens", return_value=50
        ):
            assert classifier.classify(msgs) == "moderate"

    def test_empty_messages_simple(self, classifier):
        """Empty message list produces score 0 → simple."""
        with patch(
            "litellm.proxy.hooks.cost_router._count_tokens", return_value=0
        ):
            assert classifier.classify([]) == "simple"

    def test_long_context_complex(self, classifier):
        """Token count >500 (3) + reasoning keyword (1) = 4 → complex."""
        msgs = [
            {
                "role": "user",
                "content": "Please analyze " + "word " * 100,
            }
        ]
        with patch(
            "litellm.proxy.hooks.cost_router._count_tokens", return_value=600
        ):
            assert classifier.classify(msgs) == "complex"

    def test_short_non_greeting_is_simple(self, classifier):
        """A short question with no special signals scores ≤1 → simple."""
        msgs = [{"role": "user", "content": "What time is it in Tokyo?"}]
        with patch(
            "litellm.proxy.hooks.cost_router._count_tokens", return_value=10
        ):
            assert classifier.classify(msgs) == "simple"


# -------------------------------------------------------------------
# CostRouter
# -------------------------------------------------------------------


class TestCostRouter:
    @pytest.fixture
    def router(self):
        return CostRouter(
            model_tiers={
                "simple": ["gpt-4o-mini"],
                "moderate": ["gpt-4o"],
                "complex": ["gpt-4"],
            },
        )

    def test_route_simple_to_cheap_tier(self, router):
        msgs = [{"role": "user", "content": "Hello!"}]
        result = router.route(msgs, default_model="gpt-4")
        assert result == "gpt-4o-mini"

    def test_route_complex_to_premium_tier(self, router):
        msgs = [
            {
                "role": "user",
                "content": (
                    "```python\ndef solve(x): pass\n```\n"
                    "Analyze step by step"
                ),
            }
        ]
        with patch(
            "litellm.proxy.hooks.cost_router._count_tokens", return_value=50
        ):
            result = router.route(msgs, default_model="gpt-4o-mini")
        assert result == "gpt-4"

    def test_default_model_when_no_tier_match(self):
        """Tier 'complex' has no configured models → fallback to default."""
        router = CostRouter(model_tiers={"simple": ["gpt-4o-mini"]})
        msgs = [
            {
                "role": "user",
                "content": (
                    "```python\ndef solve(x): pass\n```\n"
                    "Analyze step by step"
                ),
            }
        ]
        with patch(
            "litellm.proxy.hooks.cost_router._count_tokens", return_value=50
        ):
            result = router.route(msgs, default_model="gpt-4-fallback")
        assert result == "gpt-4-fallback"

    def test_stats_tracking(self, router):
        msgs = [{"role": "user", "content": "Hello!"}]
        router.route(msgs, default_model="gpt-4")
        router.route(msgs, default_model="gpt-4")
        stats = router.get_stats()
        assert stats["total_requests"] == 2
        assert stats["requests_per_tier"]["simple"] == 2

    def test_stats_initial_zero(self, router):
        stats = router.get_stats()
        assert stats["total_requests"] == 0
        assert stats["requests_per_tier"]["simple"] == 0
        assert stats["requests_per_tier"]["moderate"] == 0
        assert stats["requests_per_tier"]["complex"] == 0


# -------------------------------------------------------------------
# CostRouterConfig
# -------------------------------------------------------------------


class TestCostRouterConfig:
    def test_default_values(self):
        config = CostRouterConfig()
        assert config.enabled is False
        assert config.model_tiers == {}
        assert config.simple_token_threshold == 100
        assert config.complex_token_threshold == 500

    def test_custom_values(self):
        config = CostRouterConfig(
            enabled=True,
            model_tiers={"simple": ["gpt-4o-mini"]},
            simple_token_threshold=50,
            complex_token_threshold=300,
        )
        assert config.enabled is True
        assert config.model_tiers == {"simple": ["gpt-4o-mini"]}
        assert config.simple_token_threshold == 50
        assert config.complex_token_threshold == 300
