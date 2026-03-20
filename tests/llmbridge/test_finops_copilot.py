"""Tests for litellm.proxy.hooks.finops_copilot – pure-logic unit tests."""

import pytest

from litellm.proxy.hooks.finops_copilot import (
    FinOpsCopilot,
    FinOpsCopilotConfig,
    FinOpsRecommendation,
    RecommendationType,
)


# ---------------------------------------------------------------------------
# FinOpsRecommendation model
# ---------------------------------------------------------------------------


class TestFinOpsRecommendation:
    def test_defaults(self):
        rec = FinOpsRecommendation(
            type=RecommendationType.MODEL_DOWNGRADE,
            title="Downgrade gpt-4",
            description="Most requests are small.",
        )
        assert rec.recommendation_id  # auto-generated
        assert rec.type == RecommendationType.MODEL_DOWNGRADE
        assert rec.estimated_savings_monthly == 0.0
        assert rec.confidence == 0.0
        assert rec.affected_entities == []

    def test_custom_values(self):
        rec = FinOpsRecommendation(
            type=RecommendationType.BUDGET_ALERT,
            title="Budget exceeded",
            description="Team X at 90%",
            estimated_savings_monthly=500.0,
            confidence=0.95,
            affected_entities=["team-x"],
        )
        assert rec.estimated_savings_monthly == 500.0
        assert rec.confidence == 0.95
        assert rec.affected_entities == ["team-x"]


# ---------------------------------------------------------------------------
# FinOpsCopilotConfig model
# ---------------------------------------------------------------------------


class TestFinOpsCopilotConfig:
    def test_defaults(self):
        cfg = FinOpsCopilotConfig()
        assert cfg.enabled is False
        assert cfg.downgrade_input_token_threshold == 100
        assert cfg.downgrade_request_pct_threshold == 50.0
        assert cfg.cache_duplicate_pct_threshold == 20.0
        assert cfg.spend_growth_pct_threshold == 30.0
        assert cfg.budget_usage_pct_threshold == 80.0
        assert "gpt-4" in cfg.expensive_models
        assert isinstance(cfg.cheap_alternative_models, dict)

    def test_custom(self):
        cfg = FinOpsCopilotConfig(
            enabled=True,
            downgrade_input_token_threshold=200,
        )
        assert cfg.enabled is True
        assert cfg.downgrade_input_token_threshold == 200


# ---------------------------------------------------------------------------
# FinOpsCopilot.get_recommendations
# ---------------------------------------------------------------------------


class TestFinOpsCopilotRecommendations:
    def test_get_recommendations_returns_list(self):
        copilot = FinOpsCopilot()
        recs = copilot.get_recommendations()
        assert isinstance(recs, list)

    def test_no_recommendations_without_data(self):
        copilot = FinOpsCopilot()
        recs = copilot.analyze_spend()
        assert isinstance(recs, list)
        # Without any recorded data, there shouldn't be recommendations
        assert len(recs) == 0

    def test_model_downgrade_recommendation(self):
        copilot = FinOpsCopilot(config=FinOpsCopilotConfig(
            downgrade_input_token_threshold=100,
            downgrade_request_pct_threshold=50.0,
        ))
        # Record many requests with small input tokens on an expensive model
        for _ in range(100):
            copilot.record_request(
                model="gpt-4",
                input_tokens=50,
                cost=0.01,
            )
        recs = copilot.analyze_spend()
        downgrade_recs = [r for r in recs if r.type == RecommendationType.MODEL_DOWNGRADE]
        assert len(downgrade_recs) > 0
        assert downgrade_recs[0].estimated_savings_monthly > 0

    def test_budget_alert_recommendation(self):
        copilot = FinOpsCopilot(config=FinOpsCopilotConfig(
            budget_usage_pct_threshold=80.0,
        ))
        copilot.set_entity_budget("team-a", 100.0)
        # Record spend that exceeds 80% of budget
        copilot.record_request(
            model="gpt-4",
            input_tokens=100,
            cost=85.0,
            entity_id="team-a",
        )
        recs = copilot.analyze_spend()
        alerts = [r for r in recs if r.type == RecommendationType.BUDGET_ALERT]
        assert len(alerts) > 0
