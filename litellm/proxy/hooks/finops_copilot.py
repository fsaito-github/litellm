"""
AI FinOps Copilot – cost-optimisation recommendations for LLM spend.

Provides:
    * ``FinOpsRecommendation``  – Pydantic model for a single recommendation.
    * ``FinOpsCopilotConfig``   – Pydantic configuration model.
    * ``FinOpsCopilot``         – rule-based analysis engine producing
      actionable cost-saving recommendations.
"""

import threading
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RecommendationType(str, Enum):
    MODEL_DOWNGRADE = "model_downgrade"
    CACHE_OPPORTUNITY = "cache_opportunity"
    BUDGET_ALERT = "budget_alert"
    ANOMALY = "anomaly"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FinOpsRecommendation(BaseModel):
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: RecommendationType
    title: str
    description: str
    estimated_savings_monthly: float = 0.0
    confidence: float = 0.0
    affected_entities: List[str] = Field(default_factory=list)


class FinOpsCopilotConfig(BaseModel):
    enabled: bool = False
    downgrade_input_token_threshold: int = 100
    downgrade_request_pct_threshold: float = 50.0
    cache_duplicate_pct_threshold: float = 20.0
    spend_growth_pct_threshold: float = 30.0
    budget_usage_pct_threshold: float = 80.0
    expensive_models: List[str] = Field(
        default_factory=lambda: [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "claude-3-opus",
            "claude-3.5-sonnet",
        ]
    )
    cheap_alternative_models: Dict[str, str] = Field(
        default_factory=lambda: {
            "gpt-4": "gpt-3.5-turbo",
            "gpt-4-turbo": "gpt-3.5-turbo",
            "gpt-4o": "gpt-4o-mini",
            "claude-3-opus": "claude-3-haiku",
            "claude-3.5-sonnet": "claude-3-haiku",
        }
    )


# ---------------------------------------------------------------------------
# Internal types for spend tracking
# ---------------------------------------------------------------------------


class _ModelRequestStats(BaseModel):
    model: str
    total_requests: int = 0
    total_cost: float = 0.0
    total_input_tokens: int = 0
    requests_under_threshold: int = 0


class _SpendSnapshot(BaseModel):
    period: str
    total_cost: float = 0.0
    cost_by_model: Dict[str, float] = Field(default_factory=dict)
    cost_by_entity: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# FinOpsCopilot
# ---------------------------------------------------------------------------


class FinOpsCopilot:
    """Rule-based FinOps recommendation engine."""

    def __init__(self, config: Optional[FinOpsCopilotConfig] = None) -> None:
        self.config = config or FinOpsCopilotConfig()

        # Mutable tracking state
        self._model_stats: Dict[str, _ModelRequestStats] = {}
        self._spend_history: List[_SpendSnapshot] = []
        self._entity_budgets: Dict[str, float] = {}
        self._entity_spend: Dict[str, float] = {}
        self._request_hashes: Dict[str, int] = {}
        self._total_requests: int = 0
        self._recommendations: List[FinOpsRecommendation] = []
        self._lock = threading.Lock()

        verbose_proxy_logger.info(
            "FinOpsCopilot initialized (enabled=%s)", self.config.enabled
        )

    # -- data ingestion (to be called by hooks) -----------------------------

    def record_request(
        self,
        model: str,
        input_tokens: int,
        cost: float,
        entity_id: Optional[str] = None,
        request_hash: Optional[str] = None,
    ) -> None:
        """Record a completed LLM request for analysis."""
        with self._lock:
            self._total_requests += 1

            stats = self._model_stats.get(model)
            if stats is None:
                stats = _ModelRequestStats(model=model)
                self._model_stats[model] = stats

            stats.total_requests += 1
            stats.total_cost += cost
            stats.total_input_tokens += input_tokens

            if input_tokens < self.config.downgrade_input_token_threshold:
                stats.requests_under_threshold += 1

            if entity_id:
                self._entity_spend[entity_id] = (
                    self._entity_spend.get(entity_id, 0.0) + cost
                )

            if request_hash:
                self._request_hashes[request_hash] = (
                    self._request_hashes.get(request_hash, 0) + 1
                )

    def set_entity_budget(self, entity_id: str, budget: float) -> None:
        with self._lock:
            self._entity_budgets[entity_id] = budget

    def record_spend_snapshot(self, snapshot: _SpendSnapshot) -> None:
        with self._lock:
            self._spend_history.append(snapshot)

    # -- analysis -----------------------------------------------------------

    def analyze_spend(
        self, org_id: Optional[str] = None
    ) -> List[FinOpsRecommendation]:
        """Run all rule-based analyses and return fresh recommendations."""
        recommendations: List[FinOpsRecommendation] = []

        with self._lock:
            recommendations.extend(self._check_model_downgrade())
            recommendations.extend(self._check_cache_opportunity())
            recommendations.extend(self._check_spend_growth())
            recommendations.extend(self._check_budget_alerts())

            self._recommendations = recommendations

        verbose_proxy_logger.debug(
            "finops_copilot.py::analyze_spend(): generated %d recommendations",
            len(recommendations),
        )
        return recommendations

    def get_recommendations(self) -> List[FinOpsRecommendation]:
        with self._lock:
            return list(self._recommendations)

    def estimate_savings(self, recommendation: FinOpsRecommendation) -> float:
        """Return the estimated monthly savings for a recommendation."""
        return recommendation.estimated_savings_monthly

    # -- private rule implementations ---------------------------------------

    def _check_model_downgrade(self) -> List[FinOpsRecommendation]:
        recs: List[FinOpsRecommendation] = []
        threshold_pct = self.config.downgrade_request_pct_threshold

        for model, stats in self._model_stats.items():
            if model not in self.config.expensive_models:
                continue
            if stats.total_requests == 0:
                continue

            pct_under = (
                stats.requests_under_threshold / stats.total_requests
            ) * 100.0

            if pct_under > threshold_pct:
                alt = self.config.cheap_alternative_models.get(model, "a cheaper model")
                estimated_savings = stats.total_cost * (pct_under / 100.0) * 0.6

                recs.append(
                    FinOpsRecommendation(
                        type=RecommendationType.MODEL_DOWNGRADE,
                        title=f"Consider downgrading {model} to {alt}",
                        description=(
                            f"{pct_under:.0f}% of requests to {model} have "
                            f"fewer than {self.config.downgrade_input_token_threshold} "
                            f"input tokens. These could be served by {alt} at "
                            f"significantly lower cost."
                        ),
                        estimated_savings_monthly=round(estimated_savings, 2),
                        confidence=min(pct_under / 100.0, 0.95),
                        affected_entities=[model],
                    )
                )
        return recs

    def _check_cache_opportunity(self) -> List[FinOpsRecommendation]:
        recs: List[FinOpsRecommendation] = []
        if self._total_requests == 0:
            return recs

        duplicate_requests = sum(
            count - 1 for count in self._request_hashes.values() if count > 1
        )
        dup_pct = (duplicate_requests / self._total_requests) * 100.0

        if dup_pct > self.config.cache_duplicate_pct_threshold:
            total_cost = sum(s.total_cost for s in self._model_stats.values())
            estimated_savings = total_cost * (dup_pct / 100.0) * 0.9

            recs.append(
                FinOpsRecommendation(
                    type=RecommendationType.CACHE_OPPORTUNITY,
                    title="Enable response caching for repeated requests",
                    description=(
                        f"{dup_pct:.0f}% of requests are near-duplicates. "
                        f"Enabling semantic caching could eliminate redundant "
                        f"LLM calls."
                    ),
                    estimated_savings_monthly=round(estimated_savings, 2),
                    confidence=min(dup_pct / 100.0, 0.90),
                    affected_entities=list(self._model_stats.keys()),
                )
            )
        return recs

    def _check_spend_growth(self) -> List[FinOpsRecommendation]:
        recs: List[FinOpsRecommendation] = []
        if len(self._spend_history) < 2:
            return recs

        prev = self._spend_history[-2]
        curr = self._spend_history[-1]

        if prev.total_cost == 0:
            return recs

        growth_pct = (
            (curr.total_cost - prev.total_cost) / prev.total_cost
        ) * 100.0

        if growth_pct > self.config.spend_growth_pct_threshold:
            recs.append(
                FinOpsRecommendation(
                    type=RecommendationType.ANOMALY,
                    title="Spend growth exceeds threshold",
                    description=(
                        f"Month-over-month spend grew {growth_pct:.0f}% "
                        f"(from ${prev.total_cost:.2f} to "
                        f"${curr.total_cost:.2f}), exceeding the "
                        f"{self.config.spend_growth_pct_threshold:.0f}% "
                        f"threshold."
                    ),
                    estimated_savings_monthly=0.0,
                    confidence=0.85,
                    affected_entities=[],
                )
            )
        return recs

    def _check_budget_alerts(self) -> List[FinOpsRecommendation]:
        recs: List[FinOpsRecommendation] = []
        threshold = self.config.budget_usage_pct_threshold

        for entity_id, budget in self._entity_budgets.items():
            if budget <= 0:
                continue
            spent = self._entity_spend.get(entity_id, 0.0)
            usage_pct = (spent / budget) * 100.0

            if usage_pct > threshold:
                recs.append(
                    FinOpsRecommendation(
                        type=RecommendationType.BUDGET_ALERT,
                        title=f"Budget alert for {entity_id}",
                        description=(
                            f"{entity_id} has consumed {usage_pct:.0f}% of its "
                            f"${budget:.2f} budget (${spent:.2f} spent). "
                            f"Exceeds the {threshold:.0f}% warning threshold."
                        ),
                        estimated_savings_monthly=0.0,
                        confidence=0.95,
                        affected_entities=[entity_id],
                    )
                )
        return recs


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_copilot_instance: Optional[FinOpsCopilot] = None
_copilot_lock = threading.Lock()


def get_finops_copilot(
    config: Optional[FinOpsCopilotConfig] = None,
) -> FinOpsCopilot:
    """Return (or lazily create) the module-level singleton."""
    global _copilot_instance
    with _copilot_lock:
        if _copilot_instance is None:
            _copilot_instance = FinOpsCopilot(config=config)
        return _copilot_instance
