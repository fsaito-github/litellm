"""
LLM Bridge Integration Middleware

Wires all LLM Bridge hooks into the LiteLLM request/response flow:

Pre-call:
  1. Content Firewall — check input for prompt injection, jailbreak, etc.
  2. Compliance PII Masking — mask PII in input (if mask_direction=input|both)
  3. Cost Router — select cheapest model based on query complexity
  4. Circuit Breaker — check if provider is healthy
  5. Semantic Cache — check for cached response

Post-call:
  1. Compliance PII Masking — mask PII in output (if mask_direction=output|both)
  2. Semantic Cache — store response for future cache hits
  3. Circuit Breaker — record success/failure
  4. Audit Logging — log the request
  5. Observability Graph — record execution node
"""

import os
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LLMBridgeConfig(BaseModel):
    """Feature-flag configuration for LLM Bridge middleware."""

    enabled: bool = False
    firewall_enabled: bool = True
    cost_router_enabled: bool = True
    circuit_breaker_enabled: bool = True
    compliance_enabled: bool = False
    audit_enabled: bool = True
    semantic_cache_enabled: bool = False
    observability_enabled: bool = False


# ---------------------------------------------------------------------------
# Callback Handler
# ---------------------------------------------------------------------------


class LLMBridgeCallbackHandler(CustomLogger):
    """
    Central callback handler that orchestrates all LLM Bridge hooks within the
    LiteLLM ``CustomLogger`` lifecycle.

    Hooks are imported lazily on first use and individually wrapped in
    try/except so that a failure in one hook never crashes the proxy.
    """

    def __init__(self, config: Optional[LLMBridgeConfig] = None) -> None:
        super().__init__()
        self.config = config or LLMBridgeConfig()
        self._initialized = False

        # Hook references — populated by ``_ensure_initialized``
        self._firewall: Any = None
        self._cost_router: Any = None
        self._circuit_breaker_mgr: Any = None
        self._compliance_pii: Any = None
        self._observability_mgr: Any = None
        self._audit_enabled: bool = False
        self._semantic_cache: Any = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Import hook singletons on first use (graceful degradation)."""
        if self._initialized:
            return

        # Content Firewall
        if self.config.firewall_enabled:
            try:
                from litellm.proxy.hooks.content_firewall import content_firewall

                self._firewall = content_firewall
                verbose_proxy_logger.debug("LLMBridge: content_firewall loaded")
            except Exception:
                verbose_proxy_logger.warning(
                    "LLMBridge: content_firewall unavailable — skipping",
                    exc_info=True,
                )

        # Cost Router
        if self.config.cost_router_enabled:
            try:
                from litellm.proxy.hooks.cost_router import cost_router_instance

                self._cost_router = cost_router_instance
                verbose_proxy_logger.debug("LLMBridge: cost_router loaded")
            except Exception:
                verbose_proxy_logger.warning(
                    "LLMBridge: cost_router unavailable — skipping",
                    exc_info=True,
                )

        # Circuit Breaker
        if self.config.circuit_breaker_enabled:
            try:
                from litellm.proxy.hooks.circuit_breaker import (
                    circuit_breaker_manager,
                )

                self._circuit_breaker_mgr = circuit_breaker_manager
                verbose_proxy_logger.debug("LLMBridge: circuit_breaker loaded")
            except Exception:
                verbose_proxy_logger.warning(
                    "LLMBridge: circuit_breaker unavailable — skipping",
                    exc_info=True,
                )

        # Compliance / PII Masking
        if self.config.compliance_enabled:
            try:
                from litellm.proxy.hooks.compliance_latam import PIIMaskingHook

                self._compliance_pii = PIIMaskingHook()
                verbose_proxy_logger.debug("LLMBridge: compliance_pii loaded")
            except Exception:
                verbose_proxy_logger.warning(
                    "LLMBridge: compliance_pii unavailable — skipping",
                    exc_info=True,
                )

        # Audit Logging
        if self.config.audit_enabled:
            try:
                from litellm.proxy.hooks import audit_log_hook  # noqa: F401

                self._audit_enabled = True
                verbose_proxy_logger.debug("LLMBridge: audit_log_hook loaded")
            except Exception:
                verbose_proxy_logger.warning(
                    "LLMBridge: audit_log_hook unavailable — skipping",
                    exc_info=True,
                )

        # Observability Graph
        if self.config.observability_enabled:
            try:
                from litellm.proxy.hooks.observability_graph import (
                    get_observability_graph_manager,
                )

                self._observability_mgr = get_observability_graph_manager()
                verbose_proxy_logger.debug("LLMBridge: observability_graph loaded")
            except Exception:
                verbose_proxy_logger.warning(
                    "LLMBridge: observability_graph unavailable — skipping",
                    exc_info=True,
                )

        # Semantic Cache (lives in litellm.caching, not hooks)
        if self.config.semantic_cache_enabled:
            try:
                from litellm.caching.semantic_cache import SemanticCache

                self._semantic_cache = SemanticCache
                verbose_proxy_logger.debug("LLMBridge: semantic_cache loaded")
            except Exception:
                verbose_proxy_logger.warning(
                    "LLMBridge: semantic_cache unavailable — skipping",
                    exc_info=True,
                )

        self._initialized = True
        verbose_proxy_logger.info(
            "LLMBridge middleware initialised (firewall=%s, cost_router=%s, "
            "circuit_breaker=%s, compliance=%s, audit=%s, observability=%s, "
            "semantic_cache=%s)",
            self._firewall is not None,
            self._cost_router is not None,
            self._circuit_breaker_mgr is not None,
            self._compliance_pii is not None,
            self._audit_enabled,
            self._observability_mgr is not None,
            self._semantic_cache is not None,
        )

    # ------------------------------------------------------------------
    # Pre-call hook
    # ------------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Executed **before** the LLM API call.

        Order:
        1. Content Firewall
        2. Compliance PII masking (input)
        3. Cost Router
        4. Circuit Breaker
        """
        try:
            self._ensure_initialized()
        except Exception:
            verbose_proxy_logger.error(
                "LLMBridge: initialisation failed — passing through",
                exc_info=True,
            )
            return data

        messages = data.get("messages", [])

        # 1. Content Firewall ------------------------------------------------
        if self._firewall:
            try:
                is_safe, violations = self._firewall.check_messages(messages)
                if not is_safe:
                    blocking = [
                        v for v in violations if getattr(v, "action", None) == "block"
                    ]
                    if blocking:
                        from litellm.proxy._types import ProxyException

                        raise ProxyException(
                            message=(
                                "Content blocked by firewall: "
                                f"{blocking[0].rule_name}"
                            ),
                            type="content_policy_violation",
                            param=None,
                            code=400,
                        )
                    # Non-blocking violations — attach to metadata for audit
                    data.setdefault("metadata", {})["firewall_warnings"] = [
                        {
                            "rule": v.rule_name,
                            "category": getattr(v, "category", "unknown"),
                        }
                        for v in violations
                    ]
            except Exception as exc:
                # Re-raise ProxyException (intentional block) but swallow
                # unexpected errors so the proxy stays alive.
                if type(exc).__name__ == "ProxyException":
                    raise
                verbose_proxy_logger.error(
                    "LLMBridge: firewall check failed — allowing request",
                    exc_info=True,
                )

        # 2. Compliance PII masking (input) -----------------------------------
        if self._compliance_pii and messages:
            try:
                masked_messages = await self._compliance_pii.check_and_mask(
                    messages, direction="input"
                )
                data["messages"] = masked_messages
                messages = masked_messages
            except Exception:
                verbose_proxy_logger.error(
                    "LLMBridge: PII input masking failed — using original messages",
                    exc_info=True,
                )

        # 3. Cost Router — model re-selection ---------------------------------
        if self._cost_router and data.get("model"):
            try:
                routed_model = self._cost_router.route(messages, data["model"])
                if routed_model != data["model"]:
                    data.setdefault("metadata", {})
                    data["metadata"]["original_model"] = data["model"]
                    data["metadata"]["cost_routed"] = True
                    data["model"] = routed_model
                    verbose_proxy_logger.debug(
                        "LLMBridge: cost_router re-routed %s → %s",
                        data["metadata"]["original_model"],
                        routed_model,
                    )
            except Exception:
                verbose_proxy_logger.error(
                    "LLMBridge: cost_router failed — keeping original model",
                    exc_info=True,
                )

        # 4. Circuit Breaker — gate check -------------------------------------
        if self._circuit_breaker_mgr:
            try:
                model = data.get("model", "unknown")
                breaker = self._circuit_breaker_mgr.get_or_create(model)
                if not breaker.allow_request():
                    from litellm.proxy._types import ProxyException

                    raise ProxyException(
                        message=f"Circuit breaker open for model {model}",
                        type="service_unavailable",
                        param=None,
                        code=503,
                    )
            except Exception as exc:
                if type(exc).__name__ == "ProxyException":
                    raise
                verbose_proxy_logger.error(
                    "LLMBridge: circuit_breaker check failed — allowing request",
                    exc_info=True,
                )

        # Record pre-call start time for observability
        data.setdefault("metadata", {})["llmbridge_start_time"] = time.time()

        return data

    # ------------------------------------------------------------------
    # Post-call success hook
    # ------------------------------------------------------------------

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """
        Executed **after** a successful LLM API call.

        Order:
        1. Compliance PII masking (output)
        2. Circuit Breaker — record success
        3. Audit Logging
        4. Observability Graph
        """
        try:
            self._ensure_initialized()
        except Exception:
            verbose_proxy_logger.error(
                "LLMBridge: initialisation failed in success hook",
                exc_info=True,
            )
            return

        model = kwargs.get("model", "unknown")
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})

        # 1. Compliance PII masking (output) ----------------------------------
        if self._compliance_pii and response_obj is not None:
            try:
                # response_obj.choices[].message.content may contain PII
                choices = getattr(response_obj, "choices", None) or []
                for choice in choices:
                    msg = getattr(choice, "message", None)
                    if msg and getattr(msg, "content", None):
                        masked_msgs = await self._compliance_pii.check_and_mask(
                            [{"role": "assistant", "content": msg.content}],
                            direction="output",
                        )
                        if masked_msgs:
                            msg.content = masked_msgs[0].get("content", msg.content)
            except Exception:
                verbose_proxy_logger.error(
                    "LLMBridge: PII output masking failed", exc_info=True
                )

        # 2. Circuit Breaker — record success ---------------------------------
        if self._circuit_breaker_mgr:
            try:
                breaker = self._circuit_breaker_mgr.get_or_create(model)
                breaker.record_success()
            except Exception:
                verbose_proxy_logger.error(
                    "LLMBridge: circuit_breaker record_success failed",
                    exc_info=True,
                )

        # 3. Audit Logging ----------------------------------------------------
        if self._audit_enabled:
            try:
                from litellm.proxy.hooks.audit_log_hook import (
                    fire_and_forget_audit_event,
                )

                fire_and_forget_audit_event(
                    action="llm.completion",
                    actor_id=metadata.get("user_api_key_user_id", "unknown"),
                    actor_type="api_key",
                    resource_type="model",
                    resource_id=model,
                    status="success",
                    details={
                        "cost_routed": metadata.get("cost_routed", False),
                        "original_model": metadata.get("original_model"),
                    },
                )
            except Exception:
                verbose_proxy_logger.error(
                    "LLMBridge: audit logging failed", exc_info=True
                )

        # 4. Observability Graph — record node --------------------------------
        if self._observability_mgr:
            try:
                from litellm.proxy.hooks.observability_graph import ExecutionNode

                trace_id = metadata.get("trace_id", str(uuid.uuid4()))
                graph = self._observability_mgr.start_trace(trace_id)

                pre_start = metadata.get("llmbridge_start_time", time.time())
                now = time.time()
                duration_ms = (now - pre_start) * 1000

                node = ExecutionNode(
                    node_type="MODEL",
                    name=model,
                    start_time=pre_start,
                    end_time=now,
                    duration_ms=duration_ms,
                    status="SUCCESS",
                    cost=getattr(response_obj, "_hidden_params", {}).get(
                        "response_cost", 0.0
                    )
                    if response_obj
                    else 0.0,
                    tokens=_safe_token_count(response_obj),
                    metadata={
                        "cost_routed": metadata.get("cost_routed", False),
                        "original_model": metadata.get("original_model"),
                    },
                )
                graph.add_node(node)
                self._observability_mgr.end_trace(trace_id)
            except Exception:
                verbose_proxy_logger.error(
                    "LLMBridge: observability graph recording failed",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Post-call failure hook
    # ------------------------------------------------------------------

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """
        Executed **after** a failed LLM API call.

        Order:
        1. Circuit Breaker — record failure
        2. Audit Logging
        3. Observability Graph
        """
        try:
            self._ensure_initialized()
        except Exception:
            verbose_proxy_logger.error(
                "LLMBridge: initialisation failed in failure hook",
                exc_info=True,
            )
            return

        model = kwargs.get("model", "unknown")
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        exception = kwargs.get("exception", None)

        # 1. Circuit Breaker — record failure ---------------------------------
        if self._circuit_breaker_mgr:
            try:
                breaker = self._circuit_breaker_mgr.get_or_create(model)
                breaker.record_failure()
            except Exception:
                verbose_proxy_logger.error(
                    "LLMBridge: circuit_breaker record_failure failed",
                    exc_info=True,
                )

        # 2. Audit Logging ----------------------------------------------------
        if self._audit_enabled:
            try:
                from litellm.proxy.hooks.audit_log_hook import (
                    fire_and_forget_audit_event,
                )

                fire_and_forget_audit_event(
                    action="llm.completion",
                    actor_id=metadata.get("user_api_key_user_id", "unknown"),
                    actor_type="api_key",
                    resource_type="model",
                    resource_id=model,
                    status="failure",
                    details={
                        "error": str(exception) if exception else "unknown",
                        "cost_routed": metadata.get("cost_routed", False),
                        "original_model": metadata.get("original_model"),
                    },
                )
            except Exception:
                verbose_proxy_logger.error(
                    "LLMBridge: audit logging (failure) failed",
                    exc_info=True,
                )

        # 3. Observability Graph — record failed node -------------------------
        if self._observability_mgr:
            try:
                from litellm.proxy.hooks.observability_graph import ExecutionNode

                trace_id = metadata.get("trace_id", str(uuid.uuid4()))
                graph = self._observability_mgr.start_trace(trace_id)

                pre_start = metadata.get("llmbridge_start_time", time.time())
                now = time.time()

                node = ExecutionNode(
                    node_type="MODEL",
                    name=model,
                    start_time=pre_start,
                    end_time=now,
                    duration_ms=(now - pre_start) * 1000,
                    status="ERROR",
                    cost=0.0,
                    tokens=0,
                    metadata={
                        "error": str(exception) if exception else "unknown",
                    },
                )
                graph.add_node(node)
                self._observability_mgr.end_trace(trace_id)
            except Exception:
                verbose_proxy_logger.error(
                    "LLMBridge: observability graph (failure) recording failed",
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_token_count(response_obj: Any) -> int:
    """Extract total token count from a response object without raising."""
    try:
        usage = getattr(response_obj, "usage", None)
        if usage:
            return getattr(usage, "total_tokens", 0) or 0
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Initialisation entry-point
# ---------------------------------------------------------------------------


def _build_config_from_env() -> LLMBridgeConfig:
    """Build an ``LLMBridgeConfig`` from ``LLMBRIDGE_*`` environment variables."""

    def _env_bool(key: str, default: bool) -> bool:
        val = os.environ.get(key, "").strip().lower()
        if val in ("1", "true", "yes"):
            return True
        if val in ("0", "false", "no"):
            return False
        return default

    return LLMBridgeConfig(
        enabled=_env_bool("LLMBRIDGE_ENABLED", False),
        firewall_enabled=_env_bool("LLMBRIDGE_FIREWALL_ENABLED", True),
        cost_router_enabled=_env_bool("LLMBRIDGE_COST_ROUTER_ENABLED", True),
        circuit_breaker_enabled=_env_bool("LLMBRIDGE_CIRCUIT_BREAKER_ENABLED", True),
        compliance_enabled=_env_bool("LLMBRIDGE_COMPLIANCE_ENABLED", False),
        audit_enabled=_env_bool("LLMBRIDGE_AUDIT_ENABLED", True),
        semantic_cache_enabled=_env_bool("LLMBRIDGE_SEMANTIC_CACHE_ENABLED", False),
        observability_enabled=_env_bool("LLMBRIDGE_OBSERVABILITY_ENABLED", False),
    )


def initialize_llmbridge(
    config: Optional[LLMBridgeConfig] = None,
) -> Optional["LLMBridgeCallbackHandler"]:
    """
    Register LLM Bridge as a LiteLLM callback handler.

    Called during proxy startup when ``LLMBRIDGE_ENABLED=true``.
    Returns the handler instance (or ``None`` if disabled / on error).
    """
    try:
        resolved_config = config or _build_config_from_env()

        if not resolved_config.enabled:
            verbose_proxy_logger.debug(
                "LLMBridge: disabled by config (LLMBRIDGE_ENABLED != true)"
            )
            return None

        import litellm

        handler = LLMBridgeCallbackHandler(config=resolved_config)

        if isinstance(litellm.callbacks, list):
            litellm.callbacks.append(handler)
        else:
            litellm.callbacks = [handler]  # type: ignore[assignment]

        verbose_proxy_logger.info(
            "LLM Bridge middleware registered as LiteLLM callback"
        )
        return handler

    except Exception:
        verbose_proxy_logger.error(
            "LLMBridge: failed to initialise — middleware will NOT be active",
            exc_info=True,
        )
        return None
