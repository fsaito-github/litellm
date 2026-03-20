"""
LATAM / Brazilian Banking Compliance Pack for LiteLLM Proxy.

Provides:
    - PIIMaskingHook  — detects and masks Brazilian PII (CPF, CNPJ, phone,
      bank account, credit card, email) in LLM messages.
    - DataResidencyHook — enforces Azure region constraints so data stays
      within approved geographies (BACEN / LGPD).
    - ComplianceConfig — Pydantic model for YAML-driven configuration.

Usage in config.yaml:
    compliance_latam:
      enabled: true
      pii_masking: true
      data_residency: true
      allowed_regions: ["brazilsouth", "brazilsoutheast"]
"""

import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

# ---------------------------------------------------------------------------
# PII pattern definitions
# ---------------------------------------------------------------------------

_PII_PATTERNS: Dict[str, re.Pattern] = {
    "cpf": re.compile(
        r"\b(\d{3}\.\d{3}\.\d{3}-\d{2})\b"          # formatted
        r"|"
        r"\b(\d{11})\b",                              # raw digits
    ),
    "cnpj": re.compile(
        r"\b(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})\b"    # formatted
        r"|"
        r"\b(\d{14})\b",                              # raw digits
    ),
    "phone": re.compile(
        r"(\+55\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4})",
    ),
    "bank_account": re.compile(
        r"(ag\.?\s*\d{4}[\s\-]*(?:cc?\.?\s*\d{5,}))",
        re.IGNORECASE,
    ),
    "credit_card": re.compile(
        r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b",
    ),
    "email": re.compile(
        r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b",
    ),
}

_MASK_FORMATS: Dict[str, str] = {
    "cpf": "[CPF:***.***.***-**]",
    "cnpj": "[CNPJ:MASKED]",
    "phone": "[PHONE:MASKED]",
    "bank_account": "[BANK_ACCOUNT:MASKED]",
    "credit_card": "[CREDIT_CARD:****-****-****-****]",
    "email": "[EMAIL:MASKED]",
}


# ---------------------------------------------------------------------------
# In-memory statistics (thread-safe)
# ---------------------------------------------------------------------------

class _ComplianceStats:
    """Simple thread-safe counters for compliance telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pii_detections: Dict[str, int] = {}
        self.total_requests_processed: int = 0
        self.blocked_requests: int = 0
        self.residency_violations: int = 0
        self.since: datetime = datetime.now(tz=timezone.utc)

    def record_pii(self, pii_type: str, count: int = 1) -> None:
        with self._lock:
            self.pii_detections[pii_type] = (
                self.pii_detections.get(pii_type, 0) + count
            )

    def record_request(self) -> None:
        with self._lock:
            self.total_requests_processed += 1

    def record_blocked(self) -> None:
        with self._lock:
            self.blocked_requests += 1

    def record_residency_violation(self) -> None:
        with self._lock:
            self.residency_violations += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pii_detections": dict(self.pii_detections),
                "total_requests_processed": self.total_requests_processed,
                "blocked_requests": self.blocked_requests,
                "residency_violations": self.residency_violations,
                "since": self.since.isoformat(),
            }


# Module-level singleton so endpoints can read stats without coupling.
compliance_stats = _ComplianceStats()


# ---------------------------------------------------------------------------
# ComplianceConfig — Pydantic model
# ---------------------------------------------------------------------------

class ComplianceConfig(BaseModel):
    """YAML-driven configuration for the LATAM compliance pack."""

    enabled: bool = False
    pii_masking: bool = True
    pii_types: List[str] = Field(
        default=["cpf", "cnpj", "phone", "bank_account", "credit_card", "email"],
    )
    data_residency: bool = False
    allowed_regions: List[str] = Field(
        default=["brazilsouth", "brazilsoutheast"],
    )
    audit_retention_days: int = 1825  # 5 years per BACEN
    mask_direction: Literal["input", "output", "both"] = "both"


# Module-level mutable config (set at startup, read at runtime).
_active_config = ComplianceConfig()


def get_active_config() -> ComplianceConfig:
    return _active_config


def set_active_config(cfg: ComplianceConfig) -> None:
    global _active_config
    _active_config = cfg


# ---------------------------------------------------------------------------
# PIIMaskingHook
# ---------------------------------------------------------------------------

class PIIMaskingHook(CustomLogger):
    """
    Detects and masks Brazilian PII in LLM request/response messages.

    Integrates with the audit-log hook to record every detection for
    regulatory traceability (LGPD / BACEN).
    """

    def __init__(
        self,
        pii_types: Optional[List[str]] = None,
        mask_direction: Literal["input", "output", "both"] = "both",
    ) -> None:
        self.pii_types: List[str] = pii_types or list(_PII_PATTERNS.keys())
        self.mask_direction = mask_direction
        super().__init__()

    # -- public helpers -----------------------------------------------------

    def mask_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Scan *text* for Brazilian PII and return ``(masked_text, detections)``.

        Each detection dict has the shape::

            {"type": "cpf", "original": "123.456.789-00",
             "masked": "[CPF:***.***.***-**]", "position": (0, 14)}
        """
        detections: List[Dict[str, Any]] = []
        masked = text

        for pii_type in self.pii_types:
            pattern = _PII_PATTERNS.get(pii_type)
            if pattern is None:
                continue
            mask_fmt = _MASK_FORMATS[pii_type]

            for match in pattern.finditer(text):
                original = match.group(0)
                # Skip if the match was already replaced in a previous pass
                if original not in masked:
                    continue
                detections.append(
                    {
                        "type": pii_type,
                        "original": original,
                        "masked": mask_fmt,
                        "position": (match.start(), match.end()),
                    }
                )
                masked = masked.replace(original, mask_fmt, 1)

        return masked, detections

    async def check_and_mask(
        self,
        messages: List[Dict[str, Any]],
        direction: Literal["input", "output"],
    ) -> List[Dict[str, Any]]:
        """
        Walk through every message's ``content`` field, mask PII, and log
        detections via the audit-log hook.

        Returns the (possibly modified) messages list.
        """
        if self.mask_direction != "both" and self.mask_direction != direction:
            return messages

        all_detections: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            masked_text, detections = self.mask_pii(content)
            if detections:
                msg["content"] = masked_text
                all_detections.extend(detections)

        if all_detections:
            for det in all_detections:
                compliance_stats.record_pii(det["type"])

            # Fire-and-forget audit event
            try:
                from litellm.proxy.hooks.audit_log_hook import (
                    fire_and_forget_audit_event,
                )

                fire_and_forget_audit_event(
                    action="compliance_latam.pii_detected",
                    actor_id="system",
                    actor_type="compliance_hook",
                    resource_type="pii",
                    details={
                        "direction": direction,
                        "detection_count": len(all_detections),
                        "types": list({d["type"] for d in all_detections}),
                    },
                    status="masked",
                )
            except Exception:
                verbose_proxy_logger.debug(
                    "compliance_latam: audit log unavailable, skipping event"
                )

        return messages

    # -- CustomLogger hook integration -------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> None:
        cfg = get_active_config()
        if not cfg.enabled or not cfg.pii_masking:
            return

        compliance_stats.record_request()

        messages = data.get("messages")
        if messages and isinstance(messages, list):
            data["messages"] = await self.check_and_mask(messages, "input")

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        cfg = get_active_config()
        if not cfg.enabled or not cfg.pii_masking:
            return
        if cfg.mask_direction not in ("output", "both"):
            return

        # Best-effort masking of streaming/non-streaming responses
        choices = getattr(response_obj, "choices", None)
        if not choices:
            return
        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None:
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str):
                masked, detections = self.mask_pii(content)
                if detections:
                    message.content = masked
                    for det in detections:
                        compliance_stats.record_pii(det["type"])


# ---------------------------------------------------------------------------
# DataResidencyHook
# ---------------------------------------------------------------------------

class DataResidencyHook(CustomLogger):
    """
    Validates that LLM requests are routed to endpoints in approved Azure
    regions, blocking any that fall outside the allow-list.
    """

    def __init__(
        self,
        allowed_regions: Optional[List[str]] = None,
    ) -> None:
        self.allowed_regions: List[str] = allowed_regions or [
            "brazilsouth",
            "brazilsoutheast",
        ]
        super().__init__()

    def check_model_region(
        self, model: str, litellm_params: Dict[str, Any]
    ) -> bool:
        """
        Return ``True`` if the request endpoint is in an allowed region.

        Region is inferred from the ``api_base`` URL — Azure endpoints embed
        the region name as a subdomain, e.g.
        ``https://my-resource.openai.azure.com/`` where the resource name
        typically maps to a deployment region.  We also accept explicit
        ``region`` in the params.
        """
        api_base: Optional[str] = litellm_params.get("api_base") or ""
        explicit_region: Optional[str] = litellm_params.get("region")

        # Fast-path: explicit region
        if explicit_region:
            return explicit_region.lower() in [
                r.lower() for r in self.allowed_regions
            ]

        # Heuristic: check if any allowed region appears in the URL
        api_base_lower = api_base.lower()
        for region in self.allowed_regions:
            if region.lower() in api_base_lower:
                return True

        # If there is no api_base at all (e.g. OpenAI direct), we cannot
        # determine the region — treat as non-compliant when residency is on.
        if not api_base:
            return False

        return False

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> None:
        cfg = get_active_config()
        if not cfg.enabled or not cfg.data_residency:
            return

        model: str = data.get("model", "")
        litellm_params: Dict[str, Any] = data.get("litellm_params", {})

        # Also look for api_base at the top-level data dict (set by router)
        if "api_base" in data and "api_base" not in litellm_params:
            litellm_params["api_base"] = data["api_base"]

        if not self.check_model_region(model, litellm_params):
            compliance_stats.record_blocked()
            compliance_stats.record_residency_violation()

            try:
                from litellm.proxy.hooks.audit_log_hook import (
                    fire_and_forget_audit_event,
                )

                fire_and_forget_audit_event(
                    action="compliance_latam.residency_blocked",
                    actor_id=user_api_key_dict.user_id or "unknown",
                    actor_type="api_key",
                    resource_type="model",
                    resource_id=model,
                    details={
                        "api_base": litellm_params.get("api_base", ""),
                        "allowed_regions": self.allowed_regions,
                    },
                    status="blocked",
                )
            except Exception:
                verbose_proxy_logger.debug(
                    "compliance_latam: audit log unavailable, skipping event"
                )

            from fastapi import HTTPException

            raise HTTPException(
                status_code=403,
                detail=(
                    f"Data residency violation: model '{model}' is not deployed "
                    f"in an approved region. Allowed regions: "
                    f"{', '.join(self.allowed_regions)}. "
                    f"This request has been blocked to comply with BACEN / LGPD "
                    f"data-residency requirements."
                ),
            )
