"""
Audit log hook for writing structured audit events to the database.

Usage:
    from litellm.proxy.hooks.audit_log_hook import log_audit_event

    await log_audit_event(
        action="key.create",
        actor_id=user_id,
        actor_type="user",
        resource_type="key",
        resource_id=key_id,
        details={"key_alias": alias},
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent"),
    )
"""

import asyncio
import hashlib
import json
import traceback
from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of data for audit integrity."""
    try:
        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
    except Exception:
        return ""


async def log_audit_event(
    action: str,
    actor_id: str,
    actor_type: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    status: str = "success",
    # Banking enrichment fields
    request_hash: Optional[str] = None,
    response_hash: Optional[str] = None,
    model_used: Optional[str] = None,
    tokens_used: Optional[int] = None,
    cost: Optional[float] = None,
    latency_ms: Optional[float] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """
    Write an audit log entry to LiteLLM_AuditLogTable.

    This is designed to be fire-and-forget — failures are logged as warnings
    and never propagate to the caller.
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            verbose_proxy_logger.debug(
                "audit_log_hook: prisma_client is None, skipping audit log"
            )
            return

        # Enrich details with banking fields
        enriched_details = details.copy() if details else {}
        if request_hash:
            enriched_details["request_hash"] = request_hash
        if response_hash:
            enriched_details["response_hash"] = response_hash
        if model_used:
            enriched_details["model_used"] = model_used
        if tokens_used is not None:
            enriched_details["tokens_used"] = tokens_used
        if cost is not None:
            enriched_details["cost"] = cost
        if latency_ms is not None:
            enriched_details["latency_ms"] = latency_ms
        if tenant_id:
            enriched_details["tenant_id"] = tenant_id

        await prisma_client.db.litellm_auditlogtable.create(
            data={
                "action": action,
                "actor_id": actor_id,
                "actor_type": actor_type,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": enriched_details if enriched_details else None,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "status": status,
            }
        )
    except Exception:
        verbose_proxy_logger.warning(
            "audit_log_hook: failed to write audit log entry — %s",
            traceback.format_exc(),
        )


def fire_and_forget_audit_event(
    action: str,
    actor_id: str,
    actor_type: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    status: str = "success",
    # Banking enrichment fields
    request_hash: Optional[str] = None,
    response_hash: Optional[str] = None,
    model_used: Optional[str] = None,
    tokens_used: Optional[int] = None,
    cost: Optional[float] = None,
    latency_ms: Optional[float] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """Schedule an audit event write without blocking the caller."""
    kwargs = dict(
        action=action,
        actor_id=actor_id,
        actor_type=actor_type,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
        status=status,
        request_hash=request_hash,
        response_hash=response_hash,
        model_used=model_used,
        tokens_used=tokens_used,
        cost=cost,
        latency_ms=latency_ms,
        tenant_id=tenant_id,
    )
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(log_audit_event(**kwargs))
    except RuntimeError:
        # No running event loop — fall back to a thread
        import threading

        def _run():
            try:
                _loop = asyncio.new_event_loop()
                _loop.run_until_complete(log_audit_event(**kwargs))
                _loop.close()
            except Exception:
                verbose_proxy_logger.warning("audit_log_hook: thread fallback failed")

        threading.Thread(target=_run, daemon=True).start()
