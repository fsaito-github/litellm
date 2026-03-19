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
import traceback
from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger


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

        await prisma_client.db.litellm_auditlogtable.create(
            data={
                "action": action,
                "actor_id": actor_id,
                "actor_type": actor_type,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": details if details is not None else None,
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
) -> None:
    """
    Schedule an audit log write without awaiting it.

    Safe to call from sync or async contexts where you don't want to block.
    """
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(
            log_audit_event(
                action=action,
                actor_id=actor_id,
                actor_type=actor_type,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
                status=status,
            )
        )
    except Exception:
        verbose_proxy_logger.warning(
            "audit_log_hook: failed to schedule audit log task — %s",
            traceback.format_exc(),
        )
