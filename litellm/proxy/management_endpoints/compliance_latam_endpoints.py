"""
LATAM / Brazilian Banking Compliance Endpoints.

Exposes compliance status and reporting for the LATAM compliance pack
(PII masking, data residency enforcement, BACEN audit retention).

Endpoints:
    GET /compliance/latam/status — current config + runtime stats
    GET /compliance/latam/report — compliance summary report (JSON)
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


@router.get(
    "/compliance/latam/status",
    tags=["compliance"],
    dependencies=[Depends(user_api_key_auth)],
)
async def compliance_latam_status():
    """
    Return the current LATAM compliance configuration and live runtime
    statistics (PII detection counts, blocked requests, etc.).

    Requires admin API key.

    Example::

        curl -X GET "http://0.0.0.0:4000/compliance/latam/status" \\
            -H "Authorization: Bearer sk-admin-key"
    """
    from litellm.proxy.hooks.compliance_latam import (
        compliance_stats,
        get_active_config,
    )

    cfg = get_active_config()
    stats = compliance_stats.snapshot()

    return {
        "config": cfg.model_dump(),
        "stats": stats,
    }


@router.get(
    "/compliance/latam/report",
    tags=["compliance"],
    dependencies=[Depends(user_api_key_auth)],
)
async def compliance_latam_report(
    period: Optional[str] = Query(
        default="last_30d",
        description=(
            "Reporting period filter. Accepted values: last_7d, last_30d, "
            "last_90d, last_365d, all."
        ),
    ),
):
    """
    Generate a compliance summary report covering PII detections,
    data-residency violations, and audit-log coverage for the requested
    period.

    Requires admin API key.

    Example::

        curl -X GET "http://0.0.0.0:4000/compliance/latam/report?period=last_90d" \\
            -H "Authorization: Bearer sk-admin-key"
    """
    from litellm.proxy.hooks.compliance_latam import (
        compliance_stats,
        get_active_config,
    )

    cfg = get_active_config()
    stats = compliance_stats.snapshot()

    # ---- determine audit log coverage from the database ------------------
    audit_log_total: int = 0
    audit_log_compliance_events: int = 0
    period_start: Optional[str] = None

    _PERIOD_DAYS = {
        "last_7d": 7,
        "last_30d": 30,
        "last_90d": 90,
        "last_365d": 365,
        "all": None,
    }

    days = _PERIOD_DAYS.get(period)
    if days is not None:
        from datetime import timedelta

        period_start = (
            datetime.now(tz=timezone.utc) - timedelta(days=days)
        ).isoformat()

    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is not None:
            where: dict = {}
            if period_start is not None:
                where["timestamp"] = {
                    "gte": datetime.fromisoformat(period_start),
                }

            audit_log_total = await prisma_client.db.litellm_auditlogtable.count(
                where=where,
            )

            compliance_where = dict(where)
            compliance_where["action"] = {
                "startswith": "compliance_latam.",
            }
            audit_log_compliance_events = (
                await prisma_client.db.litellm_auditlogtable.count(
                    where=compliance_where,
                )
            )
    except Exception:
        verbose_proxy_logger.debug(
            "compliance_latam_report: could not query audit logs (DB may not be connected)"
        )

    return {
        "period": period,
        "period_start": period_start,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "config": {
            "enabled": cfg.enabled,
            "pii_masking": cfg.pii_masking,
            "data_residency": cfg.data_residency,
            "allowed_regions": cfg.allowed_regions,
            "audit_retention_days": cfg.audit_retention_days,
        },
        "summary": {
            "total_requests_processed": stats["total_requests_processed"],
            "pii_detections_by_type": stats["pii_detections"],
            "pii_detections_total": sum(stats["pii_detections"].values()),
            "data_residency_violations": stats["residency_violations"],
            "blocked_requests": stats["blocked_requests"],
        },
        "audit_log_coverage": {
            "total_audit_events": audit_log_total,
            "compliance_events": audit_log_compliance_events,
        },
    }
