"""
Audit log management endpoints.

Provides read access to structured audit logs stored in LiteLLM_AuditLogTable.

Endpoints:
    GET /audit/logs         — Paginated, filterable audit log query
    GET /audit/logs/export  — CSV export of audit logs
"""

import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


@router.get(
    "/audit/logs",
    tags=["audit"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_audit_logs(
    actor_id: Optional[str] = Query(default=None, description="Filter by actor ID"),
    action: Optional[str] = Query(
        default=None,
        description="Filter by action (e.g. key.create, team.update)",
    ),
    resource_type: Optional[str] = Query(
        default=None,
        description="Filter by resource type (e.g. key, team, org)",
    ),
    start_date: Optional[str] = Query(
        default=None, description="Start date in ISO format (e.g. 2024-01-01T00:00:00Z)"
    ),
    end_date: Optional[str] = Query(
        default=None, description="End date in ISO format (e.g. 2024-12-31T23:59:59Z)"
    ),
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        description="Filter by status (success, failure, blocked)",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Number of results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """
    Retrieve paginated audit log entries with optional filters.

    Requires admin API key.

    Example:
    ```
    curl -X GET "http://0.0.0.0:4000/audit/logs?action=key.create&limit=10" \\
        -H "Authorization: Bearer sk-admin-key"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected. Connect a database to your proxy.",
        )

    where_conditions: dict = {}

    if actor_id is not None:
        where_conditions["actor_id"] = actor_id
    if action is not None:
        where_conditions["action"] = action
    if resource_type is not None:
        where_conditions["resource_type"] = resource_type
    if status_filter is not None:
        where_conditions["status"] = status_filter

    # Date range filter
    if start_date is not None or end_date is not None:
        timestamp_filter: dict = {}
        if start_date is not None:
            timestamp_filter["gte"] = datetime.fromisoformat(
                start_date.replace("Z", "+00:00")
            )
        if end_date is not None:
            timestamp_filter["lte"] = datetime.fromisoformat(
                end_date.replace("Z", "+00:00")
            )
        where_conditions["timestamp"] = timestamp_filter

    logs = await prisma_client.db.litellm_auditlogtable.find_many(
        where=where_conditions,
        order={"timestamp": "desc"},
        take=limit,
        skip=offset,
    )

    total = await prisma_client.db.litellm_auditlogtable.count(
        where=where_conditions,
    )

    results = []
    for log in logs:
        try:
            entry = log.model_dump()
        except AttributeError:
            entry = log.dict()

        # Ensure timestamp is ISO-formatted string
        if isinstance(entry.get("timestamp"), datetime):
            entry["timestamp"] = entry["timestamp"].isoformat()

        results.append(entry)

    return {
        "data": results,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/audit/logs/export",
    tags=["audit"],
    dependencies=[Depends(user_api_key_auth)],
)
async def export_audit_logs(
    actor_id: Optional[str] = Query(default=None, description="Filter by actor ID"),
    action: Optional[str] = Query(default=None, description="Filter by action"),
    resource_type: Optional[str] = Query(
        default=None, description="Filter by resource type"
    ),
    start_date: Optional[str] = Query(
        default=None, description="Start date in ISO format"
    ),
    end_date: Optional[str] = Query(
        default=None, description="End date in ISO format"
    ),
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        description="Filter by status",
    ),
    limit: int = Query(default=500, ge=1, le=500, description="Max rows to export"),
):
    """
    Export audit logs as CSV.

    Requires admin API key.

    Example:
    ```
    curl -X GET "http://0.0.0.0:4000/audit/logs/export?action=key.create" \\
        -H "Authorization: Bearer sk-admin-key" -o audit.csv
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected. Connect a database to your proxy.",
        )

    where_conditions: dict = {}

    if actor_id is not None:
        where_conditions["actor_id"] = actor_id
    if action is not None:
        where_conditions["action"] = action
    if resource_type is not None:
        where_conditions["resource_type"] = resource_type
    if status_filter is not None:
        where_conditions["status"] = status_filter

    if start_date is not None or end_date is not None:
        timestamp_filter: dict = {}
        if start_date is not None:
            timestamp_filter["gte"] = datetime.fromisoformat(
                start_date.replace("Z", "+00:00")
            )
        if end_date is not None:
            timestamp_filter["lte"] = datetime.fromisoformat(
                end_date.replace("Z", "+00:00")
            )
        where_conditions["timestamp"] = timestamp_filter

    logs = await prisma_client.db.litellm_auditlogtable.find_many(
        where=where_conditions,
        order={"timestamp": "desc"},
        take=limit,
    )

    csv_columns = [
        "id",
        "timestamp",
        "action",
        "actor_id",
        "actor_type",
        "resource_type",
        "resource_id",
        "details",
        "ip_address",
        "user_agent",
        "status",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=csv_columns, extrasaction="ignore")
    writer.writeheader()

    for log in logs:
        try:
            row = log.model_dump()
        except AttributeError:
            row = log.dict()

        if isinstance(row.get("timestamp"), datetime):
            row["timestamp"] = row["timestamp"].isoformat()

        # Serialize details dict to string for CSV
        if row.get("details") is not None:
            import json

            row["details"] = json.dumps(row["details"])

        writer.writerow(row)

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_logs.csv"},
    )
