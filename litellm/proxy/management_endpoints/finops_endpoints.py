"""
FinOps Spend Analytics & Budget Alerting

/finops/summary     - Aggregated spend summary by org, team, project, model
/finops/trends      - Daily spend aggregation for charting
/finops/forecast    - Simple linear spend projection
/finops/alerts/configure - Budget alert threshold configuration
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EntitySpend(BaseModel):
    entity_id: str
    alias: Optional[str] = None
    spend: float = 0.0
    model_spend: Dict[str, float] = Field(default_factory=dict)


class ModelSpend(BaseModel):
    model: str
    total_spend: float = 0.0


class SpendSummaryResponse(BaseModel):
    total_spend: float = 0.0
    by_org: List[EntitySpend] = Field(default_factory=list)
    by_team: List[EntitySpend] = Field(default_factory=list)
    by_project: List[EntitySpend] = Field(default_factory=list)
    by_model: List[ModelSpend] = Field(default_factory=list)


class DailySpend(BaseModel):
    date: str
    spend: float = 0.0
    model_breakdown: Dict[str, float] = Field(default_factory=dict)


class SpendTrendsResponse(BaseModel):
    data: List[DailySpend] = Field(default_factory=list)


class SpendForecastResponse(BaseModel):
    current_monthly_spend: float = 0.0
    projected_monthly_spend: float = 0.0
    daily_average: float = 0.0
    trend_direction: Literal["up", "down", "stable"] = "stable"


class AlertThresholdRequest(BaseModel):
    entity_type: Literal["org", "team", "project"]
    entity_id: str
    thresholds: List[int] = Field(
        default_factory=lambda: [50, 80, 100],
        description="Budget usage percentage thresholds that trigger alerts",
    )
    notification_webhook: Optional[str] = None


class AlertConfigResponse(BaseModel):
    message: str
    entity_type: str
    entity_id: str
    thresholds: List[int]
    notification_webhook: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_model_spend(raw: Any) -> Dict[str, float]:
    """Safely parse model_spend from various stored formats."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {k: float(v) if not isinstance(v, dict) else float(v.get("spend", v.get("current_spend", 0))) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return _parse_model_spend(parsed)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _date_range_for_period(
    period: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple:
    """Return (start_dt, end_dt) as timezone-aware UTC datetimes."""
    now = datetime.now(timezone.utc)
    if period == "last_7d":
        return now - timedelta(days=7), now
    elif period == "last_30d":
        return now - timedelta(days=30), now
    elif period == "current_month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0), now
    elif period == "custom":
        if not start_date or not end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_date and end_date are required for custom period",
            )
        fmt = "%Y-%m-%d"
        sd = datetime.strptime(start_date, fmt).replace(tzinfo=timezone.utc)
        ed = (
            datetime.strptime(end_date, fmt).replace(tzinfo=timezone.utc)
            + timedelta(days=1)
            - timedelta(seconds=1)
        )
        return sd, ed
    # default to last 30 days
    return now - timedelta(days=30), now


def _aggregate_model_spends(entities: list) -> List[ModelSpend]:
    """Aggregate model_spend dicts across entities into a flat list."""
    totals: Dict[str, float] = {}
    for e in entities:
        for model, spend in e.model_spend.items():
            totals[model] = totals.get(model, 0.0) + spend
    return [ModelSpend(model=m, total_spend=s) for m, s in sorted(totals.items(), key=lambda x: x[1], reverse=True)]


# ---------------------------------------------------------------------------
# GET /finops/summary
# ---------------------------------------------------------------------------


@router.get(
    "/finops/summary",
    tags=["FinOps Analytics"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SpendSummaryResponse,
)
async def finops_spend_summary(
    period: str = "last_30d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Aggregated spend summary across orgs, teams, projects, and models.

    Period options: last_7d, last_30d, current_month, custom.
    For custom, provide start_date and end_date (YYYY-MM-DD).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected. Connect a database to your proxy.",
        )

    _date_range_for_period(period, start_date, end_date)  # validate inputs

    try:
        # Fetch entities
        orgs_raw = await prisma_client.db.litellm_organizationtable.find_many()
        teams_raw = await prisma_client.db.litellm_teamtable.find_many()

        # Projects table may not exist in all deployments
        try:
            projects_raw = await prisma_client.db.litellm_projecttable.find_many()
        except Exception:
            projects_raw = []

        # Build entity spend lists
        by_org: List[EntitySpend] = []
        for o in orgs_raw:
            ms = _parse_model_spend(getattr(o, "model_spend", None))
            by_org.append(
                EntitySpend(
                    entity_id=o.organization_id,
                    alias=getattr(o, "organization_alias", None),
                    spend=float(getattr(o, "spend", 0) or 0),
                    model_spend=ms,
                )
            )

        by_team: List[EntitySpend] = []
        for t in teams_raw:
            ms = _parse_model_spend(getattr(t, "model_spend", None))
            by_team.append(
                EntitySpend(
                    entity_id=t.team_id,
                    alias=getattr(t, "team_alias", None),
                    spend=float(getattr(t, "spend", 0) or 0),
                    model_spend=ms,
                )
            )

        by_project: List[EntitySpend] = []
        for p in projects_raw:
            ms = _parse_model_spend(getattr(p, "model_spend", None))
            by_project.append(
                EntitySpend(
                    entity_id=p.project_id,
                    alias=getattr(p, "project_alias", None),
                    spend=float(getattr(p, "spend", 0) or 0),
                    model_spend=ms,
                )
            )

        # Model-level rollup
        all_entities = by_org + by_team + by_project
        by_model = _aggregate_model_spends(all_entities)

        total_spend = sum(e.spend for e in by_org) or sum(e.spend for e in by_team)

        return SpendSummaryResponse(
            total_spend=total_spend,
            by_org=sorted(by_org, key=lambda x: x.spend, reverse=True),
            by_team=sorted(by_team, key=lambda x: x.spend, reverse=True),
            by_project=sorted(by_project, key=lambda x: x.spend, reverse=True),
            by_model=by_model,
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("FinOps summary error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)},
        )


# ---------------------------------------------------------------------------
# GET /finops/trends
# ---------------------------------------------------------------------------


@router.get(
    "/finops/trends",
    tags=["FinOps Analytics"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SpendTrendsResponse,
)
async def finops_spend_trends(
    period: str = "last_30d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    org_id: Optional[str] = None,
    team_id: Optional[str] = None,
    project_id: Optional[str] = None,
):
    """
    Daily spend aggregation for charting, sourced from SpendLogs.

    Filter by org_id, team_id, or project_id.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected. Connect a database to your proxy.",
        )

    sd, ed = _date_range_for_period(period, start_date, end_date)

    try:
        # Build WHERE clauses for SpendLogs
        conditions = [
            '"startTime" >= $1::timestamptz',
            '"startTime" <= $2::timestamptz',
        ]
        params: list = [sd.isoformat(), ed.isoformat()]
        idx = 3

        if team_id:
            conditions.append(f'"team_id" = ${idx}')
            params.append(team_id)
            idx += 1

        if org_id:
            conditions.append(f'"org_id" = ${idx}')
            params.append(org_id)
            idx += 1

        # project_id is stored in metadata for spend logs in some setups
        if project_id:
            # Escape SQL LIKE wildcards
            project_id_escaped = project_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append(f'"metadata"::text LIKE ${idx}')
            params.append(f"%{project_id_escaped}%")
            idx += 1

        where = " AND ".join(conditions)

        sql_query = f"""
            SELECT
                date_trunc('day', "startTime")::date AS day,
                COALESCE(SUM(spend), 0) AS daily_spend,
                model_group
            FROM "LiteLLM_SpendLogs"
            WHERE {where}
            GROUP BY day, model_group
            ORDER BY day ASC
        """

        db_response = await prisma_client.db.query_raw(sql_query, *params)

        # Aggregate into DailySpend objects
        days: Dict[str, DailySpend] = {}
        for row in db_response:
            day_str = str(row.get("day", ""))[:10]
            if day_str not in days:
                days[day_str] = DailySpend(date=day_str)
            ds = days[day_str]
            spend_val = float(row.get("daily_spend", 0) or 0)
            model = row.get("model_group") or "unknown"
            ds.spend += spend_val
            ds.model_breakdown[model] = ds.model_breakdown.get(model, 0.0) + spend_val

        return SpendTrendsResponse(
            data=sorted(days.values(), key=lambda x: x.date),
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("FinOps trends error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)},
        )


# ---------------------------------------------------------------------------
# GET /finops/forecast
# ---------------------------------------------------------------------------


@router.get(
    "/finops/forecast",
    tags=["FinOps Analytics"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SpendForecastResponse,
)
async def finops_spend_forecast():
    """
    Simple linear projection of spend based on the last 30 days.

    Returns current monthly spend, projected monthly spend, daily average,
    and trend direction (up / down / stable).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected. Connect a database to your proxy.",
        )

    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)

    try:
        sql_query = """
            SELECT
                date_trunc('day', "startTime")::date AS day,
                COALESCE(SUM(spend), 0) AS daily_spend
            FROM "LiteLLM_SpendLogs"
            WHERE "startTime" >= $1::timestamptz
            GROUP BY day
            ORDER BY day ASC
        """
        db_response = await prisma_client.db.query_raw(
            sql_query, thirty_days_ago.isoformat()
        )

        if not db_response:
            return SpendForecastResponse()

        daily_spends: List[float] = [
            float(row.get("daily_spend", 0) or 0) for row in db_response
        ]
        n = len(daily_spends)
        total = sum(daily_spends)
        daily_avg = total / n if n > 0 else 0.0

        # Linear regression: y = mx + b  (x = day index)
        if n >= 2:
            x_mean = (n - 1) / 2.0
            y_mean = daily_avg
            numerator = sum(
                (i - x_mean) * (daily_spends[i] - y_mean) for i in range(n)
            )
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator != 0 else 0.0
        else:
            slope = 0.0

        # Current month spend (sum of days in current month from the data)
        current_month_start = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        current_month_spend = 0.0
        for row in db_response:
            day_str = str(row.get("day", ""))[:10]
            try:
                day_dt = datetime.strptime(day_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                if day_dt >= current_month_start:
                    current_month_spend += float(row.get("daily_spend", 0) or 0)
            except ValueError:
                continue

        # Project next 30 days using linear trend
        projected_daily = [daily_avg + slope * (n + i) for i in range(30)]
        # Clamp negatives to zero
        projected_daily = [max(0.0, d) for d in projected_daily]
        projected_monthly = sum(projected_daily)

        # Determine trend direction
        if n >= 2:
            threshold = daily_avg * 0.05  # 5% of average
            if slope > threshold:
                trend = "up"
            elif slope < -threshold:
                trend = "down"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return SpendForecastResponse(
            current_monthly_spend=round(current_month_spend, 4),
            projected_monthly_spend=round(projected_monthly, 4),
            daily_average=round(daily_avg, 4),
            trend_direction=trend,
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("FinOps forecast error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)},
        )


# ---------------------------------------------------------------------------
# POST /finops/alerts/configure
# ---------------------------------------------------------------------------

# In-memory store for alert configurations. In production this would be
# persisted to the database (e.g., a dedicated table or JSON field on the
# entity). Kept lightweight here to avoid schema migrations.
_alert_configs: Dict[str, Dict[str, Any]] = {}


@router.post(
    "/finops/alerts/configure",
    tags=["FinOps Analytics"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AlertConfigResponse,
)
async def finops_configure_alerts(
    alert_config: AlertThresholdRequest,
):
    """
    Configure budget alert thresholds for an org, team, or project.

    When spend reaches the specified percentage thresholds of the entity's
    max_budget, an alert is triggered. Optionally provide a webhook URL
    for notifications.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected. Connect a database to your proxy.",
        )

    entity_type = alert_config.entity_type
    entity_id = alert_config.entity_id

    # Validate the entity exists
    try:
        if entity_type == "org":
            entity = await prisma_client.db.litellm_organizationtable.find_unique(
                where={"organization_id": entity_id}
            )
        elif entity_type == "team":
            entity = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": entity_id}
            )
        elif entity_type == "project":
            entity = await prisma_client.db.litellm_projecttable.find_unique(
                where={"project_id": entity_id}
            )
        else:
            entity = None

        if entity is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{entity_type} with id '{entity_id}' not found",
            )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("FinOps alert config lookup error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)},
        )

    # Persist alert config via entity metadata
    config_payload = {
        "thresholds": sorted(alert_config.thresholds),
        "notification_webhook": alert_config.notification_webhook,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        existing_metadata = getattr(entity, "metadata", None) or {}
        if isinstance(existing_metadata, str):
            try:
                existing_metadata = json.loads(existing_metadata)
            except (json.JSONDecodeError, TypeError):
                existing_metadata = {}

        existing_metadata["finops_alert_config"] = config_payload

        if entity_type == "org":
            await prisma_client.db.litellm_organizationtable.update(
                where={"organization_id": entity_id},
                data={"metadata": json.dumps(existing_metadata)},
            )
        elif entity_type == "team":
            await prisma_client.db.litellm_teamtable.update(
                where={"team_id": entity_id},
                data={"metadata": json.dumps(existing_metadata)},
            )
        elif entity_type == "project":
            await prisma_client.db.litellm_projecttable.update(
                where={"project_id": entity_id},
                data={"metadata": json.dumps(existing_metadata)},
            )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("FinOps alert config save error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)},
        )

    # Also keep in memory for fast access by hooks
    key = f"{entity_type}:{entity_id}"
    _alert_configs[key] = config_payload

    return AlertConfigResponse(
        message="Alert thresholds configured successfully",
        entity_type=entity_type,
        entity_id=entity_id,
        thresholds=sorted(alert_config.thresholds),
        notification_webhook=alert_config.notification_webhook,
    )
