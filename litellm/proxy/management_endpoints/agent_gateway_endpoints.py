"""
REST endpoints for the Agentic Gateway.

POST /agents/invoke      — invoke an agent
GET  /agents             — list registered agents
POST /agents/register    — register a new agent
GET  /agents/sessions/{session_id} — get session history
GET  /agents/stats       — agent usage statistics
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.hooks.agentic_gateway import (
    AgenticGateway,
    AgentDefinition,
)

router = APIRouter()

# Module-level gateway instance shared with proxy_server.
_gateway: Optional[AgenticGateway] = None


def get_gateway() -> AgenticGateway:
    global _gateway
    if _gateway is None:
        _gateway = AgenticGateway()
    return _gateway


def set_gateway(gw: AgenticGateway) -> None:
    global _gateway
    _gateway = gw


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class InvokeRequest(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    agent_id: Optional[str] = None


class InvokeResponse(BaseModel):
    response: Dict[str, Any]
    agent_id: str
    session_id: str
    turn_count: int
    tokens_used: int
    cost: float
    latency: float
    model: str


class SessionResponse(BaseModel):
    session_id: str
    agent_id: str
    messages: List[Dict[str, Any]]
    turn_count: int
    total_tokens: int
    total_cost: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/agents/invoke",
    tags=["agentic gateway"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=InvokeResponse,
)
async def invoke_agent(
    data: InvokeRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> InvokeResponse:
    """Invoke an agent for a given session.  If ``agent_id`` is omitted the
    gateway automatically routes to the best matching agent."""

    verbose_proxy_logger.debug(
        "POST /agents/invoke — session=%s agent_id=%s",
        data.session_id,
        data.agent_id,
    )

    gateway = get_gateway()

    try:
        result = await gateway.invoke(
            session_id=data.session_id,
            messages=data.messages,
            agent_id=data.agent_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        verbose_proxy_logger.exception("POST /agents/invoke failed: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return InvokeResponse(**result)


@router.get(
    "/agents",
    tags=["agentic gateway"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_agents(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[AgentDefinition]:
    """Return all registered agents."""

    verbose_proxy_logger.debug("GET /agents")
    gateway = get_gateway()
    return gateway.list_agents()


@router.post(
    "/agents/register",
    tags=["agentic gateway"],
    dependencies=[Depends(user_api_key_auth)],
)
async def register_agent(
    data: AgentDefinition,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, str]:
    """Register a new agent definition."""

    verbose_proxy_logger.debug(
        "POST /agents/register — agent_id=%s name=%s", data.agent_id, data.name
    )
    gateway = get_gateway()
    gateway.register_agent(data)
    return {"status": "ok", "agent_id": data.agent_id}


@router.get(
    "/agents/sessions/{session_id}",
    tags=["agentic gateway"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SessionResponse,
)
async def get_session(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SessionResponse:
    """Return conversation history for a session."""

    verbose_proxy_logger.debug("GET /agents/sessions/%s", session_id)
    gateway = get_gateway()
    session = gateway.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return SessionResponse(
        session_id=session.session_id,
        agent_id=session.agent_id,
        messages=session.messages,
        turn_count=session.turn_count,
        total_tokens=session.total_tokens,
        total_cost=session.total_cost,
    )


@router.get(
    "/agents/stats",
    tags=["agentic gateway"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_stats(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """Return per-agent usage statistics."""

    verbose_proxy_logger.debug("GET /agents/stats")
    gateway = get_gateway()
    return gateway.get_stats()
