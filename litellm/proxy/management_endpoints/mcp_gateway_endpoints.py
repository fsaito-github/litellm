"""
MCP Gateway Registry endpoints.

Exposes REST endpoints for registering, discovering, health-checking,
and monitoring MCP servers managed by the LiteLLM proxy.

All endpoints require authentication via ``user_api_key_auth``.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.hooks.mcp_registry import (
    MCPRegistry,
    MCPServerEntry,
    mcp_registry as _module_registry,
)

router = APIRouter(
    prefix="/mcp/registry",
    tags=["mcp-gateway"],
    dependencies=[Depends(user_api_key_auth)],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_registry() -> MCPRegistry:
    """Return the active registry or raise 503 if not initialised."""
    # Re-import at call time so late-init singletons are picked up.
    from litellm.proxy.hooks.mcp_registry import mcp_registry

    if mcp_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "MCP Registry has not been initialised."},
        )
    return mcp_registry


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------


class _ToolEntry(BaseModel):
    tool_name: str
    server_id: str
    server_name: str


class _HealthCheckResponse(BaseModel):
    server_id: str
    healthy: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/servers",
    response_model=MCPServerEntry,
    status_code=status.HTTP_201_CREATED,
    description="Register (or update) an MCP server in the gateway registry.",
)
async def register_server(
    entry: MCPServerEntry,
) -> MCPServerEntry:
    registry = _get_registry()
    verbose_proxy_logger.info(
        "mcp_gateway: register_server called for %s", entry.server_id
    )
    try:
        return registry.register(entry)
    except Exception as exc:
        verbose_proxy_logger.error(
            "mcp_gateway: failed to register server — %s", str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(exc)},
        )


@router.get(
    "/servers",
    response_model=List[MCPServerEntry],
    description="List all MCP servers currently registered in the gateway.",
)
async def list_servers() -> List[MCPServerEntry]:
    registry = _get_registry()
    verbose_proxy_logger.debug("mcp_gateway: list_servers called")
    return registry.discover()


@router.delete(
    "/servers/{server_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    description="Remove an MCP server from the gateway registry.",
)
async def unregister_server(server_id: str) -> None:
    registry = _get_registry()
    verbose_proxy_logger.info(
        "mcp_gateway: unregister_server called for %s", server_id
    )
    try:
        registry.unregister(server_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Server '{server_id}' not found in registry."},
        )


@router.post(
    "/servers/{server_id}/health",
    response_model=_HealthCheckResponse,
    description="Trigger a health check for a specific MCP server.",
)
async def trigger_health_check(server_id: str) -> _HealthCheckResponse:
    registry = _get_registry()
    verbose_proxy_logger.info(
        "mcp_gateway: health_check triggered for %s", server_id
    )
    try:
        healthy = await registry.health_check(server_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Server '{server_id}' not found in registry."},
        )
    return _HealthCheckResponse(server_id=server_id, healthy=healthy)


@router.get(
    "/stats",
    response_model=Dict[str, Any],
    description="Return usage statistics (invocation counts, error rates, latency).",
)
async def get_stats() -> Dict[str, Any]:
    registry = _get_registry()
    verbose_proxy_logger.debug("mcp_gateway: get_stats called")
    return registry.get_stats()


@router.get(
    "/tools",
    response_model=List[_ToolEntry],
    description="List all tools available across every registered MCP server.",
)
async def list_tools() -> List[_ToolEntry]:
    registry = _get_registry()
    verbose_proxy_logger.debug("mcp_gateway: list_tools called")
    result: List[_ToolEntry] = []
    for server in registry.discover():
        for tool in server.tools:
            result.append(
                _ToolEntry(
                    tool_name=tool,
                    server_id=server.server_id,
                    server_name=server.name,
                )
            )
    return result
