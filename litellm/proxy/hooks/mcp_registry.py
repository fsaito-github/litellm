"""
MCP Server Gateway & Registry for LiteLLM.

Provides in-memory registration, discovery, health checking, rate limiting,
and audit logging for MCP servers that are reachable by the proxy.

Usage:
    from litellm.proxy.hooks.mcp_registry import mcp_registry, MCPRegistry

    # Initialise once (typically at proxy startup)
    mcp_registry = MCPRegistry()
    mcp_registry.register(MCPServerEntry(...))

    # Discover / invoke
    servers = mcp_registry.discover()
    ok = await mcp_registry.health_check("my-server")
"""

import statistics
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MCPServerEntry(BaseModel):
    """Describes a single MCP server known to the gateway."""

    server_id: str
    name: str
    description: str
    endpoint_url: str
    transport: Literal["stdio", "sse", "streamable-http"] = "sse"
    tools: List[str] = Field(default_factory=list)
    auth_type: Literal["none", "api_key", "bearer", "oauth2"] = "none"
    auth_config: Optional[Dict[str, Any]] = None
    health_status: Literal["healthy", "unhealthy", "unknown"] = "unknown"
    rate_limit_rpm: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registered_at: Optional[str] = None


class MCPToolInvocation(BaseModel):
    """Audit record for a single tool invocation."""

    invocation_id: str
    server_id: str
    tool_name: str
    caller_id: str
    timestamp: str
    duration_ms: Optional[float] = None
    status: Literal["success", "error", "rate_limited"] = "success"
    error_message: Optional[str] = None


class MCPRegistryConfig(BaseModel):
    """Configuration knobs for the registry."""

    health_check_timeout_s: float = 10.0
    default_rate_limit_rpm: Optional[int] = None
    enable_audit_log: bool = True


# ---------------------------------------------------------------------------
# Registry implementation
# ---------------------------------------------------------------------------


class MCPRegistry:
    """In-memory MCP server registry with health-check, rate-limit & audit."""

    def __init__(self, config: Optional[MCPRegistryConfig] = None) -> None:
        self.config = config or MCPRegistryConfig()
        self._servers: Dict[str, MCPServerEntry] = {}
        self._invocations: deque[MCPToolInvocation] = deque(maxlen=10000)
        # rate-limit tracking: {(server_id, caller_id): [timestamps]}
        self._rate_counters: Dict[tuple, List[float]] = defaultdict(list)
        self._lock = Lock()

    # -- registration --------------------------------------------------------

    def register(self, entry: MCPServerEntry) -> MCPServerEntry:
        """Add or update an MCP server in the registry."""
        with self._lock:
            if entry.registered_at is None:
                entry.registered_at = datetime.now(timezone.utc).isoformat()
            self._servers[entry.server_id] = entry
        verbose_proxy_logger.info(
            "mcp_registry: registered server %s (%s)", entry.server_id, entry.name
        )
        return entry

    def unregister(self, server_id: str) -> None:
        """Remove an MCP server from the registry."""
        with self._lock:
            if server_id not in self._servers:
                raise KeyError(f"Server '{server_id}' not found in registry")
            del self._servers[server_id]
        verbose_proxy_logger.info(
            "mcp_registry: unregistered server %s", server_id
        )

    # -- discovery -----------------------------------------------------------

    def discover(self) -> List[MCPServerEntry]:
        """Return all registered servers."""
        with self._lock:
            return list(self._servers.values())

    def get_server(self, server_id: str) -> MCPServerEntry:
        """Return a single server or raise ``KeyError``."""
        with self._lock:
            if server_id not in self._servers:
                raise KeyError(f"Server '{server_id}' not found in registry")
            return self._servers[server_id]

    def find_by_tool(self, tool_name: str) -> Optional[MCPServerEntry]:
        """Find the first server that advertises *tool_name*."""
        with self._lock:
            for entry in self._servers.values():
                if tool_name in entry.tools:
                    return entry
        return None

    # -- health checks -------------------------------------------------------

    async def health_check(self, server_id: str) -> bool:
        """Probe a server's endpoint and update its health status."""
        entry = self.get_server(server_id)

        try:
            async with httpx.AsyncClient(
                timeout=self.config.health_check_timeout_s
            ) as client:
                resp = await client.get(entry.endpoint_url)
                healthy = resp.status_code < 500
        except Exception as exc:
            verbose_proxy_logger.warning(
                "mcp_registry: health check failed for %s — %s",
                server_id,
                str(exc),
            )
            healthy = False

        with self._lock:
            if server_id in self._servers:
                self._servers[server_id].health_status = (
                    "healthy" if healthy else "unhealthy"
                )
        return healthy

    async def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on every registered server."""
        results: Dict[str, bool] = {}
        for sid in list(self._servers):
            results[sid] = await self.health_check(sid)
        return results

    # -- rate limiting -------------------------------------------------------

    def check_rate_limit(self, server_id: str, caller_id: str) -> bool:
        """Return ``True`` if the caller is within the rate limit.

        Uses a sliding-window counter scoped per (server, caller).
        """
        entry = self.get_server(server_id)
        rpm = entry.rate_limit_rpm or self.config.default_rate_limit_rpm
        if rpm is None:
            return True  # no limit configured

        now = time.monotonic()
        key = (server_id, caller_id)

        with self._lock:
            # prune entries older than 60 s
            self._rate_counters[key] = [
                ts for ts in self._rate_counters[key] if now - ts < 60
            ]
            if len(self._rate_counters[key]) >= rpm:
                return False
            self._rate_counters[key].append(now)
        return True

    # -- audit / invocation logging ------------------------------------------

    def log_invocation(self, invocation: MCPToolInvocation) -> None:
        """Append an invocation record for auditing."""
        if not self.config.enable_audit_log:
            return
        with self._lock:
            self._invocations.append(invocation)
        verbose_proxy_logger.debug(
            "mcp_registry: logged invocation %s on %s/%s",
            invocation.invocation_id,
            invocation.server_id,
            invocation.tool_name,
        )

    # -- statistics ----------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Compute usage statistics from recorded invocations."""
        with self._lock:
            invocations = list(self._invocations)

        total = len(invocations)
        per_server: Dict[str, int] = defaultdict(int)
        per_tool: Dict[str, int] = defaultdict(int)
        errors = 0
        durations: List[float] = []

        for inv in invocations:
            per_server[inv.server_id] += 1
            per_tool[inv.tool_name] += 1
            if inv.status == "error":
                errors += 1
            if inv.duration_ms is not None:
                durations.append(inv.duration_ms)

        def _percentile(data: List[float], pct: float) -> Optional[float]:
            if not data:
                return None
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * (pct / 100)
            f = int(k)
            c = f + 1
            if c >= len(sorted_data):
                return sorted_data[f]
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

        return {
            "total_invocations": total,
            "invocations_per_server": dict(per_server),
            "invocations_per_tool": dict(per_tool),
            "error_count": errors,
            "error_rate": errors / total if total else 0.0,
            "latency_p50_ms": _percentile(durations, 50),
            "latency_p99_ms": _percentile(durations, 99),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

mcp_registry: Optional[MCPRegistry] = None
