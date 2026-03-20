"""Tests for litellm.proxy.hooks.mcp_registry – pure-logic unit tests."""

import time

import pytest

from litellm.proxy.hooks.mcp_registry import (
    MCPRegistry,
    MCPRegistryConfig,
    MCPServerEntry,
    MCPToolInvocation,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_entry(**overrides) -> MCPServerEntry:
    defaults = dict(
        server_id="srv-1",
        name="Test Server",
        description="A test MCP server",
        endpoint_url="http://localhost:8080",
        tools=["tool_a", "tool_b"],
    )
    defaults.update(overrides)
    return MCPServerEntry(**defaults)


# ---------------------------------------------------------------------------
# MCPServerEntry model
# ---------------------------------------------------------------------------


class TestMCPServerEntry:
    def test_required_fields(self):
        entry = _make_entry()
        assert entry.server_id == "srv-1"
        assert entry.name == "Test Server"
        assert entry.endpoint_url == "http://localhost:8080"
        assert entry.transport == "sse"
        assert entry.auth_type == "none"
        assert entry.health_status == "unknown"

    def test_custom_transport(self):
        entry = _make_entry(transport="streamable-http")
        assert entry.transport == "streamable-http"

    def test_tools_list(self):
        entry = _make_entry(tools=["search", "fetch"])
        assert entry.tools == ["search", "fetch"]


# ---------------------------------------------------------------------------
# MCPRegistryConfig
# ---------------------------------------------------------------------------


class TestMCPRegistryConfig:
    def test_defaults(self):
        cfg = MCPRegistryConfig()
        assert cfg.health_check_timeout_s == 10.0
        assert cfg.default_rate_limit_rpm is None
        assert cfg.enable_audit_log is True

    def test_custom(self):
        cfg = MCPRegistryConfig(health_check_timeout_s=5.0, default_rate_limit_rpm=100)
        assert cfg.health_check_timeout_s == 5.0
        assert cfg.default_rate_limit_rpm == 100


# ---------------------------------------------------------------------------
# MCPRegistry – register / discover / unregister
# ---------------------------------------------------------------------------


class TestMCPRegistryRegistration:
    def test_register_and_discover(self):
        reg = MCPRegistry()
        entry = _make_entry()
        reg.register(entry)
        servers = reg.discover()
        assert len(servers) == 1
        assert servers[0].server_id == "srv-1"

    def test_register_sets_registered_at(self):
        reg = MCPRegistry()
        entry = _make_entry()
        assert entry.registered_at is None
        reg.register(entry)
        assert entry.registered_at is not None

    def test_unregister(self):
        reg = MCPRegistry()
        reg.register(_make_entry())
        reg.unregister("srv-1")
        assert reg.discover() == []

    def test_unregister_unknown_raises(self):
        reg = MCPRegistry()
        with pytest.raises(KeyError):
            reg.unregister("nonexistent")

    def test_register_multiple(self):
        reg = MCPRegistry()
        reg.register(_make_entry(server_id="s1"))
        reg.register(_make_entry(server_id="s2"))
        assert len(reg.discover()) == 2


# ---------------------------------------------------------------------------
# get_server
# ---------------------------------------------------------------------------


class TestGetServer:
    def test_exists(self):
        reg = MCPRegistry()
        reg.register(_make_entry(server_id="s1"))
        srv = reg.get_server("s1")
        assert srv.server_id == "s1"

    def test_not_exists(self):
        reg = MCPRegistry()
        with pytest.raises(KeyError):
            reg.get_server("nope")


# ---------------------------------------------------------------------------
# find_by_tool
# ---------------------------------------------------------------------------


class TestFindByTool:
    def test_finds_matching_server(self):
        reg = MCPRegistry()
        reg.register(_make_entry(server_id="s1", tools=["search"]))
        reg.register(_make_entry(server_id="s2", tools=["fetch"]))
        result = reg.find_by_tool("search")
        assert result is not None
        assert result.server_id == "s1"

    def test_returns_none_when_no_match(self):
        reg = MCPRegistry()
        reg.register(_make_entry(server_id="s1", tools=["search"]))
        assert reg.find_by_tool("nonexistent") is None


# ---------------------------------------------------------------------------
# check_rate_limit
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    def test_under_limit(self):
        reg = MCPRegistry()
        entry = _make_entry(server_id="s1", rate_limit_rpm=10)
        reg.register(entry)
        assert reg.check_rate_limit("s1", "caller-1") is True

    def test_over_limit(self):
        reg = MCPRegistry()
        entry = _make_entry(server_id="s1", rate_limit_rpm=2)
        reg.register(entry)
        assert reg.check_rate_limit("s1", "c") is True
        assert reg.check_rate_limit("s1", "c") is True
        assert reg.check_rate_limit("s1", "c") is False

    def test_no_limit_configured(self):
        reg = MCPRegistry()
        entry = _make_entry(server_id="s1", rate_limit_rpm=None)
        reg.register(entry)
        for _ in range(100):
            assert reg.check_rate_limit("s1", "c") is True


# ---------------------------------------------------------------------------
# log_invocation
# ---------------------------------------------------------------------------


class TestLogInvocation:
    def test_log_adds_record(self):
        reg = MCPRegistry()
        inv = MCPToolInvocation(
            invocation_id="inv-1",
            server_id="s1",
            tool_name="search",
            caller_id="user-1",
            timestamp="2024-01-01T00:00:00Z",
        )
        reg.log_invocation(inv)
        assert len(reg._invocations) == 1

    def test_log_disabled_audit(self):
        cfg = MCPRegistryConfig(enable_audit_log=False)
        reg = MCPRegistry(config=cfg)
        inv = MCPToolInvocation(
            invocation_id="inv-1",
            server_id="s1",
            tool_name="search",
            caller_id="user-1",
            timestamp="2024-01-01T00:00:00Z",
        )
        reg.log_invocation(inv)
        assert len(reg._invocations) == 0


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_empty_stats(self):
        reg = MCPRegistry()
        stats = reg.get_stats()
        assert stats["total_invocations"] == 0
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0.0
        assert "invocations_per_server" in stats
        assert "invocations_per_tool" in stats
        assert "latency_p50_ms" in stats
        assert "latency_p99_ms" in stats

    def test_stats_after_invocations(self):
        reg = MCPRegistry()
        for i in range(3):
            reg.log_invocation(MCPToolInvocation(
                invocation_id=f"inv-{i}",
                server_id="s1",
                tool_name="search",
                caller_id="u1",
                timestamp="2024-01-01T00:00:00Z",
                duration_ms=float(10 * (i + 1)),
                status="success",
            ))
        reg.log_invocation(MCPToolInvocation(
            invocation_id="inv-err",
            server_id="s1",
            tool_name="search",
            caller_id="u1",
            timestamp="2024-01-01T00:00:00Z",
            status="error",
        ))
        stats = reg.get_stats()
        assert stats["total_invocations"] == 4
        assert stats["error_count"] == 1
        assert stats["invocations_per_server"]["s1"] == 4
