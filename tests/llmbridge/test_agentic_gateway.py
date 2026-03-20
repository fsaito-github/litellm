"""Tests for litellm.proxy.hooks.agentic_gateway – pure-logic unit tests."""

import pytest

from litellm.proxy.hooks.agentic_gateway import (
    AgentDefinition,
    AgentRouter,
    AgentSession,
    AgenticGateway,
    AgenticGatewayConfig,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_agent(**overrides) -> AgentDefinition:
    defaults = dict(
        agent_id="agent-1",
        name="Test Agent",
        description="A helpful test agent",
        model="gpt-4o",
        system_prompt="You are a test agent.",
    )
    defaults.update(overrides)
    return AgentDefinition(**defaults)


# ---------------------------------------------------------------------------
# AgentDefinition model
# ---------------------------------------------------------------------------


class TestAgentDefinition:
    def test_required_fields(self):
        a = _make_agent()
        assert a.agent_id == "agent-1"
        assert a.name == "Test Agent"
        assert a.model == "gpt-4o"
        assert a.tools == []
        assert a.guardrails == []
        assert a.max_budget is None
        assert a.max_turns == 10
        assert a.metadata == {}

    def test_custom_fields(self):
        a = _make_agent(
            tools=["search"],
            guardrails=["pii-filter"],
            max_budget=10.0,
            max_turns=5,
        )
        assert a.tools == ["search"]
        assert a.max_budget == 10.0
        assert a.max_turns == 5


# ---------------------------------------------------------------------------
# AgentSession
# ---------------------------------------------------------------------------


class TestAgentSession:
    def test_add_turn_and_get_context(self):
        session = AgentSession(session_id="s1", agent_id="a1")
        user_msg = {"role": "user", "content": "hello"}
        asst_msg = {"role": "assistant", "content": "hi"}
        session.add_turn(user_msg, asst_msg, tokens=100, cost=0.01)

        ctx = session.get_context()
        assert len(ctx) == 2
        assert ctx[0] == user_msg
        assert ctx[1] == asst_msg

    def test_turn_count(self):
        session = AgentSession(session_id="s1", agent_id="a1")
        assert session.turn_count == 0
        session.add_turn(
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        )
        assert session.turn_count == 1
        session.add_turn(
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        )
        assert session.turn_count == 2

    def test_total_tracking(self):
        session = AgentSession(session_id="s1", agent_id="a1")
        session.add_turn(
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            tokens=100,
            cost=0.01,
        )
        session.add_turn(
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            tokens=200,
            cost=0.02,
        )
        assert session.total_tokens == 300
        assert session.total_cost == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# AgentRouter
# ---------------------------------------------------------------------------


class TestAgentRouter:
    def _code_agent(self):
        return _make_agent(
            agent_id="code-assistant",
            name="Code Assistant",
            description="help write code programming debug",
            model="gpt-4o",
        )

    def _translator_agent(self):
        return _make_agent(
            agent_id="translator",
            name="Translator",
            description="translate language text translation",
            model="gpt-4o",
        )

    def _build_router(self):
        agents = {}
        for a in [self._code_agent(), self._translator_agent()]:
            agents[a.agent_id] = a
        return AgentRouter(agents)

    def test_route_code_query(self):
        router = self._build_router()
        msgs = [{"role": "user", "content": "help me write code"}]
        agent_id = router.route(msgs)
        assert agent_id == "code-assistant"

    def test_route_translate_query(self):
        router = self._build_router()
        msgs = [{"role": "user", "content": "translate this text"}]
        agent_id = router.route(msgs)
        assert agent_id == "translator"

    def test_fallback_when_no_match(self):
        router = self._build_router()
        msgs = [{"role": "user", "content": "xyzzy nonsense"}]
        agent_id = router.route(msgs)
        # Should fall back to some registered agent
        assert agent_id in ("code-assistant", "translator")

    def test_fallback_with_empty_messages(self):
        router = self._build_router()
        agent_id = router.route([])
        assert agent_id in ("code-assistant", "translator")

    def test_list_agents(self):
        router = self._build_router()
        agents = router.list_agents()
        assert len(agents) == 2


# ---------------------------------------------------------------------------
# AgenticGateway
# ---------------------------------------------------------------------------


class TestAgenticGateway:
    def test_register_agent(self):
        gw = AgenticGateway()
        agent = _make_agent(agent_id="a1")
        gw.register_agent(agent)
        assert len(gw.list_agents()) == 1
        assert gw.list_agents()[0].agent_id == "a1"

    def test_list_agents_multiple(self):
        gw = AgenticGateway()
        gw.register_agent(_make_agent(agent_id="a1", name="One", description="first"))
        gw.register_agent(_make_agent(agent_id="a2", name="Two", description="second"))
        assert len(gw.list_agents()) == 2


# ---------------------------------------------------------------------------
# AgenticGatewayConfig
# ---------------------------------------------------------------------------


class TestAgenticGatewayConfig:
    def test_defaults(self):
        cfg = AgenticGatewayConfig()
        assert cfg.enabled is False
        assert cfg.agents == []
        assert cfg.default_agent_id is None
        assert cfg.session_ttl_minutes == 60

    def test_custom(self):
        agent = _make_agent()
        cfg = AgenticGatewayConfig(
            enabled=True,
            agents=[agent],
            default_agent_id="agent-1",
            session_ttl_minutes=30,
        )
        assert cfg.enabled is True
        assert len(cfg.agents) == 1
        assert cfg.session_ttl_minutes == 30
