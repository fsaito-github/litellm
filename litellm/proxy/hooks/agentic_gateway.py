"""
Agentic Gateway — multi-agent orchestration layer for LiteLLM.

Provides agent registration, semantic intent routing, session tracking,
and proxied completion calls through named agent definitions.
"""

import time
import traceback
import uuid as _uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AgentDefinition(BaseModel):
    """Declarative description of a single agent."""

    agent_id: str
    name: str
    description: str
    model: str  # e.g. "gpt-4o", "claude-sonnet"
    system_prompt: str
    tools: List[str] = []  # MCP tool names
    guardrails: List[str] = []  # guardrail rule IDs
    max_budget: Optional[float] = None
    max_turns: int = 10
    metadata: Dict[str, Any] = {}


class AgenticGatewayConfig(BaseModel):
    """Top-level configuration consumed by the proxy."""

    enabled: bool = False
    agents: List[AgentDefinition] = []
    default_agent_id: Optional[str] = None
    session_ttl_minutes: int = 60


# ---------------------------------------------------------------------------
# Session tracking
# ---------------------------------------------------------------------------


class AgentSession:
    """Tracks multi-turn conversation context for a single session."""

    def __init__(self, session_id: str, agent_id: str) -> None:
        self.session_id: str = session_id
        self.agent_id: str = agent_id
        self.messages: List[Dict[str, Any]] = []
        self.turn_count: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.created_at: float = time.time()
        self._created_at: float = time.time()
        self.updated_at: float = time.time()

    def add_turn(
        self,
        user_msg: Dict[str, Any],
        assistant_msg: Dict[str, Any],
        tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        self.messages.append(user_msg)
        self.messages.append(assistant_msg)
        self.turn_count += 1
        self.total_tokens += tokens
        self.total_cost += cost
        self.updated_at = time.time()

    def get_context(self) -> List[Dict[str, Any]]:
        """Return the full message history."""
        return list(self.messages)


# ---------------------------------------------------------------------------
# Router — semantic intent routing
# ---------------------------------------------------------------------------


class AgentRouter:
    """Routes incoming messages to the best-matching agent using keyword
    overlap scoring against agent descriptions."""

    def __init__(self, agents: Dict[str, AgentDefinition]) -> None:
        self._agents = agents

    def route(self, messages: List[Dict[str, Any]]) -> str:
        """Return the ``agent_id`` whose description best matches the last
        user message.  Falls back to the first registered agent."""

        user_text = self._extract_user_text(messages)
        if not user_text:
            return next(iter(self._agents))

        user_words = set(user_text.lower().split())

        best_id: Optional[str] = None
        best_score: float = 0.0

        for agent_id, agent in self._agents.items():
            desc_words = set(agent.description.lower().split())
            name_words = set(agent.name.lower().split())
            candidate_words = desc_words | name_words
            overlap = len(user_words & candidate_words)
            if overlap > best_score:
                best_score = overlap
                best_id = agent_id

        if best_id is None:
            best_id = next(iter(self._agents))

        verbose_proxy_logger.debug(
            "AgentRouter.route: selected agent_id=%s (score=%s)", best_id, best_score
        )
        return best_id

    def get_agent(self, agent_id: str) -> AgentDefinition:
        return self._agents[agent_id]

    def list_agents(self) -> List[AgentDefinition]:
        return list(self._agents.values())

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
        """Pull the text from the last user message."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    return " ".join(parts)
        return ""


# ---------------------------------------------------------------------------
# Agentic Gateway — main orchestration class
# ---------------------------------------------------------------------------


class AgenticGateway:
    """Central orchestration layer.

    * Maintains a registry of ``AgentDefinition`` objects.
    * Stores per-session conversation state.
    * Delegates model calls to ``litellm.acompletion()``.
    """

    _MAX_SESSIONS = 10000

    def __init__(self) -> None:
        self._agents: Dict[str, AgentDefinition] = {}
        self._sessions: Dict[str, AgentSession] = {}
        self._router: Optional[AgentRouter] = None
        self._config = type("_Cfg", (), {"session_ttl_minutes": 60})()

        # lightweight stats
        self._stats_requests: Dict[str, int] = defaultdict(int)
        self._stats_latency: Dict[str, List[float]] = defaultdict(list)
        self._stats_cost: Dict[str, float] = defaultdict(float)

    # -- agent management ---------------------------------------------------

    def register_agent(self, agent: AgentDefinition) -> None:
        self._agents[agent.agent_id] = agent
        self._router = AgentRouter(self._agents)
        verbose_proxy_logger.debug(
            "AgenticGateway: registered agent %s (%s)", agent.agent_id, agent.name
        )

    def list_agents(self) -> List[AgentDefinition]:
        return list(self._agents.values())

    # -- invocation ---------------------------------------------------------

    async def invoke(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invoke an agent for the given session.

        * If *agent_id* is ``None`` the router picks the best match.
        * The agent's system prompt is prepended to the conversation.
        * ``litellm.acompletion()`` is called with the agent's model.
        * Token/cost bookkeeping is updated in the session.
        """

        if not self._agents:
            raise ValueError("No agents registered in the gateway")

        # --- resolve agent ---
        if agent_id and agent_id not in self._agents:
            raise ValueError(f"Unknown agent_id: {agent_id}")

        if agent_id is None:
            if self._router is None:
                self._router = AgentRouter(self._agents)
            agent_id = self._router.route(messages)

        agent = self._agents[agent_id]

        # --- session ---
        session = self._get_or_create_session(session_id, agent_id)

        if agent.max_budget is not None and session.total_cost >= agent.max_budget:
            raise ValueError(
                f"Session {session_id} exceeded max budget "
                f"({session.total_cost:.4f} >= {agent.max_budget})"
            )
        if session.turn_count >= agent.max_turns:
            raise ValueError(
                f"Session {session_id} exceeded max turns "
                f"({session.turn_count} >= {agent.max_turns})"
            )

        # --- build prompt ---
        full_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": agent.system_prompt},
        ]
        full_messages.extend(session.get_context())
        full_messages.extend(messages)

        # --- call litellm ---
        start = time.time()
        try:
            response = await litellm.acompletion(
                model=agent.model,
                messages=full_messages,
            )
        except Exception:
            verbose_proxy_logger.exception(
                "AgenticGateway: litellm.acompletion failed for agent %s — %s",
                agent_id,
                traceback.format_exc(),
            )
            raise

        elapsed = time.time() - start

        # --- extract response data ---
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) if usage else 0
        cost = 0.0
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            pass

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": ""}
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0:
            msg_obj = getattr(choices[0], "message", None)
            if msg_obj:
                assistant_msg = {
                    "role": "assistant",
                    "content": getattr(msg_obj, "content", "") or "",
                }

        # --- bookkeeping ---
        last_user_msg = messages[-1] if messages else {"role": "user", "content": ""}
        session.add_turn(last_user_msg, assistant_msg, tokens=tokens, cost=cost)

        self._stats_requests[agent_id] += 1
        self._stats_latency[agent_id].append(elapsed)
        self._stats_cost[agent_id] += cost

        verbose_proxy_logger.debug(
            "AgenticGateway: invoke agent=%s session=%s tokens=%d cost=%.6f latency=%.3fs",
            agent_id,
            session_id,
            tokens,
            cost,
            elapsed,
        )

        return {
            "response": assistant_msg,
            "agent_id": agent_id,
            "session_id": session_id,
            "turn_count": session.turn_count,
            "tokens_used": tokens,
            "cost": cost,
            "latency": round(elapsed, 4),
            "model": agent.model,
        }

    # -- session management -------------------------------------------------

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        return self._sessions.get(session_id)

    def _cleanup_expired_sessions(self) -> None:
        """Remove sessions older than TTL."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s._created_at > self._config.session_ttl_minutes * 60]
        for sid in expired:
            del self._sessions[sid]

    def _get_or_create_session(
        self, session_id: str, agent_id: str
    ) -> AgentSession:
        self._cleanup_expired_sessions()
        if session_id not in self._sessions:
            self._sessions[session_id] = AgentSession(
                session_id=session_id, agent_id=agent_id
            )
            verbose_proxy_logger.debug(
                "AgenticGateway: created session %s for agent %s",
                session_id,
                agent_id,
            )
        # Cap total sessions after cleanup
        if len(self._sessions) > self._MAX_SESSIONS:
            oldest = sorted(self._sessions, key=lambda s: self._sessions[s]._created_at)
            for sid in oldest[:len(self._sessions) - self._MAX_SESSIONS]:
                del self._sessions[sid]
        return self._sessions[session_id]

    # -- stats --------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for agent_id in self._agents:
            latencies = self._stats_latency.get(agent_id, [])
            stats[agent_id] = {
                "requests": self._stats_requests.get(agent_id, 0),
                "avg_latency": (
                    round(sum(latencies) / len(latencies), 4) if latencies else 0.0
                ),
                "total_cost": round(self._stats_cost.get(agent_id, 0.0), 6),
            }
        return stats
