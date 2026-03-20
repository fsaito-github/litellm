"""Tests for litellm.proxy.hooks.observability_graph – pure-logic unit tests."""

import time

import pytest

from litellm.proxy.hooks.observability_graph import (
    ExecutionGraph,
    ExecutionNode,
    NodeStatus,
    NodeType,
    ObservabilityGraphManager,
)


# ---------------------------------------------------------------------------
# ExecutionNode
# ---------------------------------------------------------------------------


class TestExecutionNode:
    def test_required_fields(self):
        node = ExecutionNode(node_type=NodeType.MODEL, name="gpt-4")
        assert node.name == "gpt-4"
        assert node.node_type == NodeType.MODEL
        assert node.node_id  # auto-generated
        assert node.status == NodeStatus.PENDING
        assert node.cost == 0.0
        assert node.tokens == 0

    def test_custom_fields(self):
        node = ExecutionNode(
            node_id="n1",
            node_type=NodeType.TOOL,
            name="calculator",
            start_time=100.0,
            end_time=200.0,
            duration_ms=100_000.0,
            status=NodeStatus.SUCCESS,
            cost=0.05,
            tokens=500,
            parent_node_id="p1",
            metadata={"key": "val"},
        )
        assert node.node_id == "n1"
        assert node.parent_node_id == "p1"
        assert node.metadata["key"] == "val"


# ---------------------------------------------------------------------------
# ExecutionGraph – add_node / add_edge
# ---------------------------------------------------------------------------


class TestExecutionGraphBasics:
    def test_add_node(self):
        g = ExecutionGraph(trace_id="t1")
        node = ExecutionNode(node_id="a", node_type=NodeType.AGENT, name="agent1")
        g.add_node(node)
        assert "a" in g.nodes

    def test_add_edge(self):
        g = ExecutionGraph(trace_id="t1")
        g.add_node(ExecutionNode(node_id="a", node_type=NodeType.AGENT, name="a"))
        g.add_node(ExecutionNode(node_id="b", node_type=NodeType.TOOL, name="b"))
        g.add_edge("a", "b")
        assert ("a", "b") in g.edges

    def test_add_node_with_parent_creates_edge(self):
        g = ExecutionGraph(trace_id="t1")
        parent = ExecutionNode(node_id="p", node_type=NodeType.AGENT, name="parent")
        child = ExecutionNode(
            node_id="c", node_type=NodeType.TOOL, name="child", parent_node_id="p"
        )
        g.add_node(parent)
        g.add_node(child)
        assert ("p", "c") in g.edges

    def test_add_edge_missing_node_no_crash(self):
        g = ExecutionGraph(trace_id="t1")
        g.add_node(ExecutionNode(node_id="a", node_type=NodeType.AGENT, name="a"))
        g.add_edge("a", "nonexistent")  # should log warning, not crash


# ---------------------------------------------------------------------------
# to_dag
# ---------------------------------------------------------------------------


class TestToDag:
    def test_structure(self):
        g = ExecutionGraph(trace_id="t1")
        g.add_node(ExecutionNode(node_id="a", node_type=NodeType.MODEL, name="m1"))
        g.add_node(ExecutionNode(node_id="b", node_type=NodeType.TOOL, name="t1"))
        g.add_edge("a", "b")

        dag = g.to_dag()
        assert dag["trace_id"] == "t1"
        assert "a" in dag["nodes"]
        assert "b" in dag["nodes"]
        assert len(dag["edges"]) == 1
        assert dag["metadata"]["node_count"] == 2
        assert dag["metadata"]["edge_count"] == 1


# ---------------------------------------------------------------------------
# get_critical_path – linear chain
# ---------------------------------------------------------------------------


class TestCriticalPath:
    def test_linear_chain(self):
        """A → B → C: critical path is [A, B, C]."""
        g = ExecutionGraph(trace_id="t1")
        a = ExecutionNode(
            node_id="A", node_type=NodeType.MODEL, name="A",
            duration_ms=100.0, status=NodeStatus.SUCCESS,
        )
        b = ExecutionNode(
            node_id="B", node_type=NodeType.TOOL, name="B",
            duration_ms=200.0, status=NodeStatus.SUCCESS, parent_node_id="A",
        )
        c = ExecutionNode(
            node_id="C", node_type=NodeType.GUARDRAIL, name="C",
            duration_ms=50.0, status=NodeStatus.SUCCESS, parent_node_id="B",
        )
        g.add_node(a)
        g.add_node(b)
        g.add_node(c)

        path = g.get_critical_path()
        ids = [n.node_id for n in path]
        assert ids == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# get_bottlenecks
# ---------------------------------------------------------------------------


class TestBottlenecks:
    def test_with_threshold(self):
        g = ExecutionGraph(trace_id="t1")
        slow = ExecutionNode(
            node_id="slow", node_type=NodeType.MODEL, name="slow",
            duration_ms=5000.0,
        )
        fast = ExecutionNode(
            node_id="fast", node_type=NodeType.TOOL, name="fast",
            duration_ms=10.0,
        )
        g.add_node(slow)
        g.add_node(fast)

        bottlenecks = g.get_bottlenecks(threshold_ms=1000.0)
        assert len(bottlenecks) == 1
        assert bottlenecks[0].node_id == "slow"

    def test_no_bottlenecks(self):
        g = ExecutionGraph(trace_id="t1")
        g.add_node(ExecutionNode(
            node_id="f", node_type=NodeType.TOOL, name="fast", duration_ms=10.0
        ))
        assert g.get_bottlenecks(threshold_ms=100.0) == []


# ---------------------------------------------------------------------------
# get_cost_attribution
# ---------------------------------------------------------------------------


class TestCostAttribution:
    def test_returns_nodes_with_cost(self):
        g = ExecutionGraph(trace_id="t1")
        g.add_node(ExecutionNode(
            node_id="paid", node_type=NodeType.MODEL, name="m", cost=0.05
        ))
        g.add_node(ExecutionNode(
            node_id="free", node_type=NodeType.TOOL, name="t", cost=0.0
        ))

        costs = g.get_cost_attribution()
        assert "paid" in costs
        assert costs["paid"] == pytest.approx(0.05)
        assert "free" not in costs


# ---------------------------------------------------------------------------
# ObservabilityGraphManager
# ---------------------------------------------------------------------------


class TestGraphManager:
    def test_start_and_get_trace(self):
        mgr = ObservabilityGraphManager()
        graph = mgr.start_trace("trace-1")
        assert isinstance(graph, ExecutionGraph)
        assert mgr.get_trace("trace-1") is graph

    def test_end_trace(self):
        mgr = ObservabilityGraphManager()
        mgr.start_trace("trace-1")
        mgr.end_trace("trace-1")
        graph = mgr.get_trace("trace-1")
        assert graph.ended_at is not None

    def test_get_nonexistent_trace(self):
        mgr = ObservabilityGraphManager()
        assert mgr.get_trace("nope") is None

    def test_end_nonexistent_trace_no_crash(self):
        mgr = ObservabilityGraphManager()
        mgr.end_trace("nope")  # should not raise


# ---------------------------------------------------------------------------
# query_traces
# ---------------------------------------------------------------------------


class TestQueryTraces:
    def test_filter_by_time(self):
        mgr = ObservabilityGraphManager()
        g = mgr.start_trace("t1")
        # Manipulate created_at for deterministic test
        now = time.time()
        g.created_at = now - 10
        g.ended_at = now - 5

        # Trace within range
        results = mgr.query_traces({"time_start": now - 20, "time_end": now})
        assert len(results) == 1
        assert results[0]["trace_id"] == "t1"

        # Trace outside range
        results = mgr.query_traces({"time_start": now + 100, "time_end": now + 200})
        assert len(results) == 0


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    def test_detect_slow_trace(self):
        mgr = ObservabilityGraphManager()
        now = time.time()

        # Create several "normal" traces
        for i in range(5):
            g = mgr.start_trace(f"normal-{i}")
            g.created_at = now - 100
            g.ended_at = now - 99  # ~1 second

        # Create an anomalously slow trace
        slow = mgr.start_trace("slow-1")
        slow.created_at = now - 100
        slow.ended_at = now  # 100 seconds

        anomalies = mgr.detect_anomalies()
        trace_ids = [a["trace_id"] for a in anomalies]
        assert "slow-1" in trace_ids

    def test_no_anomalies_when_all_similar(self):
        mgr = ObservabilityGraphManager()
        now = time.time()
        for i in range(5):
            g = mgr.start_trace(f"t-{i}")
            g.created_at = now - 10
            g.ended_at = now - 9  # all ~1 second
        anomalies = mgr.detect_anomalies()
        assert len(anomalies) == 0
