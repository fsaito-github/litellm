"""
Observability Graph – execution DAG for tracing agent/tool/model/guardrail calls.

Provides:
    * ``ExecutionNode``   – Pydantic model for a single node in the graph.
    * ``ExecutionGraph``  – DAG of nodes with critical-path & bottleneck analysis.
    * ``ObservabilityGraphManager`` – trace lifecycle management and querying.
"""

import threading
import time
import traceback
import uuid
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    AGENT = "agent"
    TOOL = "tool"
    MODEL = "model"
    GUARDRAIL = "guardrail"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ExecutionNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType
    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: NodeStatus = NodeStatus.PENDING
    cost: float = 0.0
    tokens: int = 0
    parent_node_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExecutionGraph
# ---------------------------------------------------------------------------


class ExecutionGraph:
    """In-memory DAG representing a single trace of execution nodes."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id: str = trace_id
        self.nodes: Dict[str, ExecutionNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self.created_at: float = time.time()
        self.ended_at: Optional[float] = None
        self._lock = threading.Lock()

    # -- mutators -----------------------------------------------------------

    def add_node(self, node: ExecutionNode) -> None:
        with self._lock:
            self.nodes[node.node_id] = node
            if node.parent_node_id and node.parent_node_id in self.nodes:
                self.edges.append((node.parent_node_id, node.node_id))

    def add_edge(self, parent_id: str, child_id: str) -> None:
        with self._lock:
            if parent_id in self.nodes and child_id in self.nodes:
                self.edges.append((parent_id, child_id))
            else:
                verbose_proxy_logger.warning(
                    "observability_graph.py::add_edge(): one or both nodes "
                    "not found — parent=%s child=%s",
                    parent_id,
                    child_id,
                )

    # -- analysis -----------------------------------------------------------

    def get_critical_path(self) -> List[ExecutionNode]:
        """Return the longest path (by cumulative duration) through the DAG."""
        with self._lock:
            children: Dict[str, List[str]] = defaultdict(list)
            parents: set = set()
            for parent_id, child_id in self.edges:
                children[parent_id].append(child_id)
                parents.add(child_id)

            roots = [nid for nid in self.nodes if nid not in parents]
            if not roots:
                roots = list(self.nodes.keys())[:1]

            best_path: List[str] = []
            best_cost: float = 0.0

            def _dfs(nid: str, path: List[str], cost: float) -> None:
                nonlocal best_path, best_cost
                node = self.nodes[nid]
                dur = node.duration_ms or 0.0
                path.append(nid)
                cost += dur
                if not children.get(nid):
                    if cost > best_cost:
                        best_cost = cost
                        best_path = list(path)
                else:
                    for cid in children[nid]:
                        _dfs(cid, path, cost)
                path.pop()

            for root in roots:
                _dfs(root, [], 0.0)

            return [self.nodes[nid] for nid in best_path]

    def get_bottlenecks(self, threshold_ms: float) -> List[ExecutionNode]:
        """Return nodes whose duration exceeds *threshold_ms*."""
        with self._lock:
            return [
                n
                for n in self.nodes.values()
                if n.duration_ms is not None and n.duration_ms > threshold_ms
            ]

    def to_dag(self) -> Dict[str, Any]:
        """Return a JSON-serializable DAG representation."""
        with self._lock:
            return {
                "trace_id": self.trace_id,
                "nodes": {
                    nid: n.model_dump() for nid, n in self.nodes.items()
                },
                "edges": [
                    {"parent": p, "child": c} for p, c in self.edges
                ],
                "metadata": {
                    "created_at": self.created_at,
                    "ended_at": self.ended_at,
                    "node_count": len(self.nodes),
                    "edge_count": len(self.edges),
                },
            }

    def get_cost_attribution(self) -> Dict[str, float]:
        """Return per-node cost attribution ``{node_id: cost}``."""
        with self._lock:
            return {
                nid: n.cost for nid, n in self.nodes.items() if n.cost > 0.0
            }


# ---------------------------------------------------------------------------
# ObservabilityGraphManager
# ---------------------------------------------------------------------------


class ObservabilityGraphManager:
    """Manages the lifecycle of execution traces."""

    def __init__(self) -> None:
        self._traces: OrderedDict[str, ExecutionGraph] = OrderedDict()
        self._max_traces = 5000
        self._lock = threading.Lock()
        verbose_proxy_logger.info("ObservabilityGraphManager initialized")

    def start_trace(self, trace_id: str) -> ExecutionGraph:
        with self._lock:
            graph = ExecutionGraph(trace_id=trace_id)
            self._traces[trace_id] = graph
            while len(self._traces) > self._max_traces:
                self._traces.popitem(last=False)
            verbose_proxy_logger.debug(
                "observability_graph.py::start_trace(): trace_id=%s", trace_id
            )
            return graph

    def end_trace(self, trace_id: str) -> None:
        with self._lock:
            graph = self._traces.get(trace_id)
            if graph is None:
                verbose_proxy_logger.warning(
                    "observability_graph.py::end_trace(): unknown trace_id=%s",
                    trace_id,
                )
                return
            graph.ended_at = time.time()
            verbose_proxy_logger.debug(
                "observability_graph.py::end_trace(): trace_id=%s ended", trace_id
            )

    def get_trace(self, trace_id: str) -> Optional[ExecutionGraph]:
        with self._lock:
            return self._traces.get(trace_id)

    def query_traces(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search traces by *time_range*, *status*, and *min_duration*."""
        with self._lock:
            results: List[Dict[str, Any]] = []
            time_start = filters.get("time_start")
            time_end = filters.get("time_end")
            status_filter = filters.get("status")
            min_duration = filters.get("min_duration")

            for graph in self._traces.values():
                # time range filter
                if time_start and graph.created_at < time_start:
                    continue
                if time_end and graph.created_at > time_end:
                    continue

                # status filter — at least one node must match
                if status_filter:
                    statuses = {n.status.value for n in graph.nodes.values()}
                    if status_filter not in statuses:
                        continue

                # min_duration filter (trace wall-clock time)
                if min_duration is not None and graph.ended_at is not None:
                    trace_dur_ms = (graph.ended_at - graph.created_at) * 1000.0
                    if trace_dur_ms < min_duration:
                        continue

                results.append(graph.to_dag())

            return results

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect traces with unusual duration or error patterns."""
        anomalies: List[Dict[str, Any]] = []
        with self._lock:
            # Collect durations of completed traces
            durations: List[float] = []
            for g in self._traces.values():
                if g.ended_at is not None:
                    durations.append((g.ended_at - g.created_at) * 1000.0)

            if not durations:
                return anomalies

            avg_dur = sum(durations) / len(durations)
            # Threshold: 2× average
            dur_threshold = avg_dur * 2.0

            for trace_id, g in self._traces.items():
                issues: List[str] = []

                if g.ended_at is not None:
                    trace_dur = (g.ended_at - g.created_at) * 1000.0
                    if trace_dur > dur_threshold:
                        issues.append(
                            f"duration {trace_dur:.1f}ms exceeds 2x avg "
                            f"({avg_dur:.1f}ms)"
                        )

                error_count = sum(
                    1
                    for n in g.nodes.values()
                    if n.status == NodeStatus.ERROR
                )
                if error_count > 0:
                    total = len(g.nodes)
                    error_pct = (error_count / total) * 100.0 if total else 0.0
                    if error_pct > 25.0:
                        issues.append(
                            f"high error rate: {error_count}/{total} nodes "
                            f"({error_pct:.0f}%)"
                        )

                if issues:
                    anomalies.append(
                        {
                            "trace_id": trace_id,
                            "issues": issues,
                            "node_count": len(g.nodes),
                        }
                    )

        return anomalies


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_manager_instance: Optional[ObservabilityGraphManager] = None
_manager_lock = threading.Lock()


def get_observability_graph_manager() -> ObservabilityGraphManager:
    """Return (or lazily create) the module-level singleton."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = ObservabilityGraphManager()
        return _manager_instance
