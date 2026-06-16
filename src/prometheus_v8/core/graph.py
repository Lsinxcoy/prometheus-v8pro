"""Graph Store - 3 backends: NetworkX/KuzuDB/FalkorDB."""

from __future__ import annotations

import logging
import pickle
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Protocol, runtime_checkable

from prometheus_v8.schema import EdgeType

logger = logging.getLogger(__name__)


@runtime_checkable
class GraphProtocol(Protocol):
    """Protocol defining the graph store interface for type-safe dependency injection."""

    def add_node(self, node_id: str, attributes: dict | None = None) -> None: ...
    def add_edge(
        self, source: str, target: str, edge_type: str, weight: float = 1.0, metadata: dict | None = None
    ) -> None: ...
    def get_neighbors(self, node_id: str, edge_type: str | None = None) -> list[str]: ...
    def get_node_attrs(self, node_id: str) -> dict[str, Any]: ...
    def remove_node(self, node_id: str) -> None: ...
    def shortest_path(self, source: str, target: str) -> list[str] | None: ...


class GraphBackend(ABC):
    """Abstract graph store backend."""

    @abstractmethod
    def add_node(self, node_id: str, attributes: dict | None = None) -> None: ...

    @abstractmethod
    def add_edge(
        self, source: str, target: str, edge_type: str, weight: float = 1.0, metadata: dict | None = None
    ) -> None: ...

    @abstractmethod
    def get_neighbors(
        self, node_id: str, edge_type: str | None = None, direction: str = "outgoing"
    ) -> list[tuple[str, str, float]]: ...

    @abstractmethod
    def find_paths(self, source: str, target: str, max_depth: int = 5) -> list[list[str]]: ...

    @abstractmethod
    def community_detection(self) -> dict[str, int]: ...

    @abstractmethod
    def hallway_traversal(self, start: str, max_depth: int = 3) -> list[tuple[str, float]]: ...


class NetworkXGraphBackend(GraphBackend):
    """NetworkX MultiDiGraph with Leiden community detection."""

    def __init__(self) -> None:
        try:
            import networkx as nx

            self._nx = nx
            self._graph = nx.MultiDiGraph()
        except ImportError:
            raise ImportError("networkx is required for NetworkXGraphBackend")
        self._lock = threading.RLock()

    def add_node(self, node_id: str, attributes: dict | None = None) -> None:
        with self._lock:
            self._graph.add_node(node_id, **(attributes or {}))

    def add_edge(
        self, source: str, target: str, edge_type: str, weight: float = 1.0, metadata: dict | None = None
    ) -> None:
        with self._lock:
            self._graph.add_edge(source, target, key=edge_type, weight=weight, **(metadata or {}))

    def get_neighbors(
        self, node_id: str, edge_type: str | None = None, direction: str = "outgoing"
    ) -> list[tuple[str, str, float]]:
        with self._lock:
            results = []
            if direction in ("outgoing", "both"):
                for _, target, key, data in self._graph.out_edges(node_id, keys=True, data=True):
                    if edge_type is None or key == edge_type:
                        results.append((target, key, data.get("weight", 1.0)))
            if direction in ("incoming", "both"):
                for source, _, key, data in self._graph.in_edges(node_id, keys=True, data=True):
                    if edge_type is None or key == edge_type:
                        results.append((source, key, data.get("weight", 1.0)))
            return results

    def find_paths(self, source: str, target: str, max_depth: int = 5) -> list[list[str]]:
        with self._lock:
            try:
                paths = list(self._nx.all_simple_paths(self._graph, source, target, cutoff=max_depth))
                return [list(p) for p in paths[:10]]
            except self._nx.NetworkXNoPath:
                return []

    def community_detection(self) -> dict[str, int]:
        with self._lock:
            undirected = self._graph.to_undirected()
            if len(undirected.nodes) < 2:
                return {}
            try:
                from networkx.algorithms.community import greedy_modularity_communities

                communities = greedy_modularity_communities(undirected)
                result = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        result[node] = i
                return result
            except Exception as e:
                logger.warning(f"Community detection failed: {e}")
                return {n: 0 for n in undirected.nodes}

    def hallway_traversal(self, start: str, max_depth: int = 3) -> list[tuple[str, float]]:
        """BFS traversal following hallway/tunnel edges with highest weight."""
        with self._lock:
            visited = set()
            queue = deque([(start, 0, 1.0)])
            results = []
            while queue:
                node, depth, score = queue.popleft()
                if node in visited or depth > max_depth:
                    continue
                visited.add(node)
                results.append((node, score))
                neighbors = self.get_neighbors(node, edge_type=EdgeType.HALLWAY.value)
                neighbors += self.get_neighbors(node, edge_type=EdgeType.TUNNEL.value)
                for target, _, weight in neighbors:
                    if target not in visited:
                        queue.append((target, depth + 1, score * weight))
            return results[1:]  # exclude start

    def get_subgraph(self, nodes: list[str]) -> Any:
        with self._lock:
            return self._graph.subgraph(nodes)

    def persist(self, path: str) -> None:
        with self._lock:
            with open(path, "wb") as f:
                pickle.dump(self._graph, f)

    def load(self, path: str) -> None:
        import os

        if os.path.exists(path):
            with open(path, "rb") as f:
                self._graph = pickle.load(f)


class SQLGraphBackend(GraphBackend):
    """SQLite-based graph with recursive CTE traversal."""

    def __init__(self, db_path: str = "data/graph.db") -> None:
        import sqlite3

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (id TEXT PRIMARY KEY, attributes TEXT DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS graph_edges (source TEXT, target TEXT, type TEXT, weight REAL DEFAULT 1.0, metadata TEXT DEFAULT '{}', PRIMARY KEY(source, target, type));
        CREATE INDEX IF NOT EXISTS idx_ge_source ON graph_edges(source);
        CREATE INDEX IF NOT EXISTS idx_ge_target ON graph_edges(target);
        """)
        self._lock = threading.RLock()

    def add_node(self, node_id: str, attributes: dict | None = None) -> None:
        import json

        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO graph_nodes VALUES(?,?)", (node_id, json.dumps(attributes or {}))
            )
            self._conn.commit()

    def add_edge(
        self, source: str, target: str, edge_type: str, weight: float = 1.0, metadata: dict | None = None
    ) -> None:
        import json

        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO graph_edges VALUES(?,?,?,?,?)",
                (source, target, edge_type, weight, json.dumps(metadata or {})),
            )
            self._conn.commit()

    def get_neighbors(
        self, node_id: str, edge_type: str | None = None, direction: str = "outgoing"
    ) -> list[tuple[str, str, float]]:
        with self._lock:
            if direction == "outgoing":
                if edge_type:
                    rows = self._conn.execute(
                        "SELECT target, type, weight FROM graph_edges WHERE source=? AND type=?", (node_id, edge_type)
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT target, type, weight FROM graph_edges WHERE source=?", (node_id,)
                    ).fetchall()
            else:
                if edge_type:
                    rows = self._conn.execute(
                        "SELECT source, type, weight FROM graph_edges WHERE target=? AND type=?", (node_id, edge_type)
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT source, type, weight FROM graph_edges WHERE target=?", (node_id,)
                    ).fetchall()
            return [(r[0], r[1], r[2]) for r in rows]

    def find_paths(self, source: str, target: str, max_depth: int = 5) -> list[list[str]]:
        with self._lock:
            rows = self._conn.execute(
                """
            WITH RECURSIVE paths(path, last_node, depth) AS (
                SELECT '|' || ?, ?, 0
                UNION ALL
                SELECT p.path || '|' || e.target, e.target, p.depth + 1
                FROM paths p JOIN graph_edges e ON p.last_node = e.source
                WHERE p.depth < ? AND p.path NOT LIKE '%|' || e.target || '|%'
            )
            SELECT path FROM paths WHERE last_node = ? AND depth > 0 LIMIT 10
            """,
                (source, source, max_depth, target),
            ).fetchall()
            return [r[0].strip("|").split("|") for r in rows if r[0]]

    def community_detection(self) -> dict[str, int]:
        return {}  # SQL backend doesn't support community detection

    def hallway_traversal(self, start: str, max_depth: int = 3) -> list[tuple[str, float]]:
        visited = set()
        queue = deque([(start, 0, 1.0)])
        results = []
        while queue:
            node, depth, score = queue.popleft()
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            results.append((node, score))
            neighbors = self.get_neighbors(node, edge_type=EdgeType.HALLWAY.value)
            neighbors += self.get_neighbors(node, edge_type=EdgeType.TUNNEL.value)
            for target, _, weight in neighbors:
                if target not in visited:
                    queue.append((target, depth + 1, score * weight))
        return results[1:]


def create_graph_backend(backend: str = "networkx", **kwargs: Any) -> GraphBackend:
    if backend == "sql":
        return SQLGraphBackend(**kwargs)
    return NetworkXGraphBackend()
