"""SQLite Store with WAL, FTS5, ACID, version control."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

from prometheus_v8.schema import (
    Edge,
    EdgeType,
    MemoryLayer,
    MemoryScope,
    Node,
    NodePayload,
    NodeType,
    Provenance,
    TrustLevel,
    Veracity,
    WeibullParams,
)

logger = logging.getLogger(__name__)


class SQLiteStore:
    """Thread-safe SQLite store with WAL, FTS5, and version control."""

    def __init__(
        self, db_path: str = "data/prometheus_v8.db", cache_size_mb: int = 64, mmap_size_mb: int = 512,
        event_bus=None,
    ) -> None:
        self._db_path = db_path
        self._cache_size = cache_size_mb
        self._mmap_size = mmap_size_mb
        self._local = threading.local()
        self._lock = threading.RLock()
        self._event_bus = event_bus
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute(f"PRAGMA cache_size=-{self._cache_size * 1024}")
            conn.execute(f"PRAGMA mmap_size={self._mmap_size * 1024 * 1024}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            logger.debug(f"Transaction rolled back: {e}")
            conn.rollback()
            raise

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id BLOB PRIMARY KEY,
            type TEXT NOT NULL,
            layer TEXT NOT NULL,
            scope TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            embedding BLOB,
            checksum TEXT NOT NULL DEFAULT '',
            importance REAL NOT NULL DEFAULT 0.5,
            confidence REAL NOT NULL DEFAULT 0.5,
            veracity TEXT NOT NULL DEFAULT 'unverified',
            trust_level TEXT NOT NULL DEFAULT 'pending',
            access_count INTEGER NOT NULL DEFAULT 0,
            consecutive_hits INTEGER NOT NULL DEFAULT 0,
            weibull_lam REAL NOT NULL DEFAULT 7.0,
            weibull_k REAL NOT NULL DEFAULT 0.8,
            provenance_source TEXT DEFAULT 'agent_output',
            provenance_agent_id TEXT,
            provenance_confidence REAL DEFAULT 0.5,
            tags TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            branch TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            accessed_at REAL NOT NULL,
            valid_from REAL NOT NULL DEFAULT 0,
            valid_to REAL NOT NULL DEFAULT 9e18
        );

        CREATE TABLE IF NOT EXISTS edges (
            id BLOB PRIMARY KEY,
            source_id BLOB NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            target_id BLOB NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            UNIQUE(source_id, target_id, type)
        );

        CREATE TABLE IF NOT EXISTS node_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id BLOB NOT NULL,
            content TEXT NOT NULL,
            importance REAL NOT NULL,
            confidence REAL NOT NULL,
            timestamp REAL NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            content, tags, tokenize='unicode61'
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
        CREATE INDEX IF NOT EXISTS idx_nodes_layer ON nodes(layer);
        CREATE INDEX IF NOT EXISTS idx_nodes_importance ON nodes(importance);
        CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
        CREATE INDEX IF NOT EXISTS idx_nodes_valid_from ON nodes(valid_from);
        CREATE INDEX IF NOT EXISTS idx_nodes_valid_to ON nodes(valid_to);
        """)

    def add_node(self, node: Node) -> bytes:
        with self._transaction() as conn:
            emb = node.payload.embedding
            conn.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    node.id,
                    node.type.value,
                    node.layer.value,
                    node.scope.value,
                    node.payload.content,
                    emb,
                    node.payload.checksum,
                    node.importance,
                    node.confidence,
                    node.veracity.value,
                    node.trust_level.value,
                    node.access_count,
                    node.consecutive_hits,
                    node.weibull.lam,
                    node.weibull.k,
                    node.provenance.source.value,
                    node.provenance.agent_id,
                    node.provenance.confidence,
                    json.dumps(node.tags),
                    json.dumps(node.metadata),
                    node.branch,
                    node.created_at,
                    node.updated_at,
                    node.accessed_at,
                    node.valid_from,
                    node.valid_to,
                ),
            )
            # FTS5 insert - use auto-increment rowid
            conn.execute("INSERT INTO nodes_fts(content,tags) VALUES(?,?)", (node.payload.content, " ".join(node.tags)))
        # Publish node_created event
        if self._event_bus:
            try:
                self._event_bus.publish("node_created", {"node": node, "node_id": node.id.hex()})
            except Exception as e:
                logger.debug(f"Event publish failed: {e}")
        return node.id

    def get_node(self, node_id: bytes) -> Optional[Node]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            return None
        node = self._row_to_node(row)
        node.touch()
        conn.execute(
            "UPDATE nodes SET access_count=?, consecutive_hits=?, accessed_at=?, updated_at=? WHERE id=?",
            (node.access_count, node.consecutive_hits, node.accessed_at, node.updated_at, node_id),
        )
        # Publish node_accessed event
        if self._event_bus:
            try:
                self._event_bus.publish("node_accessed", {"node": node, "node_id": node_id.hex()})
            except Exception as e:
                logger.debug(f"Event publish failed: {e}")
        return node

    def update_node(self, node: Node) -> None:
        node.updated_at = time.time()
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO node_versions(node_id,content,importance,confidence,timestamp) VALUES(?,?,?,?,?)",
                (node.id, node.payload.content, node.importance, node.confidence, time.time()),
            )
            # Direct upsert instead of calling add_node (avoids nested transaction)
            emb = node.payload.embedding
            conn.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    node.id,
                    node.type.value,
                    node.layer.value,
                    node.scope.value,
                    node.payload.content,
                    emb,
                    node.payload.checksum,
                    node.importance,
                    node.confidence,
                    node.veracity.value,
                    node.trust_level.value,
                    node.access_count,
                    node.consecutive_hits,
                    node.weibull.lam,
                    node.weibull.k,
                    node.provenance.source.value
                    if hasattr(node.provenance.source, "value")
                    else node.provenance.source,
                    node.provenance.agent_id,
                    node.provenance.confidence,
                    json.dumps(node.tags),
                    json.dumps(node.metadata),
                    node.branch,
                    node.created_at,
                    node.updated_at,
                    node.accessed_at,
                    node.valid_from,
                    node.valid_to,
                ),
            )
            conn.execute("INSERT INTO nodes_fts(content,tags) VALUES(?,?)", (node.payload.content, " ".join(node.tags)))

    def delete_node(self, node_id: bytes) -> bool:
        with self._transaction() as conn:
            conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
            return conn.total_changes > 0

    def search_fts(self, query: str, limit: int = 20) -> list[Node]:
        conn = self._get_conn()
        # Use FTS rowid to join with nodes table (efficient, avoids content mismatch)
        try:
            rows = conn.execute(
                "SELECT n.* FROM nodes_fts f JOIN nodes n ON n.rowid = f.rowid "
                "WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
            return [self._row_to_node(dict(r)) for r in rows if r]
        except Exception as e:
            # Fallback: content-based lookup if rowid join fails (e.g. schema mismatch)
            logger.debug(f"FTS rowid join failed, falling back to content lookup: {e}")
            fts_rows = conn.execute(
                "SELECT content FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?", (query, limit)
            ).fetchall()
            if not fts_rows:
                return []
            results = []
            for row in fts_rows:
                content = row["content"] if hasattr(row, "keys") else row[0]
                node_rows = conn.execute(
                    "SELECT * FROM nodes WHERE content=? ORDER BY importance DESC LIMIT 1", (content,)
                ).fetchall()
                for nr in node_rows:
                    try:
                        results.append(self._row_to_node(nr))
                    except Exception as e:
                        logger.debug(f"Row-to-node conversion failed: {e}")
            return results[:limit]

    def get_nodes_by_type(self, type_: NodeType, limit: int = 100) -> list[Node]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM nodes WHERE type=? ORDER BY importance DESC LIMIT ?", (type_.value, limit)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_nodes_by_layer(self, layer: MemoryLayer, limit: int = 100) -> list[Node]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM nodes WHERE layer=? ORDER BY importance DESC LIMIT ?", (layer.value, limit)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def count_nodes(self, layer: Optional[MemoryLayer] = None) -> int:
        conn = self._get_conn()
        if layer:
            row = conn.execute("SELECT COUNT(*) FROM nodes WHERE layer=?", (layer.value,)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return row[0]

    def add_edge(self, edge: Edge) -> bytes:
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?,?)",
                (
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.type.value,
                    edge.weight,
                    json.dumps(edge.metadata),
                    edge.created_at,
                ),
            )
        return edge.id

    def get_edges(self, node_id: bytes, direction: str = "outgoing") -> list[Edge]:
        conn = self._get_conn()
        if direction == "outgoing":
            rows = conn.execute("SELECT * FROM edges WHERE source_id=?", (node_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM edges WHERE target_id=?", (node_id,)).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_node_versions(self, node_id: bytes, limit: int = 10) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM node_versions WHERE node_id=? ORDER BY timestamp DESC LIMIT ?", (node_id, limit)
        ).fetchall()
        return [
            {"version_id": r[0], "content": r[2], "importance": r[3], "confidence": r[4], "timestamp": r[5]}
            for r in rows
        ]

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        return Node(
            id=bytes(row["id"]),
            type=NodeType(row["type"]),
            layer=MemoryLayer(row["layer"]),
            scope=MemoryScope(row["scope"]),
            payload=NodePayload(content=row["content"], embedding=row["embedding"], checksum=row["checksum"]),
            importance=row["importance"],
            confidence=row["confidence"],
            veracity=Veracity(row["veracity"]),
            trust_level=TrustLevel(row["trust_level"]),
            access_count=row["access_count"],
            consecutive_hits=row["consecutive_hits"],
            weibull=WeibullParams(lam=row["weibull_lam"], k=row["weibull_k"]),
            provenance=Provenance(
                source=row["provenance_source"] if isinstance(row["provenance_source"], str) else "agent_output",
                agent_id=row["provenance_agent_id"],
                confidence=row["provenance_confidence"],
            ),
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            branch=row["branch"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            accessed_at=row["accessed_at"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
        )

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        return Edge(
            id=bytes(row["id"]),
            source_id=bytes(row["source_id"]),
            target_id=bytes(row["target_id"]),
            type=EdgeType(row["type"]),
            weight=row["weight"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
        )

    def get_node_with_context(self, node_id: bytes, max_hops: int = 2) -> dict:
        """Get a node with enriched context: 1-2 hop neighbors, edges, and details.

        Args:
            node_id: The target node ID.
            max_hops: Maximum graph hops for neighbor expansion (1 or 2).

        Returns:
            Structured dict with node, edges, neighbors, and 2-hop neighbors.
        """
        result: dict[str, Any] = {
            "node": None,
            "edges": [],
            "neighbors": [],
            "hop2_neighbors": [],
        }

        # Get the central node
        node = self.get_node(node_id)
        if node is None:
            return result
        result["node"] = node

        # Get edges for the node
        outgoing = self.get_edges(node_id, direction="outgoing")
        incoming = self.get_edges(node_id, direction="incoming")
        all_edges = outgoing + incoming
        result["edges"] = all_edges

        # 1-hop neighbors
        neighbor_ids = set()
        neighbor_map: dict[bytes, tuple[Node, list[Edge]]] = {}
        for edge in all_edges:
            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            neighbor_ids.add(neighbor_id)

        for nid in neighbor_ids:
            neighbor = self.get_node(nid)
            if neighbor:
                neighbor_edges = self.get_edges(nid, direction="outgoing") + self.get_edges(nid, direction="incoming")
                neighbor_map[nid] = (neighbor, neighbor_edges)
                result["neighbors"].append({
                    "node": neighbor,
                    "edges": neighbor_edges,
                    "relationship": "1-hop",
                })

        # 2-hop neighbors (if max_hops >= 2)
        if max_hops >= 2:
            hop2_ids = set()
            for nid, (neighbor, neighbor_edges) in neighbor_map.items():
                for edge in neighbor_edges:
                    hop2_id = edge.target_id if edge.source_id == nid else edge.source_id
                    if hop2_id != node_id and hop2_id not in neighbor_ids:
                        hop2_ids.add(hop2_id)

            for hid in list(hop2_ids)[:20]:  # Limit to avoid explosion
                hop2_node = self.get_node(hid)
                if hop2_node:
                    result["hop2_neighbors"].append({
                        "node": hop2_node,
                        "relationship": "2-hop",
                    })

        return result

    def get_nodes_by_time_range(
        self, t_from: float, t_to: float, limit: int = 100, layer: str | None = None
    ) -> list[Node]:
        """Get nodes whose validity overlaps with [t_from, t_to].

        A node is considered active in the range if:
          valid_from <= t_to AND valid_to >= t_from
        This handles open-ended validity (valid_to=inf).

        Args:
            t_from: Start of time range (epoch seconds).
            t_to: End of time range (epoch seconds).
            limit: Maximum number of results.
            layer: Optional layer filter.

        Returns:
            List of Node objects sorted by importance descending.
        """
        conn = self._get_conn()
        if layer:
            rows = conn.execute(
                "SELECT id FROM nodes WHERE valid_from <= ? AND valid_to >= ? AND layer = ? "
                "ORDER BY importance DESC LIMIT ?",
                (t_to, t_from, layer, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id FROM nodes WHERE valid_from <= ? AND valid_to >= ? "
                "ORDER BY importance DESC LIMIT ?",
                (t_to, t_from, limit),
            ).fetchall()
        results = []
        for (nid_bytes,) in rows:
            node = self.get_node(nid_bytes)
            if node:
                results.append(node)
        return results

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


class MinervaStore:
    """Business-level store wrapper with hallway management."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def add_node(self, node: Node) -> bytes:
        return self._store.add_node(node)

    def get_node(self, node_id: bytes) -> Optional[Node]:
        return self._store.get_node(node_id)

    def update_node(self, node: Node) -> None:
        self._store.update_node(node)

    def delete_node(self, node_id: bytes) -> bool:
        return self._store.delete_node(node_id)

    def search(self, query: str, limit: int = 20) -> list[Node]:
        return self._store.search_fts(query, limit)

    def add_hallway(self, agent_id: bytes, mechanism_id: bytes, weight: float = 1.0) -> bytes:
        edge = Edge(source_id=agent_id, target_id=mechanism_id, type=EdgeType.HALLWAY, weight=weight)
        return self._store.add_edge(edge)

    def get_hallways(self, agent_id: bytes) -> list[Edge]:
        return self._store.get_edges(agent_id, "outgoing")

    @property
    def raw(self) -> SQLiteStore:
        return self._store
