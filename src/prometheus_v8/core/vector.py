"""Vector Store - 4 backends + compression + LRU cache."""

from __future__ import annotations
import hashlib
import logging
import math
import os
import pickle
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VectorBackend(ABC):
    """Abstract vector store backend."""

    @abstractmethod
    def add(self, id: str, vector: list[float], metadata: dict | None = None) -> None: ...

    @abstractmethod
    def search(self, query: list[float], k: int = 10) -> list[tuple[str, float]]: ...

    @abstractmethod
    def delete(self, id: str) -> bool: ...

    @abstractmethod
    def count(self) -> int: ...


class NumpyVectorBackend(VectorBackend):
    """Pure NumPy cosine similarity search."""

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._metadata: dict[str, dict] = {}
        self._lock = threading.RLock()

    def add(self, id: str, vector: list[float], metadata: dict | None = None) -> None:
        with self._lock:
            vec = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            if id in self._ids:
                idx = self._ids.index(id)
                self._vectors[idx] = vec
            else:
                self._ids.append(id)
                self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)]) if self._vectors.size else vec.reshape(1, -1)
            if metadata:
                self._metadata[id] = metadata

    def search(self, query: list[float], k: int = 10) -> list[tuple[str, float]]:
        with self._lock:
            if len(self._ids) == 0:
                return []
            q = np.array(query, dtype=np.float32)
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
            scores = self._vectors @ q
            top_k = min(k, len(self._ids))
            indices = np.argsort(scores)[-top_k:][::-1]
            return [(self._ids[i], float(scores[i])) for i in indices if scores[i] > 0]

    def delete(self, id: str) -> bool:
        with self._lock:
            if id not in self._ids:
                return False
            idx = self._ids.index(id)
            self._ids.pop(idx)
            self._vectors = np.delete(self._vectors, idx, axis=0)
            self._metadata.pop(id, None)
            return True

    def count(self) -> int:
        return len(self._ids)

    def persist(self, path: str) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"ids": self._ids, "vectors": self._vectors, "metadata": self._metadata}, f)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._ids = data["ids"]
            self._vectors = data["vectors"]
            self._metadata = data.get("metadata", {})


class HNSWVectorBackend(VectorBackend):
    """HNSW approximate nearest neighbor search via hnswlib."""

    def __init__(self, dimension: int = 384, max_elements: int = 100000, ef_construction: int = 200, M: int = 16) -> None:
        self._dim = dimension
        self._max_elements = max_elements
        self._index = None
        self._id_to_label: dict[str, int] = {}
        self._label_to_id: dict[int, str] = {}
        self._next_label = 0
        self._lock = threading.RLock()
        try:
            import hnswlib
            self._hnswlib = hnswlib
            self._index = hnswlib.Index(space="cosine", dim=dimension)
            self._index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        except ImportError:
            logger.warning("hnswlib not available, falling back to NumpyVectorBackend")
            self._fallback = NumpyVectorBackend(dimension)

    def add(self, id: str, vector: list[float], metadata: dict | None = None) -> None:
        with self._lock:
            if self._index is None:
                self._fallback.add(id, vector, metadata)
                return
            if id in self._id_to_label:
                label = self._id_to_label[id]
            else:
                label = self._next_label
                self._next_label += 1
            self._id_to_label[id] = label
            self._label_to_id[label] = id
            self._index.add_items([vector], [label])

    def search(self, query: list[float], k: int = 10) -> list[tuple[str, float]]:
        with self._lock:
            if self._index is None:
                return self._fallback.search(query, k)
            if self._next_label == 0:
                return []
            self._index.set_ef(min(k * 10, 500))
            labels, distances = self._index.knn_query([query], k=min(k, self._next_label))
            results = []
            for label, dist in zip(labels[0], distances[0]):
                id_ = self._label_to_id.get(int(label))
                if id_:
                    results.append((id_, 1.0 - float(dist)))
            return results

    def delete(self, id: str) -> bool:
        with self._lock:
            if id not in self._id_to_label:
                return False
            label = self._id_to_label.pop(id)
            self._label_to_id.pop(label, None)
            return True

    def count(self) -> int:
        return len(self._id_to_label)


class SQLiteVecBackend(VectorBackend):
    """SQLite-vec vector search backend."""

    def __init__(self, db_path: str = "data/vectors.db", dimension: int = 384) -> None:
        self._dim = dimension
        self._db_path = db_path
        self._conn = None
        self._lock = threading.RLock()
        try:
            import sqlite_vec
            self._sqlite_vec = sqlite_vec
            self._init_db()
        except ImportError:
            logger.warning("sqlite-vec not available, falling back to NumpyVectorBackend")
            self._fallback = NumpyVectorBackend(dimension)

    def _init_db(self) -> None:
        import sqlite3
        self._conn = sqlite3.connect(self._db_path)
        self._conn.enable_load_extension(True)
        self._conn.load_extension(self._sqlite_vec.loadable_path())
        self._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{self._dim}])")

    def add(self, id: str, vector: list[float], metadata: dict | None = None) -> None:
        with self._lock:
            if self._conn is None:
                self._fallback.add(id, vector, metadata)
                return
            emb = struct.pack(f"{self._dim}f", *vector)
            self._conn.execute("INSERT OR REPLACE INTO vec_items(rowid, embedding) VALUES(?, ?)", (hash(id), emb))
            self._conn.commit()

    def search(self, query: list[float], k: int = 10) -> list[tuple[str, float]]:
        with self._lock:
            if self._conn is None:
                return self._fallback.search(query, k)
            q = struct.pack(f"{self._dim}f", *query)
            rows = self._conn.execute(
                "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (q, k)
            ).fetchall()
            return [(str(r[0]), 1.0 - float(r[1])) for r in rows]

    def delete(self, id: str) -> bool:
        with self._lock:
            if self._conn is None:
                return self._fallback.delete(id)
            self._conn.execute("DELETE FROM vec_items WHERE rowid=?", (hash(id),))
            self._conn.commit()
            return True

    def count(self) -> int:
        if self._conn is None:
            return self._fallback.count()
        row = self._conn.execute("SELECT COUNT(*) FROM vec_items").fetchone()
        return row[0]


class LRUCache:
    """Thread-safe LRU cache with TTL."""

    def __init__(self, capacity: int = 10000, ttl_seconds: int = 300) -> None:
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._cache:
                value, ts = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(key)
                    return value
                del self._cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def create_vector_backend(backend: str = "numpy", dimension: int = 384, **kwargs: Any) -> VectorBackend:
    """Factory function for vector backends."""
    if backend == "hnsw":
        return HNSWVectorBackend(dimension=dimension, **kwargs)
    elif backend == "sqlitevec":
        return SQLiteVecBackend(dimension=dimension, **kwargs)
    else:
        return NumpyVectorBackend(dimension=dimension)
