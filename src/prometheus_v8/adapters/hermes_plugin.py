"""Hermes Agent Plugin Interface."""
from __future__ import annotations
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class HermesPluginAdapter:
    """Adapter for Hermes Agent plugin integration.
    
    Allows Prometheus V8 to be used as a Hermes memory provider.
    """
    
    def __init__(self, store=None) -> None:
        self._store = store
        self._name = "prometheus-v8"
    
    def save(self, key: str, value: str, metadata: dict | None = None) -> bool:
        """Save a memory entry (Hermes plugin interface)."""
        from prometheus_v8.schema import create_fact_node
        node = create_fact_node(content=f"{key}: {value}", tags=[key])
        if metadata:
            node.metadata = metadata
        if self._store:
            self._store.add_node(node)
        return True
    
    def recall(self, query: str, limit: int = 5) -> list[str]:
        """Recall memories matching query (Hermes plugin interface)."""
        if not self._store:
            return []
        nodes = self._store.search_fts(query, limit)
        return [n.payload.content for n in nodes]
    
    def forget(self, key: str) -> bool:
        """Remove a memory by key."""
        if not self._store:
            return False
        nodes = self._store.search_fts(key, limit=10)
        for node in nodes:
            if key in node.tags or key in node.payload.content:
                self._store.delete_node(node.id)
        return True
    
    @property
    def name(self) -> str:
        return self._name
