"""Minerva V2 Data Import Adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from prometheus_v8.schema import Node, NodeType, Provenance, ProvenanceType, create_fact_node

logger = logging.getLogger(__name__)


class MinervaImportAdapter:
    """Import data from Minerva V2 format."""

    def __init__(self, store=None) -> None:
        self._store = store
        self._imported_count = 0

    def import_file(self, path: str) -> int:
        """Import a Minerva V2 JSON file."""
        p = Path(path)
        if not p.exists():
            logger.error(f"File not found: {path}")
            return 0

        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return 0

        return self._import_data(data)

    def import_directory(self, dir_path: str) -> int:
        """Import all Minerva V2 files from a directory."""
        total = 0
        for p in Path(dir_path).rglob("*.json"):
            total += self.import_file(str(p))
        return total

    def _import_data(self, data: dict | list) -> int:
        count = 0
        items = data if isinstance(data, list) else [data]
        for item in items:
            node = self._convert_item(item)
            if node and self._store:
                self._store.add_node(node)
                count += 1
            self._imported_count += count
        return count

    def _convert_item(self, item: dict) -> Node | None:
        content = item.get("content", "") or item.get("text", "") or json.dumps(item, ensure_ascii=False)
        if not content:
            return None

        node_type_str = item.get("type", "fact")
        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            node_type = NodeType.FACT

        node = create_fact_node(
            content=content[:2000], importance=float(item.get("importance", 0.5)), tags=item.get("tags", [])
        )
        node.type = node_type
        node.provenance = Provenance(
            source=ProvenanceType.IMPORTED,
            agent_id=item.get("source_agent"),
            confidence=float(item.get("confidence", 0.5)),
        )
        return node
