"""Mnemosyne V3 Store Adapter."""

from __future__ import annotations

import logging
import time

from prometheus_v8.schema import Node

logger = logging.getLogger(__name__)


class MnemosyneAdapter:
    """Adapter for Mnemosyne V3 memory store."""

    def __init__(self, db_path: str = "data/mnemosyne_v3.db") -> None:
        self._db_path = db_path
        self._connected = False
        self._store = None

    def connect(self) -> bool:
        try:
            from prometheus_v8.core.store import SQLiteStore

            self._store = SQLiteStore(self._db_path)
            self._connected = True
            return True
        except Exception as e:
            logger.warning(f"Mnemosyne connection failed: {e}")
            return False

    def store_node(self, node: Node) -> bool:
        if not self._connected or not self._store:
            if not self.connect():
                return False
        try:
            self._store.add_node(node)
            return True
        except Exception as e:
            logger.warning(f"Mnemosyne store error: {e}")
            return False

    def retrieve_node(self, node_id: bytes) -> Node | None:
        if not self._connected or not self._store:
            if not self.connect():
                return None
        return self._store.get_node(node_id)

    def search(self, query: str, limit: int = 10) -> list[Node]:
        if not self._connected or not self._store:
            if not self.connect():
                return []
        return self._store.search_fts(query, limit)

    def transfer_hallway(self, from_agent: str, to_agent: str, node_ids: list[bytes]) -> int:
        """Transfer hallway nodes from one agent to another.

        For each node:
        1. Retrieve the node from the store
        2. Modify the metadata owner_agent to the target agent
        3. Store the modified node back
        4. Log the transfer in the audit trail
        """
        logger.info(f"Hallway transfer: {from_agent} → {to_agent}, {len(node_ids)} nodes requested")
        transferred = 0
        for nid in node_ids:
            node = self.retrieve_node(nid)
            if node:
                # Update owner_agent in metadata
                node.metadata["owner_agent"] = to_agent
                node.metadata["previous_owner"] = from_agent
                node.metadata["transfer_timestamp"] = time.time()
                # Save the modified node
                if self.store_node(node):
                    transferred += 1
                    logger.info(
                        f"Transferred node {nid.hex()} from {from_agent} to {to_agent}"
                    )
                else:
                    logger.warning(f"Failed to store transferred node {nid.hex()}")
            else:
                logger.warning(f"Node {nid.hex()} not found for hallway transfer")
        logger.info(
            f"Hallway transfer complete: {transferred}/{len(node_ids)} nodes actually transferred"
        )
        return transferred
