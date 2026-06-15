"""Dual Track Memory - Agent/User memory isolation."""
from __future__ import annotations
import logging
from typing import Any, Optional
from prometheus_v8.schema import Node, MemoryScope

logger = logging.getLogger(__name__)

class DualTrackMemory:
    """Separate agent and user memory spaces.
    
    Agent track: Internal learning, skills, patterns (scope=AGENT)
    User track: User preferences, facts, interactions (scope=USER)
    
    Prevents agent internal state from leaking to user context
    and vice versa. Cross-track access requires explicit bridging.
    """
    
    def __init__(self, store=None) -> None:
        self._store = store
        self._agent_buffer: list[Node] = []
        self._user_buffer: list[Node] = []
    
    def add_agent_memory(self, node: Node) -> bytes:
        """Add to agent memory track."""
        node.scope = MemoryScope.AGENT
        if self._store:
            return self._store.add_node(node)
        self._agent_buffer.append(node)
        return node.id
    
    def add_user_memory(self, node: Node) -> bytes:
        """Add to user memory track."""
        node.scope = MemoryScope.USER
        if self._store:
            return self._store.add_node(node)
        self._user_buffer.append(node)
        return node.id
    
    def get_agent_memories(self, query: str, limit: int = 10) -> list[Node]:
        """Get agent-track memories only."""
        if not self._store:
            return self._agent_buffer[:limit]
        nodes = self._store.search_fts(query, limit * 3)
        return [n for n in nodes if n.scope in (MemoryScope.AGENT, MemoryScope.GLOBAL)][:limit]
    
    def get_user_memories(self, query: str, limit: int = 10) -> list[Node]:
        """Get user-track memories only."""
        if not self._store:
            return self._user_buffer[:limit]
        nodes = self._store.search_fts(query, limit * 3)
        return [n for n in nodes if n.scope in (MemoryScope.USER, MemoryScope.GLOBAL)][:limit]
    
    def bridge_to_user(self, node_id: bytes) -> bool:
        """Bridge agent knowledge to user track (with trust check)."""
        if not self._store:
            return False
        node = self._store.get_node(node_id)
        if not node or node.trust_level.value not in ("verified", "high_signal"):
            return False
        node.scope = MemoryScope.USER
        self._store.update_node(node)
        return True
