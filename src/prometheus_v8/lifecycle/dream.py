"""DreamCycle - 5-stage offline consolidation."""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from prometheus_v8.schema import Node, MemoryLayer, NodeType, create_dream_node

logger = logging.getLogger(__name__)

class DreamCycle:
    """5-stage DreamCycle for offline consolidation:
    1. REPLAY: Revisit recent episodic memories
    2. ASSOCIATE: Find connections between memories
    3. CONSOLIDATE: Strengthen important, weaken unimportant
    4. GENERATE: Create new insights from patterns
    5. INTEGRATE: Merge new insights into semantic memory
    """
    
    def __init__(self, store=None, event_bus=None) -> None:
        self._store = store
        self._event_bus = event_bus
        self._dream_count = 0
        self._insights_generated = 0
    
    def dream(self, recent_nodes: list[Node] | None = None) -> list[Node]:
        """Run one dream cycle."""
        self._dream_count += 1
        start = time.time()
        
        if recent_nodes is None and self._store:
            recent_nodes = self._store.get_nodes_by_layer(MemoryLayer.EPISODIC, limit=50)
        if not recent_nodes:
            recent_nodes = []
        
        # Stage 1: REPLAY
        replayed = self._replay(recent_nodes)
        
        # Stage 2: ASSOCIATE
        associations = self._associate(replayed)
        
        # Stage 3: CONSOLIDATE
        consolidated = self._consolidate(replayed)
        
        # Stage 4: GENERATE
        insights = self._generate(replayed, associations)
        
        # Stage 5: INTEGRATE
        integrated = self._integrate(insights)
        
        elapsed = time.time() - start
        logger.info(f"Dream cycle #{self._dream_count} completed in {elapsed:.1f}s, {len(integrated)} insights")
        
        return integrated
    
    def _replay(self, nodes: list[Node]) -> list[Node]:
        """Stage 1: Replay recent memories, updating access counts."""
        for node in nodes:
            node.touch()
            if self._store:
                self._store.update_node(node)
        return nodes
    
    def _associate(self, nodes: list[Node]) -> list[tuple[Node, Node, float]]:
        """Stage 2: Find associations between memories based on content overlap."""
        associations = []
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                score = self._compute_association(n1, n2)
                if score > 0.3:
                    associations.append((n1, n2, score))
        return associations
    
    def _consolidate(self, nodes: list[Node]) -> list[Node]:
        """Stage 3: Strengthen important nodes, decay unimportant."""
        for node in nodes:
            if node.importance > 0.6:
                node.importance = min(1.0, node.importance * 1.05)
            else:
                node.importance *= 0.95
            if self._store:
                self._store.update_node(node)
        return nodes
    
    def _generate(self, nodes: list[Node], associations: list[tuple[Node, Node, float]]) -> list[Node]:
        """Stage 4: Generate new insights from patterns and associations."""
        insights = []
        
        # Generate from high-association pairs
        for n1, n2, score in associations[:5]:
            if score > 0.5:
                content = f"Pattern detected: '{n1.payload.content[:50]}' ↔ '{n2.payload.content[:50]}' (strength={score:.2f})"
                insight = create_dream_node(content=content, importance=score * 0.8)
                insight.layer = MemoryLayer.SEMANTIC
                insights.append(insight)
        
        self._insights_generated += len(insights)
        return insights
    
    def _integrate(self, insights: list[Node]) -> list[Node]:
        """Stage 5: Integrate insights into semantic memory."""
        integrated = []
        for insight in insights:
            if self._store:
                self._store.add_node(insight)
            integrated.append(insight)
        return integrated
    
    def _compute_association(self, n1: Node, n2: Node) -> float:
        """Compute association score between two nodes."""
        # Content overlap (Jaccard)
        w1 = set(n1.payload.content.lower().split())
        w2 = set(n2.payload.content.lower().split())
        if not w1 or not w2:
            return 0.0
        jaccard = len(w1 & w2) / len(w1 | w2)
        
        # Tag overlap
        tag_score = len(set(n1.tags) & set(n2.tags)) / max(1, len(set(n1.tags) | set(n2.tags)))
        
        # Type bonus
        type_bonus = 0.1 if n1.type == n2.type else 0.0
        
        return 0.5 * jaccard + 0.3 * tag_score + 0.2 * type_bonus
    
    @property
    def stats(self) -> dict[str, int]:
        return {"dream_cycles": self._dream_count, "insights_generated": self._insights_generated}
