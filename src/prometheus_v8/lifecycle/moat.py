"""Memory Moat - 3-layer defense (data/structure/governance).

From MiMo insights: Memory competitive moat has 3 layers:
1. Data moat: Unique data (easily copied)
2. Structure moat: How data connects (hard to copy)
3. Governance moat: Rules of engagement (impossible to copy)
"""
from __future__ import annotations
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from prometheus_v8.schema import Node, MemoryLayer, TrustLevel

logger = logging.getLogger(__name__)

@dataclass
class MoatAssessment:
    """Assessment of memory moat strength."""
    data_moat_score: float = 0.0
    structure_moat_score: float = 0.0
    governance_moat_score: float = 0.0
    composite_score: float = 0.0
    vulnerabilities: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

class MemoryMoat:
    """3-layer memory defense system.
    
    Data Layer: Unique knowledge that competitors don't have
    Structure Layer: How knowledge is connected (hallway/tunnel graph)
    Governance Layer: Trust levels, access rules, action hooks
    """
    
    def __init__(self, store=None) -> None:
        self._store = store
        self._assessment_history: list[MoatAssessment] = []
    
    def assess(self) -> MoatAssessment:
        """Assess the strength of the 3-layer memory moat."""
        assessment = MoatAssessment()
        
        total_nodes = 0
        verified_nodes = 0
        connected_nodes = 0
        governed_nodes = 0
        
        if self._store:
            from prometheus_v8.schema import MemoryLayer
            for layer in MemoryLayer:
                nodes = self._store.get_nodes_by_layer(layer, limit=100)
                total_nodes += len(nodes)
                for node in nodes:
                    if node.trust_level == TrustLevel.VERIFIED:
                        verified_nodes += 1
                    edges = self._store.get_edges(node.id)
                    if edges:
                        connected_nodes += 1
                    if node.action_hook or node.trust_level != TrustLevel.PENDING:
                        governed_nodes += 1
        
        # Data moat: ratio of unique verified knowledge
        assessment.data_moat_score = verified_nodes / max(1, total_nodes) * 100
        
        # Structure moat: ratio of connected knowledge
        assessment.structure_moat_score = connected_nodes / max(1, total_nodes) * 100
        
        # Governance moat: ratio of governed knowledge
        assessment.governance_moat_score = governed_nodes / max(1, total_nodes) * 100
        
        # Composite
        assessment.composite_score = (
            0.2 * assessment.data_moat_score +
            0.3 * assessment.structure_moat_score +
            0.5 * assessment.governance_moat_score
        )
        
        # Vulnerabilities
        if assessment.data_moat_score < 30:
            assessment.vulnerabilities.append("Low verified knowledge ratio")
            assessment.recommendations.append("Increase knowledge verification rate")
        if assessment.structure_moat_score < 30:
            assessment.vulnerabilities.append("Low graph connectivity")
            assessment.recommendations.append("Add more hallway/tunnel edges between related nodes")
        if assessment.governance_moat_score < 30:
            assessment.vulnerabilities.append("Low governance coverage")
            assessment.recommendations.append("Add action hooks and trust annotations to more nodes")
        
        self._assessment_history.append(assessment)
        return assessment
    
    @property
    def stats(self) -> dict[str, Any]:
        if not self._assessment_history:
            return {"assessments": 0}
        latest = self._assessment_history[-1]
        return {
            "assessments": len(self._assessment_history),
            "data_moat": latest.data_moat_score,
            "structure_moat": latest.structure_moat_score,
            "governance_moat": latest.governance_moat_score,
            "composite": latest.composite_score,
        }
