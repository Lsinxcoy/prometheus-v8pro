"""Unified Knowledge Layer with gap detection."""
from __future__ import annotations
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional
from prometheus_v8.schema import Node, NodeType, MemoryLayer, TrustLevel

logger = logging.getLogger(__name__)

class KnowledgeLayer:
    """Unified knowledge management with gap detection and trust-aware access."""
    
    def __init__(self, store=None) -> None:
        self._store = store
        self._gap_history: list[dict] = []
    
    def get_verified(self, query: str, limit: int = 10) -> list[Node]:
        """Get only verified knowledge for decision-making."""
        nodes = self._store.search_fts(query, limit * 3) if self._store else []
        return [n for n in nodes if n.trust_level == TrustLevel.VERIFIED][:limit]
    
    def get_reference(self, query: str, limit: int = 10) -> list[Node]:
        """Get verified + high-signal knowledge for reference."""
        nodes = self._store.search_fts(query, limit * 2) if self._store else []
        return [n for n in nodes if n.trust_level in (TrustLevel.VERIFIED, TrustLevel.HIGH_SIGNAL)][:limit]
    
    def detect_gaps(self) -> list[dict]:
        """Detect knowledge gaps by analyzing coverage of key domains."""
        if not self._store:
            return []
        
        domains = {
            "architecture": ["pattern", "design", "structure", "module"],
            "safety": ["security", "vulnerability", "attack", "defense"],
            "performance": ["optimization", "speed", "memory", "efficiency"],
            "testing": ["test", "validation", "verification", "coverage"],
            "evolution": ["mutation", "crossover", "fitness", "selection"],
        }
        
        gaps = []
        for domain, keywords in domains.items():
            total = 0
            verified = 0
            for kw in keywords:
                nodes = self._store.search_fts(kw, limit=10)
                total += len(nodes)
                verified += sum(1 for n in nodes if n.trust_level == TrustLevel.VERIFIED)
            
            coverage = verified / max(1, total)
            if coverage < 0.3:
                gaps.append({
                    "domain": domain, "coverage": coverage,
                    "total_nodes": total, "verified_nodes": verified,
                    "recommendation": f"Need more verified knowledge in {domain}",
                })
        
        self._gap_history.extend(gaps)
        return gaps
    
    @property
    def stats(self) -> dict[str, Any]:
        return {"gaps_detected": len(self._gap_history)}
