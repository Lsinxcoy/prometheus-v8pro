"""Unified Knowledge Layer with gap detection, trust-aware access, and knowledge conversion pipeline.

From knowledge-conversion solution:
- 3-level trust: verified / high_signal / pending
- Action hooks: When X, do Y
- Revision rounds every 5 operations
- Structured <-> Unstructured bidirectional conversion
"""
from __future__ import annotations
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional
from prometheus_v8.schema import Node, NodeType, MemoryLayer, TrustLevel, create_fact_node, create_insight_node

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGap:
    """A detected gap in knowledge coverage."""
    domain: str = ""
    coverage: float = 0.0
    total_nodes: int = 0
    verified_nodes: int = 0
    recommendation: str = ""
    priority: int = 5

@dataclass
class ActionHook:
    """An action hook attached to knowledge: When X, do Y."""
    trigger: str = ""  # When condition
    action: str = ""   # Do action
    confidence: float = 0.5
    source_node_id: bytes = b""
    execution_count: int = 0
    last_triggered: float = 0.0

    def matches(self, context: str) -> bool:
        """Check if the trigger matches the current context."""
        return self.trigger.lower() in context.lower()

    def execute(self) -> str:
        """Execute the action hook and return the action description."""
        self.execution_count += 1
        self.last_triggered = time.time()
        return self.action

@dataclass
class ConversionResult:
    """Result of a knowledge conversion operation."""
    source_type: str = ""
    target_type: str = ""
    input_content: str = ""
    output_content: str = ""
    success: bool = False
    fidelity: float = 0.0  # How well the conversion preserved meaning
    metadata: dict = field(default_factory=dict)

class KnowledgeLayer:
    """Unified knowledge management with gap detection, trust-aware access,
    action hooks, and structured/unstructured conversion.
    
    Features:
    - Trust-aware access: get_verified(), get_reference(), get_all()
    - Gap detection: identify domains with low verified knowledge coverage
    - Action hooks: When X, do Y patterns attached to knowledge nodes
    - Knowledge conversion: structured <-> unstructured bidirectional
    - Revision rounds: forced quality review every N operations
    """

    def __init__(self, store=None, llm=None, revision_interval: int = 5) -> None:
        self._store = store
        self._llm = llm
        self._revision_interval = revision_interval
        self._operation_count = 0
        self._revision_count = 0
        self._gap_history: list[KnowledgeGap] = []
        self._action_hooks: list[ActionHook] = []
        self._conversion_history: list[ConversionResult] = []
        
        # Domain keywords for gap detection
        self._domains: dict[str, list[str]] = {
            "architecture": ["pattern", "design", "structure", "module", "component", "system"],
            "safety": ["security", "vulnerability", "attack", "defense", "threat", "risk"],
            "performance": ["optimization", "speed", "memory", "efficiency", "latency", "throughput"],
            "testing": ["test", "validation", "verification", "coverage", "assertion", "mock"],
            "evolution": ["mutation", "crossover", "fitness", "selection", "population", "genome"],
            "memory": ["consolidation", "decay", "retention", "recall", "forgetting", "weibull"],
            "governance": ["autonomy", "trust", "initiative", "curiosity", "broadcast", "moat"],
            "communication": ["redis", "stream", "message", "agent", "bus", "event"],
        }

    # ── Trust-Aware Access ─────────────────────────────────────

    def get_verified(self, query: str, limit: int = 10) -> list[Node]:
        """Get only verified knowledge for decision-making."""
        nodes = self._search(query, limit * 3)
        return [n for n in nodes if n.trust_level == TrustLevel.VERIFIED][:limit]

    def get_reference(self, query: str, limit: int = 10) -> list[Node]:
        """Get verified + high-signal knowledge for reference."""
        nodes = self._search(query, limit * 2)
        return [n for n in nodes if n.trust_level in (TrustLevel.VERIFIED, TrustLevel.HIGH_SIGNAL)][:limit]

    def get_all(self, query: str, limit: int = 20) -> list[Node]:
        """Get all knowledge matching query, regardless of trust level."""
        return self._search(query, limit)

    def _search(self, query: str, limit: int) -> list[Node]:
        if not self._store:
            return []
        return self._store.search_fts(query, limit)

    # ── Gap Detection ──────────────────────────────────────────

    def detect_gaps(self) -> list[KnowledgeGap]:
        """Detect knowledge gaps by analyzing coverage of key domains."""
        gaps = []
        for domain, keywords in self._domains.items():
            total = 0
            verified = 0
            for kw in keywords:
                nodes = self._search(kw, limit=20)
                total += len(nodes)
                verified += sum(1 for n in nodes if n.trust_level == TrustLevel.VERIFIED)
            
            coverage = verified / max(1, total)
            if coverage < 0.3:
                gaps.append(KnowledgeGap(
                    domain=domain, coverage=coverage,
                    total_nodes=total, verified_nodes=verified,
                    recommendation=f"Need more verified knowledge in {domain} (coverage: {coverage:.1%})",
                    priority=int((1 - coverage) * 10),
                ))
        
        self._gap_history.extend(gaps)
        return sorted(gaps, key=lambda g: g.priority, reverse=True)

    def get_gap_recommendations(self) -> list[str]:
        """Get recommendations for filling knowledge gaps."""
        gaps = self.detect_gaps()
        return [g.recommendation for g in gaps[:5]]

    # ── Action Hooks ───────────────────────────────────────────

    def add_action_hook(self, trigger: str, action: str, confidence: float = 0.5,
                        source_node_id: bytes = b"") -> ActionHook:
        """Add an action hook: When trigger, do action."""
        hook = ActionHook(
            trigger=trigger, action=action, confidence=confidence,
            source_node_id=source_node_id,
        )
        self._action_hooks.append(hook)
        return hook

    def check_action_hooks(self, context: str) -> list[ActionHook]:
        """Check which action hooks match the current context."""
        matching = []
        for hook in self._action_hooks:
            if hook.matches(context):
                matching.append(hook)
        return sorted(matching, key=lambda h: h.confidence, reverse=True)

    def execute_action_hooks(self, context: str) -> list[str]:
        """Execute all matching action hooks for a context."""
        results = []
        for hook in self.check_action_hooks(context):
            action = hook.execute()
            results.append(f"When {hook.trigger}, {action}")
        return results

    # ── Knowledge Conversion ───────────────────────────────────

    def structured_to_unstructured(self, node: Node) -> ConversionResult:
        """Convert structured knowledge (facts, rules) to natural language."""
        content = node.payload.content
        
        # Simple rule-based conversion
        if node.type == NodeType.FACT:
            output = f"It is known that {content}."
        elif node.type == NodeType.INSIGHT:
            output = f"An important insight: {content}."
        elif node.type == NodeType.MUTATION:
            output = f"A change was made: {content}."
        else:
            output = content
        
        result = ConversionResult(
            source_type=node.type.value, target_type="natural_language",
            input_content=content, output_content=output,
            success=True, fidelity=0.8,
        )
        self._conversion_history.append(result)
        self._check_revision()
        return result

    def unstructured_to_structured(self, text: str) -> ConversionResult:
        """Convert unstructured text to structured knowledge nodes."""
        # Extract potential facts using simple heuristics
        facts = []
        
        # Pattern: "X is Y" -> fact
        is_pattern = re.findall(r"([A-Z][^.]+) is ([^.]+)", text)
        for subject, definition in is_pattern:
            facts.append(f"{subject} is {definition}")
        
        # Pattern: "When X, do Y" -> action hook
        when_pattern = re.findall(r"[Ww]hen ([^,.]+),? (?:do |then |we should )?([^.]+)", text)
        for trigger, action in when_pattern:
            self.add_action_hook(trigger.strip(), action.strip())
        
        # Pattern: sentences with "important" or "key" -> insights
        insight_pattern = re.findall(r"(?:important|key|critical|essential)[^:]*[:]? ([^.]+)", text, re.IGNORECASE)
        for insight in insight_pattern:
            facts.append(insight.strip())
        
        # If no patterns found, treat whole text as a fact
        if not facts:
            facts = [text[:200]]
        
        # Store extracted facts
        for fact_content in facts:
            node = create_fact_node(content=fact_content, importance=0.5)
            if self._store:
                self._store.add_node(node)
        
        result = ConversionResult(
            source_type="natural_language", target_type="structured",
            input_content=text[:200], output_content=json.dumps(facts, ensure_ascii=False),
            success=True, fidelity=0.6, metadata={"facts_extracted": len(facts)},
        )
        self._conversion_history.append(result)
        self._check_revision()
        return result

    # ── Revision Rounds ────────────────────────────────────────

    def _check_revision(self) -> None:
        """Check if a revision round is needed (every N operations)."""
        self._operation_count += 1
        if self._operation_count % self._revision_interval == 0:
            self._run_revision()

    def _run_revision(self) -> None:
        """Run a forced revision round to verify knowledge quality."""
        self._revision_count += 1
        logger.info(f"Knowledge revision round #{self._revision_count} at operation {self._operation_count}")
        
        # Check for contradictions in recent conversions
        recent = self._conversion_history[-self._revision_interval:]
        for conv in recent:
            if conv.fidelity < 0.5:
                logger.warning(f"Low fidelity conversion: {conv.source_type} -> {conv.target_type} (fidelity={conv.fidelity:.2f})")

    # ── Statistics ─────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "operations": self._operation_count,
            "revisions": self._revision_count,
            "gaps_detected": len(self._gap_history),
            "action_hooks": len(self._action_hooks),
            "conversions": len(self._conversion_history),
            "domains_tracked": len(self._domains),
        }
