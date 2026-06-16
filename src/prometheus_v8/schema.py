"""Prometheus V8 Schema - Complete Data Model.

42+ NodeTypes, 40+ EdgeTypes, Provenance, Weibull, Trust, ActionHooks.
"""
from __future__ import annotations

import hashlib
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ── Enums ──────────────────────────────────────────────────────


class NodeType(str, Enum):
    EPISODE = "episode"
    CODE_UNIT = "code_unit"
    FACT = "fact"
    INSIGHT = "insight"
    PATTERN = "pattern"
    BELIEF = "belief"
    FORESIGHT = "foresight"
    MUTATION = "mutation"
    GENOME = "genome"
    EVOLUTION_RECORD = "evolution_record"
    SKILL = "skill"
    PROCEDURE = "procedure"
    COUNTERFACTUAL = "counterfactual"
    PREDICTION = "prediction"
    TOOL = "tool"
    AGENT = "agent"
    USER = "user"
    SESSION = "session"
    PROJECT = "project"
    HALLWAY = "hallway"
    TUNNEL = "tunnel"
    BRANCH = "branch"
    MERGE = "merge"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"
    AUDIT_LOG = "audit_log"
    DREAM_RECORD = "dream_record"
    CONSOLIDATION_RECORD = "consolidation_record"
    LEARNING_ROUND = "learning_round"
    CURIOSITY_ITEM = "curiosity_item"
    BROADCAST = "broadcast"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    MILESTONE = "milestone"
    ORGAN_OUTPUT = "organ_output"
    DIRECTION_RECORD = "direction_record"
    ANTI_EVOLUTION_ALERT = "anti_evolution_alert"
    METABOLISM_RECORD = "metabolism_record"
    AGING_REPORT = "aging_report"
    MOAT_ASSESSMENT = "moat_assessment"
    TRUST_ANNOTATION = "trust_annotation"
    KNOWLEDGE_GAP = "knowledge_gap"


class EdgeType(str, Enum):
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    REFERENTIAL = "referential"
    CONFLICT = "conflict"
    MODAL = "modal"
    CODE = "code"
    LIFECYCLE = "lifecycle"
    HALLWAY = "hallway"
    TUNNEL = "tunnel"
    RELATED_TO = "related_to"
    EVOLVES_TO = "evolves_to"
    MUTATED_FROM = "mutated_from"
    PROMOTED_TO = "promoted_to"
    DERIVED_FROM = "derived_from"
    CONSOLIDATED_FROM = "consolidated_from"
    DREAMED_FROM = "dreamed_from"
    MERGED_WITH = "merged_with"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    REFINES = "refines"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    DOCUMENTS = "documents"
    FOLLOWS = "follows"
    PRECEDES = "precedes"
    TRIGGERS = "triggers"
    FEEDBACK = "feedback"
    SUB_GOAL = "sub_goal"
    PARENT_GOAL = "parent_goal"
    SATISFIES = "satisfies"
    VIOLATES = "violates"
    ORIGINATED_FROM = "originated_from"
    APPLIED_IN = "applied_in"
    ARCHIVED_FROM = "archived_from"
    RESTORED_FROM = "restored_from"
    BRANCHES_FROM = "branches_from"
    SYNTHESIZED_FROM = "synthesized_from"


# ── Dynamic Type Registries ────────────────────────────────────


class NodeTypeRegistry:
    """Dynamic registry for NodeType values beyond the static enum.

    Allows runtime registration of new node types discovered via
    ontology generation or other dynamic mechanisms. Registered types
    are accessible via get() and can be used as NodeType-compatible strings.
    """

    _registry: dict[str, str] = {}  # name -> description
    _lock = threading.Lock() if __import__("threading") else None

    @classmethod
    def register(cls, name: str, description: str = "") -> None:
        """Register a new node type. Raises ValueError if name conflicts with static enum."""
        # Check for conflict with static enum
        for e in NodeType:
            if e.value == name:
                raise ValueError(f"Cannot register '{name}': conflicts with static NodeType enum value")

        if cls._lock:
            with cls._lock:
                cls._registry[name] = description
        else:
            cls._registry[name] = description

    @classmethod
    def get(cls, name: str) -> str | None:
        """Get a registered node type name, or None if not registered."""
        # First check static enum
        for e in NodeType:
            if e.value == name:
                return name
        return cls._registry.get(name)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a node type name is registered (static or dynamic)."""
        for e in NodeType:
            if e.value == name:
                return True
        return name in cls._registry

    @classmethod
    def all_types(cls) -> list[str]:
        """List all type names (static enum + dynamic)."""
        static = [e.value for e in NodeType]
        dynamic = list(cls._registry.keys())
        return static + dynamic

    @classmethod
    def dynamic_types(cls) -> dict[str, str]:
        """Return only dynamically registered types as {name: description}."""
        return dict(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Clear all dynamically registered types."""
        if cls._lock:
            with cls._lock:
                cls._registry.clear()
        else:
            cls._registry.clear()


class EdgeTypeRegistry:
    """Dynamic registry for EdgeType values beyond the static enum.

    Allows runtime registration of new edge types discovered via
    ontology generation or other dynamic mechanisms.
    """

    _registry: dict[str, str] = {}  # name -> description
    _lock = threading.Lock() if __import__("threading") else None

    @classmethod
    def register(cls, name: str, description: str = "") -> None:
        """Register a new edge type. Raises ValueError if name conflicts with static enum."""
        for e in EdgeType:
            if e.value == name:
                raise ValueError(f"Cannot register '{name}': conflicts with static EdgeType enum value")

        if cls._lock:
            with cls._lock:
                cls._registry[name] = description
        else:
            cls._registry[name] = description

    @classmethod
    def get(cls, name: str) -> str | None:
        """Get a registered edge type name, or None if not registered."""
        for e in EdgeType:
            if e.value == name:
                return name
        return cls._registry.get(name)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an edge type name is registered (static or dynamic)."""
        for e in EdgeType:
            if e.value == name:
                return True
        return name in cls._registry

    @classmethod
    def all_types(cls) -> list[str]:
        """List all type names (static enum + dynamic)."""
        static = [e.value for e in EdgeType]
        dynamic = list(cls._registry.keys())
        return static + dynamic

    @classmethod
    def dynamic_types(cls) -> dict[str, str]:
        """Return only dynamically registered types as {name: description}."""
        return dict(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Clear all dynamically registered types."""
        if cls._lock:
            with cls._lock:
                cls._registry.clear()
        else:
            cls._registry.clear()


class MemoryLayer(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    ARCHIVE = "archive"


class MemoryScope(str, Enum):
    GLOBAL = "global"
    AGENT = "agent"
    USER = "user"
    SESSION = "session"
    PROJECT = "project"


class ProvenanceType(str, Enum):
    USER_INPUT = "user_input"
    AGENT_OUTPUT = "agent_output"
    TOOL_RESULT = "tool_result"
    EVOLUTION = "evolution"
    CONSOLIDATION = "consolidation"
    DREAM = "dream"
    SPONTANEOUS = "spontaneous"
    IMPORTED = "imported"


class Veracity(str, Enum):
    UNVERIFIED = "unverified"
    HIGH = "high"
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    DISPUTED = "disputed"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    HYPOTHETICAL = "hypothetical"


class TrustLevel(str, Enum):
    """Three-level trust from knowledge-conversion solution."""

    PENDING = "pending"  # Single source, not yet verified
    HIGH_SIGNAL = "high_signal"  # 2+ independent cross-validated sources
    VERIFIED = "verified"  # Actually used and proven effective


# ── Helper Functions ───────────────────────────────────────────


def generate_uuidv7() -> bytes:
    """Generate UUIDv7 (time-ordered) as 16 bytes."""
    ts_ms = int(time.time() * 1000)
    # UUIDv7: 48-bit timestamp + 4-bit version + 12-bit random + 2-bit variant + 62-bit random
    rand_bits = uuid.uuid4().int
    uuid_int = (ts_ms & 0xFFFFFFFFFFFF) << 80  # 48-bit timestamp in top bits
    uuid_int |= 0x7 << 76  # version 7
    uuid_int |= 0x2 << 62  # variant 10
    uuid_int |= rand_bits & 0x3FFFFFFFFFFFFFFF  # 62 random bits
    uuid_int &= (1 << 128) - 1  # Clamp to 128 bits
    return uuid_int.to_bytes(16, "big")


def compute_checksum(data: str | bytes) -> str:
    """Compute SHA-256 checksum (first 16 hex chars)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def compute_weibull_retention(
    age_days: float, importance: float, lam: float, k: float, consecutive_hits: int = 0
) -> float:
    """Compute Weibull retention score.

    retention = importance * exp(-(age/λ)^k) * (1 + 0.1 * consecutive_hits)
    """
    import math

    if age_days <= 0:
        return min(1.0, importance * (1 + 0.1 * consecutive_hits))
    retention = importance * math.exp(-((age_days / lam) ** k))
    retention *= 1 + 0.1 * consecutive_hits
    return min(1.0, retention)


# ── Per-Layer Weibull Defaults ────────────────────────────────

WEIBULL_DEFAULTS: dict[MemoryLayer, tuple[float, float]] = {
    MemoryLayer.WORKING: (1.0, 0.5),
    MemoryLayer.EPISODIC: (7.0, 0.8),
    MemoryLayer.SEMANTIC: (30.0, 1.2),
    MemoryLayer.PROCEDURAL: (365.0, 1.5),
    MemoryLayer.ARCHIVE: (1095.0, 2.0),
}


# ── Data Classes ──────────────────────────────────────────────


@dataclass
class WeibullParams:
    """Weibull forgetting curve parameters per memory layer."""

    lam: float = 7.0  # scale (lambda)
    k: float = 0.8  # shape (kappa)

    @classmethod
    def for_layer(cls, layer: MemoryLayer) -> WeibullParams:
        defaults = WEIBULL_DEFAULTS.get(layer, (7.0, 0.8))
        return cls(lam=defaults[0], k=defaults[1])


@dataclass
class Provenance:
    """Full lineage tracking — from Protogonos Minerva V2."""

    source: ProvenanceType = ProvenanceType.AGENT_OUTPUT
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    confidence: float = 0.5
    lineage: list[str] = field(default_factory=list)  # parent_id → grandparent_id → ...


@dataclass
class TemporalTriple:
    """Time-bounded knowledge triple for temporal reasoning."""

    subject: str = ""
    predicate: str = ""
    obj: str = ""
    valid_from: float = 0.0
    valid_to: float = float("inf")

    def is_valid_at(self, timestamp: float) -> bool:
        return self.valid_from <= timestamp <= self.valid_to


@dataclass
class ActionHook:
    """When X happens, do Y — from knowledge-conversion solution.

    Not 'I should...' (wish) but 'When X, do Y' (trigger + action).
    """

    trigger: str = ""  # Condition: "When encountering X scenario"
    action: str = ""  # Specific action: "Execute Y operation"
    priority: int = 5  # 1=highest, 10=lowest
    last_triggered: float = 0.0
    trigger_count: int = 0

    def should_trigger(self, context: str) -> bool:
        """Check if context matches trigger condition."""
        keywords = [kw.strip() for kw in self.trigger.lower().split(",")]
        context_lower = context.lower()
        return any(kw in context_lower for kw in keywords if kw)

    def record_trigger(self) -> None:
        self.last_triggered = time.time()
        self.trigger_count += 1


@dataclass
class PersonaProfile:
    """Agent persona profile - MBTI, sentiment, stance, and activity patterns."""

    mbti: str = ""
    sentiment_bias: float = 0.0  # -1.0 ~ 1.0
    stance: str = "neutral"
    activity_pattern: dict = field(default_factory=dict)  # 活跃时段
    influence_weight: float = 1.0
    interested_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mbti": self.mbti,
            "sentiment_bias": self.sentiment_bias,
            "stance": self.stance,
            "activity_pattern": self.activity_pattern,
            "influence_weight": self.influence_weight,
            "interested_topics": self.interested_topics,
        }


@dataclass
class NodePayload:
    """Content payload of a memory node."""

    content: str = ""
    modality: str = "text"  # text/code/image/audio/video/table/graph
    embedding: Optional[bytes] = None  # struct.pack compressed float32 array
    checksum: str = ""

    def __post_init__(self) -> None:
        if self.content and not self.checksum:
            self.checksum = compute_checksum(self.content)

    def set_embedding(self, vec: list[float] | None) -> None:
        """Compress float list to bytes."""
        if vec is None:
            self.embedding = None
        else:
            self.embedding = struct.pack(f"{len(vec)}f", *vec)

    def get_embedding(self) -> list[float] | None:
        """Decompress bytes to float list."""
        if self.embedding is None:
            return None
        count = len(self.embedding) // 4
        return list(struct.unpack(f"{count}f", self.embedding))


@dataclass
class Node:
    """Core memory unit — the atom of Prometheus V8."""

    id: bytes = b""
    type: NodeType = NodeType.EPISODE
    layer: MemoryLayer = MemoryLayer.WORKING
    scope: MemoryScope = MemoryScope.GLOBAL
    payload: NodePayload = field(default_factory=NodePayload)
    provenance: Provenance = field(default_factory=Provenance)
    weibull: WeibullParams = field(default_factory=WeibullParams)
    temporal: Optional[TemporalTriple] = None
    action_hook: Optional[ActionHook] = None
    veracity: Veracity = Veracity.UNVERIFIED
    trust_level: TrustLevel = TrustLevel.PENDING

    importance: float = 0.5
    confidence: float = 0.5
    access_count: int = 0
    consecutive_hits: int = 0
    branch: Optional[str] = None

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    valid_from: float = field(default_factory=time.time)
    valid_to: float = float("inf")

    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = generate_uuidv7()
        if isinstance(self.type, str):
            self.type = NodeType(self.type)
        if isinstance(self.layer, str):
            self.layer = MemoryLayer(self.layer)

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.created_at) / 86400)

    @property
    def retention(self) -> float:
        return compute_weibull_retention(
            self.age_days,
            self.importance,
            self.weibull.lam,
            self.weibull.k,
            self.consecutive_hits,
        )

    def touch(self) -> None:
        """Record access — increment counters and update timestamps."""
        self.access_count += 1
        self.consecutive_hits += 1
        self.accessed_at = time.time()
        self.updated_at = time.time()

    def decay_hits(self) -> None:
        """Reset consecutive hits (e.g., on consolidation)."""
        self.consecutive_hits = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id.hex(),
            "type": self.type.value,
            "layer": self.layer.value,
            "scope": self.scope.value,
            "content": self.payload.content,
            "checksum": self.payload.checksum,
            "importance": self.importance,
            "confidence": self.confidence,
            "veracity": self.veracity.value,
            "trust_level": self.trust_level.value,
            "access_count": self.access_count,
            "consecutive_hits": self.consecutive_hits,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class Edge:
    """Directed edge between memory nodes."""

    id: bytes = b""
    source_id: bytes = b""
    target_id: bytes = b""
    type: EdgeType = EdgeType.RELATED_TO
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = generate_uuidv7()
        if isinstance(self.type, str):
            self.type = EdgeType(self.type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id.hex(),
            "source_id": self.source_id.hex(),
            "target_id": self.target_id.hex(),
            "type": self.type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class Genome:
    """Evolutionary genome — code as genotype."""

    code: str = ""
    fitness: float = 0.0
    age: int = 0
    lineage: list[str] = field(default_factory=list)
    fingerprint: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    memory_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code and not self.fingerprint:
            self.fingerprint = compute_checksum(self.code)


@dataclass
class FitnessResult:
    """3-stage fitness evaluation result."""

    composite: float = 0.0
    static_score: float = 0.0
    dynamic_score: float = 0.0
    llm_score: float = 0.0
    can_promote: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# ── Factory Functions ──────────────────────────────────────────


def _make_node(
    type_: NodeType,
    layer: MemoryLayer,
    scope: MemoryScope,
    importance: float,
    confidence: float,
    content: str,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Node:
    weibull = WeibullParams.for_layer(layer)
    payload = NodePayload(content=content)
    return Node(
        type=type_,
        layer=layer,
        scope=scope,
        importance=importance,
        confidence=confidence,
        payload=payload,
        weibull=weibull,
        tags=tags or [],
        **kwargs,
    )


def create_episode_node(content: str, importance: float = 0.4, **kw: Any) -> Node:
    return _make_node(NodeType.EPISODE, MemoryLayer.EPISODIC, MemoryScope.AGENT, importance, 0.6, content, **kw)


def create_fact_node(content: str, importance: float = 0.6, **kw: Any) -> Node:
    return _make_node(NodeType.FACT, MemoryLayer.SEMANTIC, MemoryScope.GLOBAL, importance, 0.7, content, **kw)


def create_insight_node(content: str, importance: float = 0.7, **kw: Any) -> Node:
    return _make_node(NodeType.INSIGHT, MemoryLayer.SEMANTIC, MemoryScope.GLOBAL, importance, 0.75, content, **kw)


def create_pattern_node(content: str, importance: float = 0.65, **kw: Any) -> Node:
    return _make_node(NodeType.PATTERN, MemoryLayer.SEMANTIC, MemoryScope.GLOBAL, importance, 0.7, content, **kw)


def create_belief_node(content: str, importance: float = 0.8, **kw: Any) -> Node:
    return _make_node(
        NodeType.BELIEF,
        MemoryLayer.SEMANTIC,
        MemoryScope.GLOBAL,
        importance,
        0.8,
        content,
        trust_level=TrustLevel.HIGH_SIGNAL,
        **kw,
    )


def create_foresight_node(content: str, importance: float = 0.6, **kw: Any) -> Node:
    return _make_node(
        NodeType.FORESIGHT,
        MemoryLayer.SEMANTIC,
        MemoryScope.GLOBAL,
        importance,
        0.5,
        content,
        veracity=Veracity.HYPOTHETICAL,
        **kw,
    )


def create_mutation_node(content: str, importance: float = 0.5, **kw: Any) -> Node:
    return _make_node(
        NodeType.MUTATION,
        MemoryLayer.WORKING,
        MemoryScope.AGENT,
        importance,
        0.5,
        content,
        provenance=Provenance(source=ProvenanceType.EVOLUTION),
        **kw,
    )


def create_skill_node(content: str, importance: float = 0.7, **kw: Any) -> Node:
    return _make_node(NodeType.SKILL, MemoryLayer.PROCEDURAL, MemoryScope.AGENT, importance, 0.75, content, **kw)


def create_procedure_node(content: str, importance: float = 0.65, **kw: Any) -> Node:
    return _make_node(NodeType.PROCEDURE, MemoryLayer.PROCEDURAL, MemoryScope.AGENT, importance, 0.7, content, **kw)


def create_dream_node(content: str, importance: float = 0.5, **kw: Any) -> Node:
    return _make_node(
        NodeType.DREAM_RECORD,
        MemoryLayer.EPISODIC,
        MemoryScope.AGENT,
        importance,
        0.5,
        content,
        provenance=Provenance(source=ProvenanceType.DREAM),
        **kw,
    )


def create_consolidation_node(content: str, importance: float = 0.6, **kw: Any) -> Node:
    return _make_node(
        NodeType.CONSOLIDATION_RECORD,
        MemoryLayer.SEMANTIC,
        MemoryScope.AGENT,
        importance,
        0.6,
        content,
        provenance=Provenance(source=ProvenanceType.CONSOLIDATION),
        **kw,
    )


def create_learning_node(content: str, importance: float = 0.5, **kw: Any) -> Node:
    return _make_node(
        NodeType.LEARNING_ROUND,
        MemoryLayer.EPISODIC,
        MemoryScope.AGENT,
        importance,
        0.5,
        content,
        provenance=Provenance(source=ProvenanceType.SPONTANEOUS),
        **kw,
    )


def create_curiosity_node(content: str, importance: float = 0.4, priority: int = 5, **kw: Any) -> Node:
    hook = ActionHook(trigger=content, action=f"Explore: {content}", priority=priority)
    return _make_node(
        NodeType.CURIOSITY_ITEM,
        MemoryLayer.WORKING,
        MemoryScope.AGENT,
        importance,
        0.4,
        content,
        action_hook=hook,
        **kw,
    )


def create_broadcast_node(content: str, importance: float = 0.7, **kw: Any) -> Node:
    return _make_node(NodeType.BROADCAST, MemoryLayer.WORKING, MemoryScope.GLOBAL, importance, 0.7, content, **kw)
