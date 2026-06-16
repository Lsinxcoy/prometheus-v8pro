"""Dynamic Ontology Generator - LLM-driven ontology auto-generation for Prometheus V8."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EntityTypeDefinition:
    """A dynamically registered entity type."""

    name: str
    description: str = ""
    parent: str | None = None
    attributes: dict = field(default_factory=dict)


@dataclass
class RelationTypeDefinition:
    """A dynamically registered relation type."""

    name: str
    description: str = ""
    domain: str | None = None
    range: str | None = None
    symmetric: bool = False


class DynamicOntology:
    """Manages dynamically registered entity types and relation types.

    This extends the static NodeType/EdgeType enums with runtime-defined types,
    enabling the system to discover and register new knowledge categories on the fly.
    """

    def __init__(self) -> None:
        self._entity_types: dict[str, EntityTypeDefinition] = {}
        self._relation_types: dict[str, RelationTypeDefinition] = {}

    def register_entity_type(self, name: str, description: str = "", parent: str | None = None) -> EntityTypeDefinition:
        """Register a new entity type or update an existing one."""
        etype = EntityTypeDefinition(name=name, description=description, parent=parent)
        self._entity_types[name] = etype
        logger.debug(f"Registered entity type: {name} (parent={parent})")
        return etype

    def register_relation_type(self, name: str, description: str = "", domain: str | None = None, range_: str | None = None, symmetric: bool = False) -> RelationTypeDefinition:
        """Register a new relation type or update an existing one."""
        rtype = RelationTypeDefinition(name=name, description=description, domain=domain, range=range_, symmetric=symmetric)
        self._relation_types[name] = rtype
        logger.debug(f"Registered relation type: {name}")
        return rtype

    def get_entity_type(self, name: str) -> EntityTypeDefinition | None:
        """Get an entity type by name."""
        return self._entity_types.get(name)

    def get_relation_type(self, name: str) -> RelationTypeDefinition | None:
        """Get a relation type by name."""
        return self._relation_types.get(name)

    @property
    def entity_types(self) -> dict[str, EntityTypeDefinition]:
        """All registered entity types."""
        return dict(self._entity_types)

    @property
    def relation_types(self) -> dict[str, RelationTypeDefinition]:
        """All registered relation types."""
        return dict(self._relation_types)

    def entity_type_names(self) -> list[str]:
        """List all entity type names."""
        return list(self._entity_types.keys())

    def relation_type_names(self) -> list[str]:
        """List all relation type names."""
        return list(self._relation_types.keys())

    def get_children(self, parent_name: str) -> list[EntityTypeDefinition]:
        """Get all child entity types of a given parent."""
        return [et for et in self._entity_types.values() if et.parent == parent_name]

    def merge(self, other: DynamicOntology) -> None:
        """Merge another DynamicOntology into this one (other's types override on conflict)."""
        self._entity_types.update(other._entity_types)
        self._relation_types.update(other._relation_types)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "entity_types": {
                name: {"description": et.description, "parent": et.parent}
                for name, et in self._entity_types.items()
            },
            "relation_types": {
                name: {"description": rt.description, "domain": rt.domain, "range": rt.range, "symmetric": rt.symmetric}
                for name, rt in self._relation_types.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> DynamicOntology:
        """Deserialize from dictionary."""
        ont = cls()
        for name, info in data.get("entity_types", {}).items():
            ont.register_entity_type(name, description=info.get("description", ""), parent=info.get("parent"))
        for name, info in data.get("relation_types", {}).items():
            ont.register_relation_type(
                name,
                description=info.get("description", ""),
                domain=info.get("domain"),
                range_=info.get("range"),
                symmetric=info.get("symmetric", False),
            )
        return ont

    def __len__(self) -> int:
        return len(self._entity_types) + len(self._relation_types)

    def __repr__(self) -> str:
        return f"DynamicOntology(entities={len(self._entity_types)}, relations={len(self._relation_types)})"


class OntologyGenerator:
    """LLM-driven ontology auto-generation from text.

    Analyzes text to discover entity types and relation types,
    registering them into a DynamicOntology.
    """

    def __init__(self, llm=None) -> None:
        """Initialize with an optional LLM callable.

        The LLM should be a callable that takes a prompt string and returns a string response.
        If no LLM is provided, generate_from_text returns an empty ontology.
        """
        self._llm = llm

    def generate_from_text(self, text: str) -> DynamicOntology:
        """Analyze text and generate a DynamicOntology from discovered types.

        If no LLM is available, returns an empty DynamicOntology.
        """
        if self._llm is None:
            logger.debug("No LLM available for ontology generation, returning empty ontology")
            return DynamicOntology()

        prompt = (
            "Analyze the following text and extract:\n"
            "1. Entity types (categories of things mentioned), with a description and optional parent type\n"
            "2. Relation types (relationships between entities), with a description\n\n"
            "Return ONLY a JSON object with two keys:\n"
            '- "entity_types": array of {{"name": str, "description": str, "parent": str|null}}\n'
            '- "relation_types": array of {{"name": str, "description": str, "domain": str|null, "range": str|null, "symmetric": bool}}\n\n'
            f"Text:\n{text[:3000]}"
        )

        try:
            response = self._llm(prompt)
            text_resp = response.strip()
            # Handle markdown code blocks
            if text_resp.startswith("```"):
                text_resp = text_resp.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text_resp)

            ontology = DynamicOntology()

            for et_info in parsed.get("entity_types", []):
                name = et_info.get("name", "").strip()
                if name:
                    ontology.register_entity_type(
                        name=name,
                        description=et_info.get("description", ""),
                        parent=et_info.get("parent"),
                    )

            for rt_info in parsed.get("relation_types", []):
                name = rt_info.get("name", "").strip()
                if name:
                    ontology.register_relation_type(
                        name=name,
                        description=rt_info.get("description", ""),
                        domain=rt_info.get("domain"),
                        range_=rt_info.get("range"),
                        symmetric=rt_info.get("symmetric", False),
                    )

            logger.info(f"Generated ontology with {len(ontology.entity_types)} entity types, "
                        f"{len(ontology.relation_types)} relation types")
            return ontology

        except Exception as e:
            logger.warning(f"Ontology generation failed: {e}")
            return DynamicOntology()
