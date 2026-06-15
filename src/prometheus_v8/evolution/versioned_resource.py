"""Versioned Resources with Propose→Assess→Commit lifecycle.

Based on Autogenesis Protocol (AGP) - arXiv 2604.15034:
- All mutable components (prompts, tools, memory) are versioned resources
- Lifecycle: PROPOSED → ASSESSED → COMMITTED / ROLLED_BACK
- Every change has auditable lineage + rollback capability
"""
from __future__ import annotations
import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

class ResourceState(str, Enum):
    PROPOSED = "proposed"
    ASSESSED = "assessed"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"

@dataclass
class ResourceVersion:
    """A single version of a versioned resource."""
    version: int = 0
    content: Any = None
    fingerprint: str = ""
    state: ResourceState = ResourceState.PROPOSED
    author: str = ""
    reason: str = ""
    parent_version: int = -1
    created_at: float = field(default_factory=time.time)
    assessment_score: float = 0.0
    metadata: dict = field(default_factory=dict)

class VersionedResource:
    """A mutable component managed as a versioned resource.
    
    Supports: propose → assess → commit / rollback
    """
    
    def __init__(self, name: str, resource_type: str, initial_content: Any = None) -> None:
        self._name = name
        self._type = resource_type
        self._versions: list[ResourceVersion] = []
        self._current_version = -1
        
        if initial_content is not None:
            self._add_version(content=initial_content, author="init", reason="Initial version",
                            state=ResourceState.COMMITTED)
    
    def propose(self, content: Any, author: str = "", reason: str = "") -> ResourceVersion:
        """Propose a new version."""
        return self._add_version(content=content, author=author, reason=reason,
                               state=ResourceState.PROPOSED)
    
    def assess(self, version: int, score: float) -> bool:
        """Assess a proposed version."""
        if version < 0 or version >= len(self._versions):
            return False
        v = self._versions[version]
        if v.state != ResourceState.PROPOSED:
            return False
        v.state = ResourceState.ASSESSED
        v.assessment_score = score
        return True
    
    def commit(self, version: int) -> bool:
        """Commit an assessed version as the current version."""
        if version < 0 or version >= len(self._versions):
            return False
        v = self._versions[version]
        if v.state != ResourceState.ASSESSED:
            return False
        v.state = ResourceState.COMMITTED
        self._current_version = version
        logger.info(f"Resource {self._name} v{version} committed (score={v.assessment_score:.3f})")
        return True
    
    def rollback(self, to_version: int | None = None) -> bool:
        """Rollback to a previous committed version."""
        if to_version is None:
            to_version = self._find_last_committed()
        if to_version < 0 or to_version >= len(self._versions):
            return False
        
        # Mark current as rolled back
        if self._current_version >= 0:
            self._versions[self._current_version].state = ResourceState.ROLLED_BACK
        
        self._current_version = to_version
        self._versions[to_version].state = ResourceState.COMMITTED
        logger.info(f"Resource {self._name} rolled back to v{to_version}")
        return True
    
    def get_current(self) -> Any:
        if self._current_version < 0:
            return None
        return self._versions[self._current_version].content
    
    def get_version(self, version: int) -> ResourceVersion | None:
        if 0 <= version < len(self._versions):
            return self._versions[version]
        return None
    
    def list_versions(self) -> list[dict]:
        return [{"version": v.version, "state": v.state.value, "score": v.assessment_score,
                "author": v.author, "reason": v.reason, "created_at": v.created_at}
               for v in self._versions]
    
    def _add_version(self, content: Any, author: str, reason: str, state: ResourceState) -> ResourceVersion:
        version = len(self._versions)
        fingerprint = hashlib.sha256(str(content).encode()).hexdigest()[:16]
        rv = ResourceVersion(
            version=version, content=copy.deepcopy(content), fingerprint=fingerprint,
            state=state, author=author, reason=reason,
            parent_version=self._current_version,
        )
        self._versions.append(rv)
        if state == ResourceState.COMMITTED:
            self._current_version = version
        return rv
    
    def _find_last_committed(self) -> int:
        for i in range(len(self._versions) - 1, -1, -1):
            if self._versions[i].state in (ResourceState.COMMITTED, ResourceState.ROLLED_BACK):
                return i
        return -1
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def type(self) -> str:
        return self._type
    
    @property
    def current_version(self) -> int:
        return self._current_version

class ResourceManager:
    """Manages all versioned resources."""
    
    def __init__(self) -> None:
        self._resources: dict[str, VersionedResource] = {}
    
    def register(self, name: str, resource_type: str, initial_content: Any = None) -> VersionedResource:
        vr = VersionedResource(name, resource_type, initial_content)
        self._resources[name] = vr
        return vr
    
    def get(self, name: str) -> VersionedResource | None:
        return self._resources.get(name)
    
    def list_resources(self) -> list[dict]:
        return [{"name": r.name, "type": r.type, "current_version": r.current_version,
                "total_versions": len(r.list_versions())}
               for r in self._resources.values()]
