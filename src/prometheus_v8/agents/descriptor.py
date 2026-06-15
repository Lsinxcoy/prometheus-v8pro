"""Agent Descriptor - Agent metadata and capability description."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class AgentDescriptor:
    """Describes an agent's identity, capabilities, and configuration."""
    id: str = ""
    name: str = ""
    role: str = "worker"  # ceo/worker/explorer/judge
    model: str = ""
    capabilities: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    priority: int = 5
    status: str = "idle"  # idle/busy/dead
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    def can_handle(self, task_type: str) -> bool:
        return task_type in self.capabilities or "*" in self.capabilities
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "role": self.role,
            "model": self.model, "capabilities": self.capabilities,
            "status": self.status, "priority": self.priority,
        }
