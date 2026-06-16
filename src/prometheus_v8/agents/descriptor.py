"""Agent Descriptor and Registry - Agent metadata, capabilities, and lifecycle management."""
from __future__ import annotations
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class AgentRole(str, Enum):
    CEO = "ceo"             # Strategic decisions, uses pro model
    WORKER = "worker"       # Task execution, uses standard model
    EXPLORER = "explorer"   # Knowledge exploration and gap filling
    JUDGE = "judge"         # Fitness evaluation and quality assessment
    GUARDIAN = "guardian"   # Safety monitoring and enforcement

class AgentState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    DEAD = "dead"
    OFFLINE = "offline"

@dataclass
class AgentDescriptor:
    """Complete agent descriptor with identity, capabilities, and runtime state."""
    id: str = ""
    name: str = ""
    role: AgentRole = AgentRole.WORKER
    model: str = ""
    model_tier: str = "standard"  # pro/standard/light
    capabilities: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    priority: int = 5
    state: AgentState = AgentState.IDLE
    current_tasks: list[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        return self.state in (AgentState.IDLE, AgentState.WAITING) and len(self.current_tasks) < self.max_concurrent_tasks

    @property
    def success_rate(self) -> float:
        total = self.completed_tasks + self.failed_tasks
        return self.completed_tasks / max(1, total)

    def can_handle(self, task_type: str) -> bool:
        return task_type in self.capabilities or "*" in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "role": self.role.value,
            "model": self.model, "model_tier": self.model_tier,
            "capabilities": self.capabilities, "state": self.state.value,
            "priority": self.priority, "is_available": self.is_available,
            "success_rate": round(self.success_rate, 3),
            "current_tasks": len(self.current_tasks),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
        }

class AgentPool:
    """Pool of agent descriptors with capability-based selection.
    
    Features:
    - Register/deregister agents
    - Find available agents by capability
    - Model tier matching (CEO tasks -> pro, Worker tasks -> standard)
    - Agent health monitoring via heartbeats
    - Automatic dead agent detection
    - Load balancing across available agents
    """

    def __init__(self, heartbeat_timeout: float = 90.0,
                 dead_threshold: float = 300.0) -> None:
        self._agents: dict[str, AgentDescriptor] = {}
        self._heartbeat_timeout = heartbeat_timeout
        self._dead_threshold = dead_threshold
        self._lock = threading.RLock()
        self._event_callbacks: list[Callable[[str, AgentDescriptor], None]] = []

    def register(self, agent: AgentDescriptor) -> None:
        """Register an agent in the pool."""
        if not agent.id:
            agent.id = str(uuid.uuid4())[:8]
        with self._lock:
            self._agents[agent.id] = agent
        logger.info(f"Agent registered: {agent.name} ({agent.role.value}, tier={agent.model_tier})")
        self._fire_event("registered", agent)

    def deregister(self, agent_id: str) -> bool:
        """Deregister an agent from the pool."""
        with self._lock:
            agent = self._agents.pop(agent_id, None)
        if agent:
            logger.info(f"Agent deregistered: {agent.name}")
            self._fire_event("deregistered", agent)
            return True
        return False

    def heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat timestamp."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                agent.last_heartbeat = time.time()
                if agent.state == AgentState.OFFLINE:
                    agent.state = AgentState.IDLE
                return True
        return False

    def assign_task(self, agent_id: str, task_id: str) -> bool:
        """Assign a task to an agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent and agent.is_available:
                agent.current_tasks.append(task_id)
                agent.state = AgentState.BUSY
                return True
        return False

    def complete_task(self, agent_id: str, task_id: str, success: bool = True) -> None:
        """Mark a task as completed on an agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                if task_id in agent.current_tasks:
                    agent.current_tasks.remove(task_id)
                if success:
                    agent.completed_tasks += 1
                else:
                    agent.failed_tasks += 1
                if not agent.current_tasks:
                    agent.state = AgentState.IDLE

    def find_available(self, required_capability: str = "",
                       model_tier: str = "",
                       role: AgentRole | None = None) -> list[AgentDescriptor]:
        """Find available agents matching criteria."""
        with self._lock:
            candidates = [a for a in self._agents.values() if a.is_available]
        
        if required_capability:
            candidates = [a for a in candidates if a.can_handle(required_capability)]
        if model_tier:
            candidates = [a for a in candidates if a.model_tier == model_tier]
        if role:
            candidates = [a for a in candidates if a.role == role]
        
        # Sort by success rate (descending), then by load (ascending)
        candidates.sort(key=lambda a: (-a.success_rate, len(a.current_tasks)))
        return candidates

    def find_best(self, required_capability: str = "",
                  model_tier: str = "") -> AgentDescriptor | None:
        """Find the single best available agent for a task."""
        candidates = self.find_available(required_capability, model_tier)
        return candidates[0] if candidates else None

    def check_health(self) -> dict[str, int]:
        """Check health of all agents and mark dead ones."""
        now = time.time()
        result = {"healthy": 0, "degraded": 0, "dead": 0, "recovered": 0}
        
        with self._lock:
            for agent in self._agents.values():
                age = now - agent.last_heartbeat
                
                if age > self._dead_threshold:
                    if agent.state != AgentState.DEAD:
                        agent.state = AgentState.DEAD
                        logger.warning(f"Agent {agent.name} declared dead (no heartbeat for {age:.0f}s)")
                        self._fire_event("dead", agent)
                    result["dead"] += 1
                elif age > self._heartbeat_timeout:
                    if agent.state == AgentState.IDLE or agent.state == AgentState.BUSY:
                        agent.state = AgentState.OFFLINE
                        logger.info(f"Agent {agent.name} went offline (no heartbeat for {age:.0f}s)")
                    result["degraded"] += 1
                else:
                    if agent.state == AgentState.OFFLINE:
                        agent.state = AgentState.IDLE
                        result["recovered"] += 1
                    result["healthy"] += 1
        
        return result

    def add_event_callback(self, callback: Callable[[str, AgentDescriptor], None]) -> None:
        """Add a callback for agent lifecycle events."""
        self._event_callbacks.append(callback)

    def _fire_event(self, event_type: str, agent: AgentDescriptor) -> None:
        for cb in self._event_callbacks:
            try:
                cb(event_type, agent)
            except Exception as e:
                logger.warning(f"Agent event callback error: {e}")

    def get_all(self) -> list[AgentDescriptor]:
        with self._lock:
            return list(self._agents.values())

    def get_by_role(self, role: AgentRole) -> list[AgentDescriptor]:
        with self._lock:
            return [a for a in self._agents.values() if a.role == role]

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_agents": len(self._agents),
                "available": sum(1 for a in self._agents.values() if a.is_available),
                "busy": sum(1 for a in self._agents.values() if a.state == AgentState.BUSY),
                "dead": sum(1 for a in self._agents.values() if a.state == AgentState.DEAD),
                "by_role": {r.value: sum(1 for a in self._agents.values() if a.role == r) for r in AgentRole},
                "by_tier": defaultdict(int, {a.model_tier: sum(1 for b in self._agents.values() if b.model_tier == a.model_tier) for a in self._agents.values()}),
            }
