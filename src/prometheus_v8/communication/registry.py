"""Agent Registry - Heartbeat tracking + zombie reaping."""
from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from prometheus_v8.communication.bus import Message, MemoryBus

logger = logging.getLogger(__name__)

@dataclass
class AgentInfo:
    """Registered agent information."""
    agent_id: str = ""
    name: str = ""
    role: str = "worker"  # ceo/worker/explorer/judge
    model: str = ""
    channels: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    status: str = "active"  # active/idle/busy/dead
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    def is_alive(self, timeout: float = 60.0) -> bool:
        return time.time() - self.last_heartbeat < timeout


class AgentRegistry:
    """Registry of active agents with heartbeat monitoring and zombie reaping."""
    
    def __init__(self, heartbeat_timeout: float = 60.0, reap_interval: float = 30.0) -> None:
        self._agents: dict[str, AgentInfo] = {}
        self._heartbeat_timeout = heartbeat_timeout
        self._reap_interval = reap_interval
        self._lock = threading.RLock()
        self._reap_timer: threading.Timer | None = None
        self._on_agent_dead: list[Callable[[str], None]] = []
        self._start_reaper()
    
    def register(self, agent_id: str, name: str = "", role: str = "worker",
                 model: str = "", channels: list[str] | None = None,
                 capabilities: list[str] | None = None, **metadata: Any) -> AgentInfo:
        with self._lock:
            info = AgentInfo(
                agent_id=agent_id, name=name or agent_id, role=role, model=model,
                channels=channels or ["default"], capabilities=capabilities or [],
                metadata=metadata,
            )
            self._agents[agent_id] = info
            logger.info(f"Agent registered: {agent_id} ({role})")
            return info
    
    def unregister(self, agent_id: str) -> bool:
        with self._lock:
            info = self._agents.pop(agent_id, None)
            if info:
                logger.info(f"Agent unregistered: {agent_id}")
                return True
            return False
    
    def heartbeat(self, agent_id: str, status: str = "active") -> bool:
        with self._lock:
            info = self._agents.get(agent_id)
            if info:
                info.last_heartbeat = time.time()
                info.status = status
                return True
            return False
    
    def get(self, agent_id: str) -> AgentInfo | None:
        with self._lock:
            return self._agents.get(agent_id)
    
    def list_agents(self, role: str = "", status: str = "") -> list[AgentInfo]:
        with self._lock:
            agents = list(self._agents.values())
            if role:
                agents = [a for a in agents if a.role == role]
            if status:
                agents = [a for a in agents if a.status == status]
            return agents
    
    def find_by_capability(self, capability: str) -> list[AgentInfo]:
        with self._lock:
            return [a for a in self._agents.values() if capability in a.capabilities and a.is_alive(self._heartbeat_timeout)]
    
    def on_agent_dead(self, callback: Callable[[str], None]) -> None:
        self._on_agent_dead.append(callback)
    
    def _start_reaper(self) -> None:
        self._reap_timer = threading.Timer(self._reap_interval, self._reap_zombies)
        self._reap_timer.daemon = True
        self._reap_timer.start()
    
    def _reap_zombies(self) -> None:
        """Mark agents as dead if heartbeat timeout exceeded."""
        with self._lock:
            for agent_id, info in list(self._agents.items()):
                if info.status != "dead" and not info.is_alive(self._heartbeat_timeout):
                    info.status = "dead"
                    logger.warning(f"Agent zombie detected: {agent_id}, last heartbeat {time.time() - info.last_heartbeat:.0f}s ago")
                    for cb in self._on_agent_dead:
                        try:
                            cb(agent_id)
                        except Exception as e:
                            logger.warning(f"Agent dead callback error: {e}")
        
        # Schedule next reap
        self._start_reaper()
    
    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            active = sum(1 for a in self._agents.values() if a.status == "active")
            idle = sum(1 for a in self._agents.values() if a.status == "idle")
            busy = sum(1 for a in self._agents.values() if a.status == "busy")
            dead = sum(1 for a in self._agents.values() if a.status == "dead")
            return {"total": len(self._agents), "active": active, "idle": idle, "busy": busy, "dead": dead}
    
    def shutdown(self) -> None:
        if self._reap_timer:
            self._reap_timer.cancel()
