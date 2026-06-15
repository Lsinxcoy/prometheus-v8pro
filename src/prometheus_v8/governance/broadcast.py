"""Broadcast Manager - L0-L4 broadcast with approval/rollback."""
from __future__ import annotations
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class BroadcastLevel(str, Enum):
    L0_INFO = "l0_info"          # Informational, no action needed
    L1_ADVISORY = "l1_advisory"  # Advisory, agents should note
    L2_DIRECTIVE = "l2_directive" # Directive, agents should follow
    L3_MANDATE = "l3_mandate"    # Mandatory, agents must follow
    L4_EMERGENCY = "l4_emergency" # Emergency, immediate action required

@dataclass
class Broadcast:
    id: str = ""
    level: BroadcastLevel = BroadcastLevel.L0_INFO
    sender: str = ""
    content: str = ""
    target_agents: list[str] = field(default_factory=list)  # empty = all
    requires_ack: bool = False
    requires_approval: bool = False
    rollback_data: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    acks: list[str] = field(default_factory=list)
    approved: bool = False
    rolled_back: bool = False

class BroadcastManager:
    """L0-L4 broadcast with approval and rollback support."""
    
    def __init__(self, bus=None) -> None:
        self._bus = bus
        self._broadcasts: dict[str, Broadcast] = {}
        self._lock = threading.RLock()
        self._approval_callbacks: list[Callable[[Broadcast], bool]] = []
        self._broadcast_count = 0
    
    def send(self, level: BroadcastLevel, content: str, sender: str = "",
             target_agents: list[str] | None = None, requires_approval: bool = False,
             rollback_data: dict | None = None, ttl: float = 3600.0) -> Broadcast:
        """Send a broadcast."""
        self._broadcast_count += 1
        bc = Broadcast(
            id=f"bc_{self._broadcast_count}_{int(time.time())}",
            level=level, sender=sender, content=content,
            target_agents=target_agents or [],
            requires_approval=requires_approval or level.value >= BroadcastLevel.L3_MANDATE.value,
            rollback_data=rollback_data or {},
            expires_at=time.time() + ttl,
        )
        
        if bc.requires_approval:
            approved = self._request_approval(bc)
            if not approved:
                logger.warning(f"Broadcast {bc.id} not approved")
                return bc
        
        bc.approved = True
        with self._lock:
            self._broadcasts[bc.id] = bc
        
        if self._bus:
            from prometheus_v8.communication.bus import Message
            self._bus.publish(Message(channel=f"broadcast_{level.value}", sender=sender,
                                     type=level.value, payload={"broadcast_id": bc.id, "content": content}))
        
        return bc
    
    def acknowledge(self, broadcast_id: str, agent_id: str) -> bool:
        with self._lock:
            bc = self._broadcasts.get(broadcast_id)
            if bc and agent_id not in bc.acks:
                bc.acks.append(agent_id)
                return True
            return False
    
    def rollback(self, broadcast_id: str) -> bool:
        with self._lock:
            bc = self._broadcasts.get(broadcast_id)
            if bc and not bc.rolled_back:
                bc.rolled_back = True
                logger.info(f"Broadcast {broadcast_id} rolled back")
                return True
            return False
    
    def get_active(self, level: BroadcastLevel | None = None) -> list[Broadcast]:
        now = time.time()
        with self._lock:
            bcs = [b for b in self._broadcasts.values() if not b.rolled_back and (b.expires_at == 0 or b.expires_at > now)]
            if level:
                bcs = [b for b in bcs if b.level == level]
            return bcs
    
    def on_approval_request(self, callback: Callable[[Broadcast], bool]) -> None:
        self._approval_callbacks.append(callback)
    
    def _request_approval(self, bc: Broadcast) -> bool:
        for cb in self._approval_callbacks:
            try:
                if cb(bc):
                    return True
            except Exception:
                pass
        # Auto-approve L0-L2
        return bc.level in (BroadcastLevel.L0_INFO, BroadcastLevel.L1_ADVISORY, BroadcastLevel.L2_DIRECTIVE)
    
    @property
    def stats(self) -> dict[str, Any]:
        return {"total": self._broadcast_count, "active": len(self.get_active())}
