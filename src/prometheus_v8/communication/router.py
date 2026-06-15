"""Event Router - Cross-agent message routing with rules."""
from __future__ import annotations
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from prometheus_v8.communication.bus import Message, MemoryBus

logger = logging.getLogger(__name__)

@dataclass
class RoutingRule:
    """Rule for routing messages between agents."""
    name: str = ""
    source_pattern: str = ""  # regex for sender
    channel_pattern: str = ""  # regex for channel
    type_pattern: str = ""  # regex for message type
    target_channel: str = ""
    transform: Callable[[Message], Message] | None = None
    priority: int = 5  # lower = higher priority
    enabled: bool = True
    
    def matches(self, message: Message) -> bool:
        if not self.enabled:
            return False
        if self.source_pattern and not re.match(self.source_pattern, message.sender):
            return False
        if self.channel_pattern and not re.match(self.channel_pattern, message.channel):
            return False
        if self.type_pattern and not re.match(self.type_pattern, message.type):
            return False
        return True


class EventRouter:
    """Route messages between agents based on configurable rules."""
    
    def __init__(self, bus: MemoryBus | None = None) -> None:
        self._bus = bus or MemoryBus()
        self._rules: list[RoutingRule] = []
        self._agent_channels: dict[str, list[str]] = {}  # agent_id → subscribed channels
        self._lock = threading.RLock()
        self._stats: dict[str, int] = {"routed": 0, "dropped": 0, "transformed": 0}
    
    def add_rule(self, rule: RoutingRule) -> None:
        with self._lock:
            self._rules.append(rule)
            self._rules.sort(key=lambda r: r.priority)
    
    def remove_rule(self, name: str) -> bool:
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.name != name]
            return len(self._rules) < before
    
    def register_agent(self, agent_id: str, channels: list[str]) -> None:
        with self._lock:
            self._agent_channels[agent_id] = channels
    
    def unregister_agent(self, agent_id: str) -> None:
        with self._lock:
            self._agent_channels.pop(agent_id, None)
    
    def route(self, message: Message) -> int:
        """Route message through rules. Returns number of destinations reached."""
        total_reached = 0
        with self._lock:
            rules = list(self._rules)
        
        for rule in rules:
            if rule.matches(message):
                routed_msg = message
                if rule.transform:
                    try:
                        routed_msg = rule.transform(message)
                        self._stats["transformed"] += 1
                    except Exception as e:
                        logger.warning(f"Transform error in rule {rule.name}: {e}")
                        continue
                
                if rule.target_channel:
                    routed_msg.channel = rule.target_channel
                    reached = self._bus.publish(routed_msg)
                    total_reached += reached
                    self._stats["routed"] += reached
        
        # Direct delivery to registered agents
        if message.recipient:
            channels = self._agent_channels.get(message.recipient, [])
            for channel in channels:
                direct_msg = Message(channel=channel, sender=message.sender, recipient=message.recipient,
                                    type=message.type, payload=message.payload)
                total_reached += self._bus.publish(direct_msg)
        
        if total_reached == 0:
            self._stats["dropped"] += 1
        
        return total_reached
    
    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)
    
    def list_rules(self) -> list[dict]:
        return [{"name": r.name, "source": r.source_pattern, "channel": r.channel_pattern,
                 "type": r.type_pattern, "target": r.target_channel, "priority": r.priority,
                 "enabled": r.enabled} for r in self._rules]
