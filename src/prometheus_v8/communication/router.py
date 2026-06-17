"""Event Router - Cross-agent message routing with rules."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Callable

from prometheus_v8.communication.bus import MemoryBus, Message

logger = logging.getLogger(__name__)


@dataclass
class RoutingRule:
    """Rule for routing messages between agents."""

    name: str = ""
    source_pattern: str = ""  # regex for source
    channel_pattern: str = ""  # regex for topic
    type_pattern: str = ""  # regex for event_type
    target_channel: str = ""
    transform: Callable[[Message], Message] | None = None
    priority: int = 5  # lower = higher priority
    enabled: bool = True

    def matches(self, message: Message) -> bool:
        if not self.enabled:
            return False
        # Message extends Event: sender→source, channel→topic are unified via properties
        sender = message.sender if message.sender else message.source
        channel = message.channel if message.channel else message.topic
        msg_type = message.event_type
        if self.source_pattern and not re.match(self.source_pattern, sender):
            return False
        if self.channel_pattern and not re.match(self.channel_pattern, channel):
            return False
        if self.type_pattern and not re.match(self.type_pattern, msg_type):
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
                    # Use MemoryBus.publish() with proper signature
                    topic = rule.target_channel
                    event_type = routed_msg.event_type
                    source = routed_msg.sender if routed_msg.sender else routed_msg.source
                    self._bus.publish(
                        topic=topic,
                        event_type=event_type,
                        payload=routed_msg.payload,
                        source=source,
                        correlation_id=routed_msg.recipient if routed_msg.recipient else routed_msg.correlation_id,
                    )
                    total_reached += 1
                    self._stats["routed"] += 1

        # Direct delivery to registered agents
        recipient = message.recipient if message.recipient else message.correlation_id
        if recipient:
            channels = self._agent_channels.get(recipient, [])
            for channel in channels:
                sender = message.sender if message.sender else message.source
                msg_type = message.event_type
                self._bus.publish(
                    topic=channel, event_type=msg_type, payload=message.payload, source=sender, correlation_id=recipient
                )
                total_reached += 1

        if total_reached == 0:
            self._stats["dropped"] += 1

        return total_reached

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def list_rules(self) -> list[dict]:
        return [
            {
                "name": r.name,
                "source": r.source_pattern,
                "channel": r.channel_pattern,
                "type": r.type_pattern,
                "target": r.target_channel,
                "priority": r.priority,
                "enabled": r.enabled,
            }
            for r in self._rules
        ]
