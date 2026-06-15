"""Communication Bus - Memory + Redis with graceful degradation."""
from __future__ import annotations
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Typed message for inter-agent communication."""
    id: str = ""
    channel: str = "default"
    sender: str = ""
    recipient: str = ""  # empty = broadcast
    type: str = "info"
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0  # seconds
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl
    
    def to_json(self) -> str:
        return json.dumps({"id": self.id, "channel": self.channel, "sender": self.sender,
                          "recipient": self.recipient, "type": self.type, "payload": self.payload,
                          "timestamp": self.timestamp, "ttl": self.ttl})
    
    @classmethod
    def from_json(cls, data: str) -> Message:
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MemoryBus:
    """In-process message bus with channel subscriptions and history."""
    
    def __init__(self, max_history: int = 10000) -> None:
        self._subscribers: dict[str, list[Callable[[Message], None]]] = defaultdict(list)
        self._history: deque[Message] = deque(maxlen=max_history)
        self._lock = threading.RLock()
        self._msg_counter = 0
    
    def publish(self, message: Message) -> int:
        """Publish message to channel. Returns subscriber count reached."""
        with self._lock:
            self._msg_counter += 1
            if not message.id:
                message.id = f"msg_{self._msg_counter}_{int(time.time()*1000)}"
            self._history.append(message)
        
        reached = 0
        handlers = self._subscribers.get(message.channel, []) + self._subscribers.get("*", [])
        for handler in handlers:
            try:
                handler(message)
                reached += 1
            except Exception as e:
                logger.warning(f"Message handler error: {e}")
        return reached
    
    def subscribe(self, channel: str, handler: Callable[[Message], None]) -> None:
        with self._lock:
            self._subscribers[channel].append(handler)
    
    def unsubscribe(self, channel: str, handler: Callable[[Message], None]) -> None:
        with self._lock:
            handlers = self._subscribers.get(channel, [])
            self._subscribers[channel] = [h for h in handlers if h != handler]
    
    def get_history(self, channel: str = "", limit: int = 100) -> list[Message]:
        with self._lock:
            msgs = list(self._history)
            if channel:
                msgs = [m for m in msgs if m.channel == channel]
            return msgs[-limit:]
    
    def clear(self) -> None:
        with self._lock:
            self._history.clear()


class RedisStreamsBus:
    """Redis Streams-based message bus for multi-process communication."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", stream_prefix: str = "prometheus:") -> None:
        self._redis_url = redis_url
        self._prefix = stream_prefix
        self._redis = None
        self._local_subscribers: dict[str, list[Callable[[Message], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._connected = False
        self._fallback = MemoryBus()
        self._connect()
    
    def _connect(self) -> None:
        try:
            import redis
            self._redis = redis.from_url(self._redis_url, decode_responses=True)
            self._redis.ping()
            self._connected = True
            logger.info(f"Redis Streams bus connected: {self._redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed, using memory fallback: {e}")
            self._connected = False
    
    def publish(self, message: Message) -> int:
        """Publish to Redis stream with memory fallback."""
        if not message.id:
            message.id = f"msg_{int(time.time()*1000)}"
        
        if self._connected and self._redis:
            try:
                stream = f"{self._prefix}{message.channel}"
                self._redis.xadd(stream, {"data": message.to_json()})
                # Also notify local subscribers
                reached = 0
                for handler in self._local_subscribers.get(message.channel, []):
                    try:
                        handler(message)
                        reached += 1
                    except Exception:
                        pass
                return reached + 1
            except Exception as e:
                logger.warning(f"Redis publish failed, falling back: {e}")
                self._connected = False
        
        return self._fallback.publish(message)
    
    def subscribe(self, channel: str, handler: Callable[[Message], None]) -> None:
        with self._lock:
            self._local_subscribers[channel].append(handler)
        if self._connected and self._redis:
            # Redis consumer groups handled separately
            pass
    
    def unsubscribe(self, channel: str, handler: Callable[[Message], None]) -> None:
        with self._lock:
            handlers = self._local_subscribers.get(channel, [])
            self._local_subscribers[channel] = [h for h in handlers if h != handler]
    
    def read_stream(self, channel: str, group: str = "prometheus_group", consumer: str = "consumer_1",
                    count: int = 10, block_ms: int = 0) -> list[Message]:
        """Read from Redis stream consumer group."""
        if not self._connected or not self._redis:
            return self._fallback.get_history(channel, count)
        
        stream = f"{self._prefix}{channel}"
        try:
            # Create consumer group if not exists
            try:
                self._redis.xgroup_create(stream, group, id="0", mkstream=True)
            except Exception:
                pass
            
            results = self._redis.xreadgroup(group, consumer, {stream: ">"}, count=count, block=block_ms)
            messages = []
            for stream_name, entries in results:
                for entry_id, data in entries:
                    msg = Message.from_json(data.get("data", "{}"))
                    messages.append(msg)
                    # Acknowledge
                    self._redis.xack(stream, group, entry_id)
            return messages
        except Exception as e:
            logger.warning(f"Redis stream read error: {e}")
            return []
    
    def get_history(self, channel: str = "", limit: int = 100) -> list[Message]:
        if self._connected and self._redis:
            try:
                stream = f"{self._prefix}{channel}"
                entries = self._redis.xrange(stream, count=limit)
                return [Message.from_json(data.get("data", "{}")) for _, data in entries]
            except Exception:
                return self._fallback.get_history(channel, limit)
        return self._fallback.get_history(channel, limit)
    
    def is_connected(self) -> bool:
        if self._connected and self._redis:
            try:
                self._redis.ping()
                return True
            except Exception:
                self._connected = False
        return False


def create_bus(backend: str = "memory", **kwargs: Any) -> MemoryBus | RedisStreamsBus:
    """Factory: create message bus with graceful degradation."""
    if backend == "redis":
        bus = RedisStreamsBus(**kwargs)
        if bus.is_connected():
            return bus
        logger.warning("Redis unavailable, falling back to MemoryBus")
    return MemoryBus(**{k: v for k, v in kwargs.items() if k == "max_history"})
