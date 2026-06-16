"""EVO Bus Adapter - Redis Stream based inter-agent communication with graceful degradation."""
from __future__ import annotations
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

@dataclass
class BusMessage:
    """A message on the bus."""
    id: str = ""
    topic: str = ""
    sender: str = ""
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: str = ""
    priority: int = 0

    def serialize(self) -> dict[str, str]:
        return {
            "id": self.id, "topic": self.topic, "sender": self.sender,
            "payload": json.dumps(self.payload, ensure_ascii=False),
            "timestamp": str(self.timestamp),
            "reply_to": self.reply_to, "priority": str(self.priority),
        }

    @classmethod
    def deserialize(cls, data: dict[str, str]) -> "BusMessage":
        return cls(
            id=data.get("id", ""), topic=data.get("topic", ""),
            sender=data.get("sender", ""),
            payload=json.loads(data.get("payload", "{}")),
            timestamp=float(data.get("timestamp", 0)),
            reply_to=data.get("reply_to", ""),
            priority=int(data.get("priority", 0)),
        )

@dataclass
class AgentInfo:
    """Information about a registered agent."""
    id: str = ""
    name: str = ""
    role: str = "worker"
    model: str = ""
    capabilities: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "online"

class InMemoryBus:
    """In-memory fallback message bus when Redis is unavailable."""

    def __init__(self, max_queue_size: int = 1000) -> None:
        self._queues: dict[str, deque[BusMessage]] = defaultdict(lambda: deque(maxlen=max_queue_size))
        self._subscribers: dict[str, list[Callable[[BusMessage], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def publish(self, topic: str, message: BusMessage) -> bool:
        with self._lock:
            self._queues[topic].append(message)
            for cb in self._subscribers.get(topic, []):
                try:
                    cb(message)
                except Exception as e:
                    logger.warning(f"Subscriber callback error: {e}")
        return True

    def consume(self, topic: str, count: int = 10) -> list[BusMessage]:
        with self._lock:
            messages = []
            queue = self._queues.get(topic, deque())
            for _ in range(min(count, len(queue))):
                if queue:
                    messages.append(queue.popleft())
            return messages

    def subscribe(self, topic: str, callback: Callable[[BusMessage], None]) -> None:
        with self._lock:
            self._subscribers[topic].append(callback)

    def peek(self, topic: str, count: int = 10) -> list[BusMessage]:
        with self._lock:
            return list(self._queues.get(topic, deque()))[:count]

class EVOBusAdapter:
    """Redis Stream based message bus for inter-agent communication.
    
    Gracefully degrades to InMemoryBus when Redis is unavailable.
    Supports agent registration, topic routing, consumer groups,
    heartbeat tracking, and dead letter queue.
    """

    def __init__(self, redis_host: str = REDIS_HOST, redis_port: int = REDIS_PORT,
                 redis_db: int = REDIS_DB, redis_password: str = "",
                 agent_id: str = "", heartbeat_interval: float = 30.0) -> None:
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_db = redis_db
        self._redis_password = redis_password
        self._agent_id = agent_id or str(uuid.uuid4())[:8]
        self._heartbeat_interval = heartbeat_interval
        self._using_redis = False
        self._redis_client = None
        self._memory_bus = InMemoryBus()
        self._agents: dict[str, AgentInfo] = {}
        self._dead_letters: list[BusMessage] = []
        self._message_count = 0
        self._error_count = 0
        self._lock = threading.RLock()
        self._heartbeat_thread: threading.Thread | None = None
        self._running = False
        
        self._connect_redis()

    def _connect_redis(self) -> None:
        """Attempt to connect to Redis, fall back to in-memory."""
        if not HAS_REDIS:
            logger.info("Redis library not available, using in-memory bus")
            return
        try:
            self._redis_client = redis.Redis(
                host=self._redis_host, port=self._redis_port,
                db=self._redis_db, password=self._redis_password or None,
                socket_timeout=5.0, socket_connect_timeout=5.0,
            )
            self._redis_client.ping()
            self._using_redis = True
            logger.info(f"Connected to Redis at {self._redis_host}:{self._redis_port}")
        except Exception as e:
            logger.warning(f"Redis connection failed ({e}), using in-memory bus")
            self._redis_client = None
            self._using_redis = False

    def publish(self, topic: str, payload: dict, sender: str = "",
                reply_to: str = "", priority: int = 0) -> str:
        """Publish a message to a topic."""
        msg = BusMessage(
            id=str(uuid.uuid4()), topic=topic, sender=sender or self._agent_id,
            payload=payload, reply_to=reply_to, priority=priority,
        )
        self._message_count += 1
        
        if self._using_redis and self._redis_client:
            try:
                stream_key = f"prometheus:stream:{topic}"
                self._redis_client.xadd(stream_key, msg.serialize())
                return msg.id
            except Exception as e:
                self._error_count += 1
                logger.warning(f"Redis publish failed, falling back: {e}")
                self._dead_letters.append(msg)
        
        # In-memory fallback
        self._memory_bus.publish(topic, msg)
        return msg.id

    def consume(self, topic: str, count: int = 10, group: str = "") -> list[BusMessage]:
        """Consume messages from a topic."""
        if self._using_redis and self._redis_client:
            try:
                stream_key = f"prometheus:stream:{topic}"
                consumer_group = group or f"cg_{self._agent_id}"
                consumer_name = f"c_{self._agent_id}"
                
                # Try to create consumer group
                try:
                    self._redis_client.xgroup_create(stream_key, consumer_group, id="0", mkstream=True)
                except Exception:
                    pass
                
                results = self._redis_client.xreadgroup(
                    consumer_group, consumer_name,
                    {stream_key: ">"}, count=count, block=1000,
                )
                messages = []
                if results:
                    for stream_name, stream_messages in results:
                        for msg_id, msg_data in stream_messages:
                            messages.append(BusMessage.deserialize(msg_data))
                return messages
            except Exception as e:
                self._error_count += 1
                logger.warning(f"Redis consume failed: {e}")
        
        return self._memory_bus.consume(topic, count)

    def register_agent(self, agent_info: AgentInfo) -> None:
        """Register an agent on the bus."""
        with self._lock:
            self._agents[agent_info.id] = agent_info
        logger.info(f"Agent registered: {agent_info.name} ({agent_info.role})")
        
        if self._using_redis and self._redis_client:
            try:
                key = f"prometheus:agents:{agent_info.id}"
                self._redis_client.hset(key, mapping={
                    "name": agent_info.name, "role": agent_info.role,
                    "model": agent_info.model, "status": agent_info.status,
                    "capabilities": json.dumps(agent_info.capabilities),
                    "channels": json.dumps(agent_info.channels),
                    "last_heartbeat": str(time.time()),
                })
            except Exception as e:
                logger.warning(f"Redis agent registration failed: {e}")

    def deregister_agent(self, agent_id: str) -> None:
        """Deregister an agent from the bus."""
        with self._lock:
            self._agents.pop(agent_id, None)
        if self._using_redis and self._redis_client:
            try:
                self._redis_client.delete(f"prometheus:agents:{agent_id}")
            except Exception:
                pass

    def get_agents(self, status: str = "") -> list[AgentInfo]:
        """Get registered agents, optionally filtered by status."""
        with self._lock:
            agents = list(self._agents.values())
        if status:
            agents = [a for a in agents if a.status == status]
        return agents

    def start_heartbeat(self) -> None:
        """Start the background heartbeat thread."""
        if self._running:
            return
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Stop the heartbeat thread."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)

    def _heartbeat_loop(self) -> None:
        """Background heartbeat loop for agent liveness tracking."""
        while self._running:
            try:
                with self._lock:
                    now = time.time()
                    for agent in self._agents.values():
                        if now - agent.last_heartbeat > self._heartbeat_interval * 3:
                            if agent.status != "dead":
                                agent.status = "dead"
                                logger.warning(f"Agent {agent.name} declared dead")
                        elif now - agent.last_heartbeat > self._heartbeat_interval * 2:
                            if agent.status == "online":
                                agent.status = "offline"
                                logger.info(f"Agent {agent.name} went offline")
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
            time.sleep(self._heartbeat_interval)

    def check_agent_liveness(self, agent_id: str) -> bool:
        """Check if an agent is still alive."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return False
            return agent.status in ("online", "busy")

    def get_dead_letters(self, limit: int = 50) -> list[BusMessage]:
        """Get messages that failed to deliver."""
        return self._dead_letters[-limit:]

    def replay_dead_letter(self, index: int) -> bool:
        """Replay a dead letter message."""
        if 0 <= index < len(self._dead_letters):
            msg = self._dead_letters.pop(index)
            self.publish(msg.topic, msg.payload, msg.sender, msg.reply_to, msg.priority)
            return True
        return False

    def close(self) -> None:
        """Close the bus connection."""
        self.stop_heartbeat()
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception:
                pass

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "using_redis": self._using_redis,
            "registered_agents": len(self._agents),
            "online_agents": sum(1 for a in self._agents.values() if a.status == "online"),
            "total_messages": self._message_count,
            "total_errors": self._error_count,
            "dead_letters": len(self._dead_letters),
        }
