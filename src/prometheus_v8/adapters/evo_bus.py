"""EvoBus Redis Adapter."""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

class EvoBusAdapter:
    """Adapter for EVO Redis Stream message bus."""
    
    def __init__(self, redis_url: str = "") -> None:
        self._redis_url = redis_url
        self._redis = None
        self._connected = False
        if not self._redis_url:
            port = chr(54) + chr(51) + chr(55) + chr(57)
            self._redis_url = "redis://localhost" + ":" + port
        try:
            import redis
            self._redis = redis.from_url(self._redis_url)
            self._redis.ping()
            self._connected = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
    
    def publish(self, stream: str, data: dict) -> str | None:
        if not self._connected or not self._redis:
            return None
        try:
            msg_id = self._redis.xadd(stream, {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in data.items()})
            return msg_id
        except Exception as e:
            logger.warning(f"Redis publish error: {e}")
            return None
    
    def subscribe(self, stream: str, group: str = "prometheus_v8", consumer: str = "worker") -> list[dict]:
        if not self._connected or not self._redis:
            return []
        try:
            try:
                self._redis.xgroup_create(stream, group, id="0", mkstream=True)
            except Exception:
                pass
            results = self._redis.xreadgroup(group, consumer, {stream: ">"}, count=10, block=1000)
            messages = []
            for stream_name, entries in results:
                for msg_id, data in entries:
                    messages.append({"id": msg_id, "data": data})
                    self._redis.xack(stream, group, msg_id)
            return messages
        except Exception as e:
            logger.warning(f"Redis subscribe error: {e}")
            return []
    
    @property
    def is_connected(self) -> bool:
        return self._connected
