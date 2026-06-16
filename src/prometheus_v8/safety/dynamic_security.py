"""Dynamic Security - 4-level adaptive security."""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DynamicSecurityManager:
    """4-level adaptive security based on context and history.

    LOW: Normal operations, minimal checks
    MEDIUM: Elevated risk, additional validation
    HIGH: Suspicious activity, strict validation
    CRITICAL: Active threat, block all non-essential operations
    """

    def __init__(self, base_level: SecurityLevel = SecurityLevel.LOW) -> None:
        self._level = base_level
        self._lock = threading.RLock()
        self._incident_count = 0
        self._last_incident_time = 0.0
        self._level_history: list[tuple[float, SecurityLevel]] = [(time.time(), base_level)]

    def assess_risk(self, content: str, context: dict | None = None) -> str:
        """Assess risk level of content and return security level."""
        risk_score = 0.0
        context = context or {}

        # Content-based risk
        high_risk_patterns = ["exec(", "eval(", "os.system", "subprocess", "rm -rf", "__import__"]
        for pattern in high_risk_patterns:
            if pattern in content:
                risk_score += 0.3

        # Context-based risk
        if context.get("operation") in ("delete", "modify_core", "external_call"):
            risk_score += 0.2
        if context.get("untrusted_source"):
            risk_score += 0.3

        # History-based risk
        if self._incident_count > 3:
            risk_score += 0.2

        # Determine level
        if risk_score >= 0.7:
            level = SecurityLevel.CRITICAL
        elif risk_score >= 0.5:
            level = SecurityLevel.HIGH
        elif risk_score >= 0.3:
            level = SecurityLevel.MEDIUM
        else:
            level = SecurityLevel.LOW

        with self._lock:
            self._level = level
            self._level_history.append((time.time(), level))

        return level.value

    def get_level(self) -> SecurityLevel:
        with self._lock:
            return self._level

    def record_incident(self, severity: str = "medium") -> None:
        with self._lock:
            self._incident_count += 1
            self._last_incident_time = time.time()
            if severity == "critical":
                self._level = SecurityLevel.CRITICAL
            elif severity == "high" and self._level != SecurityLevel.CRITICAL:
                self._level = SecurityLevel.HIGH

    def de_escalate(self) -> None:
        """De-escalate security level if no recent incidents."""
        with self._lock:
            if time.time() - self._last_incident_time > 300:  # 5 minutes
                order = [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH, SecurityLevel.CRITICAL]
                idx = order.index(self._level)
                if idx > 0:
                    self._level = order[idx - 1]
                    self._level_history.append((time.time(), self._level))

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "level": self._level.value,
            "incidents": self._incident_count,
            "last_incident": self._last_incident_time,
        }
