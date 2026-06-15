"""Autonomy Controller - 5-level autonomy (L0 full auto → L4 forbidden)."""
from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class AutonomyLevel(str, Enum):
    L0_FULL_AUTO = "l0_full_auto"      # No approval needed
    L1_SEMI_AUTO = "l1_semi_auto"      # Report after execution
    L2_CONFIRM = "l2_confirm"          # Confirm before execution
    L3_APPROVAL = "l3_approval"        # Explicit approval required
    L4_FORBIDDEN = "l4_forbidden"      # Never execute

@dataclass
class AutonomyRule:
    """Maps operation categories to autonomy levels."""
    category: str = ""
    level: AutonomyLevel = AutonomyLevel.L1_SEMI_AUTO
    description: str = ""
    max_frequency: int = 0  # 0=unlimited
    time_window: list[str] = field(default_factory=list)  # e.g. ["06:00-22:00"]

DEFAULT_AUTONOMY_RULES = {
    "knowledge_compress": AutonomyRule("knowledge_compress", AutonomyLevel.L0_FULL_AUTO, "Compress knowledge", 10),
    "index_update": AutonomyRule("index_update", AutonomyLevel.L0_FULL_AUTO, "Update indices", 5),
    "log_cleanup": AutonomyRule("log_cleanup", AutonomyLevel.L0_FULL_AUTO, "Clean logs", 3),
    "learn_direction": AutonomyRule("learn_direction", AutonomyLevel.L1_SEMI_AUTO, "Learn new direction", 5),
    "create_skill": AutonomyRule("create_skill", AutonomyLevel.L1_SEMI_AUTO, "Create skill", 3),
    "system_inspect": AutonomyRule("system_inspect", AutonomyLevel.L1_SEMI_AUTO, "System inspection", 10),
    "modify_core": AutonomyRule("modify_core", AutonomyLevel.L2_CONFIRM, "Modify core files", 2),
    "delete_knowledge": AutonomyRule("delete_knowledge", AutonomyLevel.L2_CONFIRM, "Delete knowledge", 3),
    "adjust_cron": AutonomyRule("adjust_cron", AutonomyLevel.L2_CONFIRM, "Adjust cron tasks", 2),
    "send_message": AutonomyRule("send_message", AutonomyLevel.L3_APPROVAL, "Send external message", 3),
    "modify_safety": AutonomyRule("modify_safety", AutonomyLevel.L3_APPROVAL, "Modify safety rules", 1),
    "access_sensitive": AutonomyRule("access_sensitive", AutonomyLevel.L3_APPROVAL, "Access sensitive data", 2),
    "leak_core": AutonomyRule("leak_core", AutonomyLevel.L4_FORBIDDEN, "Leak core files"),
    "dangerous_cmd": AutonomyRule("dangerous_cmd", AutonomyLevel.L4_FORBIDDEN, "Execute dangerous commands"),
    "bypass_safety": AutonomyRule("bypass_safety", AutonomyLevel.L4_FORBIDDEN, "Bypass safety rules"),
}

class AutonomyController:
    """5-level autonomy controller with per-category rules."""
    
    def __init__(self, default_level: AutonomyLevel = AutonomyLevel.L1_SEMI_AUTO) -> None:
        self._rules = dict(DEFAULT_AUTONOMY_RULES)
        self._default_level = default_level
        self._lock = threading.RLock()
        self._execution_log: list[dict] = []
        self._confirmation_callbacks: list[Callable[[str, str], bool]] = []
        self._approval_callbacks: list[Callable[[str, str], bool]] = []
    
    def can_execute(self, category: str, context: str = "") -> tuple[bool, AutonomyLevel, str]:
        """Check if an operation can be executed at its autonomy level."""
        rule = self._rules.get(category, AutonomyRule(category, self._default_level))
        level = rule.level
        
        # Check time window
        if rule.time_window:
            now = time.strftime("%H:%M")
            in_window = any(start <= now <= end for start, end in 
                          [tw.split("-") for tw in rule.time_window if "-" in tw])
            if not in_window:
                return False, level, f"Outside time window: {rule.time_window}"
        
        # Check frequency
        if rule.max_frequency > 0:
            recent = sum(1 for e in self._execution_log[-100:] 
                        if e.get("category") == category and time.time() - e.get("timestamp", 0) < 3600)
            if recent >= rule.max_frequency:
                return False, level, f"Frequency limit reached: {recent}/{rule.max_frequency}/hour"
        
        # Level-based decision
        if level == AutonomyLevel.L4_FORBIDDEN:
            return False, level, "Operation is forbidden"
        elif level == AutonomyLevel.L3_APPROVAL:
            approved = self._request_approval(category, context)
            return approved, level, "Approval " + ("granted" if approved else "denied")
        elif level == AutonomyLevel.L2_CONFIRM:
            confirmed = self._request_confirmation(category, context)
            return confirmed, level, "Confirmation " + ("granted" if confirmed else "denied")
        elif level == AutonomyLevel.L1_SEMI_AUTO:
            return True, level, "Semi-auto: report after execution"
        else:  # L0
            return True, level, "Full auto: no approval needed"
    
    def record_execution(self, category: str, success: bool = True) -> None:
        with self._lock:
            self._execution_log.append({"category": category, "success": success, "timestamp": time.time()})
    
    def on_confirmation(self, callback: Callable[[str, str], bool]) -> None:
        self._confirmation_callbacks.append(callback)
    
    def on_approval(self, callback: Callable[[str, str], bool]) -> None:
        self._approval_callbacks.append(callback)
    
    def _request_confirmation(self, category: str, context: str) -> bool:
        for cb in self._confirmation_callbacks:
            try:
                if cb(category, context):
                    return True
            except Exception:
                pass
        return False  # Default: deny if no handler
    
    def _request_approval(self, category: str, context: str) -> bool:
        for cb in self._approval_callbacks:
            try:
                if cb(category, context):
                    return True
            except Exception:
                pass
        return False
    
    def set_level(self, category: str, level: AutonomyLevel) -> None:
        with self._lock:
            if category in self._rules:
                self._rules[category].level = level
            else:
                self._rules[category] = AutonomyRule(category, level)
    
    @property
    def rules(self) -> dict[str, dict]:
        return {cat: {"level": r.level.value, "description": r.description, "max_freq": r.max_frequency}
                for cat, r in self._rules.items()}
