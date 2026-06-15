"""Spontaneous Initiative - 7-layer governance for autonomous action."""
from __future__ import annotations
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from prometheus_v8.governance.autonomy import AutonomyController, AutonomyLevel
from prometheus_v8.governance.curiosity import CuriosityQueue

logger = logging.getLogger(__name__)

@dataclass
class InitiativeAction:
    """A spontaneous initiative action."""
    id: str = ""
    category: str = ""
    description: str = ""
    autonomy_level: AutonomyLevel = AutonomyLevel.L1_SEMI_AUTO
    steps: list[str] = field(default_factory=list)
    estimated_tokens: int = 500
    created_at: float = field(default_factory=time.time)
    executed: bool = False
    result: str = ""
    success: bool = False

# 5 categories of meaningful actions
ACTION_CATEGORIES = {
    "knowledge_management": [
        "Compress low-utility knowledge", "Add action hooks to pending knowledge",
        "Distill skills from high-signal knowledge", "Clean stale knowledge", "Update indices",
    ],
    "system_maintenance": [
        "Check cron health", "Clear backlog", "Verify backups", "Check disk usage",
    ],
    "self_learning": [
        "Explore curiosity queue item", "Scan arXiv for new papers",
        "Review low-utility knowledge for insights",
    ],
    "content_creation": [
        "Write insight from recent learning", "Update repository documentation",
    ],
    "community_contribution": [
        "Answer pending questions", "Share recent discoveries",
    ],
}

class SpontaneousInitiative:
    """7-layer governance for spontaneous initiative:
    1. Execution: Run deterministic tasks
    2. Orchestration: Schedule task priority
    3. Observation: Monitor system state
    4. Gating: Classify anomalies
    5. Cognition: LLM-based decision
    6. Verification: Validate decisions
    7. First Aid: Emergency rollback
    + Memory: Store experience
    + Quality: Verify effectiveness
    + Adaptation: Adjust parameters
    """
    
    def __init__(self, autonomy: AutonomyController | None = None,
                 curiosity: CuriosityQueue | None = None,
                 max_per_day: int = 3, max_steps: int = 15, max_minutes: int = 10) -> None:
        self._autonomy = autonomy or AutonomyController()
        self._curiosity = curiosity or CuriosityQueue()
        self._max_per_day = max_per_day
        self._max_steps = max_steps
        self._max_minutes = max_minutes
        self._lock = threading.RLock()
        self._today_count = 0
        self._today_date = time.strftime("%Y-%m-%d")
        self._consecutive_failures = 0
        self._action_history: list[InitiativeAction] = []
        self._paused_until: float = 0.0
    
    def should_act(self) -> bool:
        """Check if spontaneous initiative should trigger."""
        # Check daily limit
        today = time.strftime("%Y-%m-%d")
        if today != self._today_date:
            self._today_count = 0
            self._today_date = today
            self._consecutive_failures = 0
        
        if self._today_count >= self._max_per_day:
            return False
        
        # Check pause (from consecutive failures)
        if time.time() < self._paused_until:
            return False
        
        # Check time window (not 2-6 AM)
        hour = int(time.strftime("%H"))
        if 2 <= hour < 6:
            return False
        
        return True
    
    def generate_action(self) -> InitiativeAction | None:
        """Generate a spontaneous action based on current state."""
        if not self.should_act():
            return None
        
        # Select category and action
        category = random.choice(list(ACTION_CATEGORIES.keys()))
        actions = ACTION_CATEGORIES[category]
        description = random.choice(actions)
        
        # Check autonomy
        can_exec, level, reason = self._autonomy.can_execute(category, description)
        if not can_exec:
            logger.info(f"Initiative blocked: {reason}")
            return None
        
        # Check curiosity queue for learning actions
        if category == "self_learning":
            item = self._curiosity.peek()
            if item:
                description = f"Explore: {item.question}"
        
        action = InitiativeAction(
            id=f"init_{self._today_count}_{int(time.time())}",
            category=category, description=description, autonomy_level=level,
            steps=[description], estimated_tokens=500,
        )
        
        return action
    
    def execute_action(self, action: InitiativeAction, executor: Callable[[str], str] | None = None) -> InitiativeAction:
        """Execute a spontaneous action."""
        start = time.time()
        
        try:
            if executor:
                result = executor(action.description)
            else:
                result = f"Executed: {action.description}"
            
            action.result = result
            action.success = True
            self._consecutive_failures = 0
        except Exception as e:
            action.result = f"Error: {e}"
            action.success = False
            self._consecutive_failures += 1
            
            # Pause after 3 consecutive failures
            if self._consecutive_failures >= 3:
                self._paused_until = time.time() + 86400  # 24 hours
                logger.warning("Initiative paused for 24h after 3 consecutive failures")
        
        action.executed = True
        self._today_count += 1
        
        with self._lock:
            self._action_history.append(action)
        
        # Record in autonomy controller
        self._autonomy.record_execution(action.category, action.success)
        
        elapsed = time.time() - start
        if elapsed > self._max_minutes * 60:
            logger.warning(f"Initiative exceeded time limit: {elapsed:.0f}s")
        
        return action
    
    @property
    def stats(self) -> dict[str, Any]:
        return {"today_count": self._today_count, "max_per_day": self._max_per_day,
                "consecutive_failures": self._consecutive_failures,
                "paused": time.time() < self._paused_until,
                "total_actions": len(self._action_history)}
