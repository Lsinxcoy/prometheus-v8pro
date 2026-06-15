"""Goal System - Goal lifecycle: PENDING→ACTIVE→COMPLETED/FAILED."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

class GoalState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Goal:
    """Evolution goal with lifecycle tracking."""
    id: str = ""
    name: str = ""
    description: str = ""
    state: GoalState = GoalState.PENDING
    priority: int = 5  # 1=highest
    parent_id: str = ""
    sub_goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    progress: float = 0.0  # 0-1
    fitness_target: float = 0.8
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deadline: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def activate(self) -> None:
        self.state = GoalState.ACTIVE
        self.updated_at = time.time()
    
    def complete(self, progress: float = 1.0) -> None:
        self.state = GoalState.COMPLETED
        self.progress = progress
        self.updated_at = time.time()
    
    def fail(self, reason: str = "") -> None:
        self.state = GoalState.FAILED
        self.metadata["failure_reason"] = reason
        self.updated_at = time.time()
    
    def cancel(self) -> None:
        self.state = GoalState.CANCELLED
        self.updated_at = time.time()
    
    def update_progress(self, progress: float) -> None:
        self.progress = min(1.0, max(0.0, progress))
        self.updated_at = time.time()
        if self.progress >= 1.0:
            self.complete()
    
    @property
    def is_expired(self) -> bool:
        return self.deadline > 0 and time.time() > self.deadline

class GoalSystem:
    """Manage evolution goals with lifecycle and priority."""
    
    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}
        self._active_stack: list[str] = []  # Stack of active goal IDs
    
    def create_goal(self, name: str, description: str = "", priority: int = 5,
                    fitness_target: float = 0.8, parent_id: str = "",
                    constraints: list[str] | None = None, deadline: float = 0.0) -> Goal:
        goal_id = f"goal_{len(self._goals)}_{int(time.time())}"
        goal = Goal(id=goal_id, name=name, description=description, priority=priority,
                   fitness_target=fitness_target, parent_id=parent_id,
                   constraints=constraints or [], deadline=deadline)
        self._goals[goal_id] = goal
        
        if parent_id and parent_id in self._goals:
            self._goals[parent_id].sub_goals.append(goal_id)
        
        return goal
    
    def activate_goal(self, goal_id: str) -> bool:
        goal = self._goals.get(goal_id)
        if not goal or goal.state != GoalState.PENDING:
            return False
        goal.activate()
        self._active_stack.append(goal_id)
        return True
    
    def get_active_goal(self) -> Goal | None:
        while self._active_stack:
            goal_id = self._active_stack[-1]
            goal = self._goals.get(goal_id)
            if goal and goal.state == GoalState.ACTIVE:
                return goal
            self._active_stack.pop()
        return None
    
    def update_progress(self, goal_id: str, progress: float) -> None:
        goal = self._goals.get(goal_id)
        if goal:
            goal.update_progress(progress)
    
    def complete_goal(self, goal_id: str, progress: float = 1.0) -> None:
        goal = self._goals.get(goal_id)
        if goal:
            goal.complete(progress)
            if goal_id in self._active_stack:
                self._active_stack.remove(goal_id)
    
    def fail_goal(self, goal_id: str, reason: str = "") -> None:
        goal = self._goals.get(goal_id)
        if goal:
            goal.fail(reason)
            if goal_id in self._active_stack:
                self._active_stack.remove(goal_id)
    
    def list_goals(self, state: GoalState | None = None) -> list[Goal]:
        goals = list(self._goals.values())
        if state:
            goals = [g for g in goals if g.state == state]
        return sorted(goals, key=lambda g: g.priority)
    
    def get_next_pending(self) -> Goal | None:
        pending = [g for g in self._goals.values() if g.state == GoalState.PENDING and not g.is_expired]
        return min(pending, key=lambda g: g.priority) if pending else None
