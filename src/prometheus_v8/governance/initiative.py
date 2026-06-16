"""Spontaneous Initiative System - 7-layer governance for agent self-driven actions.

From spontaneous-initiative-design document:
- 4-step initiative process: Trigger -> Evaluate -> Plan -> Execute
- Max steps per initiative: 15
- Max initiatives per day: 3
- Safety validation at each step
- Full audit trail
"""
from __future__ import annotations
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class InitiativeState(str, Enum):
    TRIGGERED = "triggered"
    EVALUATING = "evaluating"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class InitiativeTrigger(str, Enum):
    CURIOSITY = "curiosity"          # Knowledge gap detected
    OPPORTUNITY = "opportunity"      # Improvement opportunity found
    ANOMALY = "anomaly"              # Anomaly detected in metrics
    SCHEDULED = "scheduled"          # Scheduled maintenance task
    USER_REQUEST = "user_request"    # User asked for something
    EVOLUTION_SIGNAL = "evolution"   # Evolution produced a candidate

@dataclass
class InitiativeStep:
    """A single step in an initiative execution plan."""
    index: int = 0
    action: str = ""
    description: str = ""
    status: str = "pending"  # pending/running/done/failed/skipped
    result: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    safety_approved: bool = False

@dataclass
class Initiative:
    """A spontaneous initiative with full lifecycle tracking."""
    id: str = ""
    trigger: InitiativeTrigger = InitiativeTrigger.CURIOSITY
    reason: str = ""
    state: InitiativeState = InitiativeState.TRIGGERED
    steps: list[InitiativeStep] = field(default_factory=list)
    current_step: int = 0
    max_steps: int = 15
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    result_summary: str = ""
    audit_trail: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status in ("done", "skipped"))
        return completed / len(self.steps)

class InitiativeSystem:
    """7-layer governance for spontaneous initiative.
    
    Governance layers:
    1. Trigger Detection - What initiates the action
    2. Safety Evaluation - Is it safe to proceed
    3. Resource Check - Do we have capacity
    4. Autonomy Gate - Is this action within our autonomy level
    5. Plan Validation - Is the plan sound
    6. Execution Monitoring - Track progress and safety
    7. Result Audit - Verify outcome and log
    
    Constraints:
    - Max 3 initiatives per day
    - Max 15 steps per initiative
    - Safety validation at each step
    - Full audit trail for every initiative
    """

    def __init__(self, safety_manager=None, autonomy_controller=None,
                 max_per_day: int = 3, max_steps: int = 15) -> None:
        self._safety = safety_manager
        self._autonomy = autonomy_controller
        self._max_per_day = max_per_day
        self._max_steps = max_steps
        self._initiatives: dict[str, Initiative] = {}
        self._today_count = 0
        self._today = time.strftime("%Y-%m-%d")
        self._history: deque[Initiative] = deque(maxlen=100)
        self._step_executor: Callable[[str, dict], str] | None = None
        self._lock = threading.RLock()

    def set_step_executor(self, fn: Callable[[str, dict], str]) -> None:
        """Set the function that executes individual initiative steps."""
        self._step_executor = fn

    def trigger(self, trigger_type: InitiativeTrigger, reason: str,
                priority: int = 5, metadata: dict | None = None) -> Initiative | None:
        """Layer 1: Trigger a new initiative.
        
        Returns the initiative if accepted, None if rejected.
        """
        self._check_day_reset()
        
        if self._today_count >= self._max_per_day:
            logger.info(f"Initiative rejected: daily limit reached ({self._max_per_day})")
            return None
        
        initiative = Initiative(
            id=str(uuid.uuid4())[:8],
            trigger=trigger_type, reason=reason,
            priority=priority, metadata=metadata or {},
        )
        initiative.audit_trail.append(f"[{time.time():.0f}] Triggered: {trigger_type.value} - {reason}")
        
        with self._lock:
            self._initiatives[initiative.id] = initiative
        
        logger.info(f"Initiative triggered: {initiative.id} ({trigger_type.value}) - {reason}")
        return initiative

    def evaluate(self, initiative_id: str) -> tuple[bool, str]:
        """Layer 2-4: Evaluate initiative for safety, resources, and autonomy.
        
        Returns (approved, reason).
        """
        initiative = self._initiatives.get(initiative_id)
        if not initiative:
            return False, "Initiative not found"
        
        # Layer 2: Safety Evaluation
        if self._safety:
            verdict = self._safety.check(initiative.reason)
            if not verdict.allowed:
                initiative.state = InitiativeState.CANCELLED
                initiative.audit_trail.append(f"[{time.time():.0f}] Safety rejected: {verdict.reason}")
                return False, f"Safety: {verdict.reason}"
        
        # Layer 3: Resource Check (simplified - check daily quota)
        if self._today_count >= self._max_per_day:
            initiative.state = InitiativeState.CANCELLED
            initiative.audit_trail.append(f"[{time.time():.0f}] Resource limit: daily quota reached")
            return False, "Daily quota reached"
        
        # Layer 4: Autonomy Gate
        if self._autonomy:
            can, level, reason = self._autonomy.can_execute("initiative")
            if not can:
                initiative.state = InitiativeState.CANCELLED
                initiative.audit_trail.append(f"[{time.time():.0f}] Autonomy rejected: {reason}")
                return False, f"Autonomy: {reason}"
        
        initiative.state = InitiativeState.EVALUATING
        initiative.audit_trail.append(f"[{time.time():.0f}] Evaluation passed")
        return True, "Approved"

    def plan(self, initiative_id: str, steps: list[dict]) -> tuple[bool, str]:
        """Layer 5: Create execution plan with validation.
        
        Each step dict: {"action": str, "description": str}
        """
        initiative = self._initiatives.get(initiative_id)
        if not initiative:
            return False, "Initiative not found"
        
        if len(steps) > self._max_steps:
            return False, f"Too many steps ({len(steps)} > {self._max_steps})"
        
        # Validate each step for safety
        for i, step_dict in enumerate(steps):
            action = step_dict.get("action", "")
            if self._safety:
                verdict = self._safety.check(action)
                if not verdict.allowed:
                    return False, f"Step {i} unsafe: {verdict.reason}"
        
        initiative.steps = [
            InitiativeStep(index=i, action=s.get("action", ""),
                          description=s.get("description", ""),
                          safety_approved=True)
            for i, s in enumerate(steps)
        ]
        initiative.state = InitiativeState.PLANNING
        initiative.audit_trail.append(f"[{time.time():.0f}] Plan created: {len(steps)} steps")
        return True, "Plan approved"

    def execute(self, initiative_id: str) -> tuple[bool, str]:
        """Layer 6-7: Execute the initiative step by step with monitoring and audit.
        
        Returns (success, summary).
        """
        initiative = self._initiatives.get(initiative_id)
        if not initiative:
            return False, "Initiative not found"
        
        initiative.state = InitiativeState.EXECUTING
        initiative.started_at = time.time()
        self._today_count += 1
        
        for step in initiative.steps:
            # Layer 6: Execution Monitoring
            step.status = "running"
            step.started_at = time.time()
            initiative.current_step = step.index
            
            try:
                if self._step_executor:
                    result = self._step_executor(step.action, initiative.metadata)
                else:
                    result = f"Executed: {step.action}"
                
                step.result = result
                step.status = "done"
                step.completed_at = time.time()
                initiative.audit_trail.append(
                    f"[{time.time():.0f}] Step {step.index} done: {step.action} -> {result[:50]}"
                )
            except Exception as e:
                step.result = str(e)
                step.status = "failed"
                step.completed_at = time.time()
                initiative.audit_trail.append(
                    f"[{time.time():.0f}] Step {step.index} failed: {e}"
                )
                initiative.state = InitiativeState.FAILED
                break
        
        # Layer 7: Result Audit
        if initiative.state != InitiativeState.FAILED:
            initiative.state = InitiativeState.COMPLETED
        
        initiative.completed_at = time.time()
        initiative.result_summary = (
            f"{'Completed' if initiative.state == InitiativeState.COMPLETED else 'Failed'}: "
            f"{sum(1 for s in initiative.steps if s.status == 'done')}/{len(initiative.steps)} steps, "
            f"duration={initiative.duration:.1f}s"
        )
        initiative.audit_trail.append(f"[{time.time():.0f}] {initiative.result_summary}")
        
        with self._lock:
            self._history.append(initiative)
        
        return initiative.state == InitiativeState.COMPLETED, initiative.result_summary

    def cancel(self, initiative_id: str) -> bool:
        """Cancel an initiative."""
        initiative = self._initiatives.get(initiative_id)
        if not initiative:
            return False
        initiative.state = InitiativeState.CANCELLED
        initiative.audit_trail.append(f"[{time.time():.0f}] Cancelled")
        return True

    def get_initiative(self, initiative_id: str) -> Initiative | None:
        return self._initiatives.get(initiative_id)

    def get_active(self) -> list[Initiative]:
        """Get all active (non-terminal) initiatives."""
        return [i for i in self._initiatives.values()
                if i.state not in (InitiativeState.COMPLETED, InitiativeState.FAILED, InitiativeState.CANCELLED)]

    def get_history(self, limit: int = 20) -> list[Initiative]:
        return list(self._history)[-limit:]

    def _check_day_reset(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._today:
            self._today = today
            self._today_count = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "active": len(self.get_active()),
            "today_count": self._today_count,
            "max_per_day": self._max_per_day,
            "total_history": len(self._history),
            "by_trigger": dict(
                (t.value, sum(1 for i in self._history if i.trigger == t))
                for t in InitiativeTrigger
            ),
        }
