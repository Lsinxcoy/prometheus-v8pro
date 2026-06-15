"""Reasoning Budget - Token/entropy/time triple budget control."""
from __future__ import annotations
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class BudgetState:
    """Current budget state."""
    tokens_used: int = 0
    tokens_limit: int = 4000
    time_used: float = 0.0
    time_limit: float = 240.0
    entropy: float = 1.0  # 0=deterministic, 1=maximum randomness
    steps_taken: int = 0
    max_steps: int = 50
    
    @property
    def tokens_remaining(self) -> int:
        return max(0, self.tokens_limit - self.tokens_used)
    
    @property
    def time_remaining(self) -> float:
        return max(0.0, self.time_limit - self.time_used)
    
    @property
    def is_exhausted(self) -> bool:
        return (self.tokens_used >= self.tokens_limit or 
                self.time_used >= self.time_limit or
                self.steps_taken >= self.max_steps)
    
    @property
    def utilization(self) -> float:
        return max(self.tokens_used / max(1, self.tokens_limit),
                   self.time_used / max(0.1, self.time_limit),
                   self.steps_taken / max(1, self.max_steps))

class ReasoningBudget:
    """Triple budget controller: tokens + time + entropy.
    
    From MiMo insights: reasoning efficiency crisis - problems aren't too much reasoning
    but misallocated reasoning (over-reasoning simple, under-reasoning hard).
    """
    
    def __init__(self, tokens: int = 4000, time_seconds: int = 240,
                 max_steps: int = 50, initial_entropy: float = 1.0) -> None:
        self._state = BudgetState(
            tokens_limit=tokens, time_limit=float(time_seconds),
            max_steps=max_steps, entropy=initial_entropy,
        )
        self._start_time = time.time()
        self._information_gains: list[float] = []
    
    def allocate(self, task_complexity: float = 0.5) -> BudgetState:
        """Allocate budget based on task complexity.
        
        Simple tasks (complexity < 0.3): 5-10 steps, 1000 tokens
        Medium tasks (0.3-0.7): 15-30 steps, 4000 tokens
        Hard tasks (>0.7): 30-50 steps, 8000 tokens
        """
        if task_complexity < 0.3:
            self._state.max_steps = 10
            self._state.tokens_limit = 1000
            self._state.time_limit = 60.0
        elif task_complexity < 0.7:
            self._state.max_steps = 30
            self._state.tokens_limit = 4000
            self._state.time_limit = 240.0
        else:
            self._state.max_steps = 50
            self._state.tokens_limit = 8000
            self._state.time_limit = 600.0
        
        return self._state
    
    def step(self, tokens_used: int = 0, information_gain: float = 0.0) -> BudgetState:
        """Record a reasoning step."""
        self._state.steps_taken += 1
        self._state.tokens_used += tokens_used
        self._state.time_used = time.time() - self._start_time
        self._information_gains.append(information_gain)
        
        # Decay entropy based on information gain
        if information_gain < 0.01:
            self._state.entropy *= 0.9  # Converge toward deterministic
        else:
            self._state.entropy = min(1.0, self._state.entropy * 1.1)
        
        # Auto-terminate: 3 consecutive zero-gain steps
        if len(self._information_gains) >= 3:
            recent_3 = self._information_gains[-3:]
            if all(g < 0.01 for g in recent_3):
                logger.info("Auto-terminate: 3 consecutive zero-gain steps")
                self._state.entropy = 0.0
        
        return self._state
    
    def should_continue(self) -> bool:
        """Check if budget allows continuing."""
        return not self._state.is_exhausted and self._state.entropy > 0.01
    
    @property
    def state(self) -> BudgetState:
        self._state.time_used = time.time() - self._start_time
        return self._state
    
    def reset(self) -> None:
        self._state = BudgetState(tokens_limit=self._state.tokens_limit,
                                  time_limit=self._state.time_limit,
                                  max_steps=self._state.max_steps)
        self._start_time = time.time()
        self._information_gains.clear()
