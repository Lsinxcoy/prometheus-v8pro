"""Exploration Quota + Curiosity Queue Management.

From MiMo insights:
- Daily exploration quota: max 20 rounds
- After 10 rounds: must insert revision round
- Curiosity queue: priority-based exploration questions
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class QuotaState:
    daily_limit: int = 20
    used_today: int = 0
    revision_interval: int = 10
    revision_count: int = 0
    today: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))

class ExplorationQuota:
    """Manage exploration quotas with mandatory revision rounds."""
    
    def __init__(self, daily_limit: int = 20, revision_interval: int = 10) -> None:
        self._state = QuotaState(daily_limit=daily_limit, revision_interval=revision_interval)
    
    def can_explore(self) -> tuple[bool, str]:
        """Check if exploration is allowed."""
        today = time.strftime("%Y-%m-%d")
        if today != self._state.today:
            self._state.today = today
            self._state.used_today = 0
            self._state.revision_count = 0
        
        if self._state.used_today >= self._state.daily_limit:
            return False, "Daily limit reached"
        
        if self._state.used_today > 0 and self._state.used_today % self._state.revision_interval == 0:
            if self._state.revision_count == 0:
                return False, "Revision round required before more exploration"
        
        return True, "OK"
    
    def record_exploration(self) -> None:
        self._state.used_today += 1
    
    def record_revision(self) -> None:
        self._state.revision_count += 1
    
    def remaining(self) -> int:
        return max(0, self._state.daily_limit - self._state.used_today)
    
    @property
    def stats(self) -> dict[str, Any]:
        return {"used_today": self._state.used_today, "daily_limit": self._state.daily_limit,
                "remaining": self.remaining(), "revisions": self._state.revision_count}
