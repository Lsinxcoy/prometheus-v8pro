"""Life Pipeline - Taotie→Nuwa→Darwin→Pool→Guard pipeline orchestrator."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from prometheus_v8.organs.base import BaseOrgan, OrganContext, OrganResult, LLMClient, OrganEnv
from prometheus_v8.organs.taotie import TaotieOrgan
from prometheus_v8.organs.nuwa import NuwaOrgan
from prometheus_v8.organs.darwin import DarwinOrgan
from prometheus_v8.organs.pool import PoolOrgan
from prometheus_v8.organs.guard import GuardOrgan

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Result from the full life pipeline."""
    success: bool = False
    stages_completed: list[str] = field(default_factory=list)
    final_output: Any = None
    errors: list[str] = field(default_factory=list)
    total_time: float = 0.0
    stage_times: dict[str, float] = field(default_factory=dict)

class LifePipeline:
    """Orchestrate the 5-organ pipeline: Taotie→Nuwa→Darwin→Pool→Guard."""
    
    def __init__(self, llm: LLMClient | None = None, env: OrganEnv | None = None) -> None:
        self._llm = llm or LLMClient()
        self._env = env or OrganEnv()
        self._taotie = TaotieOrgan(self._llm, self._env)
        self._nuwa = NuwaOrgan(self._llm, self._env)
        self._darwin = DarwinOrgan(self._llm, self._env)
        self._pool = PoolOrgan(self._llm, self._env)
        self._guard = GuardOrgan(self._llm, self._env)
        self._pipeline = [self._taotie, self._nuwa, self._darwin, self._pool, self._guard]
    
    def run(self, task: str, inputs: dict | None = None, constraints: list[str] | None = None) -> PipelineResult:
        """Run the full pipeline."""
        result = PipelineResult()
        start = time.time()
        context = OrganContext(task=task, inputs=inputs or {}, constraints=constraints or [])
        current_data: Any = inputs or {}
        
        for organ in self._pipeline:
            stage_start = time.time()
            try:
                context.inputs = current_data if isinstance(current_data, dict) else {"data": current_data}
                organ_result = organ._timed_execute(context)
                result.stage_times[organ.name] = time.time() - stage_start
                
                if organ_result.success:
                    result.stages_completed.append(organ.name)
                    current_data = organ_result.output
                else:
                    result.errors.append(f"{organ.name}: {organ_result.error}")
                    if organ.name in ("pool", "guard"):
                        break  # Critical failure
            except Exception as e:
                result.errors.append(f"{organ.name}: {str(e)}")
                result.stage_times[organ.name] = time.time() - stage_start
                break
        
        result.success = len(result.stages_completed) >= 3  # At least 3/5 stages
        result.final_output = current_data
        result.total_time = time.time() - start
        return result
    
    def run_single(self, organ_name: str, context: OrganContext) -> OrganResult:
        """Run a single organ by name."""
        organ_map = {o.name: o for o in self._pipeline}
        organ = organ_map.get(organ_name)
        if not organ:
            return OrganResult(success=False, error=f"Organ not found: {organ_name}")
        return organ._timed_execute(context)
    
    @property
    def organs(self) -> dict[str, BaseOrgan]:
        return {o.name: o for o in self._pipeline}
