"""Plan Validator - 3-layer: step + combination + topology attack detection."""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class PlanValidationResult:
    layer: str = ""
    passed: bool = True
    risk_score: float = 0.0
    details: str = ""

class PlanValidator:
    """3-layer plan validation:
    1. Step validation: Each step individually safe?
    2. Combination validation: Steps combined create attack?
    3. Topology validation: Step sequence forms attack topology?
    """
    
    def __init__(self) -> None:
        self._dangerous_combos = [
            {"read_file", "eval"},  # Read + eval = code injection
            {"search", "exec"},     # Search + exec = arbitrary execution
            {"download", "run"},    # Download + run = supply chain attack
        ]
        self._attack_topologies = [
            ["recon", "exploit", "exfiltrate"],  # Classic attack chain
            ["authenticate", "escalate", "persist"],  # Persistence attack
        ]
    
    def validate_step(self, step: str) -> tuple[bool, str]:
        """Layer 1: Validate a single step."""
        dangerous = ["exec(", "eval(", "os.system", "rm -rf", "__import__", "subprocess.call"]
        for pattern in dangerous:
            if pattern in step:
                return False, f"Dangerous pattern: {pattern}"
        return True, ""
    
    def validate_combination(self, steps: list[str]) -> tuple[bool, str]:
        """Layer 2: Check for dangerous step combinations."""
        step_lower = [s.lower() for s in steps]
        
        for combo in self._dangerous_combos:
            if all(any(kw in s for s in step_lower) for kw in combo):
                return False, f"Dangerous combination: {' + '.join(combo)}"
        return True, ""
    
    def validate_topology(self, steps: list[str]) -> tuple[bool, str]:
        """Layer 3: Check for attack topology patterns."""
        step_lower = [s.lower() for s in steps]
        
        for topology in self._attack_topologies:
            # Check if steps follow the topology pattern in order
            last_idx = -1
            matched = []
            for phase in topology:
                for i, s in enumerate(step_lower):
                    if i > last_idx and phase in s:
                        last_idx = i
                        matched.append(phase)
                        break
            if len(matched) == len(topology):
                return False, f"Attack topology detected: {' → '.join(topology)}"
        return True, ""
    
    def validate_plan(self, steps: list[str]) -> tuple[bool, str]:
        """Run all 3 validation layers."""
        # Layer 1: Individual steps
        for i, step in enumerate(steps):
            ok, reason = self.validate_step(step)
            if not ok:
                return False, f"Step {i}: {reason}"
        
        # Layer 2: Combinations
        ok, reason = self.validate_combination(steps)
        if not ok:
            return False, reason
        
        # Layer 3: Topology
        ok, reason = self.validate_topology(steps)
        if not ok:
            return False, reason
        
        return True, ""
