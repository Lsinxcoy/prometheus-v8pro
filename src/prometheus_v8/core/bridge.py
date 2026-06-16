"""Organ-Evolution Bridge - Connects L2 organ pipeline with L3 evolution engine.

Canonical location: core.bridge (re-exported from organs.bridge for backward compat).
"""

from prometheus_v8.organs.bridge import OrganEvolutionBridge as OrganEvolutionBridge

__all__ = ["OrganEvolutionBridge"]
