"""Harness System - Architecture-level meta-evolution (evolution of the evolution system itself).

The Harness is the missing meta-evolution layer from the Prometheus design:
- It monitors the evolution engine's own architecture
- Detects when the engine's structure needs to change (not just parameters)
- Proposes and validates architectural modifications
- Implements the "evolution of evolution" feedback loop

This is distinct from L5 MetaEvolution (which handles stagnation within the engine).
The Harness operates at a higher level: it can restructure the engine itself.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from prometheus_v8.schema import Genome

logger = logging.getLogger(__name__)


class HarnessAction(str, Enum):
    """Types of architectural actions the Harness can take."""
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    REORDER_LAYERS = "reorder_layers"
    MODIFY_LAYER_WEIGHT = "modify_layer_weight"
    ADD_ORGAN = "add_organ"
    RECONFIGURE_ORGAN = "reconfigure_organ"
    ADAPT_STRATEGY = "adapt_strategy"
    EXPAND_SEARCH_SPACE = "expand_search_space"
    CONTRACT_SEARCH_SPACE = "contract_search_space"


class HarnessComponent(str, Enum):
    """Components the Harness can modify."""
    ENGINE = "engine"
    FITNESS = "fitness"
    ORGANS = "organs"
    LIFECYCLE = "lifecycle"
    SAFETY = "safety"
    MEMORY = "memory"


@dataclass
class HarnessDesign:
    """A proposed architectural modification."""
    action: HarnessAction = HarnessAction.ADAPT_STRATEGY
    component: HarnessComponent = HarnessComponent.ENGINE
    description: str = ""
    rationale: str = ""
    expected_impact: float = 0.0
    risk_level: float = 0.5  # 0=safe, 1=dangerous
    preconditions: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    proposed_at: float = field(default_factory=time.time)


@dataclass
class HarnessEvaluation:
    """Result of evaluating a Harness design."""
    design: HarnessDesign = field(default_factory=HarnessDesign)
    approved: bool = False
    actual_impact: float = 0.0
    side_effects: list[str] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)


@dataclass
class ArchitectureSnapshot:
    """Snapshot of the current system architecture for analysis."""
    layer_count: int = 0
    layer_names: list[str] = field(default_factory=list)
    layer_execution_counts: list[int] = field(default_factory=list)
    layer_avg_deltas: list[float] = field(default_factory=list)
    organ_count: int = 0
    organ_names: list[str] = field(default_factory=list)
    fitness_components: dict[str, float] = field(default_factory=dict)
    total_generations: int = 0
    stagnation_episodes: int = 0
    restart_count: int = 0
    timestamp: float = field(default_factory=time.time)


class HarnessDesignSystem:
    """Analyzes system architecture and proposes structural modifications.

    This is the "brain" of the Harness: it looks at how the evolution engine
    is performing and decides whether the engine's own structure needs to change.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm
        self._design_history: list[HarnessDesign] = []
        self._evaluation_history: list[HarnessEvaluation] = []
        self._architecture_snapshots: list[ArchitectureSnapshot] = []

    def analyze(self, snapshot: ArchitectureSnapshot) -> list[HarnessDesign]:
        """Analyze architecture snapshot and propose designs."""
        self._architecture_snapshots.append(snapshot)
        designs = []

        # 1. Detect underperforming layers
        designs.extend(self._detect_underperforming_layers(snapshot))

        # 2. Detect missing capabilities
        designs.extend(self._detect_missing_capabilities(snapshot))

        # 3. Detect structural imbalances
        designs.extend(self._detect_structural_imbalances(snapshot))

        # 4. LLM-based architectural analysis if available
        if self._llm:
            llm_designs = self._llm_analyze(snapshot)
            designs.extend(llm_designs)

        # Sort by expected impact (highest first)
        designs.sort(key=lambda d: d.expected_impact, reverse=True)
        self._design_history.extend(designs)
        return designs

    def evaluate(self, design: HarnessDesign, before_fitness: float, after_fitness: float) -> HarnessEvaluation:
        """Evaluate the impact of an applied design."""
        actual_impact = after_fitness - before_fitness
        approved = actual_impact >= design.expected_impact * 0.5  # At least 50% of expected

        # Detect side effects
        side_effects = []
        if actual_impact < 0:
            side_effects.append("fitness_degradation")
        if actual_impact < design.expected_impact * 0.3:
            side_effects.append("underperformance")

        evaluation = HarnessEvaluation(
            design=design,
            approved=approved,
            actual_impact=actual_impact,
            side_effects=side_effects,
        )
        self._evaluation_history.append(evaluation)
        return evaluation

    def _detect_underperforming_layers(self, snapshot: ArchitectureSnapshot) -> list[HarnessDesign]:
        """Detect layers that consistently produce low fitness deltas."""
        designs = []
        for i, (name, delta) in enumerate(zip(snapshot.layer_names, snapshot.layer_avg_deltas)):
            if delta < 0.001 and snapshot.layer_execution_counts[i] > 10:
                # Layer has been executed many times but contributes almost nothing
                designs.append(HarnessDesign(
                    action=HarnessAction.MODIFY_LAYER_WEIGHT,
                    component=HarnessComponent.ENGINE,
                    description=f"Reduce weight of underperforming layer L{i}:{name} (avg_delta={delta:.4f})",
                    rationale=f"Layer {name} has avg delta {delta:.4f} after {snapshot.layer_execution_counts[i]} executions",
                    expected_impact=0.02,
                    risk_level=0.2,
                    parameters={"layer_id": i, "action": "reduce_weight"},
                ))
        return designs

    def _detect_missing_capabilities(self, snapshot: ArchitectureSnapshot) -> list[HarnessDesign]:
        """Detect when the system is missing important capabilities."""
        designs = []

        # If stagnation episodes are high, we may need new exploration strategies
        if snapshot.stagnation_episodes > 3:
            designs.append(HarnessDesign(
                action=HarnessAction.EXPAND_SEARCH_SPACE,
                component=HarnessComponent.ENGINE,
                description="Expand search space to escape repeated stagnation",
                rationale=f"{snapshot.stagnation_episodes} stagnation episodes detected",
                expected_impact=0.05,
                risk_level=0.3,
                parameters={"mutation_rate_boost": 0.2, "add_direction": "orthogonal"},
            ))

        # If no organs are registered, suggest adding them
        if snapshot.organ_count == 0:
            designs.append(HarnessDesign(
                action=HarnessAction.ADD_ORGAN,
                component=HarnessComponent.ORGANS,
                description="Add specialized organs for task decomposition and code generation",
                rationale="No organs registered; organs provide specialized processing",
                expected_impact=0.1,
                risk_level=0.1,
                parameters={"organs": ["taotie", "nuwa"]},
            ))

        return designs

    def _detect_structural_imbalances(self, snapshot: ArchitectureSnapshot) -> list[HarnessDesign]:
        """Detect when the architecture is structurally imbalanced."""
        designs = []

        # Check if some layers are over-executed relative to others
        if snapshot.layer_execution_counts:
            max_exec = max(snapshot.layer_execution_counts)
            min_exec = min(snapshot.layer_execution_counts)
            if max_exec > 0 and min_exec / max_exec < 0.1:
                # Severe execution imbalance
                under_executed_idx = snapshot.layer_execution_counts.index(min_exec)
                designs.append(HarnessDesign(
                    action=HarnessAction.REORDER_LAYERS,
                    component=HarnessComponent.ENGINE,
                    description=f"Reorder layers to give L{under_executed_idx} more execution opportunity",
                    rationale=f"Execution imbalance: max={max_exec}, min={min_exec}",
                    expected_impact=0.03,
                    risk_level=0.3,
                    parameters={"promote_layer": under_executed_idx},
                ))

        return designs

    def _llm_analyze(self, snapshot: ArchitectureSnapshot) -> list[HarnessDesign]:
        """Use LLM for architectural analysis."""
        try:
            prompt = (
                f"Analyze this AI evolution system architecture and suggest improvements:\n"
                f"Layers: {list(zip(snapshot.layer_names, snapshot.layer_avg_deltas))}\n"
                f"Organs: {snapshot.organ_names}\n"
                f"Fitness: {snapshot.fitness_components}\n"
                f"Stagnation episodes: {snapshot.stagnation_episodes}\n"
                f"Return a JSON array of objects with keys: action, component, description, expected_impact (0-1), risk_level (0-1)"
            )
            response = self._llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=300,
            )
            suggestions = json.loads(response.strip())
            designs = []
            for s in suggestions:
                if isinstance(s, dict):
                    try:
                        designs.append(HarnessDesign(
                            action=HarnessAction(s.get("action", "adapt_strategy")),
                            component=HarnessComponent(s.get("component", "engine")),
                            description=s.get("description", ""),
                            expected_impact=float(s.get("expected_impact", 0.0)),
                            risk_level=float(s.get("risk_level", 0.5)),
                        ))
                    except (ValueError, KeyError):
                        pass
            return designs
        except Exception:
            return []


class HarnessEvolutionEngine:
    """The Harness: meta-evolution engine that evolves the evolution system itself.

    This operates at a higher level than the UnifiedEvolutionEngine:
    - UnifiedEvolutionEngine evolves genomes (code, prompts, skills)
    - HarnessEvolutionEngine evolves the engine's own architecture

    The Harness runs periodically (e.g., every 100 generations) and:
    1. Takes a snapshot of the current architecture
    2. Analyzes it for structural issues
    3. Proposes modifications
    4. Validates modifications against safety constraints
    5. Applies approved modifications
    6. Evaluates the impact
    """

    def __init__(
        self,
        design_system: HarnessDesignSystem | None = None,
        safety_checker: Any = None,
        llm=None,
        evaluation_interval_generations: int = 100,
        max_risk_level: float = 0.7,
    ) -> None:
        self._design_system = design_system or HarnessDesignSystem(llm=llm)
        self._safety = safety_checker
        self._llm = llm
        self._eval_interval = evaluation_interval_generations
        self._max_risk = max_risk_level
        self._last_eval_generation: int = 0
        self._applied_designs: list[HarnessDesign] = []
        self._rejected_designs: list[HarnessDesign] = []
        self._generation_counter: int = 0

    def take_snapshot(self, engine: Any) -> ArchitectureSnapshot:
        """Take a snapshot of the current engine architecture."""
        snapshot = ArchitectureSnapshot()

        # Extract layer information
        try:
            for layer in engine._layers:
                snapshot.layer_names.append(layer.name)
                snapshot.layer_execution_counts.append(layer._execution_count)
                snapshot.layer_avg_deltas.append(
                    layer._total_fitness_delta / max(1, layer._execution_count)
                )
            snapshot.layer_count = len(engine._layers)
        except Exception:
            pass

        # Extract organ information
        try:
            if hasattr(engine, "_organs") and engine._organs:
                for organ in engine._organs:
                    snapshot.organ_names.append(organ.name if hasattr(organ, "name") else str(organ))
                snapshot.organ_count = len(engine._organs)
        except Exception:
            pass

        # Extract fitness information
        try:
            if engine._best_genome:
                snapshot.fitness_components = {
                    "best_fitness": engine._best_genome.fitness,
                    "skills_count": len(engine._best_genome.skills),
                    "tools_count": len(engine._best_genome.tools),
                    "code_length": len(engine._best_genome.code),
                }
        except Exception:
            pass

        snapshot.total_generations = getattr(engine, "_generation", 0)

        return snapshot

    def evaluate_and_apply(self, engine: Any, genome: Genome) -> list[HarnessEvaluation]:
        """Run a Harness evaluation cycle: analyze, propose, validate, apply."""
        self._generation_counter += 1

        # Only evaluate at intervals
        if self._generation_counter - self._last_eval_generation < self._eval_interval:
            return []

        self._last_eval_generation = self._generation_counter

        # Take snapshot
        snapshot = self.take_snapshot(engine)

        # Propose designs
        designs = self._design_system.analyze(snapshot)
        if not designs:
            return []

        # Get current fitness for comparison
        before_fitness = genome.fitness

        evaluations = []
        for design in designs:
            # Safety check
            if design.risk_level > self._max_risk:
                logger.info(f"Rejecting high-risk design: {design.description} (risk={design.risk_level:.2f})")
                self._rejected_designs.append(design)
                continue

            # Apply the design
            applied = self._apply_design(engine, genome, design)
            if applied:
                # Evaluate impact
                after_fitness = genome.fitness
                evaluation = self._design_system.evaluate(design, before_fitness, after_fitness)
                evaluations.append(evaluation)

                if evaluation.approved:
                    self._applied_designs.append(design)
                    logger.info(f"Applied Harness design: {design.description} (impact={evaluation.actual_impact:.4f})")
                else:
                    # Rollback
                    self._rollback_design(engine, genome, design)
                    self._rejected_designs.append(design)
                    logger.info(f"Rolled back Harness design: {design.description}")

                before_fitness = after_fitness  # Update for next design

        return evaluations

    def _apply_design(self, engine: Any, genome: Genome, design: HarnessDesign) -> bool:
        """Apply a Harness design to the engine/genome."""
        try:
            if design.action == HarnessAction.MODIFY_LAYER_WEIGHT:
                layer_id = design.parameters.get("layer_id", -1)
                action = design.parameters.get("action", "reduce_weight")
                if 0 <= layer_id < len(engine._layers):
                    if action == "reduce_weight":
                        # Skip this layer more often by reducing its execution
                        engine._layers[layer_id]._execution_count = max(
                            1, engine._layers[layer_id]._execution_count // 2
                        )
                    return True

            elif design.action == HarnessAction.EXPAND_SEARCH_SPACE:
                boost = design.parameters.get("mutation_rate_boost", 0.1)
                genome.config["mutation_rate"] = min(
                    0.8, genome.config.get("mutation_rate", 0.3) + boost
                )
                return True

            elif design.action == HarnessAction.ADAPT_STRATEGY:
                # Adjust engine-level strategy parameters
                if "mutation_rate_boost" in design.parameters:
                    genome.config["mutation_rate"] = min(
                        0.8, genome.config.get("mutation_rate", 0.3) + design.parameters["mutation_rate_boost"]
                    )
                return True

            elif design.action == HarnessAction.ADD_ORGAN:
                # This would require the engine to support dynamic organ registration
                # For now, log the recommendation
                logger.info(f"Harness recommends adding organs: {design.parameters.get('organs', [])}")
                return True

            elif design.action == HarnessAction.REORDER_LAYERS:
                # Promote a layer by moving it earlier in execution order
                promote_idx = design.parameters.get("promote_layer", -1)
                if 0 < promote_idx < len(engine._layers):
                    # Swap with the layer before it
                    engine._layers[promote_idx - 1], engine._layers[promote_idx] = (
                        engine._layers[promote_idx],
                        engine._layers[promote_idx - 1],
                    )
                    return True

            else:
                logger.debug(f"Harness action not yet implemented: {design.action}")
                return False

        except Exception as e:
            logger.warning(f"Failed to apply Harness design: {e}")
            return False

    def _rollback_design(self, engine: Any, genome: Genome, design: HarnessDesign) -> None:
        """Rollback a design that didn't produce expected results."""
        try:
            if design.action == HarnessAction.EXPAND_SEARCH_SPACE:
                boost = design.parameters.get("mutation_rate_boost", 0.1)
                genome.config["mutation_rate"] = max(
                    0.1, genome.config.get("mutation_rate", 0.3) - boost
                )
            elif design.action == HarnessAction.REORDER_LAYERS:
                promote_idx = design.parameters.get("promote_layer", -1)
                if 0 < promote_idx < len(engine._layers):
                    engine._layers[promote_idx - 1], engine._layers[promote_idx] = (
                        engine._layers[promote_idx],
                        engine._layers[promote_idx - 1],
                    )
        except Exception as e:
            logger.warning(f"Rollback failed: {e}")

    @property
    def stats(self) -> dict[str, Any]:
        """Get Harness statistics."""
        return {
            "generation_counter": self._generation_counter,
            "last_eval_generation": self._last_eval_generation,
            "applied_count": len(self._applied_designs),
            "rejected_count": len(self._rejected_designs),
            "design_history_count": len(self._design_system._design_history),
            "evaluation_history_count": len(self._design_system._evaluation_history),
        }
