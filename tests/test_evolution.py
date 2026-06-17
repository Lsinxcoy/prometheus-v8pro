"""Evolution layer tests."""

from prometheus_v8.schema import Genome
from prometheus_v8.evolution.engine import UnifiedEvolutionEngine
from prometheus_v8.evolution.fitness import ThreeStageFitness
from prometheus_v8.core.direction_selector import DirectionSelector
from prometheus_v8.evolution.anti_evolution import AntiEvolutionDetector
from prometheus_v8.evolution.goal_system import GoalSystem, GoalState
from prometheus_v8.evolution.reasoning_budget import ReasoningBudget
from prometheus_v8.evolution.parallel import ParallelIslands


def test_three_stage_fitness():
    fitness = ThreeStageFitness()
    genome = Genome(code="def hello():\n    return 'world'", fitness=0.5)
    result = fitness.evaluate(genome)
    assert 0 <= result.composite <= 1.0
    assert 0 <= result.static_score <= 1.0
    assert 0 <= result.dynamic_score <= 1.0


def test_direction_selector():
    ds = DirectionSelector()
    direction = ds.select()
    assert direction in ("forward", "lateral", "reverse")
    ds.update(direction, 0.8)


def test_anti_evolution():
    detector = AntiEvolutionDetector()
    alerts = detector.check(0.5, "def test(): pass", 0.5, 0.8)
    assert isinstance(alerts, list)


def test_goal_system():
    gs = GoalSystem()
    goal = gs.create_goal("Improve fitness", fitness_target=0.8)
    assert goal.state == GoalState.PENDING
    gs.activate_goal(goal.id)
    assert gs.get_active_goal() is not None


def test_reasoning_budget():
    budget = ReasoningBudget(tokens=4000, time_seconds=60, max_steps=10)
    state = budget.allocate(0.5)
    assert state.max_steps == 30  # medium complexity
    assert budget.should_continue()


def test_parallel_islands():
    islands = ParallelIslands(island_count=2, population_per_island=3)
    genome = Genome(code="def test(): pass", fitness=0.4)
    islands.initialize(genome)
    results = islands.evolve_one_generation()
    assert "island_0" in results
