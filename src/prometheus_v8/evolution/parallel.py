"""Parallel Island Model with Ring Migration."""
from __future__ import annotations
import copy
import logging
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Optional
from prometheus_v8.schema import Genome
from prometheus_v8.evolution.crossover import ASTCrossover
from prometheus_v8.evolution.evolution_memory import EvolutionMemory

logger = logging.getLogger(__name__)

@dataclass
class Island:
    """A single evolution island with its own population."""
    id: int = 0
    population: list[Genome] = field(default_factory=list)
    best_fitness: float = 0.0
    generation: int = 0

class ParallelIslands:
    """Island model with ring migration between islands.
    
    Islands evolve independently, periodically exchange best individuals
    through ring topology migration.
    """
    
    def __init__(self, island_count: int = 4, population_per_island: int = 5,
                 migration_interval: int = 10, migrant_count: int = 1) -> None:
        self._island_count = island_count
        self._pop_size = population_per_island
        self._migration_interval = migration_interval
        self._migrant_count = migrant_count
        self._islands: list[Island] = [Island(id=i) for i in range(island_count)]
        self._crossover = ASTCrossover()
        self._memory = EvolutionMemory()
        self._lock = threading.RLock()
    
    def initialize(self, base_genome: Genome) -> None:
        """Initialize all islands with variants of the base genome."""
        for island in self._islands:
            island.population = []
            for _ in range(self._pop_size):
                variant = copy.deepcopy(base_genome)
                # Add random variation
                variant.fitness *= random.uniform(0.8, 1.0)
                island.population.append(variant)
            island.population.sort(key=lambda g: g.fitness, reverse=True)
    
    def evolve_one_generation(self) -> dict[str, Any]:
        """Evolve each island for one generation and check for migration."""
        results = {}
        for island in self._islands:
            island.generation += 1
            self._evolve_island(island)
            results[f"island_{island.id}"] = {
                "best_fitness": island.population[0].fitness if island.population else 0,
                "avg_fitness": sum(g.fitness for g in island.population) / max(1, len(island.population)),
                "generation": island.generation,
            }
        
        # Check migration
        any_migrated = False
        for island in self._islands:
            if island.generation % self._migration_interval == 0:
                self._migrate(island.id)
                any_migrated = True
        
        results["migration"] = any_migrated
        return results
    
    def _evolve_island(self, island: Island) -> None:
        """Evolve a single island for one generation."""
        if len(island.population) < 2:
            return
        
        # Selection: keep top 50%
        elite_count = max(1, len(island.population) // 2)
        new_pop = island.population[:elite_count]
        
        # Crossover to fill remaining
        while len(new_pop) < self._pop_size:
            p1 = random.choice(island.population[:elite_count])
            p2 = random.choice(island.population)
            child = self._crossover.crossover(p1, p2)
            new_pop.append(child)
        
        island.population = new_pop
        island.population.sort(key=lambda g: g.fitness, reverse=True)
        island.best_fitness = island.population[0].fitness
        
        # Store in evolution memory
        if island.population:
            self._memory.add_short_term(island.population[0], island.generation)
            self._memory.add_medium_term(f"island_{island.id}", island.population[0], island.generation)
    
    def _migrate(self, from_island_id: int) -> None:
        """Ring migration: send best from island to next island."""
        from_island = self._islands[from_island_id]
        to_island = self._islands[(from_island_id + 1) % self._island_count]
        
        migrants = from_island.population[:self._migrant_count]
        for migrant in migrants:
            migrant_copy = copy.deepcopy(migrant)
            to_island.population.append(migrant_copy)
        
        to_island.population.sort(key=lambda g: g.fitness, reverse=True)
        to_island.population = to_island.population[:self._pop_size]
        
        logger.debug(f"Migration: island {from_island_id} → {to_island.id}, {len(migrants)} migrants")
    
    def get_best_genome(self) -> Genome | None:
        best = None
        for island in self._islands:
            if island.population and (best is None or island.population[0].fitness > best.fitness):
                best = island.population[0]
        return best
    
    @property
    def stats(self) -> dict[str, Any]:
        return {
            "islands": self._island_count,
            "total_population": sum(len(i.population) for i in self._islands),
            "best_fitness": max((i.best_fitness for i in self._islands), default=0),
            "memory": self._memory.stats,
        }
