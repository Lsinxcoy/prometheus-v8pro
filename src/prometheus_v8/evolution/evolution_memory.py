"""Evolution Memory - 3-layer memory with TTL.

From V7 dropped feature, restored:
- Short-term: Last N generations (fast access)
- Medium-term: Best individuals per island (1 hour TTL)
- Long-term: Hall-of-fame (persistent, no TTL)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v8.schema import Genome

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    genome: Genome
    fitness: float
    generation: int
    timestamp: float = field(default_factory=time.time)
    ttl: float = 0.0  # 0 = never expires

    def is_expired(self) -> bool:
        return self.ttl > 0 and time.time() - self.timestamp > self.ttl


class EvolutionMemory:
    """3-layer evolution memory with TTL.

    Short-term: Last 50 individuals (no TTL)
    Medium-term: Best per island (1h TTL)
    Long-term: Hall-of-fame (persistent)
    """

    def __init__(self, short_term_size: int = 50, medium_term_ttl: float = 3600.0) -> None:
        self._short_term: list[MemoryEntry] = []
        self._medium_term: dict[str, MemoryEntry] = {}  # island_id → entry
        self._hall_of_fame: list[MemoryEntry] = []
        self._short_term_size = short_term_size
        self._medium_term_ttl = medium_term_ttl
        self._lock = threading.RLock()

    def add_short_term(self, genome: Genome, generation: int) -> None:
        with self._lock:
            self._short_term.append(MemoryEntry(genome=genome, fitness=genome.fitness, generation=generation))
            if len(self._short_term) > self._short_term_size:
                self._short_term = self._short_term[-self._short_term_size :]

    def add_medium_term(self, island_id: str, genome: Genome, generation: int) -> None:
        with self._lock:
            existing = self._medium_term.get(island_id)
            if not existing or genome.fitness > existing.fitness:
                self._medium_term[island_id] = MemoryEntry(
                    genome=genome,
                    fitness=genome.fitness,
                    generation=generation,
                    ttl=self._medium_term_ttl,
                )

    def add_hall_of_fame(self, genome: Genome, generation: int) -> None:
        with self._lock:
            # Only add if better than worst in hall (or hall < 10)
            if len(self._hall_of_fame) < 10 or genome.fitness > self._hall_of_fame[-1].fitness:
                self._hall_of_fame.append(MemoryEntry(genome=genome, fitness=genome.fitness, generation=generation))
                self._hall_of_fame.sort(key=lambda e: e.fitness, reverse=True)
                if len(self._hall_of_fame) > 10:
                    self._hall_of_fame = self._hall_of_fame[:10]

    def get_best(self, n: int = 1) -> list[Genome]:
        with self._lock:
            all_entries = self._short_term + list(self._medium_term.values()) + self._hall_of_fame
            # Filter expired
            valid = [e for e in all_entries if not e.is_expired()]
            valid.sort(key=lambda e: e.fitness, reverse=True)
            return [e.genome for e in valid[:n]]

    def get_diverse(self, n: int = 3) -> list[Genome]:
        """Get n diverse individuals from different memory layers."""
        genomes = []
        # One from short-term
        if self._short_term:
            genomes.append(self._short_term[-1].genome)
        # One from medium-term
        for entry in self._medium_term.values():
            if not entry.is_expired():
                genomes.append(entry.genome)
                break
        # One from hall-of-fame
        if self._hall_of_fame:
            genomes.append(self._hall_of_fame[0].genome)
        return genomes[:n]

    def cleanup(self) -> int:
        """Remove expired medium-term entries. Returns count removed."""
        with self._lock:
            before = len(self._medium_term)
            self._medium_term = {k: v for k, v in self._medium_term.items() if not v.is_expired()}
            return before - len(self._medium_term)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "short_term": len(self._short_term),
            "medium_term": len(self._medium_term),
            "hall_of_fame": len(self._hall_of_fame),
        }
