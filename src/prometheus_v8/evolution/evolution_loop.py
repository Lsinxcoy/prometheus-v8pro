"""Evolution Loop - Continuous evolution with checkpoint, stagnation restart, and signal handling.

This is the missing Harness Loop component from V7 that was lost in V8.
It provides:
- Periodic evolution cycles with configurable interval
- Automatic checkpoint save/restore
- Stagnation detection with restart strategy
- Signal handling for graceful shutdown
- Organ feedback integration (organs report fitness signals)
"""

from __future__ import annotations

import copy
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from prometheus_v8.evolution.checkpoint import EvolutionCheckpoint
from prometheus_v8.schema import Genome

logger = logging.getLogger(__name__)


class LoopState(str, Enum):
    """State of the evolution loop."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    ERROR = "error"


@dataclass
class LoopConfig:
    """Configuration for the evolution loop."""
    cycle_interval_seconds: float = 60.0  # Time between evolution cycles
    max_generations_per_cycle: int = 5  # Generations per cycle
    checkpoint_interval: int = 10  # Save checkpoint every N generations
    stagnation_restart_threshold: int = 30  # Restart after N stagnant generations
    stagnation_fitness_delta: float = 0.01  # Minimum improvement to not count as stagnant
    fitness_target: float = 0.95  # Stop when this fitness is reached
    max_total_generations: int = 1000  # Hard cap on total generations
    enable_signal_handler: bool = True  # Handle SIGINT/SIGTERM for graceful shutdown


@dataclass
class LoopStatus:
    """Current status of the evolution loop."""
    state: LoopState = LoopState.IDLE
    total_cycles: int = 0
    total_generations: int = 0
    current_generation: int = 0
    best_fitness: float = 0.0
    stagnation_count: int = 0
    restart_count: int = 0
    last_cycle_time: float = 0.0
    total_time_seconds: float = 0.0
    checkpoints_saved: int = 0


class OrganFeedbackCollector:
    """Collects feedback signals from organs to feed back into evolution.

    Organs (Taotie/Nuwa/Darwin/etc.) can register feedback callbacks
    that report fitness-relevant signals. The loop aggregates these
    and passes them to the evolution engine as kwargs.
    """

    def __init__(self) -> None:
        self._callbacks: dict[str, Callable[[], dict[str, Any]]] = {}
        self._last_feedback: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register(self, organ_name: str, feedback_fn: Callable[[], dict[str, Any]]) -> None:
        """Register a feedback callback from an organ."""
        with self._lock:
            self._callbacks[organ_name] = feedback_fn

    def unregister(self, organ_name: str) -> None:
        """Unregister an organ's feedback callback."""
        with self._lock:
            self._callbacks.pop(organ_name, None)

    def collect(self) -> dict[str, Any]:
        """Collect feedback from all registered organs."""
        feedback: dict[str, Any] = {}
        with self._lock:
            for name, fn in self._callbacks.items():
                try:
                    result = fn()
                    self._last_feedback[name] = result
                    feedback[name] = result
                except Exception as e:
                    logger.warning(f"Organ feedback error from {name}: {e}")
        return feedback

    @property
    def registered_organs(self) -> list[str]:
        with self._lock:
            return list(self._callbacks.keys())


class EvolutionLoop:
    """Continuous evolution loop with checkpoint, stagnation restart, and organ feedback.

    Usage:
        engine = UnifiedEvolutionEngine(llm=llm)
        loop = EvolutionLoop(engine, genome=genome)
        loop.start()  # Runs in background thread
        # ... later ...
        loop.stop()   # Graceful shutdown
    """

    def __init__(
        self,
        engine: Any,  # UnifiedEvolutionEngine
        genome: Genome,
        config: LoopConfig | None = None,
        checkpoint_dir: str = "data/checkpoints",
        organ_feedback: OrganFeedbackCollector | None = None,
    ) -> None:
        self._engine = engine
        self._genome = genome
        self._config = config or LoopConfig()
        self._checkpoint = EvolutionCheckpoint(
            checkpoint_dir=checkpoint_dir,
            interval_generations=self._config.checkpoint_interval,
        )
        self._organ_feedback = organ_feedback or OrganFeedbackCollector()
        self._status = LoopStatus()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_fitness_improved: list[Callable[[float, int], None]] = []
        self._on_stagnation: list[Callable[[int], None]] = []
        self._on_restart: list[Callable[[int], None]] = []

    def start(self) -> None:
        """Start the evolution loop in a background thread."""
        with self._lock:
            if self._status.state == LoopState.RUNNING:
                return
            self._stop_event.clear()
            self._pause_event.clear()

            # Try to restore from checkpoint
            restored = self._try_restore()
            if restored:
                logger.info(f"Resumed from checkpoint at generation {self._status.current_generation}")

            # Register signal handlers if enabled
            if self._config.enable_signal_handler:
                self._register_signal_handlers()

            self._status.state = LoopState.RUNNING
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="evolution-loop")
            self._thread.start()
            logger.info("Evolution loop started")

    def stop(self) -> None:
        """Gracefully stop the evolution loop."""
        logger.info("Stopping evolution loop...")
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused
        with self._lock:
            self._status.state = LoopState.STOPPED
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        # Final checkpoint
        self._save_checkpoint()
        logger.info(f"Evolution loop stopped at generation {self._status.current_generation}")

    def pause(self) -> None:
        """Pause the evolution loop."""
        with self._lock:
            if self._status.state == LoopState.RUNNING:
                self._status.state = LoopState.PAUSED
                self._pause_event.set()
                self._save_checkpoint()
                logger.info("Evolution loop paused")

    def resume(self) -> None:
        """Resume a paused evolution loop."""
        with self._lock:
            if self._status.state == LoopState.PAUSED:
                self._status.state = LoopState.RUNNING
                self._pause_event.clear()
                logger.info("Evolution loop resumed")

    @property
    def status(self) -> LoopStatus:
        return self._status

    @property
    def organ_feedback(self) -> OrganFeedbackCollector:
        return self._organ_feedback

    def on_fitness_improved(self, callback: Callable[[float, int], None]) -> None:
        """Register callback for fitness improvement events."""
        self._on_fitness_improved.append(callback)

    def on_stagnation(self, callback: Callable[[int], None]) -> None:
        """Register callback for stagnation events."""
        self._on_stagnation.append(callback)

    def on_restart(self, callback: Callable[[int], None]) -> None:
        """Register callback for restart events."""
        self._on_restart.append(callback)

    def _run_loop(self) -> None:
        """Main loop executed in background thread."""
        start_time = time.time()

        while not self._stop_event.is_set():
            # Check pause
            while self._pause_event.is_set() and not self._stop_event.is_set():
                time.sleep(0.5)
            if self._stop_event.is_set():
                break

            # Check hard generation cap
            if self._status.total_generations >= self._config.max_total_generations:
                logger.info(f"Max total generations reached: {self._config.max_total_generations}")
                break

            # Check fitness target
            if self._status.best_fitness >= self._config.fitness_target:
                logger.info(f"Fitness target reached: {self._status.best_fitness:.4f}")
                break

            try:
                self._run_cycle()
            except Exception as e:
                logger.error(f"Evolution cycle error: {e}", exc_info=True)
                with self._lock:
                    self._status.state = LoopState.ERROR
                # Save checkpoint on error for recovery
                self._save_checkpoint()
                break

            # Wait for next cycle
            cycle_time = time.time() - start_time
            self._status.total_time_seconds = cycle_time
            if self._config.cycle_interval_seconds > 0:
                self._stop_event.wait(timeout=self._config.cycle_interval_seconds)

        # Final status update
        with self._lock:
            if self._status.state != LoopState.ERROR:
                self._status.state = LoopState.STOPPED
        self._status.total_time_seconds = time.time() - start_time

    def _run_cycle(self) -> None:
        """Execute one evolution cycle (multiple generations)."""
        # Collect organ feedback
        organ_data = self._organ_feedback.collect()

        prev_best = self._status.best_fitness

        for _ in range(self._config.max_generations_per_cycle):
            if self._stop_event.is_set():
                break

            # Run evolution step with organ feedback as kwargs
            kwargs = {"organ_feedback": organ_data} if organ_data else {}
            result = self._engine.evolve_single_step(self._genome, **kwargs)

            self._status.current_generation += 1
            self._status.total_generations += 1

            # Update best fitness
            if self._genome.fitness > self._status.best_fitness:
                self._status.best_fitness = self._genome.fitness
                self._status.stagnation_count = 0
                # Notify listeners
                for cb in self._on_fitness_improved:
                    try:
                        cb(self._status.best_fitness, self._status.current_generation)
                    except Exception:
                        pass
            else:
                # Check stagnation
                if self._genome.fitness - prev_best < self._config.stagnation_fitness_delta:
                    self._status.stagnation_count += 1
                else:
                    self._status.stagnation_count = 0

            # Stagnation restart
            if self._status.stagnation_count >= self._config.stagnation_restart_threshold:
                self._handle_stagnation()
                break

            # Periodic checkpoint
            if self._checkpoint.should_save(self._status.current_generation, self._status.best_fitness):
                self._save_checkpoint()
                self._status.checkpoints_saved += 1

        self._status.total_cycles += 1
        self._status.last_cycle_time = time.time()

    def _handle_stagnation(self) -> None:
        """Handle stagnation by resetting the genome with preserved knowledge."""
        logger.warning(
            f"Stagnation detected at generation {self._status.current_generation} "
            f"(stagnation_count={self._status.stagnation_count})"
        )

        # Save current state before restart
        self._save_checkpoint()

        # Restart strategy: reset genome but preserve learned skills/config
        preserved_skills = list(self._genome.skills)
        preserved_config = dict(self._genome.config)
        preserved_tools = list(self._genome.tools)

        # Reset genome to fresh state with preserved knowledge
        self._genome.fitness = 0.0
        self._genome.code = ""
        self._genome.prompts = []
        self._genome.skills = preserved_skills  # Keep learned skills
        self._genome.config = {
            "mutation_rate": 0.5,  # Higher for fresh exploration
            "crossover_rate": preserved_config.get("crossover_rate", 0.7),
            "elite_ratio": 0.05,  # Lower elitism for more exploration
        }
        self._genome.tools = preserved_tools

        self._status.stagnation_count = 0
        self._status.restart_count += 1

        # Notify listeners
        for cb in self._on_stagnation:
            try:
                cb(self._status.current_generation)
            except Exception:
                pass
        for cb in self._on_restart:
            try:
                cb(self._status.restart_count)
            except Exception:
                pass

        logger.info(f"Stagnation restart #{self._status.restart_count}")

    def _save_checkpoint(self) -> None:
        """Save current evolution state to checkpoint."""
        genome_data = {}
        try:
            genome_data = {
                "fitness": self._genome.fitness,
                "code_length": len(self._genome.code),
                "skills": list(self._genome.skills),
                "tools": list(self._genome.tools),
                "config": dict(self._genome.config),
                "prompts_count": len(self._genome.prompts),
                "fingerprint": self._genome.fingerprint,
            }
        except Exception:
            pass

        layer_stats = []
        try:
            layer_stats = self._engine.layer_stats
        except Exception:
            pass

        history_tail = []
        try:
            history_tail = self._engine.history[-20:]
        except Exception:
            pass

        self._checkpoint.save(
            generation=self._status.current_generation,
            best_fitness=self._status.best_fitness,
            genome_data=genome_data,
            layer_stats=layer_stats,
            history_tail=history_tail,
            metadata={
                "restart_count": self._status.restart_count,
                "stagnation_count": self._status.stagnation_count,
                "total_cycles": self._status.total_cycles,
            },
        )

    def _try_restore(self) -> bool:
        """Try to restore from the latest checkpoint."""
        checkpoint = self._checkpoint.load_latest()
        if not checkpoint:
            return False

        # Restore genome state from checkpoint
        gd = checkpoint.genome_data
        if gd:
            self._genome.fitness = gd.get("fitness", 0.0)
            self._genome.skills = gd.get("skills", [])
            self._genome.tools = gd.get("tools", [])
            self._genome.config = gd.get("config", {})
            self._genome.fingerprint = gd.get("fingerprint", "")

        # Restore loop status
        self._status.current_generation = checkpoint.generation
        self._status.best_fitness = checkpoint.best_fitness
        self._status.total_generations = checkpoint.generation
        md = checkpoint.metadata
        self._status.restart_count = md.get("restart_count", 0)
        self._status.total_cycles = md.get("total_cycles", 0)

        # Update checkpoint manager's internal state
        self._checkpoint._last_save_gen = checkpoint.generation
        self._checkpoint._last_milestone_fitness = checkpoint.best_fitness

        return True

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (OSError, ValueError):
            # Can't set signal handler (not in main thread, or no signals on Windows)
            pass

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, stopping evolution loop...")
        self.stop()
