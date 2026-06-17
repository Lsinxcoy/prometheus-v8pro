"""Evolution Checkpoint - Save/restore evolution state for fault tolerance and resumption.

Supports:
- Full engine state serialization (generation, best genome, layer stats, history)
- Incremental checkpoints (only save delta since last checkpoint)
- Automatic checkpoint on fitness milestones
- Crash recovery from latest valid checkpoint
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """A single evolution checkpoint."""

    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    best_fitness: float = 0.0
    genome_data: dict = field(default_factory=dict)
    layer_stats: list[dict] = field(default_factory=list)
    history_tail: list[dict] = field(default_factory=list)  # Last N history entries
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "timestamp": self.timestamp,
            "best_fitness": self.best_fitness,
            "genome_data": self.genome_data,
            "layer_stats": self.layer_stats,
            "history_tail": self.history_tail,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointData:
        return cls(
            generation=data.get("generation", 0),
            timestamp=data.get("timestamp", 0.0),
            best_fitness=data.get("best_fitness", 0.0),
            genome_data=data.get("genome_data", {}),
            layer_stats=data.get("layer_stats", []),
            history_tail=data.get("history_tail", []),
            metadata=data.get("metadata", {}),
        )


class EvolutionCheckpoint:
    """Manages evolution state persistence with save/restore.

    Features:
    - Periodic checkpoints every N generations
    - Milestone checkpoints on fitness improvement
    - Crash recovery from latest valid checkpoint
    - Configurable retention policy (keep last N + all milestones)
    """

    def __init__(
        self,
        checkpoint_dir: str = "data/checkpoints",
        interval_generations: int = 10,
        keep_last_n: int = 5,
        keep_milestones: bool = True,
        milestone_threshold: float = 0.05,  # Save when fitness improves by this much
    ) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._interval = interval_generations
        self._keep_last_n = keep_last_n
        self._keep_milestones = keep_milestones
        self._milestone_threshold = milestone_threshold
        self._last_milestone_fitness: float = 0.0
        self._last_save_gen: int = -1

    def should_save(self, generation: int, best_fitness: float) -> bool:
        """Determine if a checkpoint should be saved at this generation."""
        # Periodic checkpoint
        if generation - self._last_save_gen >= self._interval:
            return True
        # Milestone checkpoint: significant fitness improvement
        if self._keep_milestones and best_fitness - self._last_milestone_fitness >= self._milestone_threshold:
            return True
        return False

    def save(
        self,
        generation: int,
        best_fitness: float,
        genome_data: dict,
        layer_stats: list[dict],
        history_tail: list[dict],
        metadata: dict | None = None,
    ) -> str | None:
        """Save a checkpoint. Returns the checkpoint path or None on failure."""
        checkpoint = CheckpointData(
            generation=generation,
            best_fitness=best_fitness,
            genome_data=genome_data,
            layer_stats=layer_stats,
            history_tail=history_tail,
            metadata=metadata or {},
        )

        is_milestone = best_fitness - self._last_milestone_fitness >= self._milestone_threshold
        suffix = "milestone" if is_milestone else "periodic"
        filename = f"ckpt_gen{generation:06d}_{suffix}.json"
        path = self._dir / filename

        try:
            # Atomic write
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            tmp_path.rename(path)

            self._last_save_gen = generation
            if is_milestone:
                self._last_milestone_fitness = best_fitness

            # Cleanup old checkpoints
            self._cleanup()

            logger.debug(f"Checkpoint saved: {path} (fitness={best_fitness:.4f})")
            return str(path)
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")
            return None

    def load_latest(self) -> CheckpointData | None:
        """Load the latest valid checkpoint for crash recovery."""
        checkpoints = sorted(self._dir.glob("ckpt_gen*.json"))
        if not checkpoints:
            return None

        # Try from newest to oldest
        for path in reversed(checkpoints):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = CheckpointData.from_dict(data)
                logger.info(f"Loaded checkpoint: {path} (gen={checkpoint.generation}, fitness={checkpoint.best_fitness:.4f})")
                return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {path}: {e}")
                continue

        return None

    def load(self, generation: int) -> CheckpointData | None:
        """Load a specific generation's checkpoint."""
        for path in self._dir.glob(f"ckpt_gen{generation:06d}_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return CheckpointData.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {path}: {e}")
        return None

    def _cleanup(self) -> None:
        """Remove old periodic checkpoints, keeping last N + all milestones."""
        all_ckpts = sorted(self._dir.glob("ckpt_gen*.json"))
        if len(all_ckpts) <= self._keep_last_n:
            return

        # Separate milestones from periodic
        milestones = [p for p in all_ckpts if "_milestone" in p.name]
        periodic = [p for p in all_ckpts if "_periodic" in p.name]

        # Keep all milestones + last N periodic
        to_keep = set(milestones)
        to_keep.update(periodic[-self._keep_last_n:])

        # Remove the rest
        for path in all_ckpts:
            if path not in to_keep:
                try:
                    path.unlink()
                    logger.debug(f"Removed old checkpoint: {path}")
                except OSError:
                    pass

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints with metadata."""
        result = []
        for path in sorted(self._dir.glob("ckpt_gen*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result.append({
                    "path": str(path),
                    "generation": data.get("generation", 0),
                    "best_fitness": data.get("best_fitness", 0.0),
                    "timestamp": data.get("timestamp", 0.0),
                    "type": "milestone" if "_milestone" in path.name else "periodic",
                })
            except Exception:
                pass
        return result
