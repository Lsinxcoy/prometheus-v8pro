"""Orchestrator Adapter - Task decomposition, worker assignment, and result aggregation."""

from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class WorkerState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    DEAD = "dead"


@dataclass
class SubTask:
    """A decomposed subtask."""

    id: str = ""
    parent_id: str = ""
    name: str = ""
    description: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    priority: int = 5
    state: TaskState = TaskState.PENDING
    assigned_worker: str = ""
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    timeout_seconds: float = 300.0
    max_retries: int = 2
    retry_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED, TaskState.TIMEOUT)


@dataclass
class Worker:
    """A worker agent that can execute subtasks."""

    id: str = ""
    name: str = ""
    role: str = "worker"  # ceo/worker/explorer/judge
    model: str = ""
    model_tier: str = "standard"  # pro/standard/light
    capabilities: list[str] = field(default_factory=list)
    max_concurrent: int = 3
    state: WorkerState = WorkerState.IDLE
    current_tasks: list[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    avg_task_duration: float = 0.0

    @property
    def is_available(self) -> bool:
        return self.state == WorkerState.IDLE and len(self.current_tasks) < self.max_concurrent

    @property
    def success_rate(self) -> float:
        total = self.completed_tasks + self.failed_tasks
        return self.completed_tasks / max(1, total)

    def can_handle(self, required_capabilities: list[str]) -> bool:
        if "*" in self.capabilities:
            return True
        return all(c in self.capabilities for c in required_capabilities)


@dataclass
class TaskResult:
    """Aggregated result from a decomposed task."""

    task_id: str = ""
    subtask_results: dict[str, Any] = field(default_factory=dict)
    subtask_errors: dict[str, str] = field(default_factory=dict)
    total_subtasks: int = 0
    completed_subtasks: int = 0
    failed_subtasks: int = 0
    total_duration: float = 0.0
    success: bool = False
    summary: str = ""

    @property
    def completion_rate(self) -> float:
        return self.completed_subtasks / max(1, self.total_subtasks)


class OrchestratorAdapter:
    """Task orchestrator with decomposition, worker assignment, and result aggregation.

    Features:
    - Task decomposition into subtasks
    - Worker assignment based on capabilities and model tier
    - Task lifecycle management (pending/running/completed/failed)
    - Result aggregation from multiple workers
    - Timeout and retry logic
    - Priority queue for tasks
    - Worker health monitoring
    - Model tier assignment (CEO=pro, Worker=standard)
    """

    def __init__(
        self, default_timeout: float = 300.0, max_retries: int = 2, health_check_interval: float = 60.0
    ) -> None:
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._health_interval = health_check_interval
        self._tasks: dict[str, SubTask] = {}
        self._workers: dict[str, Worker] = {}
        self._task_queue: list[tuple[int, float, str]] = []  # (priority, created, task_id)
        self._results: dict[str, TaskResult] = {}
        self._lock = threading.RLock()
        self._decompose_fn: Callable[[str, dict], list[SubTask]] | None = None
        self._execute_fn: Callable[[SubTask, Worker], Any] | None = None
        self._running = False

    def set_decompose_fn(self, fn: Callable[[str, dict], list[SubTask]]) -> None:
        """Set the task decomposition function."""
        self._decompose_fn = fn

    def set_execute_fn(self, fn: Callable[[SubTask, Worker], Any]) -> None:
        """Set the subtask execution function."""
        self._execute_fn = fn

    def register_worker(self, worker: Worker) -> None:
        """Register a worker agent."""
        with self._lock:
            self._workers[worker.id] = worker
        logger.info(f"Worker registered: {worker.name} (role={worker.role}, tier={worker.model_tier})")

    def deregister_worker(self, worker_id: str) -> None:
        """Deregister a worker agent."""
        with self._lock:
            self._workers.pop(worker_id, None)

    def submit_task(
        self,
        name: str,
        description: str = "",
        required_capabilities: list[str] | None = None,
        priority: int = 5,
        timeout: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Submit a task for decomposition and execution."""
        task_id = str(uuid.uuid4())[:8]
        task = SubTask(
            id=task_id,
            name=name,
            description=description,
            required_capabilities=required_capabilities or [],
            priority=priority,
            timeout_seconds=timeout or self._default_timeout,
            metadata=metadata or {},
        )
        with self._lock:
            self._tasks[task_id] = task
            heapq.heappush(self._task_queue, (priority, task.created_at, task_id))
        logger.info(f"Task submitted: {name} (id={task_id}, priority={priority})")
        return task_id

    def decompose_task(self, task_id: str) -> list[SubTask]:
        """Decompose a task into subtasks."""
        task = self._tasks.get(task_id)
        if not task:
            return []

        if self._decompose_fn:
            subtasks = self._decompose_fn(task.name, task.metadata)
        else:
            # Default decomposition: single subtask
            subtasks = [
                SubTask(
                    id=f"{task_id}_0",
                    parent_id=task_id,
                    name=task.name,
                    description=task.description,
                    required_capabilities=task.required_capabilities,
                    priority=task.priority,
                    timeout_seconds=task.timeout_seconds,
                )
            ]

        for st in subtasks:
            with self._lock:
                self._tasks[st.id] = st

        return subtasks

    def assign_worker(self, subtask: SubTask) -> Worker | None:
        """Assign the best available worker for a subtask."""
        with self._lock:
            candidates = [
                w for w in self._workers.values() if w.is_available and w.can_handle(subtask.required_capabilities)
            ]

        if not candidates:
            return None

        # Prefer workers with matching model tier for the task type
        # CEO tasks -> pro tier, Worker tasks -> standard tier
        if subtask.required_capabilities and "strategic" in subtask.required_capabilities:
            pro_workers = [w for w in candidates if w.model_tier == "pro"]
            if pro_workers:
                return max(pro_workers, key=lambda w: w.success_rate)

        # Otherwise pick the worker with best success rate
        return max(candidates, key=lambda w: (w.success_rate, -len(w.current_tasks)))

    def execute_subtask(self, subtask_id: str) -> Any:
        """Execute a single subtask with timeout and retry."""
        subtask = self._tasks.get(subtask_id)
        if not subtask:
            return None

        worker = self.assign_worker(subtask)
        if not worker:
            subtask.state = TaskState.PENDING
            logger.warning(f"No available worker for subtask {subtask.name}")
            return None

        # Assign worker
        subtask.assigned_worker = worker.id
        subtask.state = TaskState.RUNNING
        subtask.started_at = time.time()
        worker.state = WorkerState.BUSY
        worker.current_tasks.append(subtask_id)

        try:
            if self._execute_fn:
                result = self._execute_fn(subtask, worker)
            else:
                # Default: mark as completed
                result = f"Completed: {subtask.name}"

            subtask.result = result
            subtask.state = TaskState.COMPLETED
            subtask.completed_at = time.time()
            worker.completed_tasks += 1
            worker.avg_task_duration = (
                worker.avg_task_duration * (worker.completed_tasks - 1) + subtask.duration
            ) / worker.completed_tasks
            return result

        except Exception as e:
            subtask.error = str(e)
            subtask.retry_count += 1

            if subtask.retry_count < subtask.max_retries:
                subtask.state = TaskState.PENDING
                logger.warning(f"Subtask {subtask.name} failed, retrying ({subtask.retry_count}/{subtask.max_retries})")
            else:
                subtask.state = TaskState.FAILED
                subtask.completed_at = time.time()
                worker.failed_tasks += 1
                logger.error(f"Subtask {subtask.name} failed after {subtask.max_retries} retries: {e}")
            return None

        finally:
            if subtask_id in worker.current_tasks:
                worker.current_tasks.remove(subtask_id)
            if not worker.current_tasks:
                worker.state = WorkerState.IDLE

    def aggregate_results(self, parent_task_id: str) -> TaskResult:
        """Aggregate results from all subtasks of a parent task."""
        subtasks = [t for t in self._tasks.values() if t.parent_id == parent_task_id]

        result = TaskResult(
            task_id=parent_task_id,
            total_subtasks=len(subtasks),
        )

        for st in subtasks:
            if st.state == TaskState.COMPLETED:
                result.completed_subtasks += 1
                result.subtask_results[st.id] = st.result
            elif st.state in (TaskState.FAILED, TaskState.TIMEOUT):
                result.failed_subtasks += 1
                result.subtask_errors[st.id] = st.error
            result.total_duration = max(result.total_duration, st.duration)

        result.success = result.completed_subtasks == result.total_subtasks
        result.summary = f"Completed {result.completed_subtasks}/{result.total_subtasks} subtasks"

        self._results[parent_task_id] = result
        return result

    def get_task(self, task_id: str) -> SubTask | None:
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    def get_pending_tasks(self) -> list[SubTask]:
        with self._lock:
            return [t for t in self._tasks.values() if t.state == TaskState.PENDING]

    def get_workers(self, state: WorkerState | None = None) -> list[Worker]:
        with self._lock:
            workers = list(self._workers.values())
        if state:
            workers = [w for w in workers if w.state == state]
        return workers

    def check_timeouts(self) -> list[str]:
        """Check for timed-out tasks and mark them."""
        timed_out = []
        now = time.time()
        for task in self._tasks.values():
            if task.state == TaskState.RUNNING and task.started_at:
                if now - task.started_at > task.timeout_seconds:
                    task.state = TaskState.TIMEOUT
                    task.completed_at = now
                    task.error = "Task timed out"
                    timed_out.append(task.id)
                    # Free up worker
                    if task.assigned_worker:
                        worker = self._workers.get(task.assigned_worker)
                        if worker and task.id in worker.current_tasks:
                            worker.current_tasks.remove(task.id)
                            if not worker.current_tasks:
                                worker.state = WorkerState.IDLE
        return timed_out

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_tasks": len(self._tasks),
                "pending_tasks": sum(1 for t in self._tasks.values() if t.state == TaskState.PENDING),
                "running_tasks": sum(1 for t in self._tasks.values() if t.state == TaskState.RUNNING),
                "completed_tasks": sum(1 for t in self._tasks.values() if t.state == TaskState.COMPLETED),
                "failed_tasks": sum(
                    1 for t in self._tasks.values() if t.state in (TaskState.FAILED, TaskState.TIMEOUT)
                ),
                "registered_workers": len(self._workers),
                "available_workers": sum(1 for w in self._workers.values() if w.is_available),
            }
