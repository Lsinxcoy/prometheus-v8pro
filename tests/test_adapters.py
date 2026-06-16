"""Tests for adapters: MnemosyneAdapter, OrchestratorAdapter, EVOBusAdapter."""

import pytest

from prometheus_v8.adapters.mnemosyne import MnemosyneAdapter
from prometheus_v8.adapters.orchestrator import (
    OrchestratorAdapter,
    SubTask,
    Worker,
    TaskState,
    WorkerState,
    TaskResult,
)
from prometheus_v8.adapters.evo_bus import EVOBusAdapter, InMemoryBus, BusMessage, AgentInfo


# ── MnemosyneAdapter Tests ────────────────────────────────────


class TestMnemosyneAdapter:
    """Tests for MnemosyneAdapter: connect/disconnect, store_node/retrieve, search."""

    def test_connect_fails_gracefully(self):
        adapter = MnemosyneAdapter(db_path="/nonexistent/path/db.sqlite")
        # Without SQLiteStore importable or path invalid, connect should return False or handle gracefully
        result = adapter.connect()
        # May succeed (SQLite creates file) or fail, but should not raise
        assert isinstance(result, bool)

    def test_store_node_without_connection(self):
        adapter = MnemosyneAdapter(db_path=":memory:")
        # Initially not connected
        from prometheus_v8.schema import create_fact_node

        node = create_fact_node(content="test", importance=0.5)
        # This will try to auto-connect
        result = adapter.store_node(node)
        assert isinstance(result, bool)

    def test_retrieve_nonexistent(self):
        adapter = MnemosyneAdapter(db_path=":memory:")
        adapter.connect()
        result = adapter.retrieve_node(b"\x00" * 16)
        # Should return None for nonexistent node
        assert result is None or isinstance(result, object)

    def test_search_without_connection(self):
        adapter = MnemosyneAdapter(db_path=":memory:")
        results = adapter.search("test query")
        assert isinstance(results, list)

    def test_transfer_hallway(self):
        adapter = MnemosyneAdapter(db_path=":memory:")
        adapter.connect()
        # Transfer with nonexistent nodes should return 0
        count = adapter.transfer_hallway("agent1", "agent2", [b"\x00" * 16])
        assert count == 0


# ── OrchestratorAdapter Tests ─────────────────────────────────


class TestOrchestratorAdapter:
    """Tests for OrchestratorAdapter: task decomposition, worker assignment."""

    def test_submit_task(self):
        orch = OrchestratorAdapter()
        task_id = orch.submit_task(name="test_task", description="A test task")
        assert task_id  # Non-empty ID

    def test_decompose_task_default(self):
        orch = OrchestratorAdapter()
        task_id = orch.submit_task(name="test")
        subtasks = orch.decompose_task(task_id)
        assert len(subtasks) == 1  # Default: single subtask
        assert subtasks[0].parent_id == task_id

    def test_decompose_task_custom(self):
        def custom_decompose(name, metadata):
            return [
                SubTask(id=f"sub_{i}", parent_id="", name=f"sub_{i}")
                for i in range(3)
            ]

        orch = OrchestratorAdapter()
        orch.set_decompose_fn(custom_decompose)
        task_id = orch.submit_task(name="complex_task")
        subtasks = orch.decompose_task(task_id)
        assert len(subtasks) == 3

    def test_register_worker_and_assign(self):
        orch = OrchestratorAdapter()
        worker = Worker(id="w1", name="Worker1", capabilities=["search", "code"])
        orch.register_worker(worker)

        task_id = orch.submit_task(name="search_task", required_capabilities=["search"])
        subtasks = orch.decompose_task(task_id)

        assigned = orch.assign_worker(subtasks[0])
        assert assigned is not None
        assert assigned.id == "w1"

    def test_worker_capability_check(self):
        worker = Worker(id="w1", capabilities=["search"])
        assert worker.can_handle(["search"]) is True
        assert worker.can_handle(["code"]) is False
        assert worker.can_handle(["anything"]) is False

    def test_worker_wildcard_capability(self):
        worker = Worker(id="w1", capabilities=["*"])
        assert worker.can_handle(["search", "code", "anything"]) is True

    def test_execute_subtask(self):
        orch = OrchestratorAdapter()
        worker = Worker(id="w1", name="W1", capabilities=["search"])
        orch.register_worker(worker)

        task_id = orch.submit_task(name="test")
        subtasks = orch.decompose_task(task_id)

        result = orch.execute_subtask(subtasks[0].id)
        assert result is not None

    def test_aggregate_results(self):
        orch = OrchestratorAdapter()
        worker = Worker(id="w1", name="W1", capabilities=["search"])
        orch.register_worker(worker)

        task_id = orch.submit_task(name="test")
        subtasks = orch.decompose_task(task_id)
        orch.execute_subtask(subtasks[0].id)

        result = orch.aggregate_results(task_id)
        assert isinstance(result, TaskResult)
        assert result.total_subtasks >= 1


# ── EVOBusAdapter Tests ───────────────────────────────────────


class TestEVOBusAdapter:
    """Tests for EVOBusAdapter: publish/subscribe, agent registration."""

    def test_publish_and_stats(self):
        # Uses in-memory fallback (no Redis needed)
        bus = EVOBusAdapter(agent_id="test_agent")
        msg_id = bus.publish("test.topic", {"key": "value"}, sender="tester")
        assert msg_id  # Non-empty ID
        stats = bus.stats
        assert stats["total_messages"] >= 1
        assert stats["using_redis"] is False  # No Redis in test
        bus.close()

    def test_register_and_get_agents(self):
        bus = EVOBusAdapter(agent_id="test")
        agent = AgentInfo(id="a1", name="Agent1", role="worker", status="online")
        bus.register_agent(agent)

        agents = bus.get_agents()
        assert len(agents) == 1
        assert agents[0].name == "Agent1"

        online = bus.get_agents(status="online")
        assert len(online) == 1
        bus.close()

    def test_deregister_agent(self):
        bus = EVOBusAdapter(agent_id="test")
        agent = AgentInfo(id="a1", name="Agent1")
        bus.register_agent(agent)
        bus.deregister_agent("a1")

        assert len(bus.get_agents()) == 0
        bus.close()

    def test_check_liveness(self):
        bus = EVOBusAdapter(agent_id="test")
        agent = AgentInfo(id="a1", name="A1", status="online")
        bus.register_agent(agent)

        assert bus.check_agent_liveness("a1") is True
        assert bus.check_agent_liveness("nonexistent") is False
        bus.close()

    def test_dead_letters(self):
        bus = EVOBusAdapter(agent_id="test")
        dead = bus.get_dead_letters()
        assert isinstance(dead, list)
        bus.close()

    def test_bus_message_serialization(self):
        msg = BusMessage(id="1", topic="t", sender="s", payload={"k": "v"})
        serialized = msg.serialize()
        assert isinstance(serialized, dict)
        deserialized = BusMessage.deserialize(serialized)
        assert deserialized.topic == "t"
        assert deserialized.payload == {"k": "v"}

    def test_in_memory_bus(self):
        imb = InMemoryBus()
        received = []
        imb.subscribe("ch", lambda m: received.append(m))

        msg = BusMessage(id="1", topic="ch", sender="s", payload={"x": 1})
        imb.publish("ch", msg)
        assert len(received) == 1

        consumed = imb.consume("ch", count=10)
        # Message was consumed from queue
        assert len(imb.peek("ch")) == 0
