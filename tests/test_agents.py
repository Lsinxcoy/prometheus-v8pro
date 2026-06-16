"""Tests for agents/descriptor: AgentDescriptor, AgentPool."""

import time

import pytest

from prometheus_v8.agents.descriptor import (
    AgentDescriptor,
    AgentPool,
    AgentRole,
    AgentState,
)


class TestAgentDescriptor:
    """Tests for AgentDescriptor creation and matching."""

    def test_creation_defaults(self):
        desc = AgentDescriptor(id="a1", name="Worker1")
        assert desc.id == "a1"
        assert desc.name == "Worker1"
        assert desc.role == AgentRole.WORKER
        assert desc.state == AgentState.IDLE
        assert desc.model_tier == "standard"

    def test_creation_with_role(self):
        desc = AgentDescriptor(id="ceo1", name="CEO", role=AgentRole.CEO, model_tier="pro")
        assert desc.role == AgentRole.CEO
        assert desc.model_tier == "pro"

    def test_is_available(self):
        desc = AgentDescriptor(id="a1", name="W1")
        assert desc.is_available is True

        desc.state = AgentState.BUSY
        assert desc.is_available is False

    def test_is_available_max_concurrent(self):
        desc = AgentDescriptor(id="a1", name="W1", max_concurrent_tasks=2)
        desc.current_tasks = ["t1", "t2"]
        assert desc.is_available is False

    def test_can_handle(self):
        desc = AgentDescriptor(id="a1", capabilities=["search", "code"])
        assert desc.can_handle("search") is True
        assert desc.can_handle("write") is False

    def test_can_handle_wildcard(self):
        desc = AgentDescriptor(id="a1", capabilities=["*"])
        assert desc.can_handle("anything") is True

    def test_success_rate(self):
        desc = AgentDescriptor(id="a1", completed_tasks=8, failed_tasks=2)
        assert desc.success_rate == 0.8

    def test_success_rate_zero(self):
        desc = AgentDescriptor(id="a1", completed_tasks=0, failed_tasks=0)
        assert desc.success_rate == 0.0

    def test_to_dict(self):
        desc = AgentDescriptor(id="a1", name="Test", role=AgentRole.WORKER)
        d = desc.to_dict()
        assert d["id"] == "a1"
        assert d["role"] == "worker"
        assert "is_available" in d


class TestAgentPool:
    """Tests for AgentPool: register, find, health."""

    def test_register_and_find(self):
        pool = AgentPool()
        agent = AgentDescriptor(id="a1", name="Worker1", capabilities=["search"])
        pool.register(agent)

        found = pool.find_available(required_capability="search")
        assert len(found) == 1
        assert found[0].name == "Worker1"

    def test_deregister(self):
        pool = AgentPool()
        agent = AgentDescriptor(id="a1", name="W1")
        pool.register(agent)
        assert pool.deregister("a1") is True
        assert pool.deregister("a1") is False

    def test_find_best_by_tier(self):
        pool = AgentPool()
        pool.register(AgentDescriptor(id="w1", name="W1", model_tier="standard", capabilities=["search"]))
        pool.register(AgentDescriptor(id="p1", name="P1", model_tier="pro", capabilities=["search"]))

        best = pool.find_best(required_capability="search", model_tier="pro")
        assert best is not None
        assert best.model_tier == "pro"

    def test_heartbeat(self):
        pool = AgentPool()
        agent = AgentDescriptor(id="a1", name="W1", state=AgentState.OFFLINE)
        pool.register(agent)
        pool.heartbeat("a1")
        assert pool.get_all()[0].state == AgentState.IDLE

    def test_assign_and_complete_task(self):
        pool = AgentPool()
        agent = AgentDescriptor(id="a1", name="W1", max_concurrent_tasks=3)
        pool.register(agent)

        assert pool.assign_task("a1", "t1") is True
        assert pool.get_all()[0].state == AgentState.BUSY

        pool.complete_task("a1", "t1", success=True)
        assert pool.get_all()[0].state == AgentState.IDLE
        assert pool.get_all()[0].completed_tasks == 1

    def test_check_health_dead_agent(self):
        pool = AgentPool(heartbeat_timeout=0.01, dead_threshold=0.02)
        agent = AgentDescriptor(id="a1", name="W1")
        pool.register(agent)

        # Make heartbeat very old
        pool.get_all()[0].last_heartbeat = time.time() - 1000
        health = pool.check_health()
        assert health["dead"] == 1

    def test_event_callback(self):
        events = []
        pool = AgentPool()
        pool.add_event_callback(lambda etype, agent: events.append((etype, agent.id)))

        agent = AgentDescriptor(id="a1", name="W1")
        pool.register(agent)
        assert ("registered", "a1") in events

    def test_stats(self):
        pool = AgentPool()
        pool.register(AgentDescriptor(id="a1", name="W1", role=AgentRole.WORKER))
        pool.register(AgentDescriptor(id="c1", name="C1", role=AgentRole.CEO))

        stats = pool.stats
        assert stats["total_agents"] == 2
