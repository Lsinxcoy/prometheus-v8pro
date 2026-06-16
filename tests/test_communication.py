"""Tests for communication layer: MemoryBus, EventRouter, EventBus, AgentRegistry."""

import time
import threading

import pytest

from prometheus_v8.communication.bus import Event, MemoryBus, Message, Subscription
from prometheus_v8.communication.router import EventRouter, RoutingRule
from prometheus_v8.communication.registry import AgentRegistry, AgentInfo
from prometheus_v8.events import (
    EventBus,
    EventType,
    EventLogger,
    MetricsCollector,
    ConsolidationTrigger,
)


# ── MemoryBus Tests ───────────────────────────────────────────


class TestMemoryBus:
    """Tests for MemoryBus publish/subscribe/unsubscribe and wildcard matching."""

    def test_publish_and_subscribe(self):
        bus = MemoryBus()
        received = []
        sub_id = bus.subscribe("test.topic", lambda e: received.append(e))
        bus.publish("test.topic", "test_event", {"key": "value"}, source="tester")

        assert len(received) == 1
        assert received[0].topic == "test.topic"
        assert received[0].event_type == "test_event"
        assert received[0].payload == {"key": "value"}
        assert received[0].source == "tester"

    def test_unsubscribe(self):
        bus = MemoryBus()
        received = []
        sub_id = bus.subscribe("topic", lambda e: received.append(e))
        bus.publish("topic", "t1", {})

        assert bus.unsubscribe(sub_id) is True
        bus.publish("topic", "t2", {})
        assert len(received) == 1  # only the first event

    def test_unsubscribe_nonexistent(self):
        bus = MemoryBus()
        assert bus.unsubscribe("nonexistent") is False

    def test_wildcard_topic_matching(self):
        bus = MemoryBus()
        received = []
        bus.subscribe("evolution.*", lambda e: received.append(e))

        bus.publish("evolution.mutation", "mut", {})
        bus.publish("evolution.crossover", "cross", {})
        bus.publish("safety.check", "safe", {})

        assert len(received) == 2
        assert received[0].topic == "evolution.mutation"
        assert received[1].topic == "evolution.crossover"

    def test_star_topic_matches_all(self):
        bus = MemoryBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))

        bus.publish("a", "t1", {})
        bus.publish("b", "t2", {})

        assert len(received) == 2

    def test_filter_type(self):
        bus = MemoryBus()
        received = []
        bus.subscribe("topic", lambda e: received.append(e), filter_type="important")

        bus.publish("topic", "important", {"x": 1})
        bus.publish("topic", "trivial", {"x": 2})

        assert len(received) == 1
        assert received[0].payload == {"x": 1}

    def test_dead_letters_accessible(self):
        bus = MemoryBus()
        bus.publish("orphan.topic", "type1", {})

        # dead_letters method exists and returns a list
        dead = bus.get_dead_letters()
        assert isinstance(dead, list)

    def test_event_history(self):
        bus = MemoryBus()
        bus.subscribe("t", lambda e: None)
        bus.publish("t", "a", {"i": 1})
        bus.publish("t", "b", {"i": 2})

        history = bus.get_history(topic="t")
        assert len(history) == 2

    def test_stats(self):
        bus = MemoryBus()
        bus.subscribe("t", lambda e: None)
        bus.publish("t", "x", {})

        s = bus.stats
        assert s["total_events"] == 1
        assert s["subscriptions"] == 1

    def test_callback_error_counted(self):
        bus = MemoryBus()

        def bad_cb(e):
            raise RuntimeError("boom")

        bus.subscribe("t", bad_cb)
        bus.publish("t", "x", {})
        assert bus.stats["errors"] == 1

    def test_max_subscriptions_eviction(self):
        bus = MemoryBus(max_subscriptions=2)
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        bus.subscribe("c", lambda e: None)
        assert bus.stats["subscriptions"] == 2


# ── EventRouter Tests ─────────────────────────────────────────


class TestEventRouter:
    """Tests for EventRouter: rules, priority, transform."""

    def test_route_with_rule(self):
        bus = MemoryBus()
        received = []
        bus.subscribe("output", lambda e: received.append(e))

        router = EventRouter(bus=bus)
        rule = RoutingRule(
            name="fwd",
            source_pattern="agent1",
            target_channel="output",
            priority=1,
        )
        router.add_rule(rule)

        msg = Message(sender="agent1", channel="input", event_type="task", payload={"k": "v"})
        reached = router.route(msg)
        assert reached >= 1

    def test_rule_priority_ordering(self):
        router = EventRouter()
        r_low = RoutingRule(name="low", priority=10)
        r_high = RoutingRule(name="high", priority=1)
        router.add_rule(r_low)
        router.add_rule(r_high)

        rules = router.list_rules()
        assert rules[0]["name"] == "high"
        assert rules[1]["name"] == "low"

    def test_rule_transform(self):
        bus = MemoryBus()
        received = []
        bus.subscribe("out", lambda e: received.append(e))

        router = EventRouter(bus=bus)

        def upper_payload(msg):
            msg.payload = {k: v.upper() if isinstance(v, str) else v for k, v in msg.payload.items()}
            return msg

        rule = RoutingRule(
            name="transformer",
            target_channel="out",
            transform=upper_payload,
            priority=1,
        )
        router.add_rule(rule)

        msg = Message(sender="s", event_type="t", payload={"val": "hello"})
        router.route(msg)
        assert len(received) == 1
        assert received[0].payload["val"] == "HELLO"

    def test_remove_rule(self):
        router = EventRouter()
        router.add_rule(RoutingRule(name="r1"))
        assert router.remove_rule("r1") is True
        assert router.remove_rule("nonexistent") is False

    def test_disabled_rule_skipped(self):
        router = EventRouter()
        rule = RoutingRule(name="disabled", target_channel="out", enabled=False)
        router.add_rule(rule)

        msg = Message(sender="s", event_type="t")
        reached = router.route(msg)
        assert reached == 0

    def test_router_stats(self):
        router = EventRouter()
        router.add_rule(RoutingRule(name="r", source_pattern=".*", target_channel="ch"))
        msg = Message(sender="agent", event_type="t")
        router.route(msg)

        stats = router.get_stats()
        assert stats["routed"] >= 1

    def test_register_agent_and_route(self):
        bus = MemoryBus()
        received = []
        bus.subscribe("agent_ch", lambda e: received.append(e))

        router = EventRouter(bus=bus)
        router.register_agent("worker1", ["agent_ch"])

        msg = Message(sender="ceo", event_type="task", recipient="worker1", payload={"job": "do"})
        reached = router.route(msg)
        assert reached >= 1


# ── EventBus Tests ────────────────────────────────────────────


class TestEventBus:
    """Tests for EventBus: emit, subscribe, EventLogger, MetricsCollector."""

    def test_emit_and_subscribe(self):
        eb = EventBus()
        received = []
        eb.subscribe(EventType.NODE_CREATED, lambda e: received.append(e))

        event = Event(event_type=EventType.NODE_CREATED.value, payload={"id": "n1"})
        eb.emit(event)

        assert len(received) == 1

    def test_subscribe_multiple_types(self):
        eb = EventBus()
        created = []
        deleted = []

        eb.subscribe(EventType.NODE_CREATED, lambda e: created.append(e))
        eb.subscribe(EventType.NODE_DELETED, lambda e: deleted.append(e))

        eb.emit(Event(event_type=EventType.NODE_CREATED.value))
        eb.emit(Event(event_type=EventType.NODE_DELETED.value))

        assert len(created) == 1
        assert len(deleted) == 1

    def test_unsubscribe(self):
        eb = EventBus()
        received = []

        handler = lambda e: received.append(e)  # noqa: E731
        eb.subscribe(EventType.NODE_CREATED, handler)
        eb.emit(Event(event_type=EventType.NODE_CREATED.value))
        assert len(received) == 1

        eb.unsubscribe(EventType.NODE_CREATED, handler)
        eb.emit(Event(event_type=EventType.NODE_CREATED.value))
        assert len(received) == 1

    def test_event_logger(self, tmp_path):
        log_file = tmp_path / "events.jsonl"
        logger = EventLogger(log_path=str(log_file))

        eb = EventBus()
        eb.add_handler(logger)
        eb.emit(Event(event_type=EventType.NODE_CREATED.value, payload={"x": 1}))

        assert log_file.exists()
        content = log_file.read_text()
        assert "node_created" in content

    def test_metrics_collector(self):
        mc = MetricsCollector()
        eb = EventBus()
        eb.add_handler(mc)

        eb.emit(Event(event_type=EventType.NODE_CREATED.value, correlation_id="agent1"))
        eb.emit(Event(event_type=EventType.NODE_CREATED.value, correlation_id="agent1"))
        eb.emit(Event(event_type=EventType.MUTATION_APPLIED.value, correlation_id="agent2"))

        a1 = mc.get_agent_metrics("agent1")
        assert a1.get("node_created", 0) == 2

        all_m = mc.get_all_metrics()
        assert "agent2" in all_m

    def test_consolidation_trigger(self):
        triggered = []
        trigger = ConsolidationTrigger(threshold=3, callback=lambda e: triggered.append(e))

        eb = EventBus()
        eb.add_handler(trigger)

        for _ in range(3):
            eb.emit(Event(event_type=EventType.NODE_CREATED.value))

        assert len(triggered) == 1

    def test_event_history(self):
        eb = EventBus()
        eb.emit(Event(event_type=EventType.NODE_CREATED.value))
        eb.emit(Event(event_type=EventType.NODE_DELETED.value))

        history = eb.get_history()
        assert len(history) == 2

        only_created = eb.get_history(event_type=EventType.NODE_CREATED)
        assert len(only_created) == 1

    def test_event_metrics(self):
        eb = EventBus()
        eb.emit(Event(event_type=EventType.NODE_CREATED.value))
        eb.emit(Event(event_type=EventType.NODE_CREATED.value))

        metrics = eb.get_metrics()
        assert metrics.get("node_created") == 2


# ── AgentRegistry Tests ───────────────────────────────────────


class TestAgentRegistry:
    """Tests for AgentRegistry: register, unregister, heartbeat, zombie."""

    def test_register_and_get(self):
        reg = AgentRegistry(heartbeat_timeout=9999, reap_interval=9999)
        info = reg.register("a1", name="Agent1", role="worker", capabilities=["search"])
        assert info.agent_id == "a1"
        assert info.role == "worker"

        got = reg.get("a1")
        assert got is not None
        assert got.name == "Agent1"

    def test_unregister(self):
        reg = AgentRegistry(heartbeat_timeout=9999, reap_interval=9999)
        reg.register("a1")
        assert reg.unregister("a1") is True
        assert reg.get("a1") is None
        assert reg.unregister("a1") is False

    def test_heartbeat(self):
        reg = AgentRegistry(heartbeat_timeout=9999, reap_interval=9999)
        reg.register("a1")
        assert reg.heartbeat("a1", status="busy") is True
        info = reg.get("a1")
        assert info.status == "busy"

        assert reg.heartbeat("nonexistent") is False

    def test_list_and_find(self):
        reg = AgentRegistry(heartbeat_timeout=9999, reap_interval=9999)
        reg.register("a1", role="worker", capabilities=["search"])
        reg.register("a2", role="ceo", capabilities=["strategic"])
        reg.register("a3", role="worker", capabilities=["search", "code"])

        workers = reg.list_agents(role="worker")
        assert len(workers) == 2

        searchers = reg.find_by_capability("search")
        assert len(searchers) == 2

    def test_zombie_detection(self):
        reg = AgentRegistry(heartbeat_timeout=0.01, reap_interval=9999)
        reg.register("z1")
        info = reg.get("z1")
        info.last_heartbeat = time.time() - 100

        reg._reap_zombies()
        assert reg.get("z1").status == "dead"

    def test_stats(self):
        reg = AgentRegistry(heartbeat_timeout=9999, reap_interval=9999)
        reg.register("a1", role="worker")
        reg.register("a2", role="ceo")

        stats = reg.get_stats()
        assert stats["total"] == 2
        assert stats["active"] == 2

    def test_on_agent_dead_callback(self):
        reg = AgentRegistry(heartbeat_timeout=0.01, reap_interval=9999)
        dead_agents = []
        reg.on_agent_dead(lambda aid: dead_agents.append(aid))

        reg.register("z1")
        info = reg.get("z1")
        info.last_heartbeat = time.time() - 100
        reg._reap_zombies()

        assert "z1" in dead_agents
