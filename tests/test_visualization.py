"""Tests for visualization: DashboardMetric, DashboardPanel, DashboardProvider."""

import time

import pytest

from prometheus_v8.visualization.dashboard import (
    DashboardMetric,
    DashboardPanel,
    DashboardProvider,
)


class TestDashboardMetric:
    """Tests for DashboardMetric data structure."""

    def test_creation_defaults(self):
        m = DashboardMetric()
        assert m.name == ""
        assert m.value == 0.0
        assert m.trend == "stable"
        assert m.status == "normal"

    def test_creation_with_values(self):
        m = DashboardMetric(
            name="Total Nodes",
            value=42,
            unit="nodes",
            trend="increasing",
            change_pct=10.5,
            sparkline=[1.0, 2.0, 3.0],
            status="normal",
        )
        assert m.name == "Total Nodes"
        assert m.value == 42
        assert m.unit == "nodes"
        assert m.trend == "increasing"
        assert m.change_pct == 10.5
        assert len(m.sparkline) == 3
        assert m.status == "normal"

    def test_status_values(self):
        for status in ("normal", "warning", "critical"):
            m = DashboardMetric(status=status)
            assert m.status == status

    def test_trend_values(self):
        for trend in ("stable", "increasing", "decreasing"):
            m = DashboardMetric(trend=trend)
            assert m.trend == trend


class TestDashboardPanel:
    """Tests for DashboardPanel data structure."""

    def test_creation_defaults(self):
        p = DashboardPanel()
        assert p.title == ""
        assert p.panel_type == "metrics"
        assert p.data is None

    def test_creation_with_values(self):
        metrics = [DashboardMetric(name="M1"), DashboardMetric(name="M2")]
        p = DashboardPanel(
            title="System Overview",
            panel_type="metrics",
            data=metrics,
        )
        assert p.title == "System Overview"
        assert len(p.data) == 2

    def test_panel_types(self):
        for pt in ("metrics", "chart", "table", "gauge"):
            p = DashboardPanel(panel_type=pt)
            assert p.panel_type == pt

    def test_updated_at_timestamp(self):
        before = time.time()
        p = DashboardPanel()
        after = time.time()
        assert before <= p.updated_at <= after


class TestDashboardProvider:
    """Tests for DashboardProvider: overview, health, metrics."""

    def test_overview_without_components(self):
        dp = DashboardProvider()
        overview = dp.get_overview()
        assert overview["version"] == "8.0.0"
        assert overview["nodes"]["total"] == 0
        assert overview["evolution"]["generation"] == 0

    def test_system_health_without_components(self):
        dp = DashboardProvider()
        health = dp.get_system_health()
        assert health["status"] in ("healthy", "degraded", "critical")
        assert "score" in health

    def test_safety_dashboard_without_safety(self):
        dp = DashboardProvider()
        safety = dp.get_safety_dashboard()
        assert "checks" in safety
        assert safety["checks"] == 0

    def test_evolution_progress_without_engine(self):
        dp = DashboardProvider()
        progress = dp.get_evolution_progress()
        assert progress["generation"] == 0
        assert progress["best_fitness"] == 0

    def test_memory_distribution_without_store(self):
        dp = DashboardProvider()
        distribution = dp.get_memory_distribution()
        assert "by_layer" in distribution
        assert "by_type" in distribution
        assert "by_trust" in distribution
