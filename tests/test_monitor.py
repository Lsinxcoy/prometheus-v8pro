"""Tests for monitor modules."""

import pytest


class TestKPICollector:
    def test_counter(self):
        from prometheus_v8.monitor.kpi import KPICollector

        kpi = KPICollector()
        kpi.record("test_metric", 1.0)
        kpi.record("test_metric", 2.0)
        stats = kpi.compute_stats("test_metric")
        assert stats.count == 2

    def test_export(self):
        from prometheus_v8.monitor.kpi import KPICollector

        kpi = KPICollector()
        kpi.record("m1", 1.0)
        exported = kpi.export()
        assert isinstance(exported, dict)
        assert "m1" in exported


class TestAnomalyDetector:
    def test_observe(self):
        from prometheus_v8.monitor.anomaly import AnomalyDetector

        ad = AnomalyDetector(z_threshold=2.0)
        for i in range(20):
            ad.observe("test", 1.0)
        anomalies = ad.get_anomalies()
        assert isinstance(anomalies, list)

    def test_set_threshold(self):
        from prometheus_v8.monitor.anomaly import AnomalyDetector

        ad = AnomalyDetector()
        ad.set_threshold("custom_metric", 3.0)
        assert "custom_metric" in ad._metric_thresholds


class TestMonitorManager:
    def test_record_heartbeat(self):
        from prometheus_v8.monitor.manager import MonitorManager

        mm = MonitorManager()
        mm.record_heartbeat("agent1", "healthy")
        assert mm.check("agent1") == "healthy"

    def test_record_kpi(self):
        from prometheus_v8.monitor.manager import MonitorManager

        mm = MonitorManager()
        mm.record_kpi("test_kpi", 42.0)
        data = mm.get_dashboard_data()
        assert isinstance(data, dict)

    def test_check_unknown_agent(self):
        from prometheus_v8.monitor.manager import MonitorManager

        mm = MonitorManager()
        assert mm.check("nonexistent") in ("unknown", "degraded")
