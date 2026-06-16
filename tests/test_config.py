"""Tests for config system: ConfigManager, 16 sub-configs, env override, hot reload."""

import json
import os

import pytest

from prometheus_v8.config import (
    ConfigManager,
    PrometheusConfig,
    StoreConfig,
    VectorConfig,
    GraphConfig,
    SearchConfig,
    EvolutionConfig,
    OrganConfig,
    LifecycleConfig,
    SafetyConfig,
    MonitorConfig,
    CommunicationConfig,
    GovernanceConfig,
    LLMConfig,
    EmbeddingConfig,
    LoggingConfig,
    DashboardConfig,
    TrustConfig,
)


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset ConfigManager singleton between tests."""
    ConfigManager._config = None
    ConfigManager._instance = None
    yield
    ConfigManager._config = None
    ConfigManager._instance = None


class TestConfigManager:
    """Tests for ConfigManager: load, reload, env override."""

    def test_default_load(self):
        cfg = ConfigManager.load(config_path="/nonexistent/config.yaml")
        assert isinstance(cfg, PrometheusConfig)
        assert cfg.store.db_path == "data/prometheus_v8.db"
        assert cfg.evolution.population_size == 20

    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "prometheus.yaml"
        config_file.write_text(
            "store:\n  db_path: custom/path.db\n  cache_size_mb: 128\nevolution:\n  population_size: 50\n"
        )
        cfg = ConfigManager.load(config_path=str(config_file))
        assert cfg.store.db_path == "custom/path.db"
        assert cfg.store.cache_size_mb == 128
        assert cfg.evolution.population_size == 50

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_STORE_DB_PATH", "env_override.db")
        monkeypatch.setenv("PROMETHEUS_EVOLUTION_POPULATION_SIZE", "30")

        cfg = ConfigManager.load(config_path="/nonexistent/config.yaml")
        assert cfg.store.db_path == "env_override.db"
        assert cfg.evolution.population_size == 30

    def test_env_bool_override(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_STORE_WAL_MODE", "false")
        cfg = ConfigManager.load(config_path="/nonexistent/config.yaml")
        assert cfg.store.wal_mode is False

    def test_env_float_override(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_EVOLUTION_MUTATION_RATE", "0.5")
        cfg = ConfigManager.load(config_path="/nonexistent/config.yaml")
        assert cfg.evolution.mutation_rate == 0.5

    def test_json_env_override(self, monkeypatch):
        monkeypatch.setenv(
            "PROMETHEUS_CONFIG_JSON",
            json.dumps({"evolution": {"population_size": 99}}),
        )
        cfg = ConfigManager.load(config_path="/nonexistent/config.yaml")
        assert cfg.evolution.population_size == 99

    def test_hot_reload(self):
        cfg1 = ConfigManager.get()
        ConfigManager._config = None
        cfg2 = ConfigManager.reload()
        assert isinstance(cfg2, PrometheusConfig)

    def test_get_returns_same_instance(self):
        cfg1 = ConfigManager.get()
        cfg2 = ConfigManager.get()
        assert cfg1 is cfg2


class TestSubConfigs:
    """Tests for all 16 sub-config instantiations."""

    def test_store_config(self):
        c = StoreConfig()
        assert c.db_path == "data/prometheus_v8.db"
        assert c.wal_mode is True

    def test_vector_config(self):
        c = VectorConfig()
        assert c.backend == "numpy"
        assert c.dimension == 384

    def test_graph_config(self):
        c = GraphConfig()
        assert c.backend == "networkx"

    def test_search_config(self):
        c = SearchConfig()
        assert c.rrf_k == 60

    def test_evolution_config(self):
        c = EvolutionConfig()
        assert c.max_generations == 100
        assert c.crossover_rate == 0.7

    def test_organ_config(self):
        c = OrganConfig()
        assert c.llm_provider == "openrouter"

    def test_lifecycle_config(self):
        c = LifecycleConfig()
        assert c.consolidation_threshold == 100

    def test_safety_config(self):
        c = SafetyConfig()
        assert c.circuit_breaker_threshold == 5

    def test_monitor_config(self):
        c = MonitorConfig()
        assert c.heartbeat_interval == 30

    def test_communication_config(self):
        c = CommunicationConfig()
        assert c.backend == "memory"

    def test_governance_config(self):
        c = GovernanceConfig()
        assert c.autonomy_level == 1

    def test_llm_config(self):
        c = LLMConfig()
        assert "openrouter" in c.api_base

    def test_embedding_config(self):
        c = EmbeddingConfig()
        assert c.model == "all-MiniLM-L6-v2"

    def test_logging_config(self):
        c = LoggingConfig()
        assert c.level == "INFO"

    def test_dashboard_config(self):
        c = DashboardConfig()
        assert c.port == 8082

    def test_trust_config(self):
        c = TrustConfig()
        assert c.default_level == "pending"


class TestPrometheusConfig:
    """Tests for the master config dataclass."""

    def test_all_sub_configs_present(self):
        cfg = PrometheusConfig()
        sub_configs = [
            cfg.store, cfg.vector, cfg.graph, cfg.search,
            cfg.evolution, cfg.organ, cfg.lifecycle, cfg.safety,
            cfg.monitor, cfg.communication, cfg.governance, cfg.llm,
            cfg.embedding, cfg.logging, cfg.dashboard, cfg.trust,
        ]
        assert len(sub_configs) == 16

    def test_top_level_fields(self):
        cfg = PrometheusConfig()
        assert cfg.data_dir == "data"
        assert cfg.debug is False
