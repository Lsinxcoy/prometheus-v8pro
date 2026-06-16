"""Prometheus V8 Configuration System.

16 sub-configs with YAML/ENV/dotenv loading, hot reload, singleton pattern.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class StoreConfig:
    db_path: str = "data/prometheus_v8.db"
    wal_mode: bool = True
    cache_size_mb: int = 64
    mmap_size_mb: int = 512
    fts_tokenizer: str = "unicode61"
    pool_size: int = 5


@dataclass
class VectorConfig:
    backend: str = "numpy"  # numpy/hnsw/sqlitevec/lancedb
    dimension: int = 384
    model_name: str = "all-MiniLM-L6-v2"
    cache_size: int = 10000
    cache_ttl_seconds: int = 300
    compression: str = "none"  # none/mib/int8


@dataclass
class GraphConfig:
    backend: str = "networkx"  # networkx/kuzu/falkordb
    community_algorithm: str = "leiden"
    max_depth: int = 5


@dataclass
class SearchConfig:
    rrf_k: int = 60
    mmr_lambda: float = 0.5
    max_results: int = 20
    enable_synonyms: bool = True
    cascade: bool = True


@dataclass
class EvolutionConfig:
    max_generations: int = 100
    population_size: int = 20
    elite_ratio: float = 0.1
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    direction: str = "auto"  # forward/lateral/reverse/auto
    max_stagnation: int = 20
    fitness_threshold: float = 0.95
    enable_parallel: bool = True
    island_count: int = 4
    migration_interval: int = 10


@dataclass
class OrganConfig:
    llm_provider: str = "openrouter"
    llm_model: str = "qwen/qwen3-235b-a22b:free"
    llm_temperature: float = 0.7
    sandbox_timeout: int = 10
    enable_paper_search: bool = False
    enable_github_search: bool = False


@dataclass
class LifecycleConfig:
    consolidation_threshold: int = 100
    dream_interval_hours: int = 6
    decay_rate: float = 0.95
    daily_learning_quota: int = 20
    revision_interval: int = 5
    min_importance_keep: float = 0.3


@dataclass
class SafetyConfig:
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    safe_harbor_threshold: float = 0.7
    enable_plan_validation: bool = True
    dynamic_security: bool = True


@dataclass
class MonitorConfig:
    heartbeat_interval: int = 30
    anomaly_threshold: float = 2.0
    enable_kpi: bool = True
    enable_prediction: bool = False


@dataclass
class CommunicationConfig:
    backend: str = "memory"  # memory/redis
    redis_url: str = "redis://localhost:6379"
    consumer_group: str = "prometheus_v8"
    max_history: int = 10000


@dataclass
class GovernanceConfig:
    autonomy_level: int = 1  # 0-4
    initiative_enabled: bool = True
    initiative_max_per_day: int = 3
    initiative_max_steps: int = 15
    initiative_max_minutes: int = 10
    curiosity_queue_size: int = 50


@dataclass
class LLMConfig:
    api_base: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    timeout: int = 60
    max_retries: int = 3
    fallback_model: str = ""

    @property
    def api_key_masked(self) -> str:
        """Return masked API key for display."""
        if not self.api_key or len(self.api_key) < 8:
            return "***" if self.api_key else ""
        return self.api_key[:4] + "****" + self.api_key[-4:]


@dataclass
class EmbeddingConfig:
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    file: str = ""
    max_size_mb: int = 100


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8082
    enable: bool = True


@dataclass
class TrustConfig:
    """Knowledge trust system — from knowledge-conversion solution."""

    default_level: str = "pending"
    high_signal_sources: int = 2
    verified_usage_count: int = 1
    stale_days: int = 30
    stale_utility_threshold: int = 2
    action_hook_required: bool = True


@dataclass
class PrometheusConfig:
    """Master configuration for Prometheus V8."""

    store: StoreConfig = field(default_factory=StoreConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    organ: OrganConfig = field(default_factory=OrganConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)

    data_dir: str = "data"
    debug: bool = False


class ConfigManager:
    """Singleton config manager with YAML/ENV loading and hot reload."""

    _instance: Optional[ConfigManager] = None
    _config: Optional[PrometheusConfig] = None
    _lock = threading.Lock()  # Class-level lock for thread safety

    def __new__(cls) -> ConfigManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get(cls) -> PrometheusConfig:
        with cls._lock:
            if cls._config is None:
                cls._config = cls.load()
            return cls._config

    @classmethod
    def load(cls, config_path: str | None = None) -> PrometheusConfig:
        """Load config from YAML file + environment variables."""
        cfg = PrometheusConfig()

        # Load YAML
        if config_path is None:
            config_path = os.environ.get("PROMETHEUS_CONFIG", "config/prometheus.yaml")
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            cls._apply_dict(cfg, data)

        # Override from environment variables: PROMETHEUS_<SECTION>_<FIELD>
        cls._apply_env(cfg)

        # Override from PROMETHEUS_CONFIG_JSON env var
        json_str = os.environ.get("PROMETHEUS_CONFIG_JSON", "")
        if json_str:
            try:
                cls._apply_dict(cfg, json.loads(json_str))
            except json.JSONDecodeError:
                pass

        cls._config = cfg
        return cfg

    @classmethod
    def reload(cls) -> PrometheusConfig:
        """Reload configuration (for hot reload)."""
        with cls._lock:
            old_config = cls._config
            cls._config = None
            try:
                cls._config = cls.load()
            except Exception as e:
                logger.warning(f"Config reload failed, restoring previous: {e}")
                cls._config = old_config  # Restore on failure
                raise
            return cls._config

    @staticmethod
    def _apply_dict(cfg: PrometheusConfig, data: dict[str, Any]) -> None:
        """Recursively apply dict values to config dataclass."""
        for key, value in data.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                    for k2, v2 in value.items():
                        if hasattr(attr, k2):
                            setattr(attr, k2, v2)
                else:
                    setattr(cfg, key, value)

    @staticmethod
    def _apply_env(cfg: PrometheusConfig) -> None:
        """Apply PROMETHEUS_<SECTION>_<FIELD> environment variables."""
        prefix = "PROMETHEUS_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix) :].lower().split("_", 1)
            if len(parts) == 2:
                section_name, field_name = parts
                if hasattr(cfg, section_name):
                    section = getattr(cfg, section_name)
                    if hasattr(section, field_name):
                        current = getattr(section, field_name)
                        # Type coercion
                        if isinstance(current, bool):
                            setattr(section, field_name, value.lower() in ("true", "1", "yes"))
                        elif isinstance(current, int):
                            setattr(section, field_name, int(value))
                        elif isinstance(current, float):
                            setattr(section, field_name, float(value))
                        else:
                            setattr(section, field_name, value)
            elif len(parts) == 1:
                field_name = parts[0]
                if hasattr(cfg, field_name):
                    setattr(cfg, field_name, value)

    @classmethod
    def export_safe(cls) -> dict[str, Any]:
        """Export config as dict with masked API key for safe logging/display."""
        cfg = cls.get()
        data = {}
        for section_name in ("store", "vector", "graph", "search", "evolution", "organ", "lifecycle", "safety", "monitor", "communication", "governance", "embedding", "logging", "dashboard", "trust"):
            section = getattr(cfg, section_name, None)
            if section is not None:
                data[section_name] = {k: v for k, v in section.__dict__.items() if not k.startswith("_")}
        # LLM section with masked key
        data["llm"] = {k: v for k, v in cfg.llm.__dict__.items() if not k.startswith("_")}
        data["llm"]["api_key"] = cfg.llm.api_key_masked
        data["data_dir"] = cfg.data_dir
        data["debug"] = cfg.debug
        return data


def get_config() -> PrometheusConfig:
    """Get the global configuration singleton."""
    return ConfigManager.get()
