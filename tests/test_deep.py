"""Deep tests: edge cases, concurrency, and negative tests."""
import pytest
import threading
import time

# 1. Schema edge cases
class TestSchemaEdgeCases:
    def test_empty_content_node(self):
        from prometheus_v8.schema import create_fact_node
        node = create_fact_node(content="", importance=0.0)
        assert node.payload.content == ""

    def test_unicode_content(self):
        from prometheus_v8.schema import create_fact_node
        node = create_fact_node(content="中文测试 🎉 ñ é ü", importance=0.5)
        assert "中文" in node.payload.content

    def test_very_long_content(self):
        from prometheus_v8.schema import create_fact_node
        node = create_fact_node(content="x" * 10000, importance=0.5)
        assert len(node.payload.content) == 10000

    def test_extreme_importance(self):
        from prometheus_v8.schema import create_fact_node
        node = create_fact_node(content="test", importance=-1.0)
        assert node.importance == -1.0
        node2 = create_fact_node(content="test", importance=999.0)
        assert node2.importance == 999.0

# 2. Concurrent Store access
class TestConcurrentAccess:
    def test_concurrent_store_writes(self):
        import tempfile, os
        from prometheus_v8.core.store import SQLiteStore
        from prometheus_v8.schema import create_fact_node

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = SQLiteStore(db_path)
        errors = []

        def write_node(i):
            try:
                node = create_fact_node(content=f"concurrent_{i}", importance=0.5)
                store.add_node(node)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_node, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        store.close()
        # Windows: file may still be locked briefly after close, retry unlink
        for attempt in range(5):
            try:
                os.unlink(db_path)
                break
            except PermissionError:
                time.sleep(0.1)
        assert len(errors) == 0

    def test_concurrent_kpi_counter(self):
        from prometheus_v8.monitor.kpi import KPICollector
        kpi = KPICollector()
        errors = []

        def increment():
            try:
                for _ in range(100):
                    kpi.increment("test_counter")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

# 3. Weibull edge cases
class TestWeibullEdgeCases:
    def test_zero_age(self):
        from prometheus_v8.lifecycle.weibull import WeibullRetentionCalculator
        wr = WeibullRetentionCalculator()
        r = wr.compute(age_days=0.0, importance=0.5, lam=1.0, k=0.5)
        assert r.composite >= 0.5  # Fresh memory should have meaningful retention

    def test_very_old_age(self):
        from prometheus_v8.lifecycle.weibull import WeibullRetentionCalculator
        wr = WeibullRetentionCalculator()
        r = wr.compute(age_days=10000.0, importance=0.1, lam=1.0, k=0.5)
        assert r.composite < 0.5  # Very old, low importance should decay

    def test_high_importance_preserves(self):
        from prometheus_v8.lifecycle.weibull import WeibullRetentionCalculator
        wr = WeibullRetentionCalculator()
        # High importance with slow-decay parameters (large lambda)
        r = wr.compute(age_days=100.0, importance=0.9, lam=100.0, k=0.5)
        assert r.composite > 0.3  # High importance + slow decay should preserve memory

# 4. Safety negative tests
class TestSafetyNegative:
    def test_injection_attempts(self):
        from prometheus_v8.safety.manager import SafetyManager
        sm = SafetyManager()
        dangerous = [
            "exec('import os; os.system(\"rm -rf /\")')",
            "__import__('subprocess').call(['rm', '-rf', '/'])",
            "eval(open('/etc/passwd').read())",
        ]
        for code in dangerous:
            v = sm.check(code)
            assert not v.allowed, f"Dangerous code not blocked: {code}"

    def test_circuit_breaker_trip(self):
        from prometheus_v8.safety.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(threshold=3, timeout=1.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.state.value == "open"
        assert not cb.can_execute()

    def test_circuit_breaker_recovery(self):
        from prometheus_v8.safety.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(threshold=3, timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.state.value == "open"
        time.sleep(0.2)
        assert cb.can_execute()  # Should allow in half-open after timeout
        cb.record_success()
        assert cb.state.value == "closed"

# 5. Compression roundtrip
class TestCompressionRoundtrip:
    def test_int8_roundtrip_various_dims(self):
        from prometheus_v8.core.compression import VectorCompressor
        import numpy as np
        for dim in [8, 64, 128, 384]:
            vc = VectorCompressor(method="int8")
            vec = np.random.randn(dim).astype(np.float32)
            compressed, meta = vc.compress(vec)
            decompressed = vc.decompress(compressed, meta)
            decompressed = np.array(decompressed, dtype=np.float32)
            # INT8 quantization has error but should be roughly correct
            cos_sim = np.dot(vec, decompressed) / (np.linalg.norm(vec) * np.linalg.norm(decompressed) + 1e-8)
            assert cos_sim > 0.8, f"Roundtrip too lossy for dim={dim}: cos_sim={cos_sim}"
