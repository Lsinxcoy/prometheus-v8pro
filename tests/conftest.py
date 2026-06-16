"""Test configuration and fixtures."""

import os
import tempfile
import pytest
from prometheus_v8.core.store import SQLiteStore


@pytest.fixture
def tmp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = SQLiteStore(db_path)
    yield store
    store.close()
    os.unlink(db_path)


@pytest.fixture
def sample_node():
    """Create a sample node for testing."""
    from prometheus_v8.schema import create_fact_node

    return create_fact_node(content="Test fact node", importance=0.7, tags=["test"])
