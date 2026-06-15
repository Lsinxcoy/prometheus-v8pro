"""Persistence Manager - JSON/pickle state persistence."""
from __future__ import annotations
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class PersistenceManager:
    """Manages persistent state storage with JSON and pickle backends."""
    
    def __init__(self, data_dir: str = "data/state") -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, key: str, data: Any) -> bool:
        try:
            path = self._data_dir / f"{key}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, default=str, indent=2))
            return True
        except Exception as e:
            logger.warning(f"JSON save error for {key}: {e}")
            return False
    
    def load_json(self, key: str, default: Any = None) -> Any:
        path = self._data_dir / f"{key}.json"
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"JSON load error for {key}: {e}")
            return default
    
    def save_pickle(self, key: str, data: Any) -> bool:
        try:
            path = self._data_dir / f"{key}.pkl"
            with open(path, "wb") as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.warning(f"Pickle save error for {key}: {e}")
            return False
    
    def load_pickle(self, key: str, default: Any = None) -> Any:
        path = self._data_dir / f"{key}.pkl"
        if not path.exists():
            return default
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Pickle load error for {key}: {e}")
            return default
    
    def list_keys(self) -> list[str]:
        keys = []
        for p in self._data_dir.iterdir():
            if p.suffix in (".json", ".pkl"):
                keys.append(p.stem)
        return keys
    
    def delete(self, key: str) -> bool:
        deleted = False
        for suffix in (".json", ".pkl"):
            path = self._data_dir / f"{key}{suffix}"
            if path.exists():
                path.unlink()
                deleted = True
        return deleted
