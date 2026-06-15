"""Embedder - sentence-transformers with hash fallback."""
from __future__ import annotations
import hashlib
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

class Embedder:
    """Text embedding with sentence-transformers or hash fallback."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384,
                 device: str = "cpu") -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._device = device
        self._model = None
        self._use_st = False
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._use_st = True
            logger.info(f"Loaded sentence-transformers model: {model_name}")
        except Exception as e:
            logger.warning(f"sentence-transformers not available ({e}), using hash fallback")
    
    def embed(self, text: str) -> list[float]:
        """Embed text to vector."""
        if self._use_st and self._model:
            try:
                vec = self._model.encode(text, normalize_embeddings=True)
                return vec.tolist()
            except Exception as e:
                logger.warning(f"ST embed error: {e}")
        return self._hash_embed(text)
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        if self._use_st and self._model:
            try:
                vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
                return vecs.tolist()
            except Exception:
                pass
        return [self._hash_embed(t) for t in texts]
    
    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic hash-based embedding fallback."""
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def is_using_transformers(self) -> bool:
        return self._use_st
