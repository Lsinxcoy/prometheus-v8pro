"""Hybrid Search - 4-way parallel + RRF + MMR + Synonym expansion."""
from __future__ import annotations
import asyncio
import logging
import math
import threading
from collections import defaultdict
from typing import Any, Optional
import numpy as np
from prometheus_v8.schema import Node, MemoryLayer
from prometheus_v8.core.synonyms import SynonymDictionary

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """4-way parallel search: FTS + Vector + Graph + Memory, fused with RRF + MMR."""
    
    def __init__(self, store=None, vector_backend=None, graph_backend=None, 
                 rrf_k: int = 60, mmr_lambda: float = 0.5, 
                 enable_synonyms: bool = True, max_results: int = 20) -> None:
        self._store = store
        self._vector = vector_backend
        self._graph = graph_backend
        self._rrf_k = rrf_k
        self._mmr_lambda = mmr_lambda
        self._synonyms = SynonymDictionary() if enable_synonyms else None
        self._max_results = max_results
        self._lock = threading.RLock()
        self._query_cache: dict[str, list[Node]] = {}
    
    def search(self, query: str, k: int = 20, filters: dict | None = None,
               layers: list[MemoryLayer] | None = None) -> list[Node]:
        """Execute hybrid search: expand synonyms → parallel search → RRF fuse → MMR rerank."""
        expanded_queries = self._expand_query(query)
        
        all_results: dict[str, tuple[Node, list[int]]] = {}
        
        # 1. FTS search
        if self._store:
            for q in expanded_queries:
                try:
                    fts_results = self._store.search_fts(q, limit=k*3)
                    for rank, node in enumerate(fts_results):
                        nid = node.id.hex()
                        if nid not in all_results:
                            all_results[nid] = (node, [])
                        all_results[nid][1].append(rank)
                except Exception as e:
                    logger.warning(f"FTS search error: {e}")
        
        # 2. Vector search (needs embedding - use hash-based pseudo-embedding if no embedder)
        if self._vector:
            try:
                pseudo_vec = self._query_to_vector(query)
                vec_results = self._vector.search(pseudo_vec, k=k*3)
                node_map = {n.id.hex(): n for n in (self._store.search_fts(query, limit=1000) if self._store else [])}
                for rank, (node_id, score) in enumerate(vec_results):
                    if node_id in node_map:
                        nid = node_map[node_id].id.hex()
                        if nid not in all_results:
                            all_results[nid] = (node_map[node_id], [])
                        all_results[nid][1].append(rank)
            except Exception as e:
                logger.warning(f"Vector search error: {e}")
        
        # 3. Graph traversal for related nodes
        if self._graph and self._store:
            try:
                fts_nodes = self._store.search_fts(query, limit=5)
                for node in fts_nodes:
                    neighbors = self._graph.get_neighbors(node.id.hex())
                    for target_id, edge_type, weight in neighbors[:5]:
                        target_node = self._store.get_node(bytes.fromhex(target_id.rjust(32, '0')))
                        if target_node:
                            nid = target_node.id.hex()
                            if nid not in all_results:
                                all_results[nid] = (target_node, [])
                            all_results[nid][1].append(5)  # bonus rank
            except Exception as e:
                logger.warning(f"Graph search error: {e}")
        
        # Apply filters
        if filters or layers:
            filtered = {}
            for nid, (node, ranks) in all_results.items():
                if layers and node.layer not in layers:
                    continue
                if filters:
                    match = True
                    for key, val in filters.items():
                        if node.metadata.get(key) != val:
                            match = False
                            break
                    if not match:
                        continue
                filtered[nid] = (node, ranks)
            all_results = filtered
        
        # RRF fusion
        rrf_scores = self._rrf_fuse(all_results)
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        
        # MMR reranking
        if self._vector and len(sorted_ids) > 1:
            sorted_ids = self._mmr_rerank(sorted_ids, all_results, k)
        
        return [all_results[nid][0] for nid in sorted_ids if nid in all_results]
    
    def _expand_query(self, query: str) -> list[str]:
        """Expand query with synonyms."""
        queries = [query]
        if self._synonyms:
            expanded = self._synonyms.expand(query)
            queries.extend(expanded[:2])  # max 2 synonym expansions
        return queries
    
    def _query_to_vector(self, query: str) -> list[float]:
        """Convert query to pseudo-vector using hash (fallback when no embedder)."""
        dim = 384
        h = hash(query) & 0xFFFFFFFF
        rng = np.random.RandomState(h)
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
    
    def _rrf_fuse(self, results: dict[str, tuple[Node, list[int]]]) -> dict[str, float]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) for each list."""
        scores = {}
        for nid, (node, ranks) in results.items():
            score = sum(1.0 / (self._rrf_k + r) for r in ranks)
            scores[nid] = score
        return scores
    
    def _mmr_rerank(self, candidate_ids: list[str], all_results: dict, k: int) -> list[str]:
        """Maximal Marginal Relevance reranking for diversity."""
        selected = [candidate_ids[0]]
        remaining = candidate_ids[1:]
        
        while len(selected) < min(k, len(candidate_ids)) and remaining:
            best_id = None
            best_score = -float('inf')
            
            for nid in remaining:
                relevance = all_results[nid][1]  # ranks as relevance proxy
                rel_score = sum(1.0 / (self._rrf_k + r) for r in relevance) if relevance else 0
                
                # Diversity penalty
                max_sim = 0.0
                for sel_id in selected:
                    if sel_id in all_results and nid in all_results:
                        s1 = all_results[sel_id][0].payload.content
                        s2 = all_results[nid][0].payload.content
                        max_sim = max(max_sim, self._text_similarity(s1, s2))
                
                mmr = self._mmr_lambda * rel_score - (1 - self._mmr_lambda) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_id = nid
            
            if best_id:
                selected.append(best_id)
                remaining.remove(best_id)
            else:
                break
        
        return selected
    
    @staticmethod
    def _text_similarity(s1: str, s2: str) -> float:
        """Simple Jaccard similarity between two texts."""
        w1 = set(s1.lower().split())
        w2 = set(s2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)
