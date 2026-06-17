"""Hybrid Search - 4-way parallel + RRF + MMR + Synonym expansion + InsightForge deep search."""

from __future__ import annotations

import json
import logging
import threading
import time

from prometheus_v8.core.synonyms import SynonymDictionary
from prometheus_v8.schema import MemoryLayer, Node

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """4-way parallel search: FTS + Vector + Graph + Memory, fused with RRF + MMR."""

    def __init__(
        self,
        store=None,
        vector_backend=None,
        graph_backend=None,
        rrf_k: int = 60,
        mmr_lambda: float = 0.5,
        enable_synonyms: bool = True,
        max_results: int = 20,
        llm=None,
    ) -> None:
        self._store = store
        self._vector = vector_backend
        self._graph = graph_backend
        self._rrf_k = rrf_k
        self._mmr_lambda = mmr_lambda
        self._synonyms = SynonymDictionary() if enable_synonyms else None
        self._max_results = max_results
        self._llm = llm
        self._lock = threading.RLock()
        self._query_cache: dict[str, list[Node]] = {}

    def search(
        self,
        query: str,
        k: int = 20,
        filters: dict | None = None,
        layers: list[MemoryLayer] | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> list[Node]:
        """Execute hybrid search: expand synonyms → parallel search → RRF fuse → MMR rerank.

        Args:
            query: Search query string.
            k: Maximum number of results.
            filters: Metadata key-value filters.
            layers: Memory layers to include.
            time_range: Optional (t_from, t_to) tuple to filter by valid_from/valid_to.
        """
        expanded_queries = self._expand_query(query)

        all_results: dict[str, tuple[Node, list[int]]] = {}

        # 1. FTS search
        if self._store:
            for q in expanded_queries:
                try:
                    fts_results = self._store.search_fts(q, limit=k * 3)
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
                vec_results = self._vector.search(pseudo_vec, k=k * 3)
                # Build node map from store - use targeted search instead of broad FTS
                node_map = {}
                if self._store:
                    # First try FTS with the query, limited
                    fts_nodes = self._store.search_fts(query, limit=k * 2)
                    for n in fts_nodes:
                        node_map[n.id.hex()] = n
                for rank, (node_id, score) in enumerate(vec_results):
                    # Try to get node directly from store if not in map
                    if node_id not in node_map and self._store:
                        try:
                            n = self._store.get_node(bytes.fromhex(node_id.rjust(32, "0")))
                            if n:
                                node_map[node_id] = n
                        except Exception as e:
                            logger.debug(f"Node lookup failed for {node_id}: {e}")
                            pass
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
                        target_node = self._store.get_node(bytes.fromhex(target_id.rjust(32, "0")))
                        if target_node:
                            nid = target_node.id.hex()
                            if nid not in all_results:
                                all_results[nid] = (target_node, [])
                            all_results[nid][1].append(5)  # bonus rank
            except Exception as e:
                logger.warning(f"Graph search error: {e}")

        # Apply filters
        if filters or layers or time_range:
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
                # Time range filtering
                if time_range is not None:
                    t_from, t_to = time_range
                    # Node is in range if its validity period overlaps [t_from, t_to]
                    if node.valid_from > t_to or node.valid_to < t_from:
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

        results = [all_results[nid][0] for nid in sorted_ids if nid in all_results]

        # Tag active vs historical nodes
        now = time.time()
        for node in results:
            if node.valid_to < now:
                node.metadata["_temporal_status"] = "historical"
            else:
                node.metadata["_temporal_status"] = "active"

        return results

    # ── InsightForge: Deep Search with Sub-query Decomposition ──

    def _decompose_query(self, query: str) -> list[str]:
        """Decompose a complex query into 3-5 sub-questions using LLM.

        Falls back to simple split if no LLM is available.
        """
        if self._llm is None:
            # No LLM: return the original query as-is (no decomposition)
            return [query]

        prompt = (
            "Decompose the following complex query into 3-5 simpler sub-questions "
            "that together cover the original query's intent. "
            "Return ONLY a JSON array of strings, no explanation.\n\n"
            f"Query: {query}"
        )
        try:
            response = self._llm(prompt)
            # Try to parse JSON array from response
            text = response.strip()
            # Handle cases where LLM wraps in markdown code block
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            subqueries = json.loads(text)
            if isinstance(subqueries, list) and all(isinstance(q, str) for q in subqueries):
                return subqueries[:5]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")

        # Fallback: return original query
        return [query]

    def deep_search(
        self, query: str, k: int = 20, max_subqueries: int = 5, time_range: tuple[float, float] | None = None
    ) -> list[Node]:
        """InsightForge deep search: decompose → parallel search → dedup → expand → RRF + MMR.

        1. Decompose complex query into sub-questions (requires LLM; falls back to regular search)
        2. Search each sub-question independently
        3. Deduplicate results
        4. Expand via 2-hop graph neighbors
        5. Unified RRF fusion + MMR reranking

        Fallback behavior:
        - If no LLM is available, query decomposition is skipped and this method
          delegates to ``self.search()`` (standard 4-way hybrid search without
          sub-query decomposition or graph expansion).
        - If LLM decomposition fails (JSON parse error, timeout, etc.), the
          original query is used as the sole sub-query, which effectively
          produces the same result as ``self.search()``.
        - Graph expansion is skipped if ``self._graph`` or ``self._store`` is None.
        - MMR reranking is skipped if ``self._vector`` is None.
        """
        # Step 1: Decompose query
        subqueries = self._decompose_query(query)
        if len(subqueries) > max_subqueries:
            subqueries = subqueries[:max_subqueries]

        # If no decomposition happened (no LLM), fall back to regular search
        if len(subqueries) == 1 and subqueries[0] == query:
            return self.search(query, k=k, time_range=time_range)

        # Step 2: Search each sub-query and collect results
        all_results: dict[str, tuple[Node, list[int]]] = {}

        for sub_idx, subquery in enumerate(subqueries):
            try:
                sub_results = self.search(subquery, k=k * 2, time_range=time_range)
                for rank, node in enumerate(sub_results):
                    nid = node.id.hex()
                    if nid not in all_results:
                        all_results[nid] = (node, [])
                    # Add rank with sub-query offset to distinguish sources
                    all_results[nid][1].append(rank + sub_idx * 100)
            except Exception as e:
                logger.warning(f"Sub-query search failed for '{subquery}': {e}")

        # Step 3: 2-hop graph expansion
        if self._graph and self._store:
            expanded_ids = set(all_results.keys())
            # 1st hop
            hop1_ids = set()
            for nid in list(expanded_ids):
                try:
                    neighbors = self._graph.get_neighbors(nid)
                    for target_id, edge_type, weight in neighbors[:10]:
                        hop1_ids.add(target_id)
                except Exception as e:
                    logger.debug(f"1-hop expansion failed for {nid}: {e}")

            # 2nd hop
            hop2_ids = set()
            for nid in hop1_ids:
                try:
                    neighbors = self._graph.get_neighbors(nid)
                    for target_id, edge_type, weight in neighbors[:5]:
                        if target_id not in expanded_ids:
                            hop2_ids.add(target_id)
                except Exception as e:
                    logger.debug(f"2-hop expansion failed for {nid}: {e}")

            # Add expanded nodes with bonus rank
            for target_id in hop1_ids | hop2_ids:
                if target_id not in all_results:
                    try:
                        node = self._store.get_node(bytes.fromhex(target_id.rjust(32, "0")))
                        if node:
                            # Apply time range filter to expanded nodes too
                            if time_range is not None:
                                t_from, t_to = time_range
                                if node.valid_from > t_to or node.valid_to < t_from:
                                    continue
                            all_results[target_id] = (node, [50])  # bonus rank for expansion
                    except Exception as e:
                        logger.debug(f"Expanded node lookup failed for {target_id}: {e}")

        # Step 4: Unified RRF fusion
        rrf_scores = self._rrf_fuse(all_results)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]

        # Step 5: MMR reranking
        if self._vector and len(sorted_ids) > 1:
            sorted_ids = self._mmr_rerank(sorted_ids, all_results, k)

        results = [all_results[nid][0] for nid in sorted_ids if nid in all_results]

        # Tag active vs historical
        now = time.time()
        for node in results:
            if node.valid_to < now:
                node.metadata["_temporal_status"] = "historical"
            else:
                node.metadata["_temporal_status"] = "active"

        return results

    def _expand_query(self, query: str) -> list[str]:
        """Expand query with synonyms."""
        queries = [query]
        if self._synonyms:
            expanded = self._synonyms.expand(query)
            queries.extend(expanded[:2])  # max 2 synonym expansions
        return queries

    def _query_to_vector(self, query: str) -> list[float]:
        """Convert query to pseudo-vector using hash (fallback when no embedder)."""
        from prometheus_v8.core.embedder import Embedder

        return Embedder.hash_embed(query, dimension=384)

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
            best_score = -float("inf")

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
