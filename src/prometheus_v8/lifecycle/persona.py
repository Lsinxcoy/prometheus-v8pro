"""PersonaGenerator - LLM-driven Agent persona auto-generation."""

from __future__ import annotations

import logging

from prometheus_v8.schema import Node, PersonaProfile

logger = logging.getLogger(__name__)


class PersonaGenerator:
    """Generate Agent persona profiles from node associations and neighbor analysis.

    Uses LLM for deep analysis when available, falls back to heuristic extraction.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm

    def generate_from_node(self, node: Node, store=None) -> PersonaProfile:
        """Generate a PersonaProfile from a node and its graph neighborhood.

        Args:
            node: The agent/user node to generate persona for.
            store: Optional store to look up neighbors and edges.

        Returns:
            PersonaProfile with extracted or inferred persona attributes.
        """
        if self._llm is not None:
            try:
                return self._generate_with_llm(node, store)
            except Exception as e:
                logger.warning(f"LLM persona generation failed, falling back: {e}")

        return self._generate_heuristic(node, store)

    def _generate_heuristic(self, node: Node, store=None) -> PersonaProfile:
        """Heuristic persona extraction from node content, tags, and neighbors."""
        profile = PersonaProfile()

        # Extract interested topics from tags
        profile.interested_topics = list(node.tags[:10]) if node.tags else []

        # Infer sentiment bias from content
        content_lower = node.payload.content.lower()
        positive_words = {"good", "great", "excellent", "happy", "love", "best", "amazing", "wonderful"}
        negative_words = {"bad", "terrible", "awful", "hate", "worst", "horrible", "angry", "sad"}
        pos_count = sum(1 for w in positive_words if w in content_lower)
        neg_count = sum(1 for w in negative_words if w in content_lower)
        total = pos_count + neg_count
        if total > 0:
            profile.sentiment_bias = (pos_count - neg_count) / total
        else:
            profile.sentiment_bias = 0.0

        # Infer stance from content keywords
        if any(w in content_lower for w in {"support", "agree", "favor", "endorse"}):
            profile.stance = "supportive"
        elif any(w in content_lower for w in {"oppose", "disagree", "against", "reject"}):
            profile.stance = "critical"
        else:
            profile.stance = "neutral"

        # Infer MBTI-like traits from node type and metadata
        node_type_str = node.type.value if hasattr(node.type, "value") else str(node.type)
        if node_type_str in ("agent", "user"):
            profile.mbti = "INTJ"  # Default analytical type for agents
        elif node_type_str in ("insight", "pattern"):
            profile.mbti = "INTP"
        elif node_type_str in ("episode", "fact"):
            profile.mbti = "ISTJ"
        else:
            profile.mbti = "INFJ"

        # Influence weight from importance and access count
        profile.influence_weight = min(2.0, node.importance * (1 + 0.1 * min(node.access_count, 10)))

        # Activity pattern from metadata
        if node.metadata:
            profile.activity_pattern = {
                k: v for k, v in node.metadata.items()
                if isinstance(v, (int, float, str, bool))
            }

        # Enrich from neighbors if store available
        if store:
            self._enrich_from_neighbors(node, store, profile)

        return profile

    def _enrich_from_neighbors(self, node: Node, store, profile: PersonaProfile) -> None:
        """Enrich persona from neighbor nodes and edges."""
        try:
            # Get edges for this node
            edges = store.get_edges(node.id, direction="outgoing")
            edges_in = store.get_edges(node.id, direction="incoming")
            all_edges = edges + edges_in

            # Collect neighbor node IDs
            neighbor_ids = set()
            for edge in all_edges[:20]:  # Limit to avoid overload
                neighbor_ids.add(edge.source_id)
                neighbor_ids.add(edge.target_id)
            neighbor_ids.discard(node.id)

            # Extract topics from neighbor content
            for nid in list(neighbor_ids)[:10]:
                try:
                    neighbor = store.get_node(nid)
                    if neighbor and neighbor.tags:
                        for tag in neighbor.tags[:3]:
                            if tag not in profile.interested_topics:
                                profile.interested_topics.append(tag)
                except Exception:
                    pass

            # Adjust influence weight based on edge count
            if all_edges:
                profile.influence_weight = min(2.0, profile.influence_weight + 0.05 * len(all_edges))

        except Exception as e:
            logger.debug(f"Neighbor enrichment failed: {e}")

    def _generate_with_llm(self, node: Node, store=None) -> PersonaProfile:
        """LLM-driven persona generation with structured prompt."""
        # Gather context
        neighbor_info = ""
        if store:
            try:
                edges = store.get_edges(node.id, direction="outgoing")[:10]
                for edge in edges:
                    neighbor = store.get_node(edge.target_id)
                    if neighbor:
                        neighbor_info += f"- {neighbor.payload.content[:80]} (tags: {', '.join(neighbor.tags[:3])})\n"
            except Exception:
                pass

        prompt = (
            "Analyze the following agent/user node and generate a persona profile.\n"
            "Return ONLY a JSON object with these fields:\n"
            '- "mbti": 4-letter MBTI type (e.g. "INTJ")\n'
            '- "sentiment_bias": float from -1.0 to 1.0\n'
            '- "stance": one of "supportive", "critical", "neutral"\n'
            '- "interested_topics": list of topic strings\n'
            '- "influence_weight": float 0.0 to 2.0\n\n'
            f"Node content: {node.payload.content[:200]}\n"
            f"Node tags: {', '.join(node.tags[:10])}\n"
            f"Node type: {node.type.value}\n"
            f"Importance: {node.importance}\n"
        )
        if neighbor_info:
            prompt += f"\nConnected nodes:\n{neighbor_info}\n"

        import json
        response = self._llm(prompt)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(text)
        return PersonaProfile(
            mbti=data.get("mbti", "INFJ"),
            sentiment_bias=max(-1.0, min(1.0, float(data.get("sentiment_bias", 0.0)))),
            stance=data.get("stance", "neutral"),
            interested_topics=data.get("interested_topics", []),
            influence_weight=max(0.0, min(2.0, float(data.get("influence_weight", 1.0)))),
            activity_pattern=data.get("activity_pattern", {}),
        )
