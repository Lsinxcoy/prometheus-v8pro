"""DreamCycle - 5-stage offline consolidation with ReACT reasoning."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus_v8.schema import MemoryLayer, Node, create_dream_node

logger = logging.getLogger(__name__)


@dataclass
class ReactStep:
    """A single ReACT reasoning step."""

    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""


class DreamLogger:
    """Log dream reasoning chains to JSONL files."""

    def __init__(self, log_dir: str = "data/dream_logs") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current_chain: list[dict[str, Any]] = []

    def start_chain(self, dream_id: int) -> None:
        """Start a new dream reasoning chain."""
        self._current_chain = [{"dream_id": dream_id, "start_time": time.time(), "steps": []}]

    def log_step(self, step: ReactStep) -> None:
        """Log a ReACT step."""
        if self._current_chain:
            self._current_chain[0]["steps"].append({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
            })

    def log_insight(self, insight_content: str, importance: float) -> None:
        """Log a generated insight."""
        if self._current_chain:
            self._current_chain[0].setdefault("insights", []).append({
                "content": insight_content,
                "importance": importance,
            })

    def end_chain(self, num_insights: int) -> None:
        """End the current chain and write to JSONL."""
        if not self._current_chain:
            return
        self._current_chain[0]["end_time"] = time.time()
        self._current_chain[0]["num_insights"] = num_insights
        self._current_chain[0]["duration"] = (
            self._current_chain[0]["end_time"] - self._current_chain[0]["start_time"]
        )

        # Append to JSONL file
        log_file = self._log_dir / f"dream_{time.strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._current_chain[0], ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write dream log: {e}")

        self._current_chain = []


class DreamCycle:
    """5-stage DreamCycle for offline consolidation:
    1. REPLAY: Revisit recent episodic memories
    2. ASSOCIATE: Find connections between memories
    3. CONSOLIDATE: Strengthen important, weaken unimportant
    4. GENERATE: Create new insights from patterns (with optional ReACT reasoning)
    5. INTEGRATE: Merge new insights into semantic memory
    """

    def __init__(self, store=None, event_bus=None, llm=None) -> None:
        self._store = store
        self._event_bus = event_bus
        self._llm = llm
        self._dream_count = 0
        self._insights_generated = 0
        self._dream_logger = DreamLogger()

    def dream(self, recent_nodes: list[Node] | None = None) -> list[Node]:
        """Run one dream cycle."""
        self._dream_count += 1
        start = time.time()
        self._dream_logger.start_chain(self._dream_count)

        if recent_nodes is None and self._store:
            recent_nodes = self._store.get_nodes_by_layer(MemoryLayer.EPISODIC, limit=50)
        if not recent_nodes:
            recent_nodes = []

        # Stage 1: REPLAY
        replayed = self._replay(recent_nodes)

        # Stage 2: ASSOCIATE
        associations = self._associate(replayed)

        # Stage 3: CONSOLIDATE
        self._consolidate(replayed)

        # Stage 4: GENERATE
        insights = self._generate(replayed, associations)

        # Stage 5: INTEGRATE
        integrated = self._integrate(insights)

        elapsed = time.time() - start
        logger.info(f"Dream cycle #{self._dream_count} completed in {elapsed:.1f}s, {len(integrated)} insights")

        self._dream_logger.end_chain(len(integrated))
        return integrated

    def _replay(self, nodes: list[Node]) -> list[Node]:
        """Stage 1: Replay recent memories, updating access counts."""
        for node in nodes:
            node.touch()
            if self._store:
                self._store.update_node(node)
        return nodes

    def _associate(self, nodes: list[Node]) -> list[tuple[Node, Node, float]]:
        """Stage 2: Find associations between memories based on content overlap."""
        associations = []
        # Limit to max 30 nodes to avoid O(n²) explosion
        sample = nodes[:30] if len(nodes) > 30 else nodes
        for i, n1 in enumerate(sample):
            for n2 in sample[i + 1 :]:
                score = self._compute_association(n1, n2)
                if score > 0.3:
                    associations.append((n1, n2, score))
        return associations

    def _consolidate(self, nodes: list[Node]) -> list[Node]:
        """Stage 3: Strengthen important nodes, decay unimportant."""
        for node in nodes:
            if node.importance > 0.6:
                node.importance = min(1.0, node.importance * 1.05)
            else:
                node.importance *= 0.95
            if self._store:
                self._store.update_node(node)
        return nodes

    def _generate(self, nodes: list[Node], associations: list[tuple[Node, Node, float]]) -> list[Node]:
        """Stage 4: Generate new insights from patterns and associations.

        Uses ReACT reasoning when LLM is available, falls back to pattern-based generation.
        """
        if self._llm is not None:
            try:
                return self._generate_with_react(nodes, associations)
            except Exception as e:
                logger.warning(f"ReACT generation failed, falling back: {e}")

        # Fallback: original pattern-based generation
        insights = []
        for n1, n2, score in associations[:5]:
            if score > 0.5:
                content = f"Pattern detected: '{n1.payload.content[:50]}' ↔ '{n2.payload.content[:50]}' (strength={score:.2f})"
                insight = create_dream_node(content=content, importance=score * 0.8)
                insight.layer = MemoryLayer.SEMANTIC
                insights.append(insight)
                self._dream_logger.log_insight(content, score * 0.8)

        self._insights_generated += len(insights)
        return insights

    def _generate_with_react(self, nodes: list[Node], associations: list[tuple[Node, Node, float]]) -> list[Node]:
        """ReACT-driven insight generation: Thought → Action → Observation, max 3 rounds."""
        insights = []
        max_rounds = 3

        # Build context from top associations
        top_associations = [(n1, n2, s) for n1, n2, s in associations if s > 0.4][:5]
        if not top_associations:
            return insights

        context_parts = []
        for n1, n2, score in top_associations:
            context_parts.append(
                f"- '{n1.payload.content[:60]}' ↔ '{n2.payload.content[:60]}' (score={score:.2f})"
            )
        context_str = "\n".join(context_parts)

        accumulated_observations = []

        for round_idx in range(max_rounds):
            # ── Thought ──
            thought_prompt = (
                f"You are a dream reasoning engine. Given these memory associations:\n"
                f"{context_str}\n\n"
            )
            if accumulated_observations:
                thought_prompt += "Previous observations:\n" + "\n".join(accumulated_observations) + "\n\n"
            thought_prompt += (
                "What additional information would help generate a novel insight? "
                "Respond with a brief thought (1-2 sentences) about what to search for next."
            )

            try:
                thought = self._llm(thought_prompt).strip()[:200]
            except Exception as e:
                logger.debug(f"ReACT thought failed at round {round_idx}: {e}")
                break

            # ── Action ──
            # Determine search query from thought
            action_prompt = (
                f"Based on this thought: '{thought}'\n"
                "Provide a short search query (2-5 words) to find relevant information. "
                "Return ONLY the query, nothing else."
            )
            try:
                action_query = self._llm(action_prompt).strip()[:50]
            except Exception as e:
                logger.debug(f"ReACT action failed at round {round_idx}: {e}")
                break

            # ── Observation ──
            observation = ""
            if self._store:
                try:
                    search_results = self._store.search_fts(action_query, limit=5)
                    if search_results:
                        obs_parts = [f"Found {len(search_results)} results:"]
                        for r in search_results[:3]:
                            obs_parts.append(f"  - {r.payload.content[:80]} (importance={r.importance:.2f})")
                        observation = "\n".join(obs_parts)
                    else:
                        observation = "No results found for this query."
                except Exception as e:
                    observation = f"Search error: {e}"
            else:
                observation = "No store available for search."

            # Log the ReACT step
            step = ReactStep(
                thought=thought,
                action="search",
                action_input=action_query,
                observation=observation,
            )
            self._dream_logger.log_step(step)
            accumulated_observations.append(observation)

        # Generate final insights from accumulated knowledge
        insight_prompt = (
            "Based on the following memory associations and search observations, "
            "generate 1-3 novel insights. Each insight should be a single sentence. "
            "Return a JSON array of objects with 'content' and 'importance' (0.0-1.0) fields.\n\n"
            f"Associations:\n{context_str}\n\n"
            f"Observations:\n" + "\n".join(accumulated_observations) + "\n\n"
            "Return ONLY the JSON array."
        )

        try:
            response = self._llm(insight_prompt).strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            insight_data = json.loads(response)
            if isinstance(insight_data, list):
                for item in insight_data[:3]:
                    content = item.get("content", "")
                    importance = float(item.get("importance", 0.5))
                    if content:
                        insight = create_dream_node(content=content, importance=importance)
                        insight.layer = MemoryLayer.SEMANTIC
                        insights.append(insight)
                        self._dream_logger.log_insight(content, importance)
        except Exception as e:
            logger.warning(f"ReACT insight generation failed: {e}")
            # Fallback: generate from top associations
            for n1, n2, score in top_associations[:2]:
                content = f"Pattern detected: '{n1.payload.content[:50]}' ↔ '{n2.payload.content[:50]}' (strength={score:.2f})"
                insight = create_dream_node(content=content, importance=score * 0.8)
                insight.layer = MemoryLayer.SEMANTIC
                insights.append(insight)
                self._dream_logger.log_insight(content, score * 0.8)

        self._insights_generated += len(insights)
        return insights

    def _integrate(self, insights: list[Node]) -> list[Node]:
        """Stage 5: Integrate insights into semantic memory."""
        integrated = []
        for insight in insights:
            if self._store:
                self._store.add_node(insight)
            integrated.append(insight)
            # Publish insight_generated event
            if self._event_bus:
                try:
                    self._event_bus.publish("insight_generated", {
                        "node": insight,
                        "content": insight.payload.content,
                        "importance": insight.importance,
                    })
                except Exception as e:
                    logger.debug(f"Event publish failed: {e}")
        return integrated

    def _compute_association(self, n1: Node, n2: Node) -> float:
        """Compute association score between two nodes."""
        # Content overlap (Jaccard)
        w1 = set(n1.payload.content.lower().split())
        w2 = set(n2.payload.content.lower().split())
        if not w1 or not w2:
            return 0.0
        jaccard = len(w1 & w2) / len(w1 | w2)

        # Tag overlap
        tag_score = len(set(n1.tags) & set(n2.tags)) / max(1, len(set(n1.tags) | set(n2.tags)))

        # Type bonus
        type_bonus = 0.1 if n1.type == n2.type else 0.0

        return 0.5 * jaccard + 0.3 * tag_score + 0.2 * type_bonus

    @property
    def stats(self) -> dict[str, int]:
        return {"dream_cycles": self._dream_count, "insights_generated": self._insights_generated}
