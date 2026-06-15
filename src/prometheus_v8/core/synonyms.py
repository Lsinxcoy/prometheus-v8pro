"""Synonym Dictionary - 38 AI/ML synonym groups for query expansion."""
from __future__ import annotations
import re
from typing import Optional

# 38 synonym groups covering AI/ML/agent domains
SYNONYM_GROUPS: list[list[str]] = [
    ["evolution", "evolve", "mutate", "mutation", "adapt", "adaptation"],
    ["memory", "memories", "remember", "recall", "store", "storage"],
    ["learning", "learn", "train", "training", "study", "acquire"],
    ["agent", "bot", "assistant", "ai", "model"],
    ["knowledge", "info", "information", "data", "fact", "facts"],
    ["skill", "ability", "capability", "competence", "proficiency"],
    ["consolidation", "merge", "compress", "compact", "condense"],
    ["pattern", "regularity", "trend", "tendency"],
    ["insight", "understanding", "realization", "discovery", "epiphany"],
    ["belief", "conviction", "opinion", "view", "stance"],
    ["foresight", "prediction", "forecast", "anticipation", "projection"],
    ["dream", "dreaming", "offline", "sleep", "consolidation_cycle"],
    ["code", "program", "implementation", "source", "script"],
    ["test", "testing", "validation", "verify", "verification", "check"],
    ["safety", "security", "protection", "guard", "guardrail"],
    ["evolution", "genetic", "darwin", "selection", "fitness"],
    ["prompt", "instruction", "directive", "command", "template"],
    ["tool", "instrument", "utility", "function", "capability"],
    ["graph", "network", "topology", "relation", "relationship"],
    ["embedding", "vector", "representation", "encoding"],
    ["search", "query", "find", "lookup", "retrieve", "retrieval"],
    ["node", "vertex", "point", "entity", "item"],
    ["edge", "link", "connection", "relation", "arc"],
    ["community", "cluster", "group", "module"],
    ["importance", "relevance", "weight", "priority", "significance"],
    ["confidence", "certainty", "reliability", "trust"],
    ["veracity", "truth", "accuracy", "correctness"],
    ["provenance", "origin", "source", "lineage", "history"],
    ["retention", "persistence", "durability", "persistence"],
    ["metabolism", "decay", "atrophy", "degradation", "aging"],
    ["hallway", "corridor", "channel", "path", "bridge"],
    ["tunnel", "shortcut", "fast_path", "bypass"],
    ["broadcast", "notify", "announce", "publish", "alert"],
    ["autonomy", "independence", "self-governance", "self-direction"],
    ["curiosity", "exploration", "discovery", "wonder"],
    ["trust", "reliability", "credibility", "confidence"],
    ["hook", "trigger", "callback", "action", "handler"],
    ["governance", "control", "regulation", "oversight", "management"],
]


class SynonymDictionary:
    """Bidirectional synonym lookup for query expansion."""
    
    def __init__(self, groups: list[list[str]] | None = None) -> None:
        self._word_to_group: dict[str, int] = {}
        self._groups: list[set[str]] = []
        groups = groups or SYNONYM_GROUPS
        for i, group in enumerate(groups):
            group_set = set(w.lower() for w in group)
            self._groups.append(group_set)
            for word in group_set:
                self._word_to_group[word] = i
    
    def expand(self, query: str) -> list[str]:
        """Expand query with synonyms. Returns list of expanded queries."""
        words = re.findall(r'\w+', query.lower())
        expansions = set()
        for word in words:
            gid = self._word_to_group.get(word)
            if gid is not None:
                synonyms = self._groups[gid] - {word}
                for syn in list(synonyms)[:3]:
                    new_query = re.sub(r'\b' + re.escape(word) + r'\b', syn, query, count=1, flags=re.IGNORECASE)
                    expansions.add(new_query)
        return list(expansions)[:5]
    
    def get_synonyms(self, word: str) -> set[str]:
        """Get all synonyms for a word."""
        gid = self._word_to_group.get(word.lower())
        if gid is None:
            return set()
        return self._groups[gid] - {word.lower()}
    
    def are_synonyms(self, word1: str, word2: str) -> bool:
        """Check if two words are synonyms."""
        gid1 = self._word_to_group.get(word1.lower())
        gid2 = self._word_to_group.get(word2.lower())
        return gid1 is not None and gid1 == gid2
