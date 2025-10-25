"""Lightweight bug localizer wrapping the LOCAGENT graph.

The real paper trains a full LLM policy over the graph. For integration we
provide a pragmatic implementation that can function even if the hybrid graph
has not been pre-built yet:

* If a ``LOCAGENT_Graph`` instance is supplied, we delegate to its hybrid
  semantic+sparse search helpers.
* Otherwise we fall back to a simple keyword scan across source files so the
  orchestrator always receives a ranked list of candidate entities.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


try:  # Optional import – LOCAGENT may not be available in minimal setups
    from ..locagent import LOCAGENT_Graph, HybridSearchResult
except Exception:  # pragma: no cover - fallback if optional deps missing
    LOCAGENT_Graph = None  # type: ignore
    HybridSearchResult = None  # type: ignore


@dataclass
class LocalizationCandidate:
    entity_id: str
    score: float
    location: str
    metadata: Dict[str, any]


class BugLocalizer:
    """High level wrapper that exposes a single ``localize`` entry point."""

    def __init__(self, graph: Optional[LOCAGENT_Graph] = None, use_gpu: bool = True):
        self.graph = graph
        self.use_gpu = use_gpu

    async def _ensure_graph(self, repo_path: str) -> Optional[LOCAGENT_Graph]:
        if LOCAGENT_Graph is None:
            return None
        if self.graph is None:
            logger.info("Initializing LOCAGENT graph for repo %s", repo_path)
            self.graph = LOCAGENT_Graph(use_gpu=self.use_gpu)
            await self.graph.build_from_repository(repo_path)
        return self.graph

    async def localize(
        self,
        issue_description: str,
        repo_path: str,
        top_k: int = 5,
        use_semantic: bool = True,
    ) -> List[LocalizationCandidate]:
        """Return ranked entities that are likely related to the issue."""

        graph = await self._ensure_graph(repo_path)
        if graph is not None:
            results = graph.search_hybrid(
                query=issue_description,
                limit=top_k,
                alpha=0.65 if use_semantic else 0.4,
            )
            candidates = [
                LocalizationCandidate(
                    entity_id=res.entity.entity_id,
                    score=float(res.score),
                    location=res.entity.location,
                    metadata={
                        "sparse_score": res.sparse_score,
                        "dense_score": res.dense_score,
                        **(res.entity.metadata or {}),
                    },
                )
                for res in results
            ]
            if candidates:
                return candidates

        # Fallback: quick keyword scan over repository files
        logger.debug("Falling back to keyword scan localization")
        keywords = [tok.lower() for tok in issue_description.split() if len(tok) > 3]
        if not keywords:
            keywords = issue_description.lower().split()

        repo = Path(repo_path)
        scores: Dict[str, int] = {}

        for path in repo.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            score = sum(text.count(keyword) for keyword in keywords)
            if score > 0:
                scores[str(path)] = score

        sorted_paths = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            LocalizationCandidate(
                entity_id=path,
                score=float(score),
                location=path,
                metadata={"fallback": True},
            )
            for path, score in sorted_paths
        ]

