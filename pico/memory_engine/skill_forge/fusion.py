"""Weighted Reciprocal Rank Fusion across heterogeneous skill sources."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from pico.memory_engine.skill_forge.types import RouterHit

RRF_K: int = 60


def rrf_merge_weighted(
    source_results: list[tuple[str, float, list[RouterHit]]],
    k: int,
    dedup_by: str = "name",
) -> list[RouterHit]:
    """Fuse per-source ranked lists into one top-K result."""
    rrf_scores: dict[str, float] = defaultdict(float)
    best_hit: dict[str, RouterHit] = {}
    contributing: dict[str, list[str]] = defaultdict(list)

    for source_name, weight, hits in source_results:
        for rank, hit in enumerate(hits, start=1):
            key = getattr(hit, dedup_by)
            rrf_scores[key] += weight / (RRF_K + rank)
            contributing[key].append(source_name)
            previous = best_hit.get(key)
            if previous is None or hit.score > previous.score:
                best_hit[key] = hit

    ranked = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    out: list[RouterHit] = []
    for key, score in ranked[:k]:
        representative = best_hit[key]
        out.append(
            replace(
                representative,
                meta={
                    **representative.meta,
                    "rrf_score": score,
                    "contributing_sources": list(contributing[key]),
                },
            )
        )
    return out


__all__ = ["RRF_K", "rrf_merge_weighted"]
