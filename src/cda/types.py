"""Shared data containers for the CDA pipeline.

Kept deliberately plain: these are records passed between stages, not
behavior-carrying objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Passage:
    """One term-containing sentence (or window) extracted from the corpus.

    The sentence containing the target term is the unit of analysis
    (paper §3.2). ``embedding`` is populated lazily by the clustering stage.
    """

    chunk_id: str
    source: str  # file name the passage came from
    year: str  # publication year or "unknown"
    text: str
    similarity: float  # retrieval similarity (0.0 for keyword-only hits)
    metadata: dict[str, Any] = field(default_factory=dict)
    cluster_id: str = "Cluster N/A"
    embedding: list[float] | None = None


def passage_theme_tags(passage: Passage) -> list[str]:
    """Observed source tags only; MyWork denotes authorship, not a theme."""
    raw = passage.metadata.get("zotero_tags", [])
    tags = re.split(r"[;,]", raw) if isinstance(raw, str) else raw
    if not isinstance(tags, list):
        return []
    return sorted(
        {
            str(tag).strip().lstrip("#")
            for tag in tags
            if str(tag).strip().lower() not in {"", "mywork", "#mywork"}
        }
    )


@dataclass
class ClusterSummary:
    """Result of clustering one corpus (researcher or literature).

    ``clusters`` holds one entry per cluster: cluster_id, n_passages,
    representatives (texts), sources, rep_roles. ``diagnostics`` carries the
    k-sweep log, merges, and per-cluster stats for the run's JSON diagnostics.

    Under joint clustering this is a *view* of the shared partition (see
    ``clustering/joint.py``): the cluster ids are the joint ones, so the same
    id in the researcher and literature summaries names the same region.
    """

    used: bool
    n_total_passages: int
    n_clusters: int
    silhouette: float
    note: str
    clusters: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class JointClustering:
    """One partition over the researcher's and the literature's passages pooled.

    Each entry in ``clusters`` carries the cluster's composition
    (``n_researcher`` / ``n_literature``, ``researcher_share``,
    ``literature_share``), the centroid-nearest representatives marked with
    the corpus each came from, and a per-corpus representative set used to
    build the two prompt blocks. ``silhouette`` is measured on the pooled
    passages — it describes the joint partition, not either corpus alone.
    """

    used: bool
    n_clusters: int
    silhouette: float
    note: str
    n_researcher: int
    n_literature: int
    clusters: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
