"""Computed divergence flags (paper §3.3).

A literature cluster is flagged **researcher-concentrated** when researcher
instances exceed the configured share of its combined membership while the
cluster accounts for less than the configured share of literature passages;
a cluster receiving (at most the configured number of) researcher passages is
flagged **underrepresented in researcher usage**. Two bases produce those
counts, matching the two clustering modes:

- **joint membership** (``compute_joint_flags``, the default) — the corpora
  share one partition, so "researcher instances in this cluster" is a direct
  count of members. This is the paper's §3.3 wording taken literally.
- **cross-assignment** (``compute_flags``) — with the corpora clustered
  independently there is no shared membership, so each researcher passage is
  assigned to its nearest literature-cluster centroid (cosine) first. This is
  the published pipeline's computation.

Both return the same dict shape, tagged with ``basis``, so the renderer and
the GUI treat them interchangeably and the report always says which was used.

These flags are descriptive aids for reading the report — they surface where
to look, they do not decide what a divergence means.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np

from cda.clustering.spherical import normalize_rows, unit_centroid
from cda.types import JointClustering, Passage


def _thresholds(flags_config: dict) -> tuple[float, float, int]:
    """The three flag thresholds, coerced once and named when they are wrong.

    On the pipeline path ``flag_params`` has already validated these, but the
    GUI, the benchmarks and tests call the two compute functions directly with
    a plain dict — and a bare ``float(...)`` there fails with a message that
    names no key, leaving the reader to guess which threshold was mistyped.
    """
    values: dict[str, float | int] = {}
    for key, default, cast in (
        ("researcher_share_threshold", 0.80, float),
        ("literature_share_threshold", 0.10, float),
        ("underrepresented_max_researcher_passages", 0, int),
    ):
        raw = flags_config.get(key, default)
        try:
            values[key] = cast(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"flags.{key} must be a number, got {raw!r}. See config/config.example.yaml."
            ) from exc
    for key in ("researcher_share_threshold", "literature_share_threshold"):
        _require_share(key, values[key])
    return (
        float(values["researcher_share_threshold"]),
        float(values["literature_share_threshold"]),
        int(values["underrepresented_max_researcher_passages"]),
    )


def _require_share(key: str, value: float) -> None:
    """Refuse a share threshold that is not a finite number in [0, 1]."""
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(
            f"flags.{key} must be a finite share between 0 and 1, got {value!r}. "
            "See config/config.example.yaml."
        )


_CLUSTER_ID = re.compile(r"Cluster \d+")
_BASIS_JOINT = "joint-membership"
_BASIS_CROSS = "cross-assignment"


def _flag_row(
    name: str,
    *,
    n_researcher: int,
    n_literature: int,
    total_literature: int,
    researcher_share_threshold: float,
    literature_share_threshold: float,
    underrepresented_max: int,
) -> dict:
    """One cluster's flag row, decided on exact count ratios.

    Shares are compared as fractions against the thresholds' decimal values
    (``Fraction(str(0.8)) == 4/5``), so a share just above a threshold is
    never rounded onto it; the stored shares are rounded for display only.
    """
    researcher_share = Fraction(n_researcher, n_researcher + n_literature)
    literature_share = Fraction(n_literature, total_literature)
    is_concentrated = researcher_share > Fraction(
        str(researcher_share_threshold)
    ) and literature_share < Fraction(str(literature_share_threshold))
    return {
        "cluster": name,
        "n_literature": n_literature,
        "literature_share": round(float(literature_share), 3),
        "n_researcher_assigned": n_researcher,
        "researcher_share_of_combined": round(float(researcher_share), 3),
        "researcher_concentrated": is_concentrated,
        "underrepresented_in_researcher_usage": n_researcher <= underrepresented_max,
    }


def _flags_result(basis: str, thresholds: tuple[float, float, int], rows: list[dict]) -> dict:
    researcher_share_threshold, literature_share_threshold, underrepresented_max = thresholds
    return {
        "available": True,
        "basis": basis,
        "thresholds": {
            "researcher_share_threshold": researcher_share_threshold,
            "literature_share_threshold": literature_share_threshold,
            "underrepresented_max_researcher_passages": underrepresented_max,
        },
        "rows": rows,
        "concentrated": [row["cluster"] for row in rows if row["researcher_concentrated"]],
        "underrepresented": [
            row["cluster"] for row in rows if row["underrepresented_in_researcher_usage"]
        ],
    }


def _cluster_centroids(passages: list[Passage]) -> tuple[list[str], np.ndarray] | None:
    """Normalized centroid per literature cluster (from attached cluster ids)."""
    by_cluster: dict[str, list[Passage]] = {}
    for passage in passages:
        if passage.embedding is None or not _CLUSTER_ID.fullmatch(passage.cluster_id):
            continue
        by_cluster.setdefault(passage.cluster_id, []).append(passage)
    if len(by_cluster) < 2:
        return None
    names = sorted(by_cluster, key=lambda c: int(c.split()[-1]))
    centroids = []
    for name in names:
        members = by_cluster[name]
        # In practice these vectors already passed prepare_matrix during
        # clustering, but the check is shared so there is only one definition
        # of an unusable embedding in the codebase.
        matrix = normalize_rows(
            np.asarray([p.embedding for p in members], dtype=float),
            [f"{p.chunk_id} ({name})" for p in members],
        )
        centroids.append(unit_centroid(matrix))
    return names, np.asarray(centroids)


def compute_flags(
    researcher_passages: list[Passage],
    literature_passages: list[Passage],
    flags_config: dict,
) -> dict:
    """Cross-assign researcher passages to literature clusters and flag.

    Returns {"available": bool, "rows": [...], "concentrated": [...],
    "underrepresented": [...]} — or an explanatory reason when the
    prerequisites (a usable literature clustering, researcher embeddings)
    are missing.
    """
    thresholds = _thresholds(flags_config)
    researcher_share_threshold, literature_share_threshold, underrepresented_max = thresholds

    centroid_data = _cluster_centroids(literature_passages)
    if centroid_data is None:
        return {
            "available": False,
            "reason": "literature clustering produced fewer than two usable clusters",
        }
    names, centroids = centroid_data

    embedded = [p for p in researcher_passages if p.embedding is not None]
    if not embedded:
        return {"available": False, "reason": "no researcher passages with embeddings"}
    R = normalize_rows(
        np.asarray([p.embedding for p in embedded], dtype=float),
        [p.chunk_id for p in embedded],
        what="researcher passage embeddings",
    )

    assignments = np.argmax(R @ centroids.T, axis=1)  # nearest centroid by cosine

    literature_counts = {name: 0 for name in names}
    for passage in literature_passages:
        if passage.cluster_id in literature_counts:
            literature_counts[passage.cluster_id] += 1
    total_literature = sum(literature_counts.values())

    rows = [
        _flag_row(
            name,
            n_researcher=int(np.sum(assignments == index)),
            n_literature=literature_counts[name],
            total_literature=total_literature,
            researcher_share_threshold=researcher_share_threshold,
            literature_share_threshold=literature_share_threshold,
            underrepresented_max=underrepresented_max,
        )
        for index, name in enumerate(names)
    ]
    return _flags_result(_BASIS_CROSS, thresholds, rows)


def compute_joint_flags(joint: JointClustering, flags_config: dict) -> dict:
    """Flag the shared clusters of a joint partition by direct membership.

    No geometry beyond the clustering itself: the researcher share of a
    cluster is the count of its researcher members over its total membership.
    Returns the same shape as ``compute_flags``.
    """
    thresholds = _thresholds(flags_config)
    researcher_share_threshold, literature_share_threshold, underrepresented_max = thresholds

    if not joint.used or len(joint.clusters) < 2:
        return {
            "available": False,
            "basis": _BASIS_JOINT,
            "reason": "joint clustering produced fewer than two usable clusters",
        }
    # The flags compare the two corpora, so they are undefined when one side
    # never entered the partition. With zero literature passages every cluster
    # is trivially 100% researcher and 0% of the literature — each would pass
    # the concentration test and read as a finding; with zero researcher
    # passages every cluster reads as underrepresented. Independent mode
    # already reports this honestly, and joint mode must not do worse.
    if sum(int(c["n_literature"]) for c in joint.clusters) == 0:
        return {
            "available": False,
            "basis": _BASIS_JOINT,
            "reason": "no literature passages entered the joint partition, so "
            "researcher-vs-literature shares are undefined",
        }
    if sum(int(c["n_researcher"]) for c in joint.clusters) == 0:
        return {
            "available": False,
            "basis": _BASIS_JOINT,
            "reason": "no researcher passages entered the joint partition, so "
            "researcher-vs-literature shares are undefined",
        }

    total_literature = sum(int(entry["n_literature"]) for entry in joint.clusters)
    rows = [
        _flag_row(
            f"Cluster {entry['cluster_id']}",
            n_researcher=int(entry["n_researcher"]),
            n_literature=int(entry["n_literature"]),
            total_literature=total_literature,
            researcher_share_threshold=researcher_share_threshold,
            literature_share_threshold=literature_share_threshold,
            underrepresented_max=underrepresented_max,
        )
        for entry in joint.clusters
    ]
    return _flags_result(_BASIS_JOINT, thresholds, rows)


def render_flags_markdown(flags: dict) -> str:
    if not flags.get("available"):
        return f"*Flags unavailable: {flags.get('reason', 'unknown reason')}.*"

    joint = flags.get("basis") == _BASIS_JOINT
    thresholds = flags["thresholds"]
    basis_sentence = (
        (
            "Researcher and literature passages were clustered **together**, so "
            "each row counts the researcher passages that are actually members "
            "of that shared cluster."
        )
        if joint
        else (
            "The corpora were clustered independently, so researcher passages "
            "were assigned to their nearest literature-cluster centroid (cosine)."
        )
    )
    member_word = "members" if joint else "assignments"
    lines = [
        (
            f"{basis_sentence} A cluster is **researcher-concentrated** when the "
            f"researcher share of its combined membership exceeds "
            f"{thresholds['researcher_share_threshold']:.0%} while the cluster holds under "
            f"{thresholds['literature_share_threshold']:.0%} of literature passages; "
            "clusters with "
            f"≤{thresholds['underrepresented_max_researcher_passages']} researcher "
            f"{member_word} are **underrepresented in researcher usage**."
        ),
        "",
        f"| {'Shared' if joint else 'Literature'} cluster | n lit | lit share "
        f"| n researcher {member_word} | researcher share | flags |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in flags["rows"]:
        flag_labels = []
        if row["researcher_concentrated"]:
            flag_labels.append("researcher-concentrated")
        if row["underrepresented_in_researcher_usage"]:
            flag_labels.append("underrepresented")
        lines.append(
            f"| {row['cluster']} | {row['n_literature']} | {row['literature_share']:.1%} "
            f"| {row['n_researcher_assigned']} | {row['researcher_share_of_combined']:.1%} "
            f"| {', '.join(flag_labels) if flag_labels else '—'} |"
        )
    lines.append("")
    lines.append(
        "*Flags are descriptive aids — they mark where researcher and literature "
        "usage diverge structurally, not whether that divergence is a problem.*"
    )
    return "\n".join(lines)
