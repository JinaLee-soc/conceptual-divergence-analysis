"""User-prompt assembly.

Builds the payload documented verbatim in the paper's appendix (and in the
second fenced block of prompts/register_assignment_v5.md): target term,
researcher theme-tag context, extraction notes, and the two pre-computed
cluster structures with their centroid-nearest representative passages.

The block *format* is the appendix's in both clustering modes. What differs is
what the two blocks are: under joint clustering they are two views of one
shared partition, which the EXTRACTION NOTES announce explicitly (mode line +
per-cluster composition) so the model knows that a cluster number means the
same region on both sides. Under independent clustering the notes say that
too, in the opposite direction.
"""

from __future__ import annotations

from cda.clustering.joint import composition_note
from cda.config import Config, ConfigError
from cda.corpus.timeframe import describe_timeframe
from cda.types import ClusterSummary, JointClustering, Passage, passage_theme_tags


def format_cluster_block(summary: ClusterSummary) -> str:
    """One corpus's cluster structure, exactly as the appendix template shows."""
    if not summary.used or not summary.clusters:
        return f"(Clustering unavailable: {summary.note})"

    lines = [
        f"Total passages clustered: {summary.n_total_passages} | "
        f"k={summary.n_clusters} | silhouette={summary.silhouette:.3f}",
        "",
    ]
    for entry in summary.clusters:
        lines.append(f"--- Cluster {entry['cluster_id']} (n={entry['n_passages']}) ---")
        for i, (text, source, role) in enumerate(
            zip(entry["representatives"], entry["sources"], entry["rep_roles"], strict=True), 1
        ):
            lines.append(f'Sample {i} [{role}] [{source}]: "{text}"')
            tag_sets = entry.get("rep_theme_tags", [])
            if i <= len(tag_sets) and tag_sets[i - 1]:
                tags = "; ".join(tag_sets[i - 1])
                lines.append(f"Theme tags for Sample {i} (source metadata): {tags}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _theme_line(theme) -> str:
    """One theme entry as a prompt line: a bare tag string or a {tag, description} mapping."""
    if isinstance(theme, str) and theme.strip():
        return f"  #{theme.strip().lstrip('#')}"
    if isinstance(theme, dict) and str(theme.get("tag", "")).strip():
        tag = str(theme["tag"]).strip().lstrip("#")
        description = str(theme.get("description", "")).strip()
        return f"  #{tag} — {description}" if description else f"  #{tag}"
    raise ConfigError(
        f"themes entry {theme!r} is neither a tag string nor a mapping with a 'tag'; "
        "fix or remove it."
    )


def configured_theme_lines(config: Config) -> list[str]:
    """The configured ``themes`` as prompt lines; raises ConfigError on a malformed entry.

    The pipeline calls this before any retrieval, so a typo such as
    ``themes: [2024]`` fails before extraction and clustering rather than at
    prompt assembly.
    """
    # get_list, not a bare read: `themes: evaluation-theory` (a string where a
    # list was meant) would otherwise iterate character by character and put
    # one-letter tags — "#e", "#v", "#a" — into the prompt.
    return [_theme_line(theme) for theme in config.get_list("themes", [])]


def _researcher_context(config: Config, passages: list[Passage] | None = None) -> str:
    tag_lines = configured_theme_lines(config)
    if not tag_lines:
        derived = set()
        for passage in passages or []:
            derived.update(passage_theme_tags(passage))
        tag_lines = [_theme_line(theme) for theme in sorted(derived)]
    if not tag_lines:
        return "(no theme tags provided)"
    return "Theme tags:\n" + "\n".join(tag_lines)


_RANKING_ANCHORS = {
    "term": (
        "cosine similarity to the target term's embedding (external anchor, "
        "independent of researcher and literature registers)"
    ),
    "similarity": "existing retrieval similarity scores",
}


def _saturation_note(saturation: dict) -> str:
    """The literature-selection note: the true literature count in every branch."""
    n_candidates = int(saturation["n_candidates"])
    n_selected = int(saturation["n_selected"])
    if n_candidates <= n_selected:
        return f"Literature corpus selection: all {n_selected} retrieved literature passages; no subsampling."
    ranking = str(saturation["ranking"]).strip().lower()
    if ranking not in _RANKING_ANCHORS:
        raise ValueError(f"Unknown saturation ranking {saturation['ranking']!r}.")
    reached = "reached" if saturation["saturation_reached"] else "NOT reached"
    return (
        f"Literature corpus selection: saturation-selected N*={n_selected} "
        f"from {n_candidates} candidates, ranked by {_RANKING_ANCHORS[ranking]}. "
        f"Saturation {reached} at ARI >= {saturation['ari_threshold']} for "
        f"{saturation['consecutive_increments']} consecutive increments."
    )


def _mode_note(joint: JointClustering | None) -> str:
    """State the clustering mode, and what cluster numbers mean under it."""
    if joint is None:
        return (
            "Clustering mode: INDEPENDENT — the researcher and literature corpora "
            "were partitioned separately. Cluster numbers are per-corpus labels "
            "and are NOT comparable across the two blocks below."
        )
    return (
        "Clustering mode: JOINT — researcher and literature passages were pooled "
        "into ONE spherical k-means partition "
        f"(joint k={joint.n_clusters}, silhouette={joint.silhouette:.3f} on the "
        "pooled passages). The two blocks below are two views of that single "
        "partition: the SAME cluster number denotes the SAME region of embedding "
        "space in both. A cluster missing from one block contains no passages "
        "from that corpus — treat that absence as evidence, not as missing data."
    )


def build_user_prompt(
    term: str,
    researcher_summary: ClusterSummary,
    literature_summary: ClusterSummary,
    saturation: dict,
    config: Config,
    *,
    joint: JointClustering | None = None,
    researcher_passages: list[Passage] | None = None,
    timeframe: dict | None = None,
) -> str:
    """Assemble the appendix-format payload for one run (see module docstring)."""
    notes = [
        f"Passage window mode: {config.get('retrieval.window_mode')}",
        _mode_note(joint),
        (
            f"Researcher clustering: k={researcher_summary.n_clusters}, "
            f"silhouette={researcher_summary.silhouette:.3f}, "
            f"N={researcher_summary.n_total_passages}. {researcher_summary.note}."
        ),
        (
            f"Literature clustering: k={literature_summary.n_clusters}, "
            f"silhouette={literature_summary.silhouette:.3f}, "
            f"N={literature_summary.n_total_passages}. {literature_summary.note}."
        ),
    ]
    if any(
        any(entry.get("rep_theme_tags", []))
        for summary in (researcher_summary, literature_summary)
        for entry in summary.clusters
    ):
        notes.append(
            "Per-sample theme tags are inherited from source metadata, not inferred "
            "sentence meanings. Run-level theme tags do not establish a passage association; "
            "a sample without listed tags has no observed theme association."
        )
    if joint is not None and joint.clusters:
        notes.append(composition_note(joint))
    if timeframe and (timeframe.get("window") or {}).get("active"):
        notes.append(describe_timeframe(timeframe))
    notes.append(_saturation_note(saturation))

    return (
        f'TARGET TERM: "{term}"\n\n'
        "RESEARCHER CONTEXT:\n"
        f"{_researcher_context(config, researcher_passages)}\n\n"
        "EXTRACTION NOTES:\n" + "\n".join(f"- {note}" for note in notes) + "\n\n"
        "RESEARCHER CLUSTER STRUCTURE (pre-computed from all researcher passages):\n"
        f"{format_cluster_block(researcher_summary)}\n\n"
        "LITERATURE PASSAGES (cluster summary):\n"
        f"{format_cluster_block(literature_summary)}\n\n"
        "Please perform two analyses (Polysemy Detection, "
        "Researcher vs. Literature Divergence) and return the structured "
        "CDA Report."
    )
