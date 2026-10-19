"""Timestamped markdown report rendering.

Preserves the working pipeline's report format: an HTML-comment metadata
header, the LLM's structured report body, then computed sections (divergence
flags, cluster–register audit, run parameters). Reports are written as
``drift_<term>_<timestamp>.md`` (second precision) next to the assembled
prompt (``drift_<term>_<timestamp>_prompt.md``), the system prompt actually
used (``..._system.md``), and diagnostics of the same run, which share that
timestamp — so a same-day rerun never overwrites an earlier report and a
directory of reports supports longitudinal comparison.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from cda import __version__
from cda.config import Config
from cda.corpus.timeframe import describe_timeframe
from cda.divergence.flags import render_flags_markdown
from cda.extraction.windows import safe_term_slug
from cda.outputs import unique_run_basename, write_text_immutable
from cda.registers.audit import render_audit_markdown
from cda.registers.grounding import render_grounding_markdown
from cda.types import ClusterSummary, JointClustering

# Structural boundary between the model's reply and the computed sections.
END_OF_LLM_BODY = "<!-- cda: end of LLM body; computed sections follow -->"


def silhouette_text(clustering: ClusterSummary | JointClustering) -> str:
    """The silhouette to three decimals, or "n/a" when no valid partition was found.

    A clustering without a partition carries a placeholder silhouette of 0.0,
    which would otherwise read as a measured (if weak) separation.
    """
    return f"{clustering.silhouette:.3f}" if clustering.used else "n/a"


def _header(
    model: str,
    stats: dict,
    researcher: ClusterSummary,
    literature: ClusterSummary,
    joint: JointClustering | None,
    now: datetime,
) -> str:
    mode_line = (
        f"<!-- clustering_mode: joint, joint_k={joint.n_clusters}, "
        f"joint_silhouette={silhouette_text(joint)} -->\n"
        if joint is not None
        else "<!-- clustering_mode: independent -->\n"
    )
    return (
        f"<!-- generated: {now.isoformat()} | model: {model} | cda: v{__version__} -->\n"
        f"<!-- chunks_total: {stats.get('chunks_total', 0)}, "
        f"chunks_with_term_window: {stats.get('chunks_with_term_window', 0)}, "
        f"researcher_passages: {stats.get('researcher_passages', 0)}, "
        f"literature_passages: {stats.get('literature_passages', 0)} -->\n"
        f"{mode_line}"
        f"<!-- researcher_clusters: k={researcher.n_clusters}, "
        f"sil={silhouette_text(researcher)}, N={researcher.n_total_passages} | "
        f"literature_clusters: k={literature.n_clusters}, "
        f"sil={silhouette_text(literature)}, N={literature.n_total_passages} -->\n\n"
    )


def _distance_scale_legend(config: Config) -> str:
    scale = config.section("distance_scale")

    def order(kv):
        # A non-numeric key in a user's config must not crash the renderer —
        # by the time this line runs the LLM call has already been paid for.
        try:
            return (0, int(kv[0]), "")
        except (TypeError, ValueError):
            return (1, 0, str(kv[0]))

    parts = [f"{key} = {value}" for key, value in sorted(scale.items(), key=order)]
    return "Semantic distance scale: " + "; ".join(parts) + "."


def _classification_reasons(stats: dict) -> str:
    reasons = stats.get("classification_reasons") or {}
    if not reasons:
        return "*No corpus chunks classified (drafts-only run).*"
    # Classification is per retrieved chunk (all term-sentences from a chunk
    # inherit its researcher/literature label), so the counts are chunks, not
    # passages — labeled accordingly to avoid reading them as passage totals.
    lines = [
        "*Counts are retrieved chunks; every term-sentence from a chunk "
        "inherits the chunk's signal.*",
        "",
        "| MyWork signal | chunks |",
        "|---|---:|",
    ]
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count} |")
    return "\n".join(lines)


def _saturation_line(saturation: dict) -> str:
    if not saturation.get("enabled"):
        return (
            f"Saturation selection: disabled. Full literature pool clustered "
            f"({saturation['n_candidates']} passages, no subsampling)."
        )
    if saturation.get("scope") == "prompt":
        return (
            f"Saturation scope: prompt. Full literature pool clustered "
            f"({saturation['n_candidates']} passages, no subsampling); "
            "the prompt is bounded by clusters x representatives."
        )
    if not saturation.get("ran"):
        # The pipeline records ran=False when the pool never reached the sweep
        # start (every candidate used) — distinct from a sweep that ran but
        # never converged, which falls through to the caution line below.
        return (
            f"Saturation selection: pool of {saturation.get('n_candidates', 0)} candidates "
            "below the sweep start; all passages used."
        )
    selected = saturation.get("n_selected", 0)
    candidates = saturation.get("n_candidates", 0)
    fraction = selected / candidates if candidates else 0.0
    history = saturation.get("history") or []
    evaluated = history[-1].get("n") if history else None
    stopped_early = (
        ("yes" if evaluated < min(candidates, saturation.get("max_passages", candidates)) else "no")
        if saturation.get("saturation_reached", False) and evaluated is not None
        else (
            "unknown (evaluation history unavailable)"
            if saturation.get("saturation_reached", False)
            else "no"
        )
    )
    selection_note = (
        f" Selected fraction: {selected}/{candidates} ({fraction:.1%}). "
        f"ARI stopping before the pool/cap was exhausted: {stopped_early}. "
        "Stability does not establish representativeness; review the retained passages."
    )
    reached = saturation.get("saturation_reached", False)
    status = "reached" if reached else "**NOT reached: interpret with caution**"
    return (
        f"Saturation selection: N*={saturation.get('n_selected')} of "
        f"{saturation.get('n_candidates')} candidates; saturation {status} "
        f"(ARI ≥ {saturation.get('ari_threshold')} for "
        f"{saturation.get('consecutive_increments', 2)} consecutive increments, "
        f"cap {saturation.get('max_passages')})." + selection_note
    )


def _truncate(text: str, limit: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_joint_composition_markdown(joint: JointClustering) -> str:
    """Where the researcher's passages sit inside the shared clusters.

    The point of joint clustering made readable: one row per shared cluster,
    with the researcher's own most cluster-typical sentence beside the counts,
    so a cluster the researcher never enters and a cluster only the researcher
    occupies are both visible at a glance.
    """
    if not joint.used or not joint.clusters:
        return f"*Joint clustering unavailable: {joint.note}.*"

    lines = [
        (
            f"One spherical k-means partition over {joint.n_researcher} of your "
            f"passages and {joint.n_literature} literature passages pooled "
            f"(k={joint.n_clusters}, silhouette={joint.silhouette:.3f} on the pooled "
            "passages). Each row is a shared cluster: the same region of embedding "
            "space, seen from both sides."
        ),
        "",
        "| Shared cluster | your passages | literature | your share | your most typical passage |",
        "|---|---:|---:|---:|---|",
    ]
    for entry in joint.clusters:
        reps = entry["researcher"]["representatives"]
        typical = f'"{_truncate(reps[0])}"' if reps else "— (none of your passages here)"
        lines.append(
            f"| Cluster {entry['cluster_id']} | {entry['n_researcher']} "
            f"| {entry['n_literature']} | {entry['researcher_share']:.1%} | {typical} |"
        )
    absent = [e["cluster_id"] for e in joint.clusters if not e["n_researcher"]]
    only_researcher = [e["cluster_id"] for e in joint.clusters if not e["n_literature"]]
    lines.append("")
    if absent:
        lines.append(
            "- Clusters with **none of your passages**: "
            + ", ".join(f"Cluster {c}" for c in absent)
            + " — literature usage you have not taken up."
        )
    if only_researcher:
        lines.append(
            "- Clusters with **no literature passages**: "
            + ", ".join(f"Cluster {c}" for c in only_researcher)
            + " — usage of yours that the corpus does not share."
        )
    if not absent and not only_researcher:
        lines.append("- Every shared cluster contains passages from both sides.")
    return "\n".join(lines)


def render_stability_markdown(stability: dict | None) -> str:
    """How much this run's k would vary under a different passage order.

    A canonical run is always reproducible (same passages, same order, same
    k, every time) — that is a different claim from the result being
    *robust*. Renders unconditionally, matching the Quotation Grounding /
    Cluster-Register Audit convention: a 100% match is worth stating too.
    """
    if stability is None:
        return "*Stability check skipped (dry run).*"
    if not stability.get("enabled"):
        reason = stability.get("reason")
        if reason:
            return f"*Stability check skipped: {reason}.*"
        return "*Stability check disabled (`clustering.stability_check.enabled: false`).*"

    def corpus_line(label: str, corpus: dict) -> str:
        no_partition = corpus.get("no_partition_trials", 0)
        if not corpus.get("canonical_partition_found", True):
            return (
                f"**{label}**: no valid partition was found in {no_partition} of "
                f"{corpus['trials']} orderings, including this run's own — there is "
                "no cluster structure here to call robust or fragile."
            )
        rate = corpus["k_match_rate"]
        if rate is None:
            return (
                f"**{label}**: k={corpus['canonical_k']} — no shuffled orderings were "
                "run, so nothing here says whether that k would survive a reordering."
            )
        caveat = (
            " — treat this run's k as one plausible partition among several nearby "
            "ones, not the term's one true register count."
            if rate < 1.0
            else " — every ordering agreed; this term's structure looks robust, not just reproducible."
        )
        if corpus.get("canonical_reproduced") is False:
            caveat = (
                " — **WARNING: re-clustering the canonical order did not reproduce "
                "this run's own k.** The clustering is not deterministic for this "
                "input, so these stability figures are unreliable."
            )
        k_range = (
            f"{corpus['k_seen'][0]}"
            if len(corpus["k_seen"]) == 1
            else f"{corpus['k_seen'][0]}–{corpus['k_seen'][-1]}"
        )
        missing = (
            f"; no valid partition was found in {no_partition} of {corpus['trials']} orderings"
            if no_partition
            else ""
        )
        return (
            f"**{label}**: k={corpus['canonical_k']} matched {rate:.0%} of "
            f"{corpus.get('shuffled_trials', corpus['trials'])} shuffled orderings "
            f"(k ranged {k_range}; "
            f"silhouette {corpus['silhouette_min']:.3f}–{corpus['silhouette_max']:.3f}"
            f"{missing})"
            f"{caveat}"
        )

    intro = (
        "The identical, already-selected passage set was re-clustered under "
        "independently shuffled orderings (no LLM call, no cost) to check whether "
        "this run's k is a stable answer or an artifact of which order the passages "
        "happened to be in. The count is `clustering.stability_check.trials`; "
        "`benchmarks/clustering_order_stability.py` runs the same check with a "
        "full per-trial table."
    )
    if stability.get("mode") == "joint":
        return "\n\n".join([intro, corpus_line("Joint partition", stability)])
    return "\n\n".join(
        [
            intro,
            corpus_line("Researcher corpus", stability["researcher"]),
            corpus_line("Literature corpus", stability["literature"]),
        ]
    )


def render_report(
    *,
    term: str,
    llm_body: str,
    model: str,
    stats: dict,
    researcher: ClusterSummary,
    literature: ClusterSummary,
    saturation: dict,
    flags: dict,
    audit: dict,
    config: Config,
    now: datetime,
    grounding: dict | None = None,
    joint: JointClustering | None = None,
    stability: dict | None = None,
) -> str:
    grounding_sections = (
        ["\n\n## Quotation Grounding (computed)\n", render_grounding_markdown(grounding)]
        if grounding is not None
        else []
    )
    joint_sections = (
        [
            "\n\n## Joint Clustering: where your passages sit (computed)\n",
            render_joint_composition_markdown(joint),
        ]
        if joint is not None
        else []
    )
    mode_line = (
        f"joint (researcher + literature pooled into one partition, k={joint.n_clusters})"
        if joint is not None
        else "independent (each corpus partitioned separately, legacy alternative)"
    )
    sections = [
        _header(model, stats, researcher, literature, joint, now),
        (
            "**Demo provenance:** Claude-authored demonstration response written in a Claude Code session; not an Anthropic API call.\n"
            if model == "claude-code-session"
            else ""
        ),
        llm_body.strip(),
        f"\n{END_OF_LLM_BODY}",
        "\n---\n",
        "\n\n## Clustering Stability (computed)\n",
        render_stability_markdown(stability),
        *joint_sections,
        "\n\n## Computed Divergence Flags\n",
        render_flags_markdown(flags),
        "\n\n## Cluster–Register Audit (computed)\n",
        render_audit_markdown(audit),
        *grounding_sections,
        "\n\n## Run Parameters\n",
        f"- Model: `{model}` | Embedding: `{config.get('embedding.model')}` "
        f"| Seed: {config.get('clustering.seed')} "
        f"| Representatives/cluster: {config.get('clustering.representatives_per_cluster')}",
        f"- Clustering mode: {mode_line}",
        f"- {_saturation_line(saturation)}",
        f"- {describe_timeframe(stats.get('timeframe') or {})}",
        f"- {_distance_scale_legend(config)}",
        "\n### MyWork classification\n",
        _classification_reasons(stats),
        "",
    ]
    return "\n".join(sections)


def write_outputs(
    *,
    out_dir: Path,
    term: str,
    report_text: str | None,
    prompt_text: str,
    diagnostics: dict | None,
    now: datetime,
    base: str | None = None,
    system_prompt_text: str | None = None,
    provenance: dict | None = None,
    llm_response_text: str | None = None,
) -> dict[str, str]:
    """Write the report (live runs), the assembled prompt, and diagnostics.

    Returns the paths written plus the run's shared ``base`` name. The prompt
    is always saved: being able to inspect exactly what was (or would be)
    sent is the pipeline's human-in-the-loop mechanism.

    ``system_prompt_text`` is saved beside it as ``{base}_system.md``: the
    system prompt is loaded from a prompt file that can change between an
    analysis and a later ``cda verify``, and verify re-judging under a
    *different* system prompt would show up as model disagreement. Persisting
    the exact text used lets verify replay the run as it actually ran.

    ``llm_response_text`` is the model's raw reply, saved as
    ``{base}_llm_response.txt`` before any post-processing touches it.

    The live pipeline calls this three times — before the LLM call (prompt and
    diagnostics, so a failed or interrupted call loses nothing), right after
    it (the raw reply), and after post-processing (the report). Passing ``base`` back joins the second call to the
    first run's files. Existing files must match byte-for-byte; conflicting
    content is refused before any new artifact is added.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_term_slug(term)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    if base is None:
        # The files of one run share a basename; the suffix keeps two runs
        # that finish within the same second (e.g. GUI and CLI side by side)
        # from silently replacing each other's report — the longitudinal
        # comparison the reports support depends on earlier files surviving.
        base = unique_run_basename(
            out_dir,
            f"drift_{slug}_{timestamp}",
            # ".png" is the passage map (pipeline writes it as {base}.png):
            # it belongs to the run, so a basename is only free if it is free.
            (
                ".md",
                "_prompt.md",
                "_system.md",
                "_cluster_diagnostics.json",
                ".png",
                "_manifest.json",
                "_receipt.json",
                "_llm_response.txt",
            ),
        )
    if Path(base).name != base or base in {".", ".."}:
        raise ValueError("Run base must be a filename, not a path.")
    written: dict[str, str] = {"base": base}
    artifacts = {"prompt_file": ("_prompt.md", prompt_text)}
    if system_prompt_text is not None:
        artifacts["system_file"] = ("_system.md", system_prompt_text)
    if diagnostics is not None:
        artifacts["diagnostics_file"] = (
            "_cluster_diagnostics.json",
            json.dumps(diagnostics, indent=2, default=str),
        )
    if provenance is not None:
        artifacts["manifest_file"] = ("_manifest.json", json.dumps(provenance, indent=2))
    if llm_response_text is not None:
        artifacts["llm_response_file"] = ("_llm_response.txt", llm_response_text)
    if report_text is not None:
        artifacts["report_file"] = (".md", report_text)
    # Validate the whole set before adding any new files to an existing run.
    for suffix, text in artifacts.values():
        path = out_dir / f"{base}{suffix}"
        if path.exists() and path.read_bytes() != text.encode("utf-8"):
            raise FileExistsError(f"Refusing to change existing run artifact: {path}")
    for key, (suffix, text) in artifacts.items():
        path = out_dir / f"{base}{suffix}"
        write_text_immutable(path, text)
        written[key] = str(path)
    return written
