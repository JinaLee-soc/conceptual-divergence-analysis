"""The validation report: verdict trail, agreement, curation history.

Rendered from the session alone, so it can be regenerated at any point in
the session's life (after the initial run, after each verdict batch, after
curation, after endorsement) and always reflects the full state. The
epistemic framing is fixed in the boilerplate: evidence is the researcher's
BLIND verdicts and their agreement with independent agent readings of the
ORIGINAL clustering; curation is repair, quarantined from evidence.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cda import __version__
from cda.config import Config
from cda.outputs import write_text_atomic
from cda.validation.metrics import (
    blind_verdicts,
    curation_deltas,
    to_agreement_category,
    verdict_agreement,
)
from cda.validation.session import is_revealed, partition_for


def _fmt(value) -> str:
    return "n/a" if value is None else str(value)


def _fmt_silhouettes(by_cluster: dict | None) -> str:
    """Per-cluster silhouettes as "cluster 0: 0.1234, cluster 1: n/a".

    A cluster has no silhouette (None) when the edit left fewer than two
    clusters, or a cluster of one passage.
    """
    if not by_cluster:
        return "n/a"
    return ", ".join(
        f"cluster {cluster}: {'n/a' if value is None else f'{float(value):.4f}'}"
        for cluster, value in by_cluster.items()
    )


def _verdict_rows(session: dict, side: str) -> str:
    lines = [
        "| cluster | passages | researcher verdict (blind) | agent judgment | agree? | revised after reveal |",
        "|---|---:|---|---|---|---|",
    ]
    partition = partition_for(session, side, 0)
    blind = blind_verdicts(session, side, 0)
    revised = {
        verdict["cluster_id"]: verdict
        for verdict in session["verdicts"]
        if verdict["side"] == side and verdict["revision"] == 0 and verdict["revised_after_reveal"]
    }
    readings: dict[int, list[dict]] = {}
    for reading in session["readings"]:
        if reading["side"] == side and reading["revision"] == 0:
            readings.setdefault(reading["cluster_id"], []).append(reading)

    for cluster in partition["clusters"]:
        cluster_id = cluster["cluster_id"]
        verdict = blind.get(cluster_id)
        verdict_text = "n/a"
        if verdict:
            verdict_text = verdict["verdict"]
            if verdict["verdict"] == "overlaps":
                verdict_text = f"overlaps with cluster {verdict['overlaps_with']}"
        judgments = [r["judgment"] for r in readings.get(cluster_id, []) if r["valid"]]
        # A reading is printed only after its reveal is logged in the session,
        # so the report can never show it without the trail recording it.
        shown = bool(judgments) and is_revealed(session, side, cluster_id, 0)
        if shown:
            judgment_text = ", ".join(judgments)
        elif judgments and verdict:
            judgment_text = "not yet revealed"
        elif judgments:
            judgment_text = "withheld until you record a verdict"
        else:
            judgment_text = "n/a"
        agree = "n/a"
        if verdict and shown:
            human = to_agreement_category(verdict["verdict"])
            agree = "yes" if all(j == human for j in judgments) else "no"
        lines.append(
            f"| Cluster {cluster_id} | {cluster['n_passages']} | {verdict_text} "
            f"| {judgment_text} | {agree} "
            f"| {'yes: ' + revised[cluster_id]['verdict'] if cluster_id in revised else 'no'} |"
        )
    return "\n".join(lines)


def _agreement_section(session: dict) -> list[str]:
    agreement = verdict_agreement(session, revision=0)
    lines: list[str] = []
    any_stats = False
    for side in ("researcher", "literature"):
        for model, stats in agreement["sides"].get(side, {}).items():
            any_stats = True
            unrevealed = sorted(
                {
                    pair["cluster_id"]
                    for pair in stats["pairs"]
                    if not is_revealed(session, side, pair["cluster_id"], 0)
                }
            )
            # The statistic over n clusters encodes those clusters' readings
            # (n=1 is the reading itself), so it waits for their reveals.
            if unrevealed:
                lines.append(
                    f"- {side} vs `{model}`: withheld until the readings of "
                    f"cluster(s) {', '.join(str(c) for c in unrevealed)} are revealed"
                )
                continue
            percent = (
                f"{stats['percent_agreement'] * 100:.0f}%"
                if stats["percent_agreement"] is not None
                else "n/a"
            )
            lines.append(
                f"- {side} vs `{model}`: {percent} agreement over {stats['n']} cluster(s), "
                f"Cohen's kappa {_fmt(stats['kappa'])}"
            )
    if not any_stats:
        return [
            "*No agreement statistics yet: they appear once clusters have both "
            "a blind verdict and a valid agent reading.*"
        ]
    lines.append(
        "\n*Kappa over a handful of clusters is a small-n statistic; read it "
        "with the raw agreement and the verdict table, never alone. Only "
        "blind verdicts enter these numbers; post-reveal revisions are "
        "excluded by construction.*"
    )
    if len(session["params"].get("readers", [])) < 2:
        lines.append(
            "\n*Single-reader caveat: all agent readings come from one model, "
            "so agreement reflects that model's reading, not a consensus. "
            "Configure validation.readers with a second provider for stronger "
            "evidence.*"
        )
    return lines


def _edits_section(session: dict) -> list[str]:
    if not session["edits"]:
        return ["*No curation edits.*"]
    lines = []
    deltas = {d["edit_id"]: d for d in curation_deltas(session)}
    for edit in session["edits"]:
        if edit["kind"] == "recut":
            kept = edit.get("clusters_kept", [])
            lines.append(
                f"- Edit {edit['edit_id']}: recut (dendrogram cutoff moved) to "
                f"k={edit['k']}; {len(kept)} cluster(s) unchanged with verdicts "
                f"kept, the rest re-entered the inspection queue. No agent involved."
            )
            continue
        advice = edit.get("agent_advice") or {}
        lines.append(
            f"- Edit {edit['edit_id']}: {edit['kind']} on {edit['side']} "
            f"clusters {edit['clusters']}; agent recommended "
            f"{_fmt(advice.get('recommendation'))} "
            f'("{_fmt(advice.get("argument"))}"); decision: {edit["status"]}'
            + (
                f" ({edit.get('applied_proposal')})"
                if edit["status"] == "applied" and edit.get("applied_proposal")
                else ""
            )
        )
        if edit["edit_id"] in deltas:
            delta = deltas[edit["edit_id"]]
            lines.append(
                f"  - silhouette before: {_fmt_silhouettes(delta['silhouette_before'])} | "
                f"after: {_fmt_silhouettes(delta['silhouette_after'])}"
            )
    return lines


def render_validation_report(session: dict, config: Config, now: datetime) -> str:
    endorsement = session.get("endorsement")
    sections = [
        f"<!-- generated: {now.isoformat()} | cda validate (EXPERIMENTAL) "
        f"| cda: v{__version__} -->\n",
        f'# Clustering Validation: "{session["term"]}"',
        "\n**EXPERIMENTAL** (post-paper addition; researcher-led). The validity "
        "evidence in this report is the researcher's BLIND per-cluster verdicts "
        "and their agreement with independent LLM cluster readings, both "
        "referring to the ORIGINAL clustering exactly as `cda analyze` computes "
        "it. Curation edits (merges/splits) are documented repair, not "
        "evidence: agreement is never recomputed on structures the curation "
        "itself produced.",
        "\n## Protocol\n",
        "Each cluster's verdict (coherent / mixed / overlaps / artifact) was "
        "committed BEFORE the agent's reading of the same passages was "
        "revealed; the session software refuses to reveal a reading earlier. "
        "Post-reveal revisions are recorded separately and excluded from "
        "agreement statistics.",
    ]
    for side in ("researcher", "literature"):
        info = session["sides"].get(side)
        if info is None:
            continue
        sections.append(f"\n## {side.capitalize()} clusters\n")
        if not info.get("validated"):
            sections.append(f"*Not validated: {info.get('note')}*")
            continue
        if info.get("note"):
            sections.append(f"*Note: {info['note']}*\n")
        sections.append(_verdict_rows(session, side))
    sections.append("\n## Researcher vs agent agreement (blind verdicts only)\n")
    sections.extend(_agreement_section(session))
    sections.append("\n## Curation history\n")
    sections.extend(_edits_section(session))
    sections.append("\n## Status\n")
    if endorsement:
        sections.append(
            f"**Endorsed** at {endorsement['at']} (revisions: {endorsement['revisions']})."
            + (f" Note: {endorsement['note']}" if endorsement.get("note") else "")
        )
    else:
        sections.append("Session open: not yet endorsed.")
    # The embedding model the RUN used, from the session record: this report
    # is re-rendered throughout the session's life, and the configured model
    # can change in between — the run parameters must describe the run, not
    # the current config. Sessions from before the model was recorded have
    # only the configured value to fall back on.
    embedding_model = session["params"].get("embedding_model") or config.get("embedding.model")
    sections.append(
        "\n## Run Parameters\n"
        f"- Seed: {session['params'].get('seed')} "
        f"| Passages shown per cluster: {session['params'].get('passages_shown_per_cluster')} "
        f"| Readers: {', '.join(session['params'].get('readers', [])) or 'n/a'}"
        f"\n- Embedding: `{embedding_model}`\n"
    )
    return "\n".join(sections)


def write_validation_outputs(
    session_dir: Path, session: dict, report_text: str, now: datetime
) -> dict[str, str]:
    """Write (or refresh) the session's report; the path is stable within a
    session so re-renders update one file rather than accumulating copies."""
    del now  # the session dir already carries the session timestamp
    session_dir = Path(session_dir)
    report_path = session_dir / f"{session_dir.name}.md"
    # Atomic like session.json: this file is rewritten on every verdict, and
    # a crash mid-write must leave the previous report, not half of one. The
    # shared helper is used rather than a local copy so improvements to it
    # (file permissions, fsync) reach this writer too.
    write_text_atomic(report_path, report_text)
    return {"report_file": str(report_path)}
