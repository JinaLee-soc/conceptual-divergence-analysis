"""Agreement statistics for validation sessions. Pure functions, no I/O.

The evidence a validation session produces is the researcher's blind verdict
per cluster and its agreement with independent LLM cluster readings. Both
are mapped onto a shared three-category scale before comparison:

- researcher ``coherent``  -> ``coherent``
- researcher ``mixed``     -> ``mixed``
- researcher ``artifact``  -> ``artifact``
- researcher ``overlaps``  -> ``coherent``  (an "overlaps with cluster N"
  verdict is a between-cluster judgment: the cluster itself reads as one
  usage, it just duplicates another. The reader agent sees one cluster at a
  time and has no basis to judge overlap, so the internally-coherent part is
  what the two judgments share.)

Only the ORIGINAL blind verdict enters agreement (post-reveal revisions are
reported separately): agreement between a human anchored by the agent and
the agent would be circular.

Kappa over a handful of clusters is a small-n statistic; callers must
present it with n and alongside raw percent agreement, never alone.
"""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score

AGREEMENT_SCALE = ("coherent", "mixed", "artifact")


def to_agreement_category(verdict: str) -> str | None:
    """Map a researcher verdict or reader judgment onto the shared scale."""
    if verdict in AGREEMENT_SCALE:
        return verdict
    if verdict == "overlaps":
        return "coherent"
    return None


def percent_agreement(a: list[str], b: list[str]) -> float | None:
    """Share of positions where the two raters chose the same category."""
    if len(a) != len(b):
        raise ValueError(f"rating lists differ in length ({len(a)} vs {len(b)})")
    if not a:
        return None
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa between two equal-length category sequences.

    None when kappa is undefined: empty input, or chance agreement of 1
    (both raters use one identical category).
    """
    if len(a) != len(b):
        raise ValueError(f"rating lists differ in length ({len(a)} vs {len(b)})")
    if not a or (len(set(a)) == 1 and set(a) == set(b)):
        return None
    return float(cohen_kappa_score(a, b, labels=sorted(set(a) | set(b))))


def blind_verdicts(session: dict, side: str, revision: int) -> dict[int, dict]:
    """The original blind verdict per cluster (first, never post-reveal)."""
    result: dict[int, dict] = {}
    for verdict in session["verdicts"]:
        if verdict["side"] != side or verdict["revision"] != revision:
            continue
        if verdict.get("revised_after_reveal"):
            continue
        result.setdefault(verdict["cluster_id"], verdict)
    return result


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def verdict_agreement(session: dict, revision: int = 0) -> dict:
    """Researcher-vs-reader agreement over clusters, per side and reader.

    Only clusters with both a blind verdict and a valid reading count.
    Returns per (side, reader): {"n", "percent_agreement", "kappa", "pairs"}.
    """
    agreements: dict = {"revision": revision, "sides": {}}
    for side in ("researcher", "literature"):
        verdicts = blind_verdicts(session, side, revision)
        readers: dict[str, dict] = {}
        for reading in session["readings"]:
            if reading["side"] != side or reading["revision"] != revision or not reading["valid"]:
                continue
            verdict = verdicts.get(reading["cluster_id"])
            if verdict is None:
                continue
            human = to_agreement_category(verdict["verdict"])
            agent = to_agreement_category(reading["judgment"])
            if human is None or agent is None:
                continue
            entry = readers.setdefault(reading["model"], {"human": [], "agent": [], "pairs": []})
            entry["human"].append(human)
            entry["agent"].append(agent)
            entry["pairs"].append(
                {"cluster_id": reading["cluster_id"], "human": human, "agent": agent}
            )
        agreements["sides"][side] = {
            model: {
                "n": len(entry["human"]),
                "percent_agreement": _rounded(percent_agreement(entry["human"], entry["agent"])),
                "kappa": _rounded(cohen_kappa(entry["human"], entry["agent"])),
                "pairs": entry["pairs"],
            }
            for model, entry in readers.items()
        }
    return agreements


def curation_deltas(session: dict) -> list[dict]:
    """Per-edit silhouette deltas, read from the audit trail (computed at
    apply time, when embeddings were in hand)."""
    deltas = []
    for edit in session["edits"]:
        if edit.get("status") != "applied":
            continue
        deltas.append(
            {
                "edit_id": edit["edit_id"],
                "kind": edit["kind"],
                "side": edit["side"],
                "silhouette_before": edit.get("silhouette_before"),
                "silhouette_after": edit.get("silhouette_after"),
            }
        )
    return deltas
