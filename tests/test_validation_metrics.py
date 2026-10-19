"""Agreement statistics: kappa correctness and the verdict-scale mapping."""

import pytest

from cda.validation.metrics import (
    cohen_kappa,
    curation_deltas,
    to_agreement_category,
    verdict_agreement,
)


def test_cohen_kappa_matches_sklearn():
    import random

    from sklearn.metrics import cohen_kappa_score

    rng = random.Random(42)
    categories = ["coherent", "mixed", "artifact"]
    a = [rng.choice(categories) for _ in range(60)]
    b = [rng.choice(categories) for _ in range(60)]
    assert cohen_kappa(a, b) == pytest.approx(float(cohen_kappa_score(a, b)), abs=1e-9)


def test_cohen_kappa_none_when_undefined():
    # Single shared category: chance agreement is 1, kappa undefined.
    assert cohen_kappa(["coherent"] * 5, ["coherent"] * 5) is None
    assert cohen_kappa([], []) is None


def test_cohen_kappa_perfect_and_length_check():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="differ in length"):
        cohen_kappa(["a"], ["a", "b"])


def test_verdict_scale_mapping():
    assert to_agreement_category("coherent") == "coherent"
    assert to_agreement_category("mixed") == "mixed"
    assert to_agreement_category("artifact") == "artifact"
    # A cluster that "overlaps with N" is internally coherent; the reader
    # sees one cluster at a time and cannot judge overlap.
    assert to_agreement_category("overlaps") == "coherent"
    assert to_agreement_category("nonsense") is None


def _session(verdicts, readings, edits=()):
    return {"verdicts": list(verdicts), "readings": list(readings), "edits": list(edits)}


def test_verdict_agreement_uses_only_blind_verdicts():
    session = _session(
        verdicts=[
            {"side": "literature", "cluster_id": 0, "revision": 0, "verdict": "mixed"},
            {
                "side": "literature",
                "cluster_id": 0,
                "revision": 0,
                "verdict": "coherent",
                "revised_after_reveal": True,
            },
            {"side": "literature", "cluster_id": 1, "revision": 0, "verdict": "coherent"},
        ],
        readings=[
            {
                "side": "literature",
                "cluster_id": 0,
                "revision": 0,
                "model": "m1",
                "judgment": "coherent",
                "valid": True,
            },
            {
                "side": "literature",
                "cluster_id": 1,
                "revision": 0,
                "model": "m1",
                "judgment": "coherent",
                "valid": True,
            },
        ],
    )
    stats = verdict_agreement(session)["sides"]["literature"]["m1"]
    # The blind verdict for cluster 0 was "mixed"; the post-reveal revision to
    # "coherent" must NOT count, so agreement is 1 of 2.
    assert stats["n"] == 2
    assert stats["percent_agreement"] == 0.5


def test_verdict_agreement_skips_invalid_readings_and_unverdicted_clusters():
    session = _session(
        verdicts=[{"side": "literature", "cluster_id": 0, "revision": 0, "verdict": "coherent"}],
        readings=[
            {
                "side": "literature",
                "cluster_id": 0,
                "revision": 0,
                "model": "m1",
                "judgment": "coherent",
                "valid": False,
            },
            {
                "side": "literature",
                "cluster_id": 1,
                "revision": 0,
                "model": "m1",
                "judgment": "mixed",
                "valid": True,
            },
        ],
    )
    assert verdict_agreement(session)["sides"]["literature"] == {}


def test_verdict_agreement_overlaps_counts_as_coherent():
    session = _session(
        verdicts=[
            {"side": "literature", "cluster_id": 0, "revision": 0, "verdict": "overlaps"},
        ],
        readings=[
            {
                "side": "literature",
                "cluster_id": 0,
                "revision": 0,
                "model": "m1",
                "judgment": "coherent",
                "valid": True,
            },
        ],
    )
    stats = verdict_agreement(session)["sides"]["literature"]["m1"]
    assert stats["n"] == 1 and stats["percent_agreement"] == 1.0
    assert stats["kappa"] is None  # single category: undefined, never fabricated


def test_curation_deltas_reads_applied_edits_only():
    session = _session(
        verdicts=[],
        readings=[],
        edits=[
            {
                "edit_id": 1,
                "kind": "split",
                "side": "literature",
                "status": "applied",
                "silhouette_before": {"0": 0.11},
                "silhouette_after": {"0a": 0.19, "0b": 0.22},
            },
            {"edit_id": 2, "kind": "merge", "side": "literature", "status": "rejected"},
        ],
    )
    deltas = curation_deltas(session)
    assert len(deltas) == 1 and deltas[0]["edit_id"] == 1
