"""Computed divergence flags: thresholds and edge cases."""

import numpy as np

from cda.divergence.flags import compute_flags, render_flags_markdown
from tests.conftest import make_passages

FLAGS_CONFIG = {
    "researcher_share_threshold": 0.80,
    "literature_share_threshold": 0.10,
    "underrepresented_max_researcher_passages": 0,
}


def _unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=8)
    return [float(x) for x in (v / np.linalg.norm(v))]


def build_corpora(n_big=45, n_small=2, n_res_on_small=9):
    """Literature: one big cluster (0) and one tiny cluster (1).

    Researcher passages sit on the tiny cluster → it is researcher-
    concentrated (small literature share, high researcher share) while the
    big cluster gets no researcher passages → underrepresented.
    """
    anchor_big, anchor_small = _unit(1), _unit(2)
    literature = make_passages([f"lit {i}" for i in range(n_big + n_small)])
    for i, p in enumerate(literature):
        if i < n_big:
            p.cluster_id = "Cluster 0"
            p.embedding = anchor_big
        else:
            p.cluster_id = "Cluster 1"
            p.embedding = anchor_small
    researcher = make_passages([f"res {i}" for i in range(n_res_on_small)], source="mine.docx")
    for p in researcher:
        p.embedding = anchor_small
    return researcher, literature


def test_concentrated_and_underrepresented_flags():
    researcher, literature = build_corpora()
    flags = compute_flags(researcher, literature, FLAGS_CONFIG)
    assert flags["available"]
    # Tiny cluster: researcher share 9/11 ≈ 0.82 > 0.80; lit share 2/47 < 0.10.
    assert flags["concentrated"] == ["Cluster 1"]
    # Big cluster receives zero researcher assignments.
    assert flags["underrepresented"] == ["Cluster 0"]


def test_thresholds_must_be_exceeded_not_met():
    # researcher share exactly 0.80 (8 res vs 2 lit) → NOT flagged (> required).
    researcher, literature = build_corpora(n_res_on_small=8)
    flags = compute_flags(researcher, literature, FLAGS_CONFIG)
    row = next(r for r in flags["rows"] if r["cluster"] == "Cluster 1")
    assert row["researcher_share_of_combined"] == 0.8
    assert not row["researcher_concentrated"]


def test_literature_share_condition_matters():
    # Same 9-vs-2 researcher concentration, but the "small" cluster now holds
    # 30% of literature passages → not flagged despite high researcher share.
    researcher, literature = build_corpora(n_big=5, n_small=2, n_res_on_small=9)
    flags = compute_flags(researcher, literature, FLAGS_CONFIG)
    row = next(r for r in flags["rows"] if r["cluster"] == "Cluster 1")
    assert row["researcher_share_of_combined"] > 0.8
    assert not row["researcher_concentrated"]


def test_unavailable_without_two_literature_clusters():
    researcher, literature = build_corpora()
    for p in literature:
        p.cluster_id = "Cluster 0"
    flags = compute_flags(researcher, literature, FLAGS_CONFIG)
    assert not flags["available"]
    assert "fewer than two" in flags["reason"]


def test_unavailable_without_researcher_embeddings():
    researcher, literature = build_corpora()
    for p in researcher:
        p.embedding = None
    flags = compute_flags(researcher, literature, FLAGS_CONFIG)
    assert not flags["available"]


def test_markdown_rendering_lists_flags():
    researcher, literature = build_corpora()
    text = render_flags_markdown(compute_flags(researcher, literature, FLAGS_CONFIG))
    assert "researcher-concentrated" in text
    assert "| Cluster 1 |" in text
    assert "descriptive aids" in text
