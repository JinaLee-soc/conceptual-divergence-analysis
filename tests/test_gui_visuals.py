"""Chart builders for the GUI visual summary (offline; skipped without altair).

Altair ships with streamlit's [gui] extra — dev-only environments without it
skip this module, exactly like test_gui.py does for streamlit itself.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("altair")
pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
import visuals  # noqa: E402

from tests.conftest import make_passages  # noqa: E402


def clustered_passages():
    researcher = make_passages(
        [f"researcher gamma sentence {i}" for i in range(6)], source="draft.md"
    )
    literature = make_passages(
        [f"literature alpha sentence {i}" for i in range(10)], source="paper.pdf"
    )
    for i, passage in enumerate(researcher):
        passage.embedding = [1.0, 0.0, float(i) * 0.01]
        passage.cluster_id = f"Cluster {i % 2}"
    for i, passage in enumerate(literature):
        passage.embedding = [0.0, 1.0, float(i) * 0.01]
        passage.cluster_id = f"Cluster {i % 3}"
    return {"researcher": researcher, "literature": literature}


def test_passages_frame_projects_to_2d():
    frame = visuals._passages_frame(clustered_passages())
    assert frame is not None
    assert len(frame) == 16
    assert {"x", "y", "corpus", "cluster", "text"} <= set(frame.columns)
    assert 0 < frame.attrs["explained"] <= 1
    # Long passage text is truncated for tooltips.
    assert frame["text"].str.len().max() <= 141


def test_passages_frame_needs_enough_embeddings():
    passages = clustered_passages()
    for passage in passages["researcher"] + passages["literature"]:
        passage.embedding = None
    assert visuals._passages_frame(passages) is None


def test_passage_map_chart_spec():
    frame = visuals._passages_frame(clustered_passages())
    spec = visuals.passage_map_chart(frame).to_dict()
    assert spec["mark"]["type"] == "point"  # point (not circle) so shape renders
    assert spec["mark"]["filled"] is True
    assert spec["encoding"]["color"]["field"] == "corpus"
    assert spec["encoding"]["shape"]["field"] == "corpus"
    tooltip_fields = {t["field"] for t in spec["encoding"]["tooltip"]}
    assert {"text", "source", "cluster"} <= tooltip_fields


def test_cluster_size_chart_spec():
    frame = visuals._passages_frame(clustered_passages())
    spec = visuals.cluster_size_chart(frame).to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["row"]["field"] == "corpus"


def test_saturation_chart_threshold_rule():
    saturation = {
        "history": [
            {"n": 50, "ari_vs_prev": None, "k": 2, "silhouette": 0.2},
            {"n": 75, "ari_vs_prev": 0.9, "k": 2, "silhouette": 0.21},
            {"n": 100, "ari_vs_prev": 0.95, "k": 2, "silhouette": 0.22},
        ],
        "ari_threshold": 0.8,
    }
    spec = visuals.saturation_chart(saturation).to_dict()
    assert len(spec["layer"]) == 2  # ARI line + threshold rule
    assert visuals.saturation_chart({"history": []}) is None


def test_flags_quadrant_chart_flags_and_rules():
    flags = {
        "available": True,
        "thresholds": {"researcher_share_threshold": 0.80, "literature_share_threshold": 0.10},
        "rows": [
            {
                "cluster": "Cluster 0",
                "n_literature": 40,
                "literature_share": 0.8,
                "n_researcher_assigned": 0,
                "researcher_share_of_combined": 0.0,
                "researcher_concentrated": False,
                "underrepresented_in_researcher_usage": True,
            },
            {
                "cluster": "Cluster 1",
                "n_literature": 4,
                "literature_share": 0.08,
                "n_researcher_assigned": 30,
                "researcher_share_of_combined": 0.88,
                "researcher_concentrated": True,
                "underrepresented_in_researcher_usage": False,
            },
        ],
    }
    chart = visuals.flags_quadrant_chart(flags)
    spec = chart.to_dict()
    assert len(spec["layer"]) == 3  # points + two threshold rules
    assert visuals.flags_quadrant_chart({"available": False, "reason": "x"}) is None


def test_emergence_chart_builds_from_diagnostics():
    emergence = {
        "status": "ok",
        "date_source": "zotero",
        "final": {
            "n": 30,
            "k": 2,
            "silhouette": 0.2,
            "clusters": [
                {
                    "final_cluster": 0,
                    "birth_n": 10,
                    "growth": [
                        {"n": 10, "as_of": "2019-03-18", "size": 6},
                        {"n": 20, "as_of": "2019-12-02", "size": 9},
                        {"n": 30, "as_of": "2020-08-24", "size": 14},
                    ],
                },
                {
                    "final_cluster": 1,
                    "birth_n": 20,
                    "growth": [
                        {"n": 10, "as_of": "2019-03-18", "size": 0},
                        {"n": 20, "as_of": "2019-12-02", "size": 5},
                        {"n": 30, "as_of": "2020-08-24", "size": 16},
                    ],
                },
            ],
        },
    }
    spec = visuals.emergence_chart(emergence).to_dict()
    assert len(spec["layer"]) == 2  # growth lines + birth diamonds
    encoding = spec["layer"][0]["encoding"]
    assert encoding["x"]["field"] == "as_of"
    assert encoding["x"]["type"] == "temporal"
    assert encoding["color"]["field"] == "cluster"


def _joint():
    from cda.types import JointClustering

    return JointClustering(
        used=True,
        n_clusters=3,
        silhouette=0.18,
        note="[joint] Embedding-level cluster separation detected",
        n_researcher=6,
        n_literature=10,
        clusters=[
            {
                "cluster_id": 0,
                "n_passages": 8,
                "n_researcher": 2,
                "n_literature": 6,
                "researcher_share": 0.25,
                "literature_share": 0.6,
                "researcher": {"representatives": ["mine here"], "sources": ["draft.md"]},
            },
            {
                "cluster_id": 1,
                "n_passages": 4,
                "n_researcher": 4,
                "n_literature": 0,
                "researcher_share": 1.0,
                "literature_share": 0.0,
                "researcher": {"representatives": ["only mine"], "sources": ["draft.md"]},
            },
            {
                "cluster_id": 2,
                "n_passages": 4,
                "n_researcher": 0,
                "n_literature": 4,
                "researcher_share": 0.0,
                "literature_share": 0.4,
                "researcher": {"representatives": [], "sources": []},
            },
        ],
    )


def test_joint_cluster_map_layers_literature_under_researcher():
    frame = visuals._passages_frame(clustered_passages())
    spec = visuals.joint_cluster_map_chart(frame).to_dict()
    assert len(spec["layer"]) == 2  # faint literature, then the researcher on top
    literature, researcher = spec["layer"]
    # Colour is the shared cluster (not the corpus) — that is the whole point.
    assert literature["encoding"]["color"]["field"] == "cluster"
    assert researcher["encoding"]["color"]["field"] == "cluster"
    # The researcher's own passages are drawn larger and outlined.
    assert researcher["mark"]["size"] > literature["mark"]["size"]
    assert researcher["mark"]["stroke"] == "black"


def test_joint_composition_chart_normalizes_and_sorts_by_researcher_share():
    spec = visuals.joint_composition_chart(_joint()).to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["x"]["stack"] == "normalize"
    assert spec["encoding"]["color"]["field"] == "corpus"
    # Sorted by researcher share: the researcher-only cluster first, the
    # literature-only cluster last.
    assert spec["encoding"]["y"]["sort"] == ["Cluster 1", "Cluster 0", "Cluster 2"]
    # Two rows per cluster (one per corpus).
    assert len(spec["datasets"][spec["data"]["name"]]) == 6


def test_joint_composition_chart_none_without_a_joint_partition():
    from cda.types import JointClustering

    assert visuals.joint_composition_chart(None) is None
    unused = JointClustering(
        used=False, n_clusters=1, silhouette=0.0, note="too few", n_researcher=1, n_literature=0
    )
    assert visuals.joint_composition_chart(unused) is None


def test_emergence_chart_none_when_not_ok_or_flat():
    assert visuals.emergence_chart({"status": "skipped", "reason": "x"}) is None
    single = {
        "status": "ok",
        "final": {
            "n": 10,
            "k": 1,
            "silhouette": 0.1,
            "clusters": [
                {
                    "final_cluster": 0,
                    "birth_n": 10,
                    "growth": [{"n": 10, "as_of": "2019-03-18", "size": 10}],
                }
            ],
        },
    }
    assert visuals.emergence_chart(single) is None


# --- stability note -----------------------------------------------


class _FakeSt:
    """Minimal Streamlit stand-in recording which surface each call used."""

    def __init__(self):
        self.calls = []

    def warning(self, message, icon=None):
        self.calls.append(("warning", message))

    def caption(self, message):
        self.calls.append(("caption", message))


def test_stability_note_warns_when_k_is_unstable():
    st = _FakeSt()
    visuals.render_stability_note(
        st,
        {
            "enabled": True,
            "mode": "joint",
            "canonical_k": 11,
            "trials": 5,
            "k_match_rate": 0.2,
            "k_seen": [2, 3, 11],
            "silhouette_min": 0.08,
            "silhouette_max": 0.18,
        },
    )
    kind, message = st.calls[0]
    assert kind == "warning"  # a shaky k must not read as a footnote
    assert "k=11" in message and "20%" in message
    assert "2, 3, 11" in message
    # The denominator is the shuffled orderings, not the total trials: the
    # canonical re-run is a determinism check, and saying "of 5" here would
    # contradict the written report, which says "of 4 shuffled orderings".
    assert "4 independently shuffled" in message
    assert "of 5" not in message


def test_stability_note_is_a_quiet_caption_when_fully_stable():
    st = _FakeSt()
    visuals.render_stability_note(
        st,
        {
            "enabled": True,
            "mode": "joint",
            "canonical_k": 2,
            "trials": 5,
            "k_match_rate": 1.0,
            "k_seen": [2],
            "silhouette_min": 0.34,
            "silhouette_max": 0.34,
        },
    )
    kind, message = st.calls[0]
    assert kind == "caption"
    assert "robust" in message


def test_stability_note_covers_both_corpora_in_independent_mode():
    st = _FakeSt()
    visuals.render_stability_note(
        st,
        {
            "enabled": True,
            "mode": "independent",
            "trials": 5,
            "researcher": {
                "canonical_k": 2,
                "trials": 5,
                "k_match_rate": 1.0,
                "k_seen": [2],
                "silhouette_min": 0.1,
                "silhouette_max": 0.1,
            },
            "literature": {
                "canonical_k": 4,
                "trials": 5,
                "k_match_rate": 0.4,
                "k_seen": [2, 4],
                "silhouette_min": 0.08,
                "silhouette_max": 0.12,
            },
        },
    )
    assert [kind for kind, _ in st.calls] == ["caption", "warning"]
    assert "researcher k=2" in st.calls[0][1]
    assert "literature k=4" in st.calls[1][1]


def test_stability_note_silent_when_absent_or_disabled():
    st = _FakeSt()
    visuals.render_stability_note(st, None)
    visuals.render_stability_note(st, {"enabled": False})
    assert st.calls == []


# --- grounding caption ---------------------------------------------


def test_grounding_note_all_clear_only_when_everything_was_checkable():
    note = visuals.grounding_note({"n_quotes": 4, "n_verified": 4, "unpaired_marks": 0})
    assert "4/4" in note
    assert note.endswith("✓")


def test_grounding_note_never_shows_the_check_over_unpaired_marks():
    """Regression: the caption read only n_quotes/n_verified, so unpaired
    quotation marks — text the checker could NOT verify — still rendered as
    "4/4 ✓" while the markdown report warned about them below."""
    note = visuals.grounding_note({"n_quotes": 4, "n_verified": 4, "unpaired_marks": 2})
    assert "✓" not in note
    assert "grounding section" in note  # points at the report's own warning


def test_grounding_note_warns_when_marks_exist_but_no_pair_was_checkable():
    """The n_quotes == 0 with unpaired > 0 case rendered nothing at all,
    which reads as "no quotations, nothing to check" — false: the report
    warns that whatever the marks enclose was not checked."""
    note = visuals.grounding_note({"n_quotes": 0, "n_verified": 0, "unpaired_marks": 3})
    assert note != ""
    assert "✓" not in note
    assert "checkable pair" in note  # the report's own wording
    assert "grounding section" in note


def test_grounding_note_flags_unverified_quotes():
    note = visuals.grounding_note({"n_quotes": 5, "n_verified": 3, "unpaired_marks": 0})
    assert "3/5" in note
    assert "✓" not in note
    assert "grounding section" in note


def test_grounding_note_silent_when_there_is_nothing_to_say():
    assert visuals.grounding_note(None) == ""
    assert visuals.grounding_note({}) == ""
    assert visuals.grounding_note({"n_quotes": 0, "n_verified": 0, "unpaired_marks": 0}) == ""
