"""The passage map: the joint partition drawn in the pooled embedding space."""

import numpy as np
import pytest

from cda.config import load_config
from cda.divergence.figure import PALETTE, _cluster_index, _pca_plane, draw_passage_map
from cda.types import Passage

matplotlib = pytest.importorskip("matplotlib")


def _umask_default_mode() -> int:
    """The mode a plain ``open(path, "w")`` would give a new file here."""
    import os

    current = os.umask(0)
    os.umask(current)
    return 0o666 & ~current


def passage(text, cluster, seed, dim=8):
    rng = np.random.default_rng(seed)
    return Passage(
        chunk_id=f"c{seed}",
        source="src.pdf",
        year="2020",
        text=text,
        similarity=0.5,
        cluster_id=cluster,
        embedding=list(rng.normal(size=dim)),
    )


def test_figure_is_on_by_default():
    """The map is part of a run's output, not an opt-in extra — a reader who
    never finds the flag never learns their flagged cluster was a sampling
    hole."""
    assert load_config().get("figure.enabled") is True
    assert load_config().get("figure.dpi") == 300


def test_draw_writes_a_png(tmp_path):
    researcher = [passage("mine", "Cluster 0", i) for i in range(4)]
    literature = [passage("theirs", "Cluster 1", 100 + i) for i in range(9)]
    out = draw_passage_map(
        term="novelty",
        researcher_passages=researcher,
        literature_passages=literature,
        joint=None,
        path=tmp_path / "map.png",
    )
    assert out is not None and out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # The figure is rendered to a mkstemp file (mode 0600) and renamed into
    # place, so without an explicit chmod the docker-compose host user could
    # not open the map they just generated.
    assert out.stat().st_mode & 0o777 == _umask_default_mode()


def test_too_few_passages_returns_none_rather_than_an_empty_plot(tmp_path):
    """Two passages cannot show a structure; a blank axes would look like a
    finding ('my corpus is empty') rather than an absence of one."""
    out = draw_passage_map(
        term="novelty",
        researcher_passages=[passage("mine", "Cluster 0", 1)],
        literature_passages=[],
        joint=None,
        path=tmp_path / "map.png",
    )
    assert out is None
    assert not (tmp_path / "map.png").exists()


def test_pipeline_says_why_when_the_map_is_skipped_for_too_few_passages(tmp_path):
    """--figure with under 3 embedded passages used to produce no PNG and no
    explanation — indistinguishable from a drawing failure. The skip must say
    why there is no map and that the report itself is unaffected."""
    from cda.pipeline import _write_passage_map

    messages = []
    result = _write_passage_map(
        term="novelty",
        researcher_passages=[passage("mine", "Cluster 0", 1)],
        literature_passages=[],
        joint=None,
        report_file=tmp_path / "drift_novelty.md",
        config=load_config(overrides={"figure": {"enabled": True}}),
        progress=messages.append,
    )
    assert result is None
    assert any("passage map skipped" in m and "unaffected" in m for m in messages)


def test_passages_without_embeddings_are_skipped_not_crashed_on(tmp_path):
    researcher = [passage("mine", "Cluster 0", i) for i in range(3)]
    literature = [passage("theirs", "Cluster 1", 100 + i) for i in range(5)]
    literature[0].embedding = None  # never clustered
    out = draw_passage_map(
        term="novelty",
        researcher_passages=researcher,
        literature_passages=literature,
        joint=None,
        path=tmp_path / "map.png",
    )
    assert out is not None and out.is_file()


def test_cluster_index_parses_ids_and_survives_unassigned():
    assert _cluster_index(passage("x", "Cluster 7", 1)) == 7
    assert _cluster_index(passage("x", "Cluster N/A", 1)) is None


def test_pca_plane_is_deterministic_and_reports_its_share():
    matrix = np.random.default_rng(0).normal(size=(40, 6))
    project_a, explained_a = _pca_plane(matrix)
    project_b, explained_b = _pca_plane(matrix)
    assert explained_a == explained_b
    assert 0.0 <= explained_a <= 1.0
    np.testing.assert_allclose(project_a(matrix), project_b(matrix))
    assert project_a(matrix).shape == (40, 2)


def test_palette_is_long_enough_for_the_default_k_sweep():
    """joint_max_k defaults to 15; the palette wraps rather than crashing, but
    a wrap means two clusters share a colour, so keep it wide."""
    assert len(PALETTE) >= 12
    assert len(set(PALETTE)) == len(PALETTE)
