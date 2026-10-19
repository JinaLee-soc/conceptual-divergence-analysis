"""Quotation grounding, report configuration, and figure palette regressions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cda.config import ConfigError, load_config
from cda.pipeline import analyze_term, validated_settings
from cda.registers.grounding import check_quotations, render_grounding_markdown
from cda.types import Passage

PROMPT = (
    'Sample 1 [C0] [smith2020.pdf]: "Researchers argue that novelty is rewarded unevenly '
    'across fields, and the gap widens over time."\n'
    "Sample 2 [C1] [lee.pdf]: \"The so-called 'theory penalty' falls hardest on junior "
    'women in sociology departments."\n'
)
QUOTE = "novelty is rewarded unevenly across fields"


def _counts(body: str) -> tuple[int, int, int, int]:
    result = check_quotations(body, PROMPT)
    return (
        result["n_quotes"],
        result["n_verified"],
        result["n_ambiguous"],
        result["unpaired_marks"],
    )


# ---- non-Latin neighbours of a quote mark are not ambiguous nesting ------


@pytest.mark.parametrize(
    "body",
    [
        f'They write "{QUOTE}"1 here.',
        f'They write "{QUOTE}"¹ here.',
        f'They write _"{QUOTE}"_ here.',
        f"그들은 “{QUOTE}”라고 썼다.",
        f"그들은“{QUOTE}” 라고 썼다.",
    ],
)
def test_quote_next_to_digit_underscore_or_hangul_is_checked(body):
    assert _counts(body) == (1, 1, 0, 0)


def test_latin_letter_touching_a_mark_still_blocks_the_invented_span():
    body = (
        f'They write "{QUOTE}, and the gap called "an invented phrase that never appears '
        'anywhere in the corpus at all" here."'
    )
    result = check_quotations(body, PROMPT)
    assert result["n_ambiguous"] >= 1
    assert result["n_verified"] == 0
    assert "ambiguous nesting" in render_grounding_markdown(result)


@pytest.mark.parametrize(
    "body",
    [f'They write "{QUOTE}"s here.', f'They wrote x"{QUOTE}" here.', f'Écrit é"{QUOTE}" ici.'],
)
def test_latin_letter_on_either_outer_side_is_ambiguous(body):
    assert _counts(body)[2] == 1


# ---- straight marks inside a verified curly pair are paired --------------


def test_inner_straight_quotes_of_a_verified_curly_pair_are_not_unpaired():
    body = 'They say “the so-called "theory penalty" falls hardest on junior women” here.'
    result = check_quotations(body, PROMPT)
    assert _counts(body) == (1, 1, 0, 0)
    assert "NOT checked" not in render_grounding_markdown(result)


def test_marks_inside_an_unverified_curly_pair_still_count_as_unpaired():
    body = 'They say “the so-called "theory penalty" falls hardest on senior men” here.'
    quotes, verified, _, unpaired = _counts(body)
    assert (quotes, verified) == (1, 0)
    assert unpaired == 2


# ---- final sentence punctuation inside the closing mark ------------------


@pytest.mark.parametrize("mark", [".", "?", "!"])
def test_final_sentence_punctuation_inside_the_mark_is_retried_without_it(mark):
    assert _counts(f'They write "{QUOTE}{mark}" Then more.') == (1, 1, 0, 0)


@pytest.mark.parametrize(
    "body",
    [
        'They write "argue that novelty is rewarded uneven." Then more.',
        f'They write "{QUOTE}." — lee.pdf here.',
        f'They write "{QUOTE};" Then more.',
    ],
)
def test_punctuation_retry_keeps_whole_word_and_source_checks(body):
    quotes, verified, _, _ = _counts(body)
    assert (quotes, verified) == (1, 0)


# ---- themes are validated before any retrieval ---------------------------


class _Untouchable:
    """Records any use: the pipeline must not reach retrieval with a bad config."""

    name = "untouchable"

    def __init__(self, touched: list[str]):
        self._touched = touched

    def __getattr__(self, attribute):
        self._touched.append(attribute)
        raise AssertionError(f"retrieval started before config validation ({attribute})")


def _config(tmp_path, **overrides):
    return load_config(overrides={"paths": {"output_dir": str(tmp_path / "outputs")}, **overrides})


@pytest.mark.parametrize("themes", [[2024], [{"tag": ""}], [{"description": "no tag"}]])
def test_malformed_themes_fail_in_validated_settings(tmp_path, themes):
    with pytest.raises(ConfigError, match="neither a tag string nor a mapping"):
        validated_settings(_config(tmp_path, themes=themes))


def test_malformed_themes_fail_before_extraction(tmp_path):
    touched: list[str] = []
    with pytest.raises(ConfigError, match="themes entry 2024"):
        analyze_term(
            "novelty",
            _config(tmp_path, themes=[2024]),
            mock=True,
            fixtures_dir=tmp_path,
            embedder=_Untouchable(touched),
            collection=_Untouchable(touched),
            progress=lambda _message: None,
        )
    assert touched == []


def test_wellformed_themes_pass_validation(tmp_path):
    themes = ["#evaluation", {"tag": "repair", "description": "d"}]
    settings = validated_settings(_config(tmp_path, themes=themes))
    assert settings["theme_lines"] == ["  #evaluation", "  #repair — d"]


# ---- silhouette_subsample floor -----------------------------------------


@pytest.mark.parametrize("value", [2, 50, 99])
def test_tiny_silhouette_subsample_is_a_config_error(tmp_path, value):
    config = _config(tmp_path, clustering={"silhouette_subsample": value})
    with pytest.raises(ConfigError, match=r"clustering\.silhouette_subsample.*at least 100"):
        validated_settings(config)


def test_silhouette_subsample_at_the_floor_is_accepted(tmp_path):
    config = _config(tmp_path, clustering={"silhouette_subsample": 100})
    assert validated_settings(config)["clustering"]["silhouette_subsample"] == 100


# ---- figure palettes and degenerate PCA ----------------------------------


def _clustered(side: str, n_clusters: int) -> list[Passage]:
    rng = np.random.default_rng(len(side) + n_clusters)
    return [
        Passage(
            chunk_id=f"{side}{cluster}",
            source="s",
            year="2020",
            text="t",
            similarity=0.0,
            cluster_id=f"Cluster {cluster}",
            embedding=list(rng.normal(size=4)),
        )
        for cluster in range(n_clusters)
    ]


def _drawn_colours(monkeypatch, tmp_path, joint, n_researcher, n_literature):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    from cda.divergence import figure

    drawn = []
    real_scatter = Axes.scatter

    def scatter(self, *args, **kwargs):
        drawn.append((list(kwargs["c"]), kwargs["alpha"]))
        return real_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", scatter)
    figure.draw_passage_map(
        term="novelty",
        researcher_passages=_clustered("r", n_researcher),
        literature_passages=_clustered("lit", n_literature),
        joint=joint,
        path=tmp_path / "map.png",
        dpi=30,
    )
    return drawn


def test_independent_palettes_cover_default_max_k_and_stay_apart(monkeypatch, tmp_path):
    from matplotlib.colors import to_rgb

    (literature, literature_alpha), (researcher, _) = _drawn_colours(
        monkeypatch, tmp_path, None, n_researcher=8, n_literature=15
    )
    assert len(set(literature)) == 15  # literature_max_k default
    assert len(set(researcher)) == 8  # researcher_max_k default

    def as_drawn(colour, alpha):  # over the white background
        return alpha * np.array(to_rgb(colour)) + (1 - alpha)

    closest = min(
        np.linalg.norm(as_drawn(lit, literature_alpha) - np.array(to_rgb(res)))
        for lit in literature
        for res in researcher
    )
    assert closest > 0.25  # visible separation between the two corpus palettes


def test_joint_mode_keeps_the_okabe_ito_palette(monkeypatch, tmp_path):
    from cda.divergence.figure import PALETTE

    joint = type("Joint", (), {"used": True})()
    (literature, _), (researcher, _) = _drawn_colours(
        monkeypatch, tmp_path, joint, n_researcher=3, n_literature=12
    )
    assert literature == PALETTE[:12]
    assert researcher == PALETTE[:3]


def test_pca_plane_of_identical_embeddings_explains_nothing():
    from cda.divergence.figure import _pca_plane

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        project, explained = _pca_plane(np.ones((6, 4)))
    assert explained == 0.0
    assert project(np.ones((2, 4))).shape == (2, 2)
