"""Saturation-based literature subsampling: ranking and plateau detection."""

import pytest

from cda.clustering.saturation import rank_literature, saturation_select
from tests.conftest import make_passages


def stable_pool(n=60):
    """A pool whose 2-cluster structure is present from the start (saturates).

    Similarities interleave the two blobs so similarity-mode ranking gives
    every sweep prefix both clusters — the stable-structure case. (Term-mode
    ranking with a term unrelated to either blob would front-load one blob
    and delay saturation; that behavior has its own test below.)
    """
    texts = []
    for i in range(n // 2):
        texts.append(f"An alpha framing of novelty in evaluation research, case {i}.")
        texts.append(f"The beta framing treats novelty as recombination, case {i}.")
    passages = make_passages(texts)
    for i, passage in enumerate(passages):
        passage.similarity = 1.0 - i * 0.001  # creation order = interleaved
    return passages


def select(passages, embedder, *, mode="term", **overrides):
    params = dict(
        start=10,
        step=5,
        max_passages=40,
        ari_threshold=0.8,
        consecutive_increments=2,
        max_k=4,
        n_restarts=3,
        max_iter=50,
        min_cluster_size=2,
        silhouette_subsample=200,
        seed=42,
    )
    params.update(overrides)
    ranked = rank_literature(passages, "novelty", embedder, mode=mode)
    return saturation_select(ranked, **params)


def test_saturation_reached_on_stable_structure(anchored_embedder):
    selected, info = select(stable_pool(), anchored_embedder, mode="similarity")
    assert info["saturation_reached"] is True
    assert info["n_selected"] < info["n_candidates"]
    assert len(selected) == info["n_selected"]
    # Earlier-plateau selection: N* is the first of the two high-ARI steps.
    high_ari_steps = [h for h in info["history"] if (h.get("ari_vs_prev") or 0) >= 0.8]
    assert info["n_selected"] <= high_ari_steps[-1]["n"]


def test_not_reached_flag_when_cap_hits(fake_embedder):
    # Hash embeddings have no stable cluster structure → ARI stays low.
    passages = make_passages(
        [f"novelty sentence number {i} with unrelated words" for i in range(50)]
    )
    selected, info = select(passages, fake_embedder, max_passages=30)
    assert info["saturation_reached"] is False
    assert "reason" in info
    assert len(selected) == info["n_selected"] <= 30


def test_small_pool_passes_through(fake_embedder):
    passages = make_passages([f"novelty {i}" for i in range(5)])
    selected, info = select(passages, fake_embedder, start=10)
    assert len(selected) == 5
    assert info["saturation_reached"] is False
    assert "no selection applied" in info["reason"]


def test_term_ranking_puts_similar_passages_first(anchored_embedder):
    # "alpha novelty" passages sit near the term-anchor only if the term text
    # itself contains their keyword — craft it so ranking is checkable:
    # the anchored embedder keys on "alpha", so use "alpha" as the term.
    near = [f"alpha discussion {i}" for i in range(3)]
    far = [f"completely unrelated text {i}" for i in range(3)]
    passages = make_passages(far + near)
    ranked = rank_literature(passages, "alpha", anchored_embedder, mode="term")
    assert all("alpha" in p.text for p in ranked[:3])


def test_ranked_by_similarity_mode(fake_embedder):
    passages = make_passages([f"text {i}" for i in range(4)])
    for i, p in enumerate(passages):
        p.similarity = i / 10
    ranked = rank_literature(passages, "term", fake_embedder, mode="similarity")
    assert [p.similarity for p in ranked] == sorted((p.similarity for p in passages), reverse=True)


def test_scope_prompt_returns_full_pool_untouched(fake_embedder):
    """saturation.scope=prompt: the geometry sees every retrieved passage; no
    ranking, no sweep, no subsampling."""
    from cda.config import load_config
    from cda.pipeline import select_literature_by_saturation

    config = load_config(overrides={"saturation": {"scope": "prompt"}})
    passages = make_passages([f"novelty sentence {i}" for i in range(60)])
    selected, info, ran = select_literature_by_saturation(
        "novelty", passages, config, embedder=fake_embedder
    )
    assert selected is passages  # identical list, original order, nothing dropped
    assert ran is False and info["ran"] is False
    assert info["scope"] == "prompt"
    assert info["n_candidates"] == info["n_selected"] == 60
    assert "full pool clustered" in info["reason"]


def test_scope_prompt_is_the_default_and_recorded(fake_embedder):
    """The released default clusters the full pool.

    The sweep stays `enabled`; what changed is what it governs. A run must
    still record which scope produced it, because the two are not comparable.
    """
    from cda.config import load_config
    from cda.pipeline import select_literature_by_saturation

    config = load_config()
    assert config.get("saturation.enabled") is True
    assert config.get("saturation.scope") == "prompt"
    passages = make_passages([f"novelty sentence {i}" for i in range(60)])
    selected, info, ran = select_literature_by_saturation(
        "novelty", passages, config, embedder=fake_embedder
    )
    assert info["scope"] == "prompt"
    assert ran is False  # no sweep: the geometry sees everything
    assert len(selected) == 60


def test_scope_sample_still_runs_the_paper_sweep(fake_embedder):
    """`scope: sample` must remain a faithful route back to the paper's method."""
    from cda.config import load_config
    from cda.pipeline import select_literature_by_saturation

    config = load_config(overrides={"saturation": {"scope": "sample"}})
    passages = make_passages([f"novelty sentence {i}" for i in range(60)])
    _selected, info, ran = select_literature_by_saturation(
        "novelty", passages, config, embedder=fake_embedder
    )
    assert info["scope"] == "sample"
    assert ran is True


def test_scope_invalid_is_a_config_error(fake_embedder):
    from cda.config import ConfigError, load_config
    from cda.pipeline import select_literature_by_saturation

    config = load_config(overrides={"saturation": {"scope": "geometry"}})
    with pytest.raises(ConfigError, match="sample.*prompt"):
        select_literature_by_saturation(
            "novelty", make_passages(["novelty a"]), config, embedder=fake_embedder
        )


def test_max_passages_below_start_refused_only_when_the_sweep_can_run():
    """The cross-field refusal matches its docstring: it fires only when the
    sweep can actually execute (enabled AND scope=sample). Under the shipped
    default (scope: prompt) the sweep never runs, so the same numbers must
    pass — such configs analyzed identically before the check existed."""
    from cda.config import ConfigError, load_config
    from cda.pipeline import saturation_params

    bad_numbers = {"start": 50, "max_passages": 10}

    params = saturation_params(
        load_config(overrides={"saturation": {**bad_numbers, "scope": "prompt"}})
    )
    assert params["scope"] == "prompt"  # validated and returned, not refused

    with pytest.raises(ConfigError, match="max_passages"):
        saturation_params(load_config(overrides={"saturation": {**bad_numbers, "scope": "sample"}}))

    # Disabled sweep: the cross-field rule stays out of the way...
    saturation_params(
        load_config(overrides={"saturation": {**bad_numbers, "scope": "sample", "enabled": False}})
    )
    # ...but types and spellings are still validated in every scope.
    with pytest.raises(ConfigError, match="saturation.start"):
        saturation_params(
            load_config(
                overrides={"saturation": {**bad_numbers, "scope": "prompt", "start": "fifty"}}
            )
        )
