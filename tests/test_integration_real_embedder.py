"""Integration tier: tests that need the real embedding model.

Excluded from the default suite (pyproject addopts: -m "not integration");
run explicitly with:

    pytest -m integration

First run downloads ~90 MB to the Hugging Face cache. These tests exist to
catch embedding-model regressions (a changed model silently changes every
downstream cluster) — the offline suite never loads the real model.
"""

import math

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_embedder():
    from cda.config import load_config
    from cda.corpus.embeddings import get_embedder

    return get_embedder(load_config())


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_pinned_model_loads_and_embeds(real_embedder):
    vectors = real_embedder.embed(["novelty in science"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384  # all-MiniLM-L6-v2 dimensionality
    assert real_embedder.name == "sentence-transformers/all-MiniLM-L6-v2"


def test_semantic_neighbors_beat_unrelated_text(real_embedder):
    related_a, related_b, unrelated = real_embedder.embed(
        [
            "The novelty of a scientific finding is judged by reviewers.",
            "Reviewers assess how original and new a discovery is.",
            "The recipe calls for two cups of flour and an egg.",
        ]
    )
    assert _cosine(related_a, related_b) > _cosine(related_a, unrelated)


def test_embeddings_deterministic_across_calls(real_embedder):
    first = real_embedder.embed(["consensus formation"])[0]
    second = real_embedder.embed(["consensus formation"])[0]
    assert first == second
