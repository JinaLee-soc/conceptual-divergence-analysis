"""Shared test fixtures.

The default suite is fully offline: no API key, no model download, no
network. Embeddings come from deterministic fakes; LLM responses come from
MockProvider fixtures created inside the tests.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cda.config import load_config
from cda.types import Passage

DIM = 16


def _hash_vector(text: str, dim: int = DIM) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Deterministic text → unit vector; equal texts get equal vectors."""

    name = "fake-hash-embedder"

    def embed(self, texts):
        return [[float(x) for x in _hash_vector(t)] for t in texts]


class AnchoredFakeEmbedder:
    """Texts sharing a topic keyword land near that topic's anchor vector.

    Gives tests controllable cluster structure: passages mentioning "alpha"
    form one tight cluster, "beta" another, and so on. Unkeyed texts fall
    back to hash vectors.
    """

    name = "fake-anchored-embedder"
    KEYWORDS = ("alpha", "beta", "gamma", "delta")

    def __init__(self, noise: float = 0.08):
        self.noise = noise
        self._anchors = {}
        for i, keyword in enumerate(self.KEYWORDS):
            rng = np.random.default_rng(10_000 + i)
            anchor = rng.normal(size=DIM)
            self._anchors[keyword] = anchor / np.linalg.norm(anchor)

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            anchor = next((self._anchors[k] for k in self.KEYWORDS if k in lowered), None)
            if anchor is None:
                vectors.append([float(x) for x in _hash_vector(text)])
                continue
            vector = anchor + self.noise * _hash_vector(text)
            vector = vector / np.linalg.norm(vector)
            vectors.append([float(x) for x in vector])
        return vectors


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def anchored_embedder():
    return AnchoredFakeEmbedder()


@pytest.fixture
def tmp_config(tmp_path):
    """A config rooted in tmp_path with small, test-friendly sweep sizes."""
    return load_config(
        overrides={
            "paths": {
                "corpus_dir": str(tmp_path / "corpus"),
                "drafts_dir": str(tmp_path / "drafts"),
                "output_dir": str(tmp_path / "outputs"),
            },
            "clustering": {
                "min_cluster_size": 2,
                "silhouette_subsample": 200,
                "researcher_max_k": 4,
                "literature_max_k": 5,
            },
            "saturation": {"start": 8, "step": 4, "max_passages": 60},
        }
    )


def make_passages(
    texts: list[str], *, source: str = "paper.pdf", year: str = "2020"
) -> list[Passage]:
    return [
        Passage(
            chunk_id=f"{source}::w{i}",
            source=source,
            year=year,
            text=text,
            similarity=0.5,
        )
        for i, text in enumerate(texts)
    ]
