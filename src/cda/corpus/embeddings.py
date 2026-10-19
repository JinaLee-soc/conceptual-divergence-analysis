"""Sentence-embedding access.

One embedder object is passed through the pipeline so tests can substitute a
deterministic fake (no model download) and real runs share a single loaded
model. The model and its snapshot are pinned in config
(sentence-transformers/all-MiniLM-L6-v2, the model reported in the paper, at
``embedding.revision``).
"""

from __future__ import annotations

import os
from typing import Protocol


class Embedder(Protocol):
    """Anything that turns texts into fixed-length vectors."""

    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# sentence-transformers prefixes a bare model name with this organization
# unless it is one of the original transformers checkpoints (bert-base-uncased, …).
_HUB_ORGANIZATION = "sentence-transformers"


def hub_model_id(name: str) -> str:
    """The Hugging Face repo id sentence-transformers loads for ``name``.

    "all-MiniLM-L6-v2" loads "sentence-transformers/all-MiniLM-L6-v2"; local
    paths and names that already carry an organization are unchanged.
    """
    if "/" in name or os.path.exists(name):
        return name
    try:
        from sentence_transformers.util.misc import ORIGINAL_TRANSFORMER_MODELS
    except ImportError:  # older or stubbed sentence-transformers
        ORIGINAL_TRANSFORMER_MODELS = ()
    if name.lower() in ORIGINAL_TRANSFORMER_MODELS:
        return name
    return f"{_HUB_ORGANIZATION}/{name}"


class EmbeddingModelError(RuntimeError):
    """The sentence-embedding model could not be loaded, with a next step.

    Distinct from ConfigError/LLMError so the CLI can catch it: the first
    model load is the demo's one sanctioned network use, and a download
    failure would otherwise surface as a multi-screen HuggingFace traceback.
    """


class SentenceTransformerEmbedder:
    """Real embeddings via sentence-transformers (lazy-loaded)."""

    def __init__(self, model_name: str, revision: str | None = None):
        self.name = model_name
        self.revision = revision
        self._model = None  # loaded on first use so --help etc. stay instant
        self._cache_folder: str | None = None  # where the model was loaded from

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - installation problem
                raise RuntimeError(
                    "sentence-transformers is not installed. Run: pip install -e ."
                ) from exc
            try:
                # Downloads ~90 MB to the local HF cache on first use.
                # SentenceTransformer reads this variable too; recording it
                # lets provenance look in the cache the model came from.
                self._cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")
                self._model = SentenceTransformer(
                    self.name, revision=self.revision, cache_folder=self._cache_folder
                )
            except Exception as exc:
                raise EmbeddingModelError(
                    f"Could not load the embedding model '{self.name}' "
                    f"({type(exc).__name__}: {exc}).\n"
                    "  - First run: the model is a one-time ~90 MB download — "
                    "check your network/proxy and rerun (afterwards everything "
                    "works offline; set HF_HUB_OFFLINE=1 to enforce that).\n"
                    "  - If it worked before: the local cache may be corrupt — "
                    "delete ~/.cache/huggingface and rerun once online."
                ) from exc
        return self._model

    def provenance(self) -> dict:
        """Identify the loaded snapshot from the local cache, without a network request.

        ``resolved_revision`` is the commit of the cached Hugging Face snapshot
        that the requested revision (or, with none, the default branch)
        resolves to; for a local model directory, the commit its transformers
        config records, if any.
        """
        from pathlib import Path

        from cda.provenance import digest, file_digest

        result = {
            "model_id": hub_model_id(self.name),
            "requested_revision": self.revision,
            "resolved_revision": None,
        }
        if self._model is None:
            return dict(result, status="not loaded; no embedding model used")
        root = Path(self.name)
        if root.is_dir():
            config = getattr(getattr(self._model[0], "auto_model", None), "config", None)
            result["resolved_revision"] = getattr(config, "_commit_hash", None)
        else:
            cached = self._cached_config()
            if cached is None:
                where = self._cache_folder or "the default Hugging Face cache"
                return dict(
                    result,
                    status=(
                        f"unresolved: the snapshot of '{self.name}' (revision "
                        f"{self.revision!r}) was not found in {where}, so its model "
                        "files were not hashed"
                    ),
                    device=str(self._model.device),
                )
            root = Path(cached).parent
            result["resolved_revision"] = root.name
        files = {
            str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()
        }
        result.update(
            files=files,
            files_sha256=digest(files),
            status="local model files hashed",
            device=str(self._model.device),
        )
        return result

    def _cached_config(self) -> str | None:
        """Path of the cached snapshot's config.json, in the cache the model was loaded from."""
        from huggingface_hub import try_to_load_from_cache

        options = {"revision": self.revision}
        if self._cache_folder:
            options["cache_dir"] = self._cache_folder
        cached = try_to_load_from_cache(hub_model_id(self.name), "config.json", **options)
        return cached if isinstance(cached, str) else None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, show_progress_bar=False)
        return [[float(x) for x in vec] for vec in vectors]


def get_embedder(config) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        str(config.get("embedding.model")), revision=config.get("embedding.revision", None)
    )
