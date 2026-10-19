"""Literal term-sentences from drafts; absence is not filled by semantic guesses."""

from __future__ import annotations

from cda.config import Config
from cda.corpus.drafts import normalize_drafts
from cda.extraction.windows import (
    extract_term_occurrences,
    infer_year,
    term_regex,
    window_settings,
)
from cda.types import Passage


def extract_draft_passages(
    term: str,
    draft_texts: dict[str, str],
    embedder,
    config: Config,
) -> list[Passage]:
    """Researcher passages from drafts: every literal term-sentence occurrence.

    ``embedder`` is not used: draft passages are literal matches, never
    semantic ones. It stays in the signature for the pipeline's call.
    """
    regex = term_regex(term)
    settings = window_settings(config)
    passages: list[Passage] = []
    for filename, text in normalize_drafts(draft_texts).items():
        year = infer_year(filename, {})
        for i, (sentence_index, window) in enumerate(
            extract_term_occurrences(text, regex, **settings)
        ):
            passages.append(
                Passage(
                    chunk_id=f"draft:{filename}::w{i}",
                    source=filename,
                    year=year,
                    text=window,
                    similarity=1.0,
                    metadata={
                        "draft": True,
                        "source_identity": f"draft:{filename}",
                        "sentence_index": sentence_index,
                        "self_authored": "true",
                        "draft_match": "contains",
                    },
                )
            )
    return passages
