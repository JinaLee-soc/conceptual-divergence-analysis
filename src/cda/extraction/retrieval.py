"""Literal retrieval and term-sentence extraction.

Chroma's case-insensitive ``$regex`` filter finds literal candidates. Phrases
use one literal word to locate their complete stored sentence across chunk
edges; the full phrase is then checked before applying the retrieval cap.
Whole-word extraction removes substring false positives.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

from cda.config import Config, ConfigError
from cda.corpus.zotero import normalize_key
from cda.extraction.windows import (
    SENTENCE_SEGMENTER,
    infer_year,
    normalize_text,
    sentence_windows,
    split_sentences,
    term_regex,
    window_settings,
)
from cda.types import Passage


def literal_filter(term: str) -> dict:
    """A ``where_document`` filter matching the literal term in any letter case.

    The term is NFC-normalized because chunk text is NFC-normalized at ingest.
    ``re.escape`` output is accepted by Chroma's Rust regex engine.
    """
    term = unicodedata.normalize("NFC", term)
    return {"$regex": f"(?i){re.escape(term)}"}


def keyword_hits(
    collection,
    term: str,
    *,
    batch_size: int = 1000,
    max_chunks: int = 0,
    progress: Callable[[str], None] = print,
) -> dict:
    """Literal term hits, including phrases across chunk edges, capped at ``max_chunks``."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("retrieval batch_size must be a positive integer.")
    if type(max_chunks) is not int or max_chunks < 0:
        raise ValueError("retrieval max_chunks must be a nonnegative integer.")
    hits: dict[str, dict] = {}
    words = term.split()
    phrase = len(words) > 1
    # Every word belongs to a chunk, whose sentence_context contains the full
    # sentence. One literal word finds a phrase even with zero chunk overlap;
    # only candidates containing the full literal phrase count toward the cap.
    anchor = max(words, key=len) if phrase else term
    where_document = literal_filter(anchor)
    phrase_pattern = re.compile(re.escape(unicodedata.normalize("NFC", term)), re.IGNORECASE)
    seen_ids: set[str] = set()
    offset = 0
    capped = False
    while True:
        try:
            result = collection.get(
                where_document=where_document,
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
        except Exception as exc:
            raise ConfigError(
                f"Keyword retrieval is incomplete for '{term}' at offset {offset}. "
                "No partial result will be analyzed. Check the corpus database and retry."
            ) from exc
        ids = result.get("ids") or []
        if not ids:
            break
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if len(documents) != len(ids) or len(metadatas) != len(ids):
            raise ConfigError(
                f"Keyword retrieval for '{term}' returned {len(ids)} ids but "
                f"{len(documents)} documents and {len(metadatas)} metadata records. "
                "The corpus database is inconsistent; rebuild it with cda ingest --reingest."
            )
        for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            if str(chunk_id) in seen_ids:
                raise ConfigError(
                    f"Keyword retrieval returned duplicate chunk id {chunk_id!r}. "
                    "The corpus may have changed during paging; retry on a stable store. "
                    "No partial result will be analyzed."
                )
            seen_ids.add(str(chunk_id))
            metadata = metadata or {}
            if phrase:
                context = metadata.get("sentence_context", document)
                if not isinstance(context, str):
                    raise ConfigError(
                        f"Invalid sentence context in corpus chunk {chunk_id!r}; re-ingest its source."
                    )
                if not phrase_pattern.search(context):
                    continue
            if max_chunks and len(hits) >= max_chunks:  # exact cap, even mid-batch
                capped = True
                break
            hits[str(chunk_id)] = {
                "chunk_id": str(chunk_id),
                "document": document,
                "metadata": metadata or {},
                "similarity": 0.0,
                "source": "contains",
            }
        if capped or len(ids) < batch_size:
            break
        offset += len(ids)
    if capped:
        progress(
            f"[warn] keyword retrieval for '{term}' hit the cap of {max_chunks} "
            "chunks; the analysis proceeds on that subset. Raise "
            "retrieval.keyword_max_chunks to cover more."
        )
    return hits


def _stored_sentences(context: str, lengths) -> list[str] | None:
    """The ingest-time sentences of ``context`` from their stored lengths; ``None`` if invalid.

    Sentences are joined by single spaces at ingest, so the lengths must
    tile the context exactly with one space between neighbors.
    """
    try:
        sizes = json.loads(lengths) if isinstance(lengths, str) else None
    except ValueError:
        return None
    if not isinstance(sizes, list) or any(type(n) is not int or n < 1 for n in sizes):
        return None
    sentences, cursor = [], 0
    for i, size in enumerate(sizes):
        if i and context[cursor : cursor + 1] != " ":
            return None
        cursor += bool(i)
        sentences.append(context[cursor : cursor + size])
        cursor += size
    return sentences if cursor == len(context) else None


def _chunk_sentences(chunk_id: str, document: str, metadata: dict) -> tuple[list[str], bool]:
    """The sentences of one hit, and whether it is an indexed corpus chunk.

    Indexed chunks carry their complete original sentences
    (``sentence_context``), the ingest-time sentence lengths, and the
    segmenter that produced them; they are never re-segmented, because a
    chunk's context segments differently from the whole document. Anything
    else from an indexed corpus must be re-ingested before analysis.
    """
    if not document:
        raise ConfigError(
            f"Corpus chunk {chunk_id!r} has no text; re-ingest its source with cda ingest --reingest."
        )
    context = metadata.get("sentence_context")
    if context is None:
        if "chunk_index" in metadata or "total_chunks" in metadata:
            raise ConfigError(
                "This corpus lacks original sentence boundaries. Run cda ingest "
                "again to re-ingest legacy sources before analyzing them."
            )
        return split_sentences(document), False
    if not isinstance(context, str) or normalize_text(document) not in context:
        raise ConfigError(
            "Invalid sentence context in corpus; re-ingest the source with cda ingest --reingest."
        )
    position = metadata.get("sentence_start")
    if type(position) is not int or position < 0:
        raise ConfigError("Invalid or missing sentence position; re-ingest the source.")
    segmenter = metadata.get("sentence_segmenter")
    if segmenter != SENTENCE_SEGMENTER:
        raise ConfigError(
            f"Chunk {chunk_id!r} was segmented into sentences by {segmenter or 'an older splitter'}, "
            f"but this version uses {SENTENCE_SEGMENTER}, so its sentence positions do not "
            "match. Run cda ingest again (with the same --pdf-dir) to re-ingest the corpus."
        )
    if "sentence_lengths" not in metadata:
        raise ConfigError(
            "This corpus lacks original sentence boundaries. Run cda ingest "
            "again to re-ingest legacy sources before analyzing them."
        )
    sentences = _stored_sentences(context, metadata["sentence_lengths"])
    if sentences is None:
        raise ConfigError(
            f"Chunk {chunk_id!r} has invalid stored sentence boundaries; re-ingest the "
            "source with cda ingest --reingest."
        )
    return sentences, True


def build_passages(
    term: str,
    merged_hits: dict,
    classify,
    config: Config,
) -> tuple[list[Passage], list[Passage], dict]:
    """Extract term-sentences from every hit and split by MyWork resolution.

    ``classify(filename, metadata) -> (bool, reason)`` comes from
    ``cda.extraction.mywork.mywork_classifier``. Returns (researcher_passages,
    literature_passages, stats); stats include per-signal classification
    counts, which the report surfaces so users can audit attribution.

    Overlapping chunks contain the same sentence twice. A (source, sentence
    position) pair is one occurrence; when the two chunks give it different
    context windows (a neighbor sentence lies outside one chunk), the window
    that contains the other is kept.
    """
    regex = term_regex(term)
    settings = window_settings(config)

    pools: dict[bool, list[Passage]] = {True: [], False: []}
    seen: dict[tuple[str, int], tuple[bool, int]] = {}
    stats = {
        "chunks_total": len(merged_hits),
        "chunks_with_term_window": 0,
        "classification_reasons": {},
        "overlap_occurrences_removed": 0,
    }

    for chunk_id, record in sorted(merged_hits.items()):
        metadata = record.get("metadata") or {}
        sentences, indexed = _chunk_sentences(chunk_id, record.get("document") or "", metadata)
        windows = sentence_windows(sentences, regex, **settings)
        if not windows:
            continue
        stats["chunks_with_term_window"] += 1

        filename = str(metadata.get("pdf_filename") or metadata.get("filename") or chunk_id)
        year = infer_year(filename, metadata)
        is_researcher, reason = classify(filename, metadata)
        stats["classification_reasons"][reason] = stats["classification_reasons"].get(reason, 0) + 1
        source_id = str(metadata.get("pdf_relpath") or normalize_key(filename))
        # Unindexed caller-supplied records are independent source units.
        identity = source_id if indexed else str(chunk_id)

        for i, (sentence_index, window) in enumerate(windows):
            position = metadata.get("sentence_start", 0) + sentence_index
            passage = Passage(
                chunk_id=f"{chunk_id}::w{i}",
                source=Path(filename).name,
                year=year,
                text=window,
                similarity=float(record.get("similarity") or 0.0),
                metadata=dict(metadata, sentence_index=position, source_identity=identity),
            )
            key = (identity, position)
            if key not in seen:
                seen[key] = (is_researcher, len(pools[is_researcher]))
                pools[is_researcher].append(passage)
                continue
            stats["overlap_occurrences_removed"] += 1
            kept_side, kept_index = seen[key]
            kept = pools[kept_side][kept_index].text
            if kept_side != is_researcher or (window not in kept and kept not in window):
                raise ConfigError(
                    f"Conflicting overlapping sentence records for {identity!r} at sentence "
                    f"{position}; re-ingest the source with cda ingest --reingest."
                )
            if len(window) > len(kept):
                pools[kept_side][kept_index] = passage

    researcher, literature = pools[True], pools[False]
    stats["researcher_passages"] = len(researcher)
    stats["literature_passages"] = len(literature)
    return researcher, literature, stats
