"""Literal-retrieval helpers, tested offline against a stub collection."""

import pytest

from cda.config import ConfigError, load_config
from cda.extraction.mywork import mywork_classifier
from cda.extraction.retrieval import build_passages, keyword_hits, literal_filter


def test_max_window_chars_below_four_is_rejected():
    """Truncation is `window[: max_window_chars - 3] + "..."`, so a cap under
    4 yields a window LONGER than the cap it was given."""
    config = load_config(overrides={"retrieval": {"max_window_chars": 3}})
    with pytest.raises(ConfigError, match="max_window_chars"):
        build_passages("novelty", {}, mywork_classifier(config.section("mywork")), config)


class _PagedCollection:
    """Stub yielding term-bearing chunks through paginated ``.get()`` calls."""

    def __init__(self, n: int):
        self._ids = [f"chunk{i:04d}" for i in range(n)]

    def get(self, where_document=None, include=None, limit=1000, offset=0):
        page = self._ids[offset : offset + limit]
        return {
            "ids": page,
            "documents": ["The novelty sentence sits here."] * len(page),
            "metadatas": [{} for _ in page],
        }


def test_keyword_hits_cap_is_exact_even_mid_batch():
    """max_chunks that is not a multiple of batch_size must not overshoot."""
    hits = keyword_hits(
        _PagedCollection(100),
        "novelty",
        batch_size=30,
        max_chunks=70,
        progress=lambda _msg: None,
    )
    assert len(hits) == 70


def test_literal_filter_is_a_case_insensitive_escaped_regex():
    """Chroma's $contains is case-sensitive; the filter is one (?i) regex."""
    assert literal_filter("novelty") == {"$regex": "(?i)novelty"}
    assert literal_filter("C++") == {"$regex": "(?i)C\\+\\+"}


def test_literal_filter_normalizes_nfd_terms_to_nfc():
    """Chunk text is NFC by contract (normalized at ingest), so an NFD-typed
    term (pasted from a macOS filename) must be composed before matching."""
    import unicodedata

    nfd = unicodedata.normalize("NFD", "café")
    assert literal_filter(nfd) == {"$regex": "(?i)" + unicodedata.normalize("NFC", "café")}


class _RecordingCollection:
    """Stub that records the where_document filter it was queried with."""

    def __init__(self):
        self.filters = []

    def get(self, where_document=None, include=None, limit=1000, offset=0):
        self.filters.append(where_document)
        return {"ids": [], "documents": [], "metadatas": []}


def test_keyword_hits_queries_the_case_insensitive_filter():
    collection = _RecordingCollection()
    keyword_hits(collection, "Novelty", progress=lambda _msg: None)
    assert collection.filters == [{"$regex": "(?i)Novelty"}]


def test_keyword_hits_warns_when_the_cap_binds():
    messages = []
    hits = keyword_hits(
        _PagedCollection(100),
        "novelty",
        batch_size=30,
        max_chunks=70,
        progress=messages.append,
    )
    assert len(hits) == 70
    assert any("cap" in m and "keyword_max_chunks" in m for m in messages)


def test_keyword_hits_no_warning_when_everything_fits():
    messages = []
    hits = keyword_hits(
        _PagedCollection(50),
        "novelty",
        batch_size=30,
        max_chunks=70,
        progress=messages.append,
    )
    assert len(hits) == 50
    assert not any("cap" in m for m in messages)


@pytest.mark.parametrize(
    ("term", "false_text", "valid_text"),
    [
        ("novel", "A novelist writes stories.", "A novel is a work of fiction."),
        (
            "social capital",
            "Social capitalism is an economic system.",
            "Social capital is relational.",
        ),
        (
            "novel idea",
            "That is a novel. Idea formation takes time.",
            "A novel idea changes the field.",
        ),
    ],
)
def test_cap_counts_only_chunks_with_matching_sentences(
    tmp_config, fake_embedder, term, false_text, valid_text
):
    from cda.corpus.ingest import get_collection, ingest_records

    ingest_records(
        tmp_config,
        fake_embedder,
        [
            {"source": "first.abstract", "abstract": false_text},
            {"source": "second.abstract", "abstract": valid_text},
        ],
        progress=lambda _: None,
    )
    messages = []
    hits = keyword_hits(
        get_collection(tmp_config), term, batch_size=1, max_chunks=1, progress=messages.append
    )
    _, passages, _ = build_passages(term, hits, lambda *_: (False, "literature"), tmp_config)
    assert [p.text for p in passages] == [valid_text]
    assert not any("hit the cap" in m for m in messages)
