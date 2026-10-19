"""Term-sentence extraction and helpers."""

import unicodedata

from cda.extraction.windows import (
    extract_term_occurrences,
    infer_year,
    safe_term_slug,
    split_sentences,
    term_regex,
)


def windows_of(text, regex, **settings):
    return [window for _, window in extract_term_occurrences(text, regex, **settings)]


LONG_A = "The concept of novelty has been central to organizational analysis for decades."
LONG_B = "Scholars disagree about how novelty should be operationalized in practice."


def test_extracts_only_term_sentences():
    text = f"{LONG_A} This sentence is about something else entirely. {LONG_B}"
    windows = windows_of(text, term_regex("novelty"))
    assert windows == [LONG_A, LONG_B]


def test_short_sentence_gains_context():
    text = f"{LONG_A} Novelty matters here. {LONG_B}"
    windows = windows_of(text, term_regex("novelty"), min_window_chars=40)
    # "Novelty matters here." is under 40 chars → ±1 sentence context window.
    assert any("Novelty matters here." in w and LONG_A in w for w in windows)


def test_complete_repeated_sentences_are_distinct_occurrences():
    sentence = "Novelty " + "x" * 800 + "."
    windows = windows_of(sentence + " " + sentence, term_regex("novelty"))
    assert len(windows) == 2
    assert windows[0] == sentence  # default preserves complete sentences


def test_word_boundary_matching():
    text = "Innovativeness is not the same word. True novelty appears here in this sentence."
    windows = windows_of(text, term_regex("novelty"))
    assert len(windows) == 1
    assert "innovativeness" not in windows[0].lower()


def test_term_with_trailing_non_word_chars_matches_every_occurrence():
    """ "C++" never survived a blanket \\b...\\b: the trailing \\b after "+"
    demands a following word character, so no real sentence ever matched and
    every chunk collapsed to the single first-occurrence fallback snippet."""
    text = (
        "The C++ implementation of the parser is described in this opening sentence. "
        "A second discussion of C++ template metaprogramming follows in this sentence. "
        "This closing sentence mentions neither languages nor any tooling at all."
    )
    windows = windows_of(text, term_regex("C++"))
    assert len(windows) == 2  # both term sentences, via the sentence path
    assert all("C++" in w for w in windows)
    # The word-character side still anchors: "C++" must not hit inside "ObjC++".
    assert not term_regex("C++").search("compiled as ObjC++ today")


def test_term_with_leading_non_word_char_matches():
    text = (
        "Applications built on .NET are described in this opening sentence of the chunk. "
        "An unrelated middle sentence with enough words sits between the two here. "
        "The .NET runtime behavior is analyzed further in this closing sentence."
    )
    windows = windows_of(text, term_regex(".NET"))
    assert len(windows) == 2
    # The trailing edge is a word character, so it still anchors whole-word.
    assert not term_regex(".NET").search("a high .NETwork stack")


def test_term_regex_composes_nfd_terms_to_match_nfc_text():
    """Corpus text is NFC by contract (normalized at ingest); a term pasted
    from a macOS filename arrives NFD. Both spellings must hit the same
    stored text."""
    nfc_text = unicodedata.normalize(
        "NFC", "The café study reframes conversation venues in this sentence of decent length."
    )
    nfd_term = unicodedata.normalize("NFD", "café")
    assert nfd_term != unicodedata.normalize("NFC", "café")  # really two spellings
    assert term_regex(nfd_term).pattern == term_regex(unicodedata.normalize("NFC", "café")).pattern
    assert windows_of(nfc_text, term_regex(nfd_term))


def test_context_mode_adds_neighbors():
    text = f"First filler sentence stands here. {LONG_A} Trailing sentence follows after it."
    windows = windows_of(text, term_regex("novelty"), window_mode="context")
    assert "First filler sentence" in windows[0]
    assert "Trailing sentence" in windows[0]


def test_split_sentences_basic():
    assert split_sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]


def test_infer_year_precedence():
    assert infer_year("paper_1999.pdf", {"year": "2005"}) == "2005"
    assert infer_year("paper_1999.pdf", {}) == "1999"
    assert infer_year("paper.pdf", {}) == "unknown"


def test_safe_term_slug():
    assert safe_term_slug("boundary work!") == "boundary_work"


def test_safe_term_slug_distinguishes_non_latin_terms():
    # Terms with no ASCII letters/digits used to collapse to one shared slug
    # ("term"), so two Korean terms' reports, sessions, and verify lookups
    # silently shared a filename family.
    korean = safe_term_slug("참신성")
    chinese = safe_term_slug("合意")
    assert korean != chinese
    # Deterministic across runs (file lookup depends on it) and filesystem-safe.
    assert korean == safe_term_slug("참신성")
    assert korean.startswith("term_")
    assert korean.replace("_", "").isalnum()


def test_safe_term_slug_mixed_script_keeps_ascii_part():
    assert safe_term_slug("참신성 novelty") == "novelty"


def test_safe_term_slug_is_unicode_normalization_insensitive():
    # macOS filesystems hand back decomposed (NFD) names while a terminal
    # types composed (NFC) ones, so the same term arrives in two spellings.
    # If they slugged differently, `cda verify --term <pasted from filename>`
    # would not find the run `cda analyze --term <typed>` had just written.
    for term in ("참신성", "café"):
        composed = unicodedata.normalize("NFC", term)
        decomposed = unicodedata.normalize("NFD", term)
        assert composed != decomposed  # the two spellings really do differ
        assert safe_term_slug(composed) == safe_term_slug(decomposed)
