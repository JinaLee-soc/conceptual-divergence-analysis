"""MyWork resolution: signals, priority order, and thresholds."""

import pytest

from cda.extraction.mywork import mywork_classifier


def resolver(**overrides):
    base = {
        "priority": ["filename_patterns", "metadata_flags", "zotero_tags", "profile_titles"],
        "filename_patterns": [],
        "publication_titles": [],
        "title_match_min_overlap": 3,
        "title_match_min_ratio": 0.55,
    }
    base.update(overrides)
    return mywork_classifier(base)


def test_default_is_literature():
    is_researcher, reason = resolver()("Smith_2019_Some_Paper.pdf", {})
    assert not is_researcher
    assert reason == "default:literature"


def test_filename_pattern_signal():
    r = resolver(filename_patterns=["*working_paper*"])
    is_researcher, reason = r("my_working_paper_v3.pdf", {})
    assert is_researcher
    assert reason == "pattern:*working_paper*"


def test_metadata_flag_signal():
    for key in ("self_authored", "is_self", "self", "author_type", "owner"):
        is_researcher, reason = resolver()("x.pdf", {key: "true"})
        assert is_researcher, key
        assert reason == f"metadata:{key}"
    assert not resolver()("x.pdf", {"self_authored": "false"})[0]


def test_zotero_tag_signal_with_normalization():
    for raw in ("mywork", "#MyWork", " My-Work "):
        is_researcher, reason = resolver()("x.pdf", {"zotero_tags": f"sociology; {raw}"})
        assert is_researcher, raw
        assert reason == "zotero:mywork"
    assert not resolver()("x.pdf", {"zotero_tags": "networks; theory"})[0]


def test_profile_title_signal_and_thresholds():
    r = resolver(publication_titles=["Epistemic Cultures of Novelty Evaluation in Science"])
    hit, reason = r("Epistemic_Cultures_Novelty_Evaluation_Science_2023.pdf", {})
    assert hit
    assert reason == "profile:title_match"
    # Two shared tokens is below the min_overlap=3 bar.
    miss, _ = r("Novelty_Evaluation_2020.pdf", {})
    assert not miss


def test_priority_order_first_match_wins():
    r = resolver(filename_patterns=["*draft*"])
    metadata = {"self_authored": "true"}
    _, reason = r("my_draft_paper.pdf", metadata)
    assert reason.startswith("pattern:")  # patterns outrank metadata by default

    reordered = resolver(
        filename_patterns=["*draft*"],
        priority=["metadata_flags", "filename_patterns", "zotero_tags", "profile_titles"],
    )
    _, reason = reordered("my_draft_paper.pdf", metadata)
    assert reason == "metadata:self_authored"  # order is configurable


def test_unknown_signal_rejected():
    with pytest.raises(ValueError, match="Unknown MyWork signal"):
        resolver(priority=["telepathy"])
