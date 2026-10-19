"""Method contracts from the paper, checked with fictional data only."""

import pytest

from cda.config import load_config
from cda.divergence.report import _saturation_line
from cda.extraction.draft_passages import extract_draft_passages
from cda.extraction.windows import extract_term_occurrences, term_regex
from cda.pipeline import analyze_term, extract_passages
from cda.registers.audit import audit_report
from cda.registers.grounding import check_quotations


def windows_of(text, regex, **settings):
    return [window for _, window in extract_term_occurrences(text, regex, **settings)]


class NoSemantic:
    def embed(self, texts):
        raise AssertionError("extraction must not embed a query or draft")


class LiteralCollection:
    def query(self, **kw):
        raise AssertionError("semantic query must not run")

    def get(self, **kw):
        if kw.get("offset", 0):
            return {"ids": []}
        return {
            "ids": ["a"],
            "documents": ["Novelty matters. Unrelated background."],
            "metadatas": [{"filename": "source.pdf"}],
        }


def test_extraction_is_literal_only(tmp_path):
    c = load_config(overrides={"paths": {"drafts_dir": str(tmp_path)}})
    researcher, literature, _ = extract_passages(
        "novelty", c, embedder=NoSemantic(), collection=LiteralCollection(), progress=lambda _: None
    )
    assert researcher == []
    assert [p.text for p in literature] == ["Novelty matters."]
    assert (
        extract_draft_passages(
            "novelty", {"a.md": "An entirely different subject."}, NoSemantic(), c
        )
        == []
    )


def test_default_unit_is_whole_literal_sentence():
    long = "Background " + "explanation " * 100 + "novelty ends this sentence."
    assert windows_of(long, term_regex("novelty")) == [long]
    assert windows_of("Prior context. Novelty matters. Next context.", term_regex("novelty")) == [
        "Novelty matters."
    ]
    assert windows_of("The controller manages operations.", term_regex("control")) == []


@pytest.mark.parametrize("number", [2, 3])
def test_mapping_rate_counts_changes_on_both_sides(number):
    body = f"""## 1. Polysemy Map
### Cluster A: merged
- **k-means/LLM mapping:** merges embedding clusters 0 and 1.
### Cluster B: split
- **k-means/LLM mapping:** splits embedding cluster 1 across registers A and B.
## {number}. Researcher vs. Literature Divergence
### Literature Register I: common
- **k-means/LLM mapping:** merges embedding clusters 0 and 1.
"""
    audit = audit_report(body, 2, 2)
    assert audit["researcher"]["divergence_rate"] == 1
    assert audit["literature"]["divergence_rate"] == 1


def test_grounding_checks_source_short_quotes_and_fragment_order():
    prompt = 'Sample 1 [top1] [a.pdf]: "Novelty matters. These six words form a longer passage."\nSample 2 [top1] [b.pdf]: "Other six words belong to another source entirely."'
    assert check_quotations('"Novelty matters." - a.pdf', prompt)["n_verified"] == 1
    assert check_quotations('"Novelty matters." - b.pdf', prompt)["n_verified"] == 0
    assert (
        check_quotations(
            '"These six words form a longer passage … Other six words belong to another source entirely"',
            prompt,
        )["n_verified"]
        == 0
    )
    assert (
        check_quotations(
            '"These six words form a longer passage … Novelty matters" - a.pdf', prompt
        )["n_verified"]
        == 0
    )


def test_saturation_reports_selection_without_asserting_representativeness():
    text = _saturation_line(
        {
            "enabled": True,
            "ran": True,
            "n_candidates": 1000,
            "n_selected": 75,
            "history": [{"n": 100}],
            "max_passages": 1200,
            "saturation_reached": True,
            "ari_threshold": 0.8,
        }
    )
    assert "75/1000 (7.5%)" in text
    assert "exhausted: yes" in text
    assert "does not establish representativeness" in text


def test_nonpersistent_live_call_is_refused(tmp_path):
    with pytest.raises(ValueError, match="dry-run"):
        analyze_term("novelty", load_config(), persist=False)


def test_memory_preview_writes_no_private_artifacts(tmp_path, fake_embedder, capsys):
    c = load_config(
        overrides={
            "paths": {
                "drafts_dir": str(tmp_path / "drafts"),
                "output_dir": str(tmp_path / "outputs"),
            },
            "clustering": {"min_cluster_size": 1},
        }
    )
    result = analyze_term(
        "novelty",
        c,
        dry_run=True,
        persist=False,
        draft_texts={
            "private.md": "Novelty matters in private research. Novelty measures unexpected recombinations. Novelty depends on audience repair. Novelty is a disputed criterion in reviews."
        },
        collection=LiteralCollection(),
        embedder=fake_embedder,
        progress=lambda _: None,
    )
    assert result["status"] == "dry-run"
    assert "private research" in result["user_prompt"]
    assert not list(tmp_path.iterdir())
    assert "private research" not in capsys.readouterr().out


def test_saturation_at_cap_does_not_claim_early_stopping():
    text = _saturation_line(
        {
            "enabled": True,
            "ran": True,
            "n_candidates": 1000,
            "n_selected": 75,
            "history": [{"n": 100}],
            "max_passages": 100,
            "saturation_reached": True,
        }
    )
    assert "75/1000 (7.5%)" in text
    assert "exhausted: no" in text


def test_metadata_tags_and_explicit_override():
    from cda.registers.assemble import _researcher_context
    from cda.types import Passage

    passages = [
        Passage(
            chunk_id="a",
            source="a.pdf",
            year="2020",
            text="Novelty.",
            similarity=0.0,
            metadata={"zotero_tags": "MyWork; Science;Evaluation"},
        )
    ]
    context = _researcher_context(load_config(), passages)
    assert "#Science" in context and "#Evaluation" in context and "#MyWork" not in context
    context = _researcher_context(load_config(overrides={"themes": ["Chosen"]}), passages)
    assert "#Chosen" in context and "#Science" not in context


def test_keyword_retrieval_default_has_no_4000_chunk_cap():
    from cda.extraction.retrieval import keyword_hits

    class Collection:
        def get(self, **kw):
            ids = list(range(4001))[kw["offset"] : kw["offset"] + kw["limit"]]
            return {
                "ids": ids,
                "documents": ["Novelty."] * len(ids),
                "metadatas": [None] * len(ids),
            }

    assert len(keyword_hits(Collection(), "novelty")) == 4001
