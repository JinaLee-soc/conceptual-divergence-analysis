"""Diagnostics aggregations and verification comparison — fully offline."""

from datetime import datetime

import pytest

from cda.diagnose.report import (
    corpus_works,
    field_shares,
    render_diagnosis_markdown,
    temporal_profile,
    topic_distribution,
    underrepresented_topics,
    venue_concentration,
)
from cda.registers.verify import (
    extract_register_structure,
    find_latest_run,
    render_agreement,
    verification_verdict,
)


class _FakeCollection:
    def __init__(self, metadatas):
        self._metadatas = metadatas

    def get(self, include=None, limit=1000, offset=0):
        return {"metadatas": self._metadatas[offset : offset + limit]}


def test_corpus_works_counts_distinct_relpaths_not_basenames():
    """Two `paper.pdf`s in different folders are two works, not one; extra chunks
    of the same relpath collapse. Keying on the basename would undercount."""
    metadatas = [
        {"pdf_filename": "paper.pdf", "pdf_relpath": "a/paper.pdf", "doi": "10.1/a"},
        {"pdf_filename": "paper.pdf", "pdf_relpath": "a/paper.pdf", "doi": "10.1/a"},
        {"pdf_filename": "paper.pdf", "pdf_relpath": "b/paper.pdf", "doi": "10.1/b"},
    ]
    works = corpus_works(_FakeCollection(metadatas))
    assert {w["relpath"] for w in works} == {"a/paper.pdf", "b/paper.pdf"}
    assert {w["doi"] for w in works} == {"10.1/a", "10.1/b"}


def test_corpus_works_falls_back_to_filename_without_relpath():
    """Older stores lacking pdf_relpath still key on the filename."""
    works = corpus_works(_FakeCollection([{"pdf_filename": "solo.pdf", "doi": "10.1/s"}]))
    assert len(works) == 1 and works[0]["relpath"] == "solo.pdf"


def work(venue: str, topic: str, field: str, year: int) -> dict:
    return {
        "primary_location": {"source": {"display_name": venue}},
        "primary_topic": {"display_name": topic, "field": {"display_name": field}},
        "publication_year": year,
    }


WORKS = (
    [work("Journal A", "Peer Review", "Social Sciences", 2015)] * 6
    + [work("Journal B", "Citation Analysis", "Social Sciences", 2005)] * 3
    + [work("Journal C", "Climate Modeling", "Environmental Science", 1995)]
)


def test_venue_concentration_shares_and_hhi():
    import pytest

    stats = venue_concentration(WORKS, top_n=2)
    assert stats["n_works"] == 10
    assert stats["n_venues"] == 3
    assert stats["top"][0] == ("Journal A", 6, 0.6)
    assert stats["top_share"] == pytest.approx(0.9)
    # HHI = 0.6² + 0.3² + 0.1² = 0.46
    assert stats["hhi"] == pytest.approx(0.46)


def test_topic_and_field_distributions():
    topics = topic_distribution(WORKS)
    assert topics[0] == ("Peer Review", "Social Sciences", 6)
    fields = field_shares(WORKS)
    assert fields[0] == ("Social Sciences", 9, 0.9)


def test_temporal_profile():
    profile = temporal_profile(WORKS)
    assert profile["min"] == 1995 and profile["max"] == 2015
    assert dict(profile["by_decade"])["2010s"] == 6
    assert profile["n_dated"] == 10


def test_underrepresented_topics():
    baseline = [{"display_name": name} for name in ("Peer Review", "Scientometrics", "Altmetrics")]
    missing = underrepresented_topics(baseline, {"Peer Review", "Citation Analysis"})
    assert missing == ["Scientometrics", "Altmetrics"]


def test_untouched_topic_count_is_not_the_print_limit():
    """The count must describe the corpus, not how many lines the report prints.

    The list used to be truncated before its length was reported, so a corpus
    missing thirty of a field's largest topics read as missing exactly ten.
    """
    from cda.diagnose.report import TOPICS_LISTED, render_diagnosis_markdown

    absent = [f"Topic {i}" for i in range(TOPICS_LISTED + 5)]
    assert len(underrepresented_topics([{"display_name": n} for n in absent], set())) == len(absent)

    report = render_diagnosis_markdown(
        {
            "works": [{"doi": "x"}],
            "with_doi": [{"doi": "x"}],
            "resolved": [1],
            "unresolved": [],
            "venue_stats": {
                "n_venues": 1,
                "n_works": 1,
                "n_unknown_venue": 0,
                "top": [],
                "top_share": 0.0,
                "hhi": 1.0,
            },
            "topics": [],
            "fields": [],
            "temporal": {"n_dated": 0},
            "baselines": [("Social Sciences", [{"display_name": n} for n in absent])],
            "corpus_topic_names": set(),
        },
        datetime(2026, 1, 2, 3, 4, 5),
    )
    assert f"{len(absent)} have no corpus presence" in report
    assert "… and 5 more" in report


def test_render_diagnosis_markdown_offline():
    """The report renderer is a pure function — testable without OpenAlex."""
    data = {
        "works": [{"doi": "x"}] * 10,
        "with_doi": [{"doi": "x"}] * 8,
        "resolved": [1] * 8,
        "unresolved": [{"filename": "a.pdf", "doi": "bad"}],
        "venue_stats": {
            "n_venues": 3,
            "n_works": 8,
            "n_unknown_venue": 0,
            "top": [("Journal A", 5, 0.625)],
            "top_share": 0.625,
            "hhi": 0.46,
        },
        "topics": [("Peer Review", "Social Sciences", 5)],
        "fields": [("Social Sciences", 8, 1.0)],
        "temporal": {
            "n_dated": 8,
            "min": 2000,
            "max": 2020,
            "median": 2010,
            "by_decade": [("2010s", 5)],
        },
        "baselines": [
            (
                "Social Sciences",
                [{"display_name": "Scientometrics"}, {"display_name": "Peer Review"}],
            )
        ],
        "corpus_topic_names": {"Peer Review"},
    }
    md = render_diagnosis_markdown(data, datetime(2026, 1, 2, 3, 4, 5))
    assert "# Corpus Composition Report" in md
    assert "Herfindahl" in md and "0.46" in md
    assert "| Journal A | 5 | 62.5% |" in md
    assert "Scientometrics" in md  # a declared-field topic with no corpus presence
    assert "## Temporal profile" in md and "| 2010s | 5 |" in md
    assert "a.pdf" in md  # unresolvable DOI listed


def test_render_diagnosis_no_baselines_message():
    data = {
        "works": [{"doi": "x"}],
        "with_doi": [{"doi": "x"}],
        "resolved": [1],
        "unresolved": [],
        "venue_stats": {
            "n_venues": 1,
            "n_works": 1,
            "n_unknown_venue": 0,
            "top": [],
            "top_share": 0.0,
            "hhi": 1.0,
        },
        "topics": [],
        "fields": [],
        "temporal": {"n_dated": 0},
        "baselines": [],
        "corpus_topic_names": set(),
    }
    md = render_diagnosis_markdown(data, datetime(2026, 1, 2, 3, 4, 5))
    assert "No declared fields configured" in md
    assert "## Temporal profile" not in md  # no dated works → section omitted


PRIMARY = """## 1. Polysemy Map

### Cluster A: novelty-as-repair
Semantic distance from other clusters: 4

### Cluster B: novelty-as-property
Semantic distance from Cluster A: 2
"""

SECONDARY = """## 1. Polysemy Map

### Cluster A: novelty as audience work
Semantic distance from Cluster B: 5
"""


def test_extract_register_structure():
    structure = extract_register_structure(PRIMARY)
    assert structure["researcher_register_labels"] == ["novelty-as-repair", "novelty-as-property"]
    assert structure["distance_ratings"] == [4, 2]


def _write_run(out_dir, slug, ts, suffix="", *, term=None):
    base = f"drift_{slug}_{ts}{suffix}"
    (out_dir / f"{base}.md").write_text(
        "<!-- generated: 2026-08-01T09:00:00 | model: claude-sonnet-4-6 | cda: v1 -->\nreport",
        encoding="utf-8",
    )
    (out_dir / f"{base}_prompt.md").write_text(
        f'TARGET TERM: "{term if term is not None else slug.replace("_", " ")}"\n\nprompt',
        encoding="utf-8",
    )


def test_find_latest_run_never_pairs_a_prefix_sharing_term(tmp_path):
    """A run for "novelty claims" must not shadow runs for "novelty".

    With the old prefix glob + lexicographic sort, drift_novelty_claims_*
    matched drift_novelty_* and sorted above every timestamp ('c' > '2'), so
    `cda verify --term novelty` always re-ran the wrong term's prompt.
    """
    _write_run(tmp_path, "novelty", "20260801_090000")
    _write_run(tmp_path, "novelty", "20260801_100000")
    _write_run(tmp_path, "novelty_claims", "20260801_110000")  # newer, other term
    prompt, report = find_latest_run(tmp_path, "novelty")
    assert report.name == "drift_novelty_20260801_100000.md"
    assert prompt.name == "drift_novelty_20260801_100000_prompt.md"


def test_find_latest_run_orders_by_timestamp_and_collision_suffix(tmp_path):
    _write_run(tmp_path, "novelty", "20260801_100000")
    _write_run(tmp_path, "novelty", "20260801_100000", suffix="-1")
    prompt, report = find_latest_run(tmp_path, "novelty")
    assert report.name == "drift_novelty_20260801_100000-1.md"
    assert prompt.name == "drift_novelty_20260801_100000-1_prompt.md"


def test_find_latest_run_skips_reports_missing_their_prompt(tmp_path):
    _write_run(tmp_path, "novelty", "20260801_090000")
    (tmp_path / "drift_novelty_20260801_100000.md").write_text("orphan", encoding="utf-8")
    _, report = find_latest_run(tmp_path, "novelty")
    assert report.name == "drift_novelty_20260801_090000.md"


@pytest.mark.parametrize(("wanted", "other"), [("C++", "C#"), ("social capital", "social-capital")])
def test_find_latest_run_checks_the_actual_term_when_slugs_collide(tmp_path, wanted, other):
    from cda.extraction.windows import safe_term_slug

    slug = safe_term_slug(wanted)
    assert slug == safe_term_slug(other)
    _write_run(tmp_path, slug, "20261019_090000", term=wanted)
    _write_run(tmp_path, slug, "20261019_100000", term=other)
    prompt, report = find_latest_run(tmp_path, wanted)
    assert "090000" in report.name
    assert f'TARGET TERM: "{wanted}"' in prompt.read_text()
    assert "100000" in find_latest_run(tmp_path, other)[1].name


def test_find_latest_run_refuses_a_colliding_or_unidentified_prompt(tmp_path):
    _write_run(tmp_path, "c", "20261019_090000", term="C#")
    with pytest.raises(FileNotFoundError):
        find_latest_run(tmp_path, "C++")
    (tmp_path / "drift_c_20261019_090000_prompt.md").write_text("unidentified prompt")
    with pytest.raises(FileNotFoundError):
        find_latest_run(tmp_path, "C#")


def test_render_agreement_flags_count_mismatch():
    table = render_agreement(
        extract_register_structure(PRIMARY),
        extract_register_structure(SECONDARY),
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    )
    assert "⚠️ NO" in table  # 2 registers vs 1
    assert "novelty-as-repair" in table and "novelty as audience work" in table
    assert "worth a human look" in table or "ordinal variation" in table


def test_render_agreement_structural_match():
    table = render_agreement(
        extract_register_structure(PRIMARY),
        extract_register_structure(PRIMARY),
        "m1",
        "m2",
    )
    assert "✅ yes" in table


def test_verification_verdict_agreement_and_discrepancy():
    same = extract_register_structure(PRIMARY)  # 2 registers
    other = extract_register_structure(SECONDARY)  # 1 register

    verdict, banner = verification_verdict(same, same)
    assert verdict == "agreement"
    assert "STRUCTURAL AGREEMENT" in banner and "2 researcher registers" in banner
    assert "⚠️" not in banner

    verdict, banner = verification_verdict(same, other)
    assert verdict == "discrepancy"
    assert "⚠️" in banner and "(2 vs 1)" in banner
    assert "read both reports" in banner


def test_verification_verdict_empty_is_discrepancy():
    empty = extract_register_structure("no register headings here")
    verdict, _banner = verification_verdict(empty, empty)
    assert verdict == "discrepancy"  # 0 registers on both sides is not agreement


def test_existing_chroma_store_used_directly(tmp_path, fake_embedder):
    """corpus_dir that IS a chroma store (contains chroma.sqlite3) works as-is."""
    from cda.config import load_config
    from cda.corpus.ingest import get_collection, ingest_records

    store_dir = tmp_path / "existing_store"
    nested_config = load_config(overrides={"paths": {"corpus_dir": str(store_dir)}})
    # Build a store the normal way first (lands at <corpus_dir>/chroma)…
    ingest_records(
        nested_config,
        fake_embedder,
        [{"source": "a.pdf", "abstract": "novelty text here for the record"}],
        progress=lambda _msg: None,
    )
    # …then point corpus_dir AT the store directory itself.
    direct_config = load_config(overrides={"paths": {"corpus_dir": str(store_dir / "chroma")}})
    assert get_collection(direct_config).count() >= 1


# --- diagnose acquisition-date window ---------------------------------------


class _WindowCollection:
    """Chunks from three PDFs acquired on different dates, plus one undated."""

    _ROWS = [
        {
            "pdf_relpath": "old.pdf",
            "pdf_filename": "old.pdf",
            "doi": "10.1/a",
            "date_added_fs": "2024-01-05 10:00:00",
            "date_added": "2026-07-01 10:00:00",
        },
        {
            "pdf_relpath": "old.pdf",
            "pdf_filename": "old.pdf",
            "doi": "10.1/a",
            "date_added_fs": "2024-01-05 10:00:00",
            "date_added": "2026-07-01 10:00:00",
        },
        {
            "pdf_relpath": "new.pdf",
            "pdf_filename": "new.pdf",
            "doi": "10.1/b",
            "date_added_fs": "2026-06-20 10:00:00",
            "date_added": "2026-07-01 10:00:00",
        },
        {"pdf_relpath": "undated.pdf", "pdf_filename": "undated.pdf", "doi": "10.1/c"},
    ]

    def get(self, include=None, limit=1000, offset=0):
        return {"metadatas": self._ROWS[offset : offset + limit]}


def _window(**overrides):
    base = {
        "active": True,
        "added_from": None,
        "added_to": "2026-05-01 23:59:59",
        "include_undated": False,
        "date_source": "filesystem",
    }
    base.update(overrides)
    return base


def test_corpus_works_without_a_window_keeps_everything():
    works = corpus_works(_WindowCollection())
    assert {w["relpath"] for w in works} == {"old.pdf", "new.pdf", "undated.pdf"}


def test_corpus_works_drops_works_acquired_after_the_window():
    """The point of the flag: describe the corpus an --added-to analysis saw."""
    works = corpus_works(_WindowCollection(), _window())
    assert {w["relpath"] for w in works} == {"old.pdf"}


def test_undated_works_follow_include_undated():
    kept = {w["relpath"] for w in corpus_works(_WindowCollection(), _window(include_undated=True))}
    assert kept == {"old.pdf", "undated.pdf"}


def test_date_source_selects_the_clock():
    """Under the Zotero clock every dated work is a 2026 import, so the same
    2026-05-01 bound that keeps old.pdf on the filesystem clock drops it."""
    works = corpus_works(_WindowCollection(), _window(date_source="zotero"))
    assert works == []


def test_verify_record_then_mock_replay_round_trip(tmp_path, monkeypatch):
    """cda verify --mock-llm used to be unreachable: no shipped path ever
    recorded a verification-model fixture, so the fixture-miss remedy could
    not be followed. --record-fixtures on a live verify now records one, and
    the mock rerun replays it."""
    from cda.config import load_config
    from cda.registers import verify as verify_mod

    _write_run(tmp_path, "novelty", "20260801_090000")
    config = load_config(
        overrides={
            "paths": {"output_dir": str(tmp_path)},
            "llm": {"verification_model": "claude-haiku-4-5-20251001"},
        }
    )

    class StubSecondModel:
        name = "stub"
        model = "claude-haiku-4-5-20251001"

        def complete(self, *, system, user, max_tokens, temperature):
            return "### Cluster R1: replayed-register\nSemantic distance from Cluster A: 2\n"

    monkeypatch.setattr(verify_mod, "resolve_model_spec", lambda _spec: StubSecondModel())
    fixtures = tmp_path / "fixtures"
    recorded = verify_mod.run_verify(
        "novelty",
        config,
        record_fixtures=True,
        fixtures_dir=fixtures,
        progress=lambda _msg: None,
    )
    assert any(fixtures.glob("*.json")), "the live verify must record a fixture"

    # The mock rerun must replay without touching resolve_model_spec at all.
    monkeypatch.setattr(
        verify_mod,
        "resolve_model_spec",
        lambda _spec: (_ for _ in ()).throw(AssertionError("mock path went live")),
    )
    replayed = verify_mod.run_verify(
        "novelty",
        config,
        mock=True,
        fixtures_dir=fixtures,
        progress=lambda _msg: None,
    )
    assert replayed["verdict"] == recorded["verdict"]


def test_primary_report_boilerplate_does_not_leak_into_ratings():
    """The primary side is parsed from the RENDERED report; the appended
    computed sections (distance-scale legend etc.) used to feed spurious
    'distance N' mentions into the primary side only."""
    rendered = (
        PRIMARY
        + "\n## Clustering Stability (computed)\n"
        + "stable\n\n## Run Parameters\n"
        + "- Semantic distance scale: 1 = terminological variants; "
        + "5 = incommensurable.\n"
    )
    structure = extract_register_structure(rendered)
    assert structure["distance_ratings"] == [4, 2]  # legend's 1 and 5 excluded
    assert structure["researcher_register_labels"] == [
        "novelty-as-repair",
        "novelty-as-property",
    ]


def test_openalex_200_with_non_json_body_names_the_cause(monkeypatch):
    """A captive portal or intercepting proxy answers 200 with HTML; the bare
    'Expecting value: line 1 column 1' parse error named neither OpenAlex nor
    a next step."""
    import pytest
    import requests

    from cda.diagnose import openalex as oa

    class PortalResponse:
        status_code = 200

        def json(self):
            raise requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)

    monkeypatch.setenv("OPENALEX_MAILTO", "x@example.org")
    monkeypatch.setattr(oa.requests, "get", lambda *a, **k: PortalResponse())
    with pytest.raises(oa.OpenAlexError, match="captive portal|proxy"):
        oa.get_json("works")


def _verify_config(tmp_path):
    from cda.config import load_config

    return load_config(
        overrides={
            "paths": {"output_dir": str(tmp_path)},
            "llm": {"verification_model": "claude-haiku-4-5-20251001"},
        }
    )


def test_verify_mock_refuses_an_empty_fixtures_dir_before_any_work(tmp_path):
    """The refusal must come before the run files are even looked for — no
    run exists here, and the empty-fixtures message is still the one raised."""
    from cda.llm.base import LLMError
    from cda.registers.verify import run_verify

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    with pytest.raises(LLMError, match="nothing to replay for cda verify"):
        run_verify(
            "novelty",
            _verify_config(tmp_path),
            mock=True,
            fixtures_dir=fixtures,
            progress=lambda _msg: None,
        )


def test_verify_fixture_miss_names_the_live_command_that_records_one(tmp_path):
    """No verification-model fixture ships with CDA, and `cda demo` records
    the primary model only — so the generic miss advice ("run the demo")
    would loop here. The verification path says what actually creates one."""
    from cda.llm.base import FixtureMissError
    from cda.registers.verify import run_verify

    _write_run(tmp_path, "novelty", "20260801_090000")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    # Some unrelated recording, so the emptiness precheck passes and the
    # per-prompt miss is what we exercise.
    (fixtures / "0123456789abcdef.json").write_text('{"response": "x"}', encoding="utf-8")

    with pytest.raises(FixtureMissError) as excinfo:
        run_verify(
            "novelty",
            _verify_config(tmp_path),
            mock=True,
            fixtures_dir=fixtures,
            progress=lambda _msg: None,
        )
    message = str(excinfo.value)
    assert "cda verify --term novelty --record-fixtures" in message
    assert "claude-haiku-4-5-20251001" in message


def _capturing_second_model(captured):
    class CapturingSecondModel:
        name = "stub"
        model = "claude-haiku-4-5-20251001"

        def complete(self, *, system, user, max_tokens, temperature):
            captured.append(system)
            return "### Cluster R1: some-register\nSemantic distance from Cluster A: 2\n"

    return CapturingSecondModel()


def test_verify_prefers_the_runs_saved_system_prompt(tmp_path, monkeypatch):
    """The premise is byte-identical inputs: when the run saved its system
    prompt ({base}_system.md), verification must use that copy verbatim, not
    whatever the prompt file currently says."""
    from cda.registers import verify as verify_mod

    _write_run(tmp_path, "novelty", "20260801_090000")
    saved_system = "SAVED SYSTEM PROMPT: the run's own instructions.\n"
    (tmp_path / "drift_novelty_20260801_090000_system.md").write_text(
        saved_system, encoding="utf-8"
    )
    captured: list[str] = []
    monkeypatch.setattr(
        verify_mod, "resolve_model_spec", lambda _spec: _capturing_second_model(captured)
    )
    lines: list[str] = []
    verify_mod.run_verify("novelty", _verify_config(tmp_path), progress=lines.append)
    assert captured == [saved_system]
    assert not any("predates system-prompt persistence" in line for line in lines)


def test_verify_falls_back_to_the_configured_system_prompt_with_a_note(tmp_path, monkeypatch):
    """Pre-persistence runs saved no system prompt; the configured one is the
    only candidate, and the assumption is said out loud instead of silently
    weakening the byte-identical premise."""
    from cda.registers import verify as verify_mod
    from cda.registers.prompt_loader import load_system_prompt

    _write_run(tmp_path, "novelty", "20260801_090000")  # no _system.md
    config = _verify_config(tmp_path)
    captured: list[str] = []
    monkeypatch.setattr(
        verify_mod, "resolve_model_spec", lambda _spec: _capturing_second_model(captured)
    )
    lines: list[str] = []
    verify_mod.run_verify("novelty", config, progress=lines.append)
    assert captured == [load_system_prompt(config)]
    assert any("predates system-prompt persistence" in line for line in lines)


def test_verify_reports_an_undecodable_run_file_instead_of_a_codec_error(tmp_path):
    """Damage in the chosen run's prompt body is reported with the recovery
    advice (the finder reads only the head, so the body is first read here)."""
    from cda.llm.base import LLMError
    from cda.registers import verify
    from cda.registers.verify import run_verify

    _write_run(tmp_path, "novelty", "20260801_090000")
    body = b'TARGET TERM: "novelty"\n\n' + b"x" * verify._PROMPT_HEAD_BYTES + b"caf\xc3"
    (tmp_path / "drift_novelty_20260801_090000_prompt.md").write_bytes(body)
    with pytest.raises(LLMError, match="not readable as UTF-8"):
        run_verify(
            "novelty",
            _verify_config(tmp_path),
            fixtures_dir=tmp_path / "fixtures",
            progress=lambda _msg: None,
        )


def test_verify_names_a_run_whose_prompt_head_is_undecodable(tmp_path):
    """A prompt unidentifiable from its head is skipped by the finder and
    named in the no-run error, never surfaced as a bare codec error."""
    from cda.registers.verify import run_verify

    _write_run(tmp_path, "novelty", "20260801_090000")
    (tmp_path / "drift_novelty_20260801_090000_prompt.md").write_bytes(b"prompt caf\xc3")
    with pytest.raises(FileNotFoundError) as excinfo:
        run_verify(
            "novelty",
            _verify_config(tmp_path),
            fixtures_dir=tmp_path / "fixtures",
            progress=lambda _msg: None,
        )
    message = str(excinfo.value)
    assert "1 same-slug run(s) were skipped" in message
    assert "drift_novelty_20260801_090000_prompt.md: not UTF-8" in message
