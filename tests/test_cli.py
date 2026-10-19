"""CLI wiring: exit codes and friendly errors (no network, no corpus)."""

from pathlib import Path

import pytest

from cda.cli import build_parser, main


def test_version_flag():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0


def test_all_subcommands_registered():
    parser = build_parser()
    invocations = {
        "ingest": ["ingest", "--pdf-dir", "x"],
        "analyze": ["analyze", "--term", "x"],
        "diagnose": ["diagnose"],
        "emergence": ["emergence", "--term", "x"],
        "validate": ["validate", "--term", "x"],
        "verify": ["verify", "--term", "x"],
        "benchmark": ["benchmark"],
        "demo": ["demo"],
    }
    for command, argv in invocations.items():
        args = parser.parse_args(argv)
        assert callable(args.func), command


def test_analyze_without_corpus_is_a_friendly_error(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "analyze",
            "--term",
            "novelty",
            "--dry-run",
            "--corpus-dir",
            str(tmp_path / "missing"),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "cda ingest" in err


def test_missing_pdf_dir_is_a_friendly_error(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(["ingest", "--pdf-dir", str(tmp_path / "nope")])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_analyze_fixtures_default_is_under_output_dir(tmp_path, monkeypatch):
    """Fixtures recorded from a real corpus quote corpus passages, so the
    default must live under the git-ignored output directory — never in the
    committed demo/fixtures."""
    import cda.pipeline

    captured = {}

    def fake_analyze_term(term, config, **kwargs):
        captured.update(kwargs)
        return {"term": term, "status": "ok"}

    monkeypatch.setattr(cda.pipeline, "analyze_term", fake_analyze_term)
    code = main(["analyze", "--term", "novelty", "--out-dir", str(tmp_path / "out")])
    assert code == 0
    assert captured["fixtures_dir"] == tmp_path / "out" / "fixtures"


def test_independent_flag_overrides_the_joint_default():
    from cda.cli import _config_from_args

    parser = build_parser()
    default = _config_from_args(parser.parse_args(["analyze", "--term", "x"]))
    assert default.get("clustering.mode") == "joint"
    independent = _config_from_args(parser.parse_args(["analyze", "--term", "x", "--independent"]))
    assert independent.get("clustering.mode") == "independent"


def test_timeframe_flags_become_config_overrides():
    from cda.cli import _config_from_args

    args = build_parser().parse_args(
        [
            "analyze",
            "--term",
            "x",
            "--added-from",
            "2019-01-01",
            "--added-to",
            "2026-05-01",
            "--include-undated",
        ]
    )
    config = _config_from_args(args)
    assert config.get("timeframe.added_from") == "2019-01-01"
    assert config.get("timeframe.added_to") == "2026-05-01"
    assert config.get("timeframe.include_undated") is True

    # Untouched by default, and unrelated clustering defaults survive the merge.
    plain = _config_from_args(build_parser().parse_args(["analyze", "--term", "x"]))
    assert plain.get("timeframe.added_to") is None
    assert plain.get("clustering.seed") == 42


@pytest.mark.parametrize("command", ["analyze", "emergence", "validate"])
def test_analysis_commands_accept_the_timeframe_window(command):
    """The window belongs to every command that reads the corpus for a term."""
    args = build_parser().parse_args([command, "--term", "x", "--added-to", "2026-05-01"])
    assert args.added_to == "2026-05-01"


def test_bad_timeframe_bound_is_a_friendly_error(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(["analyze", "--term", "novelty", "--dry-run", "--added-to", "last tuesday"])
    assert code == 1
    assert "Could not read 'last tuesday'" in capsys.readouterr().err


def test_demo_flag_keeps_demo_paths_but_honors_analysis_flags(tmp_path, monkeypatch):
    """--demo fixes the toy corpus paths; the analysis flags still apply."""
    from cda.cli import _demo_or_user_config

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    args = build_parser().parse_args(
        ["emergence", "--term", "x", "--demo", "--added-to", "2020-06-30"]
    )
    config = _demo_or_user_config(args)
    assert config.get("timeframe.added_to") == "2020-06-30"
    assert str(config.path("paths.corpus_dir")).endswith("_runtime/corpus")


def test_demo_with_a_path_flag_is_refused_not_ignored(tmp_path, capsys, monkeypatch):
    """--demo fixes its paths, so a path flag beside it would be silently
    discarded — the user would look in --out-dir and find nothing."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    code = main(["validate", "--term", "x", "--demo", "--out-dir", str(tmp_path / "results")])
    assert code == 1
    err = capsys.readouterr().err
    assert "--out-dir" in err and "--demo" in err


def test_validate_session_flag_reopens_and_refreshes_the_report(tmp_path, capsys, monkeypatch):
    """--session re-renders an existing session's report; it must never start
    a new session (or, live, spend money on new readings)."""
    from datetime import datetime

    from cda.validation.session import create_session

    monkeypatch.chdir(tmp_path)
    session_dir, _session = create_session(
        "novelty",
        params={"seed": 42},
        sides={"literature": {"validated": True, "note": None}},
        partitions=[
            {
                "revision": 0,
                "side": "literature",
                "kind": "original",
                "assignments": {"a": 0, "b": 0},
                "clusters": [
                    {"cluster_id": 0, "n_passages": 2, "silhouette": None, "shown": ["a", "b"]}
                ],
            }
        ],
        passages={key: {"text": f"text {key}", "side": "literature"} for key in "ab"},
        readings=[],
        output_dir=tmp_path,
        now=datetime(2026, 7, 30, 12, 0, 0),
    )
    report_path = session_dir / f"{session_dir.name}.md"
    assert not report_path.exists()

    code = main(["validate", "--term", "novelty", "--session", str(session_dir)])
    assert code == 0
    out = capsys.readouterr().out
    assert "reopened session" in out
    assert report_path.is_file()
    # Only the reopened session exists — no second session was created.
    sessions = [p for p in tmp_path.iterdir() if p.name.startswith("validation_")]
    assert sessions == [session_dir]

    # Pointing --session at a session for another term is refused clearly.
    code = main(["validate", "--term", "consensus", "--session", str(session_dir)])
    assert code == 1
    assert "'novelty'" in capsys.readouterr().err


def _ingest_dated_store(tmp_path, fake_embedder, *, dated: bool):
    from cda.config import load_config
    from cda.corpus.ingest import ingest_records

    config = load_config(
        overrides={
            "paths": {
                "corpus_dir": str(tmp_path / "corpus"),
                "drafts_dir": str(tmp_path / "drafts"),
                "output_dir": str(tmp_path / "outputs"),
            }
        }
    )
    records = [
        {
            "source": f"2020_paper_{i}.abstract",
            "abstract": f"An alpha reading of novelty, case {i}. "
            f"Novelty appears again in variant {i}.",
            **({"date_added": f"2019-0{1 + i % 9}-07 09:00:00"} if dated else {}),
        }
        for i in range(12)
    ]
    ingest_records(config, fake_embedder, records, progress=lambda _m: None)
    return config


def test_emergence_writes_report_and_diagnostics(tmp_path, monkeypatch, fake_embedder):
    import json

    import cda.divergence.emergence

    _ingest_dated_store(tmp_path, fake_embedder, dated=True)
    monkeypatch.setattr(cda.divergence.emergence, "get_embedder", lambda _config: fake_embedder)
    code = main(
        [
            "emergence",
            "--term",
            "novelty",
            "--corpus-dir",
            str(tmp_path / "corpus"),
            "--drafts-dir",
            str(tmp_path / "drafts"),
            "--out-dir",
            str(tmp_path / "outputs"),
        ]
    )
    reports = list((tmp_path / "outputs").glob("emergence_novelty_*.md"))
    diagnostics = list((tmp_path / "outputs").glob("emergence_novelty_*_diagnostics.json"))
    assert len(reports) == 1 and len(diagnostics) == 1
    payload = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert payload["term"] == "novelty"
    assert payload["status"] in ("ok", "skipped", "degraded")
    assert code == (1 if payload["status"] == "skipped" else 0)


def test_emergence_without_dates_is_a_friendly_error(tmp_path, capsys, monkeypatch, fake_embedder):
    import cda.divergence.emergence

    _ingest_dated_store(tmp_path, fake_embedder, dated=False)
    monkeypatch.setattr(cda.divergence.emergence, "get_embedder", lambda _config: fake_embedder)
    code = main(
        [
            "emergence",
            "--term",
            "novelty",
            "--corpus-dir",
            str(tmp_path / "corpus"),
            "--drafts-dir",
            str(tmp_path / "drafts"),
            "--out-dir",
            str(tmp_path / "outputs"),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "--reingest" in err
    assert "Traceback" not in err


def test_diagnose_without_mailto_is_a_friendly_error(tmp_path, capsys, monkeypatch, fake_embedder):
    """OpenAlexError must exit 1 with the next step, not a traceback.

    polite_mailto raises before any network request, so this stays offline."""
    from cda.config import load_config
    from cda.corpus.ingest import ingest_records

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    ingest_records(
        config,
        fake_embedder,
        [{"source": "a.pdf", "abstract": "novelty appears in this abstract", "doi": "10.1/x"}],
        progress=lambda _msg: None,
    )
    monkeypatch.setenv("OPENALEX_MAILTO", "")
    code = main(["diagnose", "--corpus-dir", str(tmp_path / "corpus")])
    assert code == 1
    err = capsys.readouterr().err
    assert "OPENALEX_MAILTO" in err
    assert "Traceback" not in err


# --- bounding what a recursive walk sweeps in --------------------------------


def _library(tmp_path):
    """A curated folder shaped like a real one: papers at the top, plus
    sub-folders holding things that are not literature."""
    (tmp_path / "paper_a.pdf").write_bytes(b"%PDF-")
    (tmp_path / "paper_b.pdf").write_bytes(b"%PDF-")
    for sub in ("outputs", "#Agent", "Book"):
        (tmp_path / sub).mkdir(exist_ok=True)
        (tmp_path / sub / f"{sub}_thing.pdf").write_bytes(b"%PDF-")
    nested = tmp_path / "collections" / "2024"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "deep.pdf").write_bytes(b"%PDF-")
    return tmp_path


def test_recursive_walk_takes_everything_by_default(tmp_path):
    from cda.corpus.ingest import select_pdfs

    kept, excluded = select_pdfs(_library(tmp_path))
    assert {p.name for p in kept} == {
        "paper_a.pdf",
        "paper_b.pdf",
        "outputs_thing.pdf",
        "#Agent_thing.pdf",
        "Book_thing.pdf",
        "deep.pdf",
    }
    assert excluded == 0


def test_no_recursive_takes_only_the_top_level(tmp_path):
    from cda.corpus.ingest import select_pdfs

    kept, excluded = select_pdfs(_library(tmp_path), recursive=False)
    assert {p.name for p in kept} == {"paper_a.pdf", "paper_b.pdf"}
    assert excluded == 0  # not reached by the walk at all, so not "excluded"


def test_a_directory_pattern_skips_the_whole_subtree(tmp_path):
    from cda.corpus.ingest import select_pdfs

    kept, excluded = select_pdfs(_library(tmp_path), exclude=["outputs", "#Agent", "collections"])
    assert {p.name for p in kept} == {"paper_a.pdf", "paper_b.pdf", "Book_thing.pdf"}
    assert excluded == 3  # including the nested one under collections/2024


def test_a_wildcard_pattern_matches_the_relative_path(tmp_path):
    from cda.corpus.ingest import select_pdfs

    (tmp_path / "notes.draft.pdf").write_bytes(b"%PDF-")
    kept, excluded = select_pdfs(_library(tmp_path), recursive=False, exclude=["*.draft.pdf"])
    assert {p.name for p in kept} == {"paper_a.pdf", "paper_b.pdf"}
    assert excluded == 1


def test_selection_order_is_stable(tmp_path):
    """Chunk ids and Zotero collision detection both depend on a fixed order."""
    from cda.corpus.ingest import select_pdfs

    library = _library(tmp_path)
    first, _ = select_pdfs(library)
    second, _ = select_pdfs(library)
    assert first == second == sorted(first)


@pytest.mark.parametrize("excluded", ["a-alias", "a-alias/*.pdf"])
@pytest.mark.parametrize("nested", [False, True])
def test_excluded_alias_cannot_hide_an_included_pdf(tmp_path, excluded, nested):
    from cda.corpus.ingest import select_pdfs

    target = tmp_path / ("z-parent/papers" if nested else "z-papers")
    target.mkdir(parents=True)
    pdf = target / "paper.pdf"
    pdf.write_bytes(b"%PDF-")
    (tmp_path / "a-alias").symlink_to(target, target_is_directory=True)
    (target / "cycle").symlink_to(tmp_path, target_is_directory=True)
    kept, _ = select_pdfs(tmp_path, exclude=[excluded])
    assert kept == [pdf]


def test_cli_flags_add_to_configured_exclusions(tmp_path, monkeypatch):
    """A project config carries standing exclusions; a flag adds a one-off."""
    from cda.corpus.ingest import select_pdfs

    seen = {}

    def fake_ingest(config, embedder, pdf_dir, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr("cda.corpus.ingest.ingest_pdf_dir", fake_ingest)
    monkeypatch.setattr("cda.corpus.embeddings.get_embedder", lambda config: object())
    user_config = tmp_path / "user.yaml"
    user_config.write_text("corpus:\n  exclude: ['outputs']\n", encoding="utf-8")

    main(
        [
            "ingest",
            "--pdf-dir",
            str(_library(tmp_path)),
            "--config",
            str(user_config),
            "--exclude",
            "#Agent",
            "--no-recursive",
        ]
    )
    assert seen["exclude"] == ["outputs", "#Agent"]
    assert seen["recursive"] is False
    # And the combination does what the two sources say.
    kept, _ = select_pdfs(tmp_path, recursive=False, exclude=seen["exclude"])
    assert all(p.parent == tmp_path for p in kept)


def test_mock_with_record_fixtures_is_refused(capsys):
    """Recording needs a live call; silently ignoring the flag would let a
    user believe fixtures were written."""
    code = main(["analyze", "--term", "x", "--mock-llm", "--record-fixtures"])
    assert code == 1
    assert "--record-fixtures" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "--term", "x", "--dry-run", "--record-fixtures"],
        ["demo", "--dry-run", "--record-fixtures"],
    ],
)
def test_dry_run_with_record_fixtures_is_refused(argv, capsys):
    """A dry run never builds a provider, so the flag pair used to be
    accepted silently and recorded nothing."""
    code = main(argv)
    assert code == 1
    err = capsys.readouterr().err
    assert "--dry-run" in err and "--record-fixtures" in err and "records nothing" in err


def test_session_with_record_fixtures_is_refused(tmp_path, capsys, monkeypatch):
    """Sibling of the dry-run guard: reopening a session makes no calls, so
    recording is impossible and silence would imply fixtures were written."""
    monkeypatch.chdir(tmp_path)
    code = main(["validate", "--term", "x", "--session", "somewhere", "--record-fixtures"])
    assert code == 1
    err = capsys.readouterr().err
    assert "--session" in err and "records nothing" in err


def test_unicode_encode_error_notes_that_disk_files_are_unaffected(tmp_path, capsys, monkeypatch):
    """A terminal that cannot display a character (Windows cp1252 consoles)
    fails while printing, after the run's files are already safely on disk;
    the bare "charmap codec" message read as data loss."""
    import cda.pipeline

    def printing_fails(term, config, **kwargs):
        raise UnicodeEncodeError("charmap", "→", 0, 1, "character maps to <undefined>")

    monkeypatch.setattr(cda.pipeline, "analyze_term", printing_fails)
    monkeypatch.chdir(tmp_path)
    code = main(["analyze", "--term", "novelty", "--out-dir", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "UTF-8" in err and "unaffected" in err
    assert "Traceback" not in err
