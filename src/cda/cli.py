"""The ``cda`` command line:
ingest | analyze | diagnose | emergence | validate | verify | benchmark | demo.

Every subcommand runs in two modes: live (needs ANTHROPIC_API_KEY) and
--mock-llm (fixture replay; zero network, zero cost). Errors aim to tell you
the next step, not just what went wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cda import __version__
from cda.config import Config, ConfigError, load_config
from cda.corpus.embeddings import EmbeddingModelError
from cda.corpus.ingest import CorpusLockError
from cda.diagnose.openalex import OpenAlexError
from cda.llm.base import LLMError

# Every flag whose value is a filesystem path. An unset shell variable
# (`--out-dir "$OUT"`) expands to an empty string, and each of these would
# swallow that in its own way: falling back to the default, disabling a
# feature, or resolving to `Path("")`, the current directory. They are
# checked in one place so a new path flag inherits the guard.
_PATH_FLAGS = (
    "config",
    "corpus_dir",
    "drafts_dir",
    "out_dir",
    "fixtures_dir",
    "pdf_dir",
    "zotero_csv",
    "demo_dir",
    "session",
)


def _reject_empty_path_flags(args: argparse.Namespace) -> None:
    """Refuse a path flag that was passed but empty, naming the flag."""
    for flag in _PATH_FLAGS:
        value = getattr(args, flag, None)
        if value is not None and not str(value).strip():
            raise ConfigError(
                f"--{flag.replace('_', '-')} got an empty path (did an unset "
                "shell variable expand to nothing?). Pass a real path, or drop "
                "the flag to use the default."
            )


def _expand_home_in_path_flags(args: argparse.Namespace) -> None:
    """Expand a leading ``~`` in every path flag, in one place.

    A quoted ``--zotero-csv "~/export.csv"`` reaches Python with the tilde
    intact (the shell only expands an unquoted one). Expanding it here means
    every path flag honors it, instead of each consumer deciding for itself.
    """
    for flag in _PATH_FLAGS:
        value = getattr(args, flag, None)
        if value is not None and str(value).startswith("~"):
            setattr(args, flag, str(Path(value).expanduser()))


def _cli_overrides(args: argparse.Namespace) -> dict:
    """The config overrides implied by the flags on this command."""
    _reject_empty_path_flags(args)
    _expand_home_in_path_flags(args)
    overrides: dict = {}
    paths = {}
    for flag, key in (
        ("corpus_dir", "corpus_dir"),
        ("drafts_dir", "drafts_dir"),
        ("out_dir", "output_dir"),
    ):
        value = getattr(args, flag, None)
        if value:
            paths[key] = value
    if paths:
        overrides["paths"] = paths
    if getattr(args, "independent", False):
        overrides["clustering"] = {"mode": "independent"}
    timeframe = {
        key: getattr(args, key)
        for key in ("added_from", "added_to", "date_source")
        if getattr(args, key, None)
    }
    if getattr(args, "include_undated", False):
        timeframe["include_undated"] = True
    if timeframe:
        overrides["timeframe"] = timeframe
    return overrides


def _config_from_args(args: argparse.Namespace) -> Config:
    """Load config with CLI flags folded in as highest-precedence overrides."""
    return load_config(getattr(args, "config", None), overrides=_cli_overrides(args))


def _cmd_ingest(args: argparse.Namespace) -> int:
    from cda.corpus.embeddings import get_embedder
    from cda.corpus.ingest import ingest_pdf_dir

    config = _config_from_args(args)
    # Flags add to the config rather than replacing it, so a project config can
    # carry the standing exclusions and a one-off run can add more.
    recursive = config.get_bool("corpus.recursive", True) and not args.no_recursive
    # get_list wraps a bare `exclude: outputs` string — it would otherwise
    # iterate into single letters and exclude nothing while looking configured.
    exclude = [str(p) for p in config.get_list("corpus.exclude", [])] + list(args.exclude or [])
    ingest_pdf_dir(
        config,
        get_embedder(config),
        args.pdf_dir,
        zotero_csv=args.zotero_csv,
        reingest=args.reingest,
        recursive=recursive,
        exclude=exclude,
        prune_missing=args.prune_missing,
    )
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from cda.pipeline import analyze_term

    config = _config_from_args(args)
    # Default fixtures location: under the (git-ignored) output directory.
    # Fixtures recorded from a real corpus quote corpus passages, so they
    # must not default into the committed demo/fixtures directory.
    fixtures_dir = args.fixtures_dir or config.path("paths.output_dir") / "fixtures"
    if args.mock_llm and not args.dry_run:
        # Same reasoning as the validate path: the provider layer raises this
        # too, but only after extraction and clustering — minutes on a real
        # corpus. A dry run makes no LLM call, so it needs no fixtures.
        from cda.llm.mock import require_recorded_fixtures

        require_recorded_fixtures(fixtures_dir, what="cda analyze")
    result = analyze_term(
        args.term,
        config,
        mock=args.mock_llm,
        dry_run=args.dry_run,
        record_fixtures=args.record_fixtures,
        overwrite_fixtures=args.overwrite_fixtures,
        fixtures_dir=fixtures_dir,
        drafts_only=args.drafts_only,
    )
    if result["status"] == "skipped":
        print(f"[skipped] {result['reason']}")
        return 1
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from cda.demo import run_demo

    result = run_demo(
        term=args.term,
        mock=args.mock_llm,
        dry_run=args.dry_run,
        record_fixtures=args.record_fixtures,
        overwrite_fixtures=args.overwrite_fixtures,
        demo_dir=args.demo_dir,
    )
    return 0 if result["status"] in ("ok", "dry-run") else 1


def _demo_or_user_config(args: argparse.Namespace):
    if getattr(args, "demo", False):
        from cda.demo import demo_config, find_demo_dir

        # Demo paths and the fictional researcher's themes are fixed; the
        # analysis flags this command accepts still apply. A path flag beside
        # --demo would be silently discarded — the exact behavior the `cda
        # demo` parser refuses by not registering those flags at all — so
        # refuse it here too rather than send the user hunting for output
        # that went somewhere else.
        passed = [
            flag
            for attr, flag in (
                ("corpus_dir", "--corpus-dir"),
                ("drafts_dir", "--drafts-dir"),
                ("out_dir", "--out-dir"),
                ("config", "--config"),
            )
            if getattr(args, attr, None)
        ]
        if passed:
            raise ConfigError(
                f"--demo runs entirely inside demo/_runtime with fixed paths and "
                f"parameters, so {', '.join(passed)} would be ignored. Drop the "
                "flag(s), or drop --demo to run on your own corpus."
            )
        overrides = _cli_overrides(args)
        overrides.pop("paths", None)
        return demo_config(find_demo_dir(), extra_overrides=overrides)
    return _config_from_args(args)


def _ingest_demo_corpus(config):
    """Self-heal like `cda demo`: index the toy corpus so a fresh clone can
    run any --demo command directly. Returns (demo_dir, embedder)."""
    from cda.corpus.embeddings import get_embedder
    from cda.demo import ensure_toy_corpus_ingested, find_demo_dir

    demo_dir = find_demo_dir()
    embedder = get_embedder(config)
    ensure_toy_corpus_ingested(config, demo_dir, embedder)
    return demo_dir, embedder


def _cmd_diagnose(args: argparse.Namespace) -> int:
    from cda.diagnose.report import run_diagnose

    config = _demo_or_user_config(args)
    if getattr(args, "demo", False):
        _ingest_demo_corpus(config)
    fields = [name.strip() for name in (args.fields or "").split(",") if name.strip()]
    if not fields:
        # get_list wraps a bare string — `declared_fields: Social Sciences`
        # would otherwise iterate into one-letter "fields".
        fields = [str(f) for f in config.get_list("diagnose.declared_fields", [])]
    run_diagnose(config, declared_fields=fields)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from cda.registers.verify import run_verify

    config = _demo_or_user_config(args)
    fixtures_dir = args.fixtures_dir
    if fixtures_dir is None:
        # Absolute defaults, mirroring analyze/validate: a cwd-relative
        # "demo/fixtures" string silently resolved nowhere when the command
        # ran outside the repository root.
        if getattr(args, "demo", False):
            from cda.demo import find_demo_dir

            fixtures_dir = find_demo_dir() / "fixtures"
        else:
            fixtures_dir = config.path("paths.output_dir") / "fixtures"
    run_verify(
        args.term,
        config,
        mock=args.mock_llm,
        fixtures_dir=fixtures_dir,
        record_fixtures=args.record_fixtures,
    )
    return 0


def _cmd_emergence(args: argparse.Namespace) -> int:
    from cda.divergence.emergence import run_emergence

    config = _demo_or_user_config(args)
    embedder = None
    if getattr(args, "demo", False):
        _demo_dir, embedder = _ingest_demo_corpus(config)
    result = run_emergence(args.term, config, embedder=embedder)
    if result["status"] == "skipped":
        print(f"[emergence] skipped: {result['emergence']['reason']}", file=sys.stderr)
        return 1
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config = _demo_or_user_config(args)

    if args.session:
        # Re-open an existing session: re-render its report from the recorded
        # state and stop. No clustering, no LLM call — the session directory
        # already holds everything (so this also never spends money).
        if args.record_fixtures:
            # Sibling of the --mock-llm/--dry-run guards in main(): silence
            # would let the user believe fixtures were written.
            print(
                "error: --session --record-fixtures records nothing because "
                "reopening a session makes no calls; drop one of the flags.",
                file=sys.stderr,
            )
            return 1
        from datetime import datetime

        from cda.extraction.windows import term_identity
        from cda.validation.report import render_validation_report, write_validation_outputs
        from cda.validation.session import load_session, session_lock

        session_dir = Path(args.session)
        if not session_dir.is_dir():
            raise FileNotFoundError(
                f"No validation session directory at {session_dir}. Point "
                "--session at a directory named validation_<term>_<timestamp> "
                "under your output directory, or omit --session to start one."
            )
        # The GUI may be working on this session in another process; holding
        # the same lock its handlers use keeps this load→render→write from
        # interleaving with a concurrent save.
        with session_lock(session_dir):
            session = load_session(session_dir)
            # Same comparison as result lookup: case and spacing do not make
            # a different term, so `--term AI` resumes a session saved as "ai".
            if term_identity(str(session["term"])) != term_identity(args.term):
                print(
                    f"error: the session at {session_dir} is for term "
                    f"'{session['term']}', not '{args.term}'. Point --session at a "
                    f"session for this term, or rerun with --term '{session['term']}'.",
                    file=sys.stderr,
                )
                return 1
            now = datetime.now()
            written = write_validation_outputs(
                session_dir, session, render_validation_report(session, config, now), now
            )
        print(f"[validate] reopened session: {session_dir}")
        print(f"[validate] report refreshed: {written['report_file']}")
        return 0

    embedder = None
    fixtures_dir = args.fixtures_dir
    if getattr(args, "demo", False):
        demo_dir, embedder = _ingest_demo_corpus(config)
        if fixtures_dir is None:
            fixtures_dir = demo_dir / "fixtures" / "validation"

    if fixtures_dir is None:
        # Same default location the GUI's Validate tab uses, so readings
        # recorded by either replay in the other.
        fixtures_dir = config.path("paths.output_dir") / "fixtures" / "validation"

    if args.mock_llm and not any(Path(fixtures_dir).glob("*.json")):
        # The provider layer would raise the same refusal, but only after the
        # extraction and clustering stages have run — minutes on a real
        # corpus. The condition is checkable before any work starts.
        raise LLMError(
            f"--mock-llm has nothing to replay: no recorded readings under "
            f"{fixtures_dir}. Run once live to record them (they replay free "
            "afterwards), pass --fixtures-dir PATH, or use --demo for the "
            "committed demo readings."
        )

    from cda.validation.run import run_validation

    # Live runs record by default: each completed reading lands on disk as a
    # replayable fixture, so an interrupted run resumes without paying again
    # for what already succeeded — and the demo's first live run doubles as
    # the recording run that enables everyone's offline replay.
    record = args.record_fixtures or not args.mock_llm
    run_validation(
        args.term,
        config,
        mock=args.mock_llm,
        fixtures_dir=fixtures_dir,
        record_fixtures=record,
        embedder=embedder,
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    print(
        "Benchmarks are standalone scripts (kept out of the installed package "
        "so the paper's pipeline stays lean):\n"
        "  python benchmarks/clustering_comparison.py --demo --term novelty\n"
        "  python benchmarks/clustering_order_stability.py --demo --term novelty\n"
        "  python benchmarks/stability_protocol.py --demo --term novelty --mock-llm\n"
        "See benchmarks/README.md for pointing them at your own corpus."
    )
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to a YAML config (see config/config.example.yaml)")
    parser.add_argument("--corpus-dir", help="Override paths.corpus_dir")
    parser.add_argument("--drafts-dir", help="Override paths.drafts_dir")
    parser.add_argument("--out-dir", help="Override paths.output_dir")


def _add_timeframe(parser: argparse.ArgumentParser) -> None:
    """The acquisition-date window (Zotero "Date Added"), for analysis commands."""
    group = parser.add_argument_group(
        "corpus timeframe",
        'Restrict the corpus to works added to your library in a window (Zotero "Date '
        'Added" — when you acquired them, never their publication year), so a result '
        "can be reproduced later against the same snapshot.",
    )
    group.add_argument(
        "--added-from", metavar="YYYY-MM-DD", help="Ignore works added before this date"
    )
    group.add_argument(
        "--added-to",
        metavar="YYYY-MM-DD",
        help="Ignore works added after this date (inclusive of the whole day) — "
        "e.g. --added-to 2026-05-01 analyzes the library as it stood that day",
    )
    group.add_argument(
        "--include-undated",
        action="store_true",
        help='Keep passages whose source has no "Date Added" instead of dropping '
        "them (the window is then advisory; your drafts are never filtered either way)",
    )
    group.add_argument(
        "--date-source",
        choices=["zotero", "filesystem"],
        help='Which acquisition clock the window reads: "zotero" (default) is '
        'Zotero\'s "Date Added"; "filesystem" is when the PDF arrived on disk, '
        "which survives a rebuilt Zotero library. Reports name the clock used",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cda",
        description=(
            "Conceptual Divergence Analysis: compare how you use key theoretical "
            "terms against a researcher-curated literature corpus."
        ),
    )
    parser.add_argument("--version", action="version", version=f"cda {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest", help="Build the local corpus: PDFs (+ optional Zotero CSV export) → ChromaDB"
    )
    ingest.add_argument(
        "--pdf-dir", required=True, help="Directory of PDFs (sub-folders searched by default)"
    )
    ingest.add_argument(
        "--no-recursive",
        action="store_true",
        help="Ingest only the PDFs sitting directly in --pdf-dir, not its sub-folders",
    )
    ingest.add_argument(
        "--exclude",
        action="append",
        metavar="PATTERN",
        help="Skip paths matching PATTERN (repeatable). A directory name skips that "
        "whole sub-tree, e.g. --exclude outputs; wildcards match the path relative "
        "to --pdf-dir, e.g. --exclude '*.draft.pdf'",
    )
    ingest.add_argument(
        "--zotero-csv",
        help="Zotero library export (File → Export Library… → CSV) for metadata and mywork tags",
    )
    ingest.add_argument(
        "--reingest", action="store_true", help="Re-ingest files that are already in the corpus"
    )
    ingest.add_argument(
        "--prune-missing",
        action="store_true",
        help="Remove stored sources from --pdf-dir whose PDF is no longer on disk "
        "(without it they are listed and kept)",
    )
    _add_common(ingest)
    ingest.set_defaults(func=_cmd_ingest)

    analyze = subparsers.add_parser("analyze", help="Run CDA for one term")
    analyze.add_argument("--term", required=True, help="Target theoretical term, e.g. novelty")
    analyze.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble and print the full prompt without sending anything",
    )
    analyze.add_argument(
        "--mock-llm",
        action="store_true",
        help="Replay a recorded fixture instead of calling the API (no network, no cost)",
    )
    analyze.add_argument(
        "--record-fixtures",
        action="store_true",
        help="On a live run, save the response as a fixture for future --mock-llm replay",
    )
    analyze.add_argument(
        "--overwrite-fixtures",
        action="store_true",
        help="With --record-fixtures, replace an existing replayable fixture for the same prompt",
    )
    analyze.add_argument(
        "--fixtures-dir",
        help="Where fixtures are read (--mock-llm) or written (--record-fixtures). "
        "Default: <output_dir>/fixtures — recorded responses quote your corpus, "
        "so they stay out of the repository's committed demo/fixtures",
    )
    analyze.add_argument(
        "--drafts-only",
        action="store_true",
        help="Researcher corpus = the drafts in paths.drafts_dir ONLY; drop the "
        "mywork-tagged corpus chunks (scope the analysis to one manuscript)",
    )
    analyze.add_argument(
        "--independent",
        action="store_true",
        help="Cluster the two corpora separately and locate your passages by "
        "cross-assignment, reproducing the pipeline described in the paper. "
        "Default: joint clustering (both corpora pooled into one partition)",
    )
    _add_common(analyze)
    _add_timeframe(analyze)
    analyze.set_defaults(func=_cmd_analyze)

    demo = subparsers.add_parser(
        "demo", help="End-to-end run on the committed toy corpus (works offline with --mock-llm)"
    )
    demo.add_argument(
        "--term", default="novelty", help="Demo term (fixtures cover: novelty, consensus)"
    )
    demo.add_argument(
        "--mock-llm",
        action="store_true",
        help="Replay the committed fixture instead of calling the API (no key needed)",
    )
    demo.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble and print the full prompt without sending anything",
    )
    demo.add_argument(
        "--record-fixtures",
        action="store_true",
        help="Maintainers: record a live response into demo/fixtures/",
    )
    demo.add_argument(
        "--overwrite-fixtures",
        action="store_true",
        help="With --record-fixtures, replace an existing replayable fixture for the same prompt",
    )
    demo.add_argument("--demo-dir", help="Path to the demo/ directory (default: ./demo)")
    # No _add_common(demo): the demo runs on the committed toy corpus with the
    # paper's default parameters, so --config/--corpus-dir/--drafts-dir/
    # --out-dir would be silently ignored. Point them at your own corpus via
    # `cda analyze` instead.
    demo.set_defaults(func=_cmd_demo)

    diagnose = subparsers.add_parser(
        "diagnose", help="Reflexive corpus-composition report against OpenAlex"
    )
    diagnose.add_argument(
        "--fields",
        help='Comma-separated OpenAlex field names for the coverage baseline, e.g. "Social Sciences"',
    )
    diagnose.add_argument(
        "--demo", action="store_true", help="Diagnose the toy corpus instead of your own"
    )
    _add_timeframe(diagnose)
    _add_common(diagnose)
    diagnose.set_defaults(func=_cmd_diagnose)

    emergence = subparsers.add_parser(
        "emergence",
        help="EXPERIMENTAL: how the literature clusters for a term emerged as "
        "your library grew (Zotero 'Date Added'; no LLM call, works offline)",
    )
    emergence.add_argument("--term", required=True, help="Target theoretical term, e.g. novelty")
    emergence.add_argument(
        "--demo", action="store_true", help="Run on the toy corpus instead of your own"
    )
    _add_common(emergence)
    _add_timeframe(emergence)
    emergence.set_defaults(func=_cmd_emergence)

    validate = subparsers.add_parser(
        "validate",
        help="EXPERIMENTAL: researcher-led validation of the clustering (blind "
        "verdicts vs independent LLM readings; the interactive session lives "
        "in the GUI's Validate tab)",
    )
    validate.add_argument("--term", required=True, help="Target theoretical term, e.g. novelty")
    validate.add_argument(
        "--demo", action="store_true", help="Run on the toy corpus instead of your own"
    )
    validate.add_argument(
        "--mock-llm",
        action="store_true",
        help="Replay recorded cluster readings instead of calling the API "
        "(works for the demo once its validation fixtures are recorded)",
    )
    validate.add_argument(
        "--record-fixtures",
        action="store_true",
        help="Maintainers: record live cluster readings as replayable fixtures",
    )
    validate.add_argument(
        "--fixtures-dir",
        help="Where reading fixtures are read (--mock-llm) or written "
        "(--record-fixtures). Demo default: demo/fixtures/validation",
    )
    validate.add_argument(
        "--session",
        help="Re-open an existing validation session directory and refresh its "
        "report from the recorded state (no clustering, no LLM call). "
        "Omit to start a new session.",
    )
    validate.add_argument(
        "--independent",
        action="store_true",
        help="Validate an independent-mode clustering (default: joint, matching `cda analyze`)",
    )
    _add_common(validate)
    _add_timeframe(validate)
    validate.set_defaults(func=_cmd_validate)

    verify = subparsers.add_parser(
        "verify", help="EXPERIMENTAL: re-run register assignment with a second model and compare"
    )
    verify.add_argument("--term", required=True, help="Term of a previous analysis to verify")
    verify.add_argument("--demo", action="store_true", help="Verify against the demo's outputs")
    verify.add_argument("--mock-llm", action="store_true")
    verify.add_argument(
        "--record-fixtures",
        action="store_true",
        help="On a live run, record the verification reply as a replayable fixture",
    )
    verify.add_argument(
        "--fixtures-dir",
        default=None,
        help="Fixture directory (default: <output_dir>/fixtures, or demo/fixtures with --demo)",
    )
    _add_common(verify)
    verify.set_defaults(func=_cmd_verify)

    benchmark = subparsers.add_parser(
        "benchmark", help="Clustering-comparison and stability benchmarks"
    )
    _add_common(benchmark)
    benchmark.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Load .env from the working directory so ANTHROPIC_API_KEY etc. are
    # available without shell exports (the low-friction path for non-CS users).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "mock_llm", False) and getattr(args, "record_fixtures", False):
        # Recording wraps a live provider; under --mock-llm there is no live
        # call to record, and silently ignoring the flag would let a user
        # believe fixtures were written.
        print(
            "error: --record-fixtures needs a live run; it does nothing under "
            "--mock-llm. Drop one of the two flags.",
            file=sys.stderr,
        )
        return 1
    if getattr(args, "dry_run", False) and getattr(args, "record_fixtures", False):
        # Same reasoning as above: a dry run assembles the prompt and stops,
        # so no provider is ever built and nothing would be recorded.
        print(
            "error: --dry-run --record-fixtures records nothing because a dry "
            "run makes no calls; drop one of the flags.",
            file=sys.stderr,
        )
        return 1
    if getattr(args, "overwrite_fixtures", False) and not getattr(args, "record_fixtures", False):
        # Overwriting is a modifier of recording; the two guards above already
        # refuse recording under --mock-llm and --dry-run, so this also covers
        # --overwrite-fixtures combined with either of those.
        print(
            "error: --overwrite-fixtures does nothing without --record-fixtures "
            "on a live run; add --record-fixtures or drop --overwrite-fixtures.",
            file=sys.stderr,
        )
        return 1
    try:
        # Checked here as well as in _cli_overrides so that commands which
        # build no overrides (`cda demo --demo-dir ""`) are covered too.
        _reject_empty_path_flags(args)
        _expand_home_in_path_flags(args)
        return args.func(args)
    except (
        ConfigError,
        CorpusLockError,
        LLMError,
        OpenAlexError,
        EmbeddingModelError,
        FileNotFoundError,
        TimeoutError,
        ValueError,
    ) as exc:
        message = f"\nerror: {exc}"
        if isinstance(exc, UnicodeEncodeError):
            # Raised while *printing* to a terminal whose encoding cannot show
            # a character (Windows cp1252 consoles, some SSH locales) — the
            # bare "charmap codec" text reads as data loss, so say it is not.
            message += (
                "\nThis is a display problem only: your terminal could not "
                "show a character, but the files on disk are written as UTF-8 "
                "and are unaffected."
            )
        print(message, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Most likely during the first embedding-model download or a long
        # ingest; a stack trace here reads as a crash rather than a Ctrl-C.
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
