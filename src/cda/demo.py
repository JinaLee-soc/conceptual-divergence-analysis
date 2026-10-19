"""The `cda demo` command: an end-to-end run on the committed toy corpus.

The toy corpus (demo/toy_corpus/) is CC0 OpenAlex abstracts plus clearly-
fictional synthetic researcher drafts, so a rendered report can be produced —
and shared — without any copyrighted full text. With --mock-llm the LLM
response is replayed from committed fixtures: zero network, zero cost, no API
key. First run builds a local vector store under demo/_runtime/ (downloads
the ~90 MB embedding model once).
"""

from __future__ import annotations

import json
from pathlib import Path

from cda.config import Config, load_config
from cda.corpus.embeddings import get_embedder
from cda.corpus.ingest import get_collection, ingest_records

TOY_JSONL = "toy_corpus/abstracts.jsonl"
# The committed demo fixtures were authored in a Claude Code session and are
# keyed under this name; mock replay falls back to them.
AUTHORED_FIXTURE_MODEL = "claude-code-session"


def find_demo_dir(explicit: str | Path | None = None) -> Path:
    """Locate demo/ (repository checkout required)."""
    candidates = [Path(explicit)] if explicit else [Path.cwd() / "demo"]
    for candidate in candidates:
        if (candidate / TOY_JSONL).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the toy corpus (demo/toy_corpus/abstracts.jsonl).\n"
        "Run `cda demo` from the root of the cloned repository, or pass "
        "--demo-dir PATH."
    )


def demo_config(
    demo_dir: Path,
    drafts_dir: str | Path | None = None,
    extra_overrides: dict | None = None,
    output_dir: str | Path | None = None,
) -> Config:
    """Paper-default parameters, demo paths. The demo runs the real pipeline.

    ``drafts_dir`` overrides the researcher side: the "try it on your own
    text" mode points this at a directory holding the user's pasted text, so
    their writing (instead of the fictional drafts) is analyzed against the
    toy literature corpus. ``output_dir`` overrides where the run's report and
    prompt are written; both exist so a multi-visitor deployment can give each
    session its own directory instead of sharing one (see the GUI's
    ``session_runtime_dir``). The corpus is deliberately NOT overridable here:
    the ingested toy corpus is read-only and shared, so every session reuses
    one copy. ``extra_overrides`` carries flags a --demo run may still
    legitimately set (clustering mode, the acquisition-date window) — the demo
    paths always win over them, so a ``"paths"`` key in there is refused
    rather than silently discarded: use the dedicated parameters instead.
    """
    if extra_overrides and "paths" in extra_overrides:
        raise ValueError(
            "demo_config ignores a 'paths' override (the dict literal's own "
            "paths would silently win). Use the drafts_dir/output_dir "
            "parameters instead; the corpus path is deliberately fixed."
        )
    runtime = demo_dir / "_runtime"
    return load_config(
        overrides={
            **(extra_overrides or {}),
            "paths": {
                "corpus_dir": str(runtime / "corpus"),
                "drafts_dir": str(drafts_dir or demo_dir / "toy_corpus" / "drafts"),
                "output_dir": str(output_dir or runtime / "outputs"),
            },
            # The fictional demo researcher's theme tags (see the synthetic
            # drafts) — gives the LLM the same labeling context a real
            # researcher profile would.
            "themes": [
                {
                    "tag": "science-as-maintenance",
                    "description": "novelty, consensus, and repair as budgeted collective work",
                },
                {
                    "tag": "evaluation-infrastructures",
                    "description": "how instruments, logs, and committees ratify claims",
                },
            ],
        }
    )


def load_toy_records(demo_dir: Path) -> list[dict]:
    path = demo_dir / TOY_JSONL
    records = []
    # A damaged toy corpus should point at the regeneration script the way an
    # empty one does, not surface a raw decoder traceback.
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"{path} is damaged after record {len(records)} ({exc}) — "
            "re-generate it with python demo/build_toy_corpus.py"
        ) from exc
    if not records:
        raise FileNotFoundError(
            f"{demo_dir / TOY_JSONL} is empty — re-generate it with python demo/build_toy_corpus.py"
        )
    return records


def ensure_toy_corpus_ingested(config: Config, demo_dir: Path, embedder) -> None:
    """Ingest the committed abstracts once; later runs reuse the local store.

    ``ingest_records`` skips records that are already stored, so this call is
    cheap on a complete store — and it completes a *partial* one (a first run
    interrupted mid-indexing), instead of leaving the demo permanently unable
    to match its recorded fixture.
    """
    records = load_toy_records(demo_dir)
    collection = get_collection(config, create=True)
    if collection.count() == 0:
        print(f"[demo] first run: indexing {len(records)} toy-corpus abstracts locally…")
    ingest_records(config, embedder, records, progress=lambda message: None)


def run_demo(
    *,
    term: str,
    mock: bool,
    dry_run: bool = False,
    record_fixtures: bool = False,
    overwrite_fixtures: bool = False,
    demo_dir: str | Path | None = None,
    drafts_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    return_passages: bool = False,
    draft_texts: dict[str, str] | None = None,
    persist: bool = True,
) -> dict:
    """Ingest the toy corpus if needed, then run the real pipeline on ``term``.

    ``drafts_dir`` and ``output_dir`` let a caller scope one run's inputs and
    outputs to a directory of its own; the GUI uses this to keep concurrent
    visitors' pasted text and reports apart.

    A mock run replays the fixture recorded under the configured
    ``llm.model`` (what a live ``record_fixtures`` demo writes) and falls back
    to the committed authored fixture; the report names whichever it replayed.
    """
    from cda.pipeline import analyze_term  # imported late to keep --help fast

    directory = find_demo_dir(demo_dir)
    config = demo_config(
        directory,
        drafts_dir=drafts_dir,
        output_dir=output_dir,
    )
    embedder = get_embedder(config)

    print(f"[demo] toy corpus: {directory / 'toy_corpus'}")
    print("[demo] the demo runs the full CDA pipeline with the paper's default parameters")
    ensure_toy_corpus_ingested(config, directory, embedder)

    result = analyze_term(
        term,
        config,
        mock=mock,
        dry_run=dry_run,
        record_fixtures=record_fixtures,
        overwrite_fixtures=overwrite_fixtures,
        fixtures_dir=directory / "fixtures",
        fallback_fixture_models=(AUTHORED_FIXTURE_MODEL,) if mock else (),
        embedder=embedder,
        return_passages=return_passages,
        draft_texts=draft_texts,
        persist=persist,
    )
    if result["status"] == "ok":
        print("\n[demo] done — open the report:")
        print(f"       {result['report_file']}")
        print("[demo] reference outputs for comparison: demo/expected_reports/")
    elif result["status"] == "skipped":
        print(f"[demo] skipped: {result['reason']}")
    return result
