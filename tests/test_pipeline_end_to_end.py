"""End-to-end pipeline tests: dry run and mock full run, fully offline.

Builds a small synthetic corpus in a temporary ChromaDB, adds a synthetic
draft, and runs `analyze_term` twice: once with --dry-run semantics (prompt
printed, nothing sent) and once in mock mode against a fixture created from
the very prompt the dry run assembled — which also proves the pipeline
assembles byte-identical prompts across runs (the property fixture replay
depends on).
"""

import json
from pathlib import Path

import pytest

from cda.config import ConfigError, load_config
from cda.corpus.ingest import get_collection, ingest_records
from cda.llm.base import prompt_fingerprint
from cda.pipeline import analyze_term
from cda.registers.prompt_loader import load_system_prompt

LLM_BODY = """# Semantic Drift Report: "novelty"

## Summary
Two researcher senses; researcher usage diverges from the literature.

## 1. Polysemy Map

### Cluster A: novelty-as-alpha
- **Operative definition:** alpha sense.

## 2. Temporal Trajectory
Insufficient temporal data.

## 3. Researcher vs. Literature Divergence

### Literature Register L-I
- **k-means/LLM mapping:** maps 1:1 onto embedding cluster 0.

### Literature Register L-II
- **k-means/LLM mapping:** merges embedding clusters 1 and 2.

## 4. Actionable Recommendations
- none
"""


def _literature_records():
    records = []
    for i in range(12):
        flavor = "alpha" if i % 2 == 0 else "beta"
        text = (
            f"This study examines novelty through the {flavor} lens of evaluation, case {i}. "
            f"The findings show that novelty within the {flavor} tradition behaves distinctively in sample {i}. "
            "A closing sentence without the target word keeps the abstract realistic."
        )
        records.append(
            {
                "source": f"Author{i}_2020_Study_{i}.pdf",
                "abstract": text,
                "title": f"Study {i} of novelty ({flavor})",
                "year": str(2010 + i),
            }
        )
    return records


def _researcher_records():
    return [
        {
            "source": "Lee_2022_My_Gamma_Paper.pdf",
            "abstract": (
                "Our gamma account reframes novelty as institutional repair work in study one. "
                "We argue novelty under the gamma reading is a claim about audiences, not artifacts."
            ),
            "title": "My Gamma Paper",
            "year": "2022",
            "zotero_tags": "sociology; mywork",
        }
    ]


@pytest.fixture
def populated_corpus(tmp_config, anchored_embedder, tmp_path):
    ingest_records(
        tmp_config,
        anchored_embedder,
        _literature_records() + _researcher_records(),
        progress=lambda _msg: None,
    )
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "synthetic_draft.txt").write_text(
        "In this draft I treat novelty as gamma-style repair of expectations. "
        "The gamma view of novelty guides the empirical section of this draft.",
        encoding="utf-8",
    )
    return tmp_config


def test_dry_run_prints_prompt_and_sends_nothing(populated_corpus, anchored_embedder, capsys):
    result = analyze_term(
        "novelty",
        populated_corpus,
        dry_run=True,
        embedder=anchored_embedder,
        progress=lambda _msg: None,
    )
    assert result["status"] == "dry-run"
    printed = capsys.readouterr().out
    assert "SYSTEM PROMPT" in printed
    assert 'TARGET TERM: "novelty"' in printed
    assert "nothing was sent to any API" in printed

    # The assembled prompt is saved for inspection alongside diagnostics.
    prompt_text = open(result["prompt_file"], encoding="utf-8").read()
    assert "RESEARCHER CLUSTER STRUCTURE" in prompt_text
    assert "synthetic_draft.txt" in prompt_text  # drafts entered the researcher pool
    assert "diagnostics_file" in result

    # Joint clustering is the default: the prompt says so and carries the
    # per-cluster composition, so the model can read the two blocks as views
    # of one partition.
    assert result["clustering_mode"] == "joint"
    assert "Clustering mode: JOINT" in prompt_text
    assert "Joint cluster composition" in prompt_text

    # MyWork resolution: the zotero-tagged paper classified as researcher.
    assert result["stats"]["classification_reasons"].get("zotero:mywork") == 1


def test_mock_run_replays_fixture_and_writes_report(populated_corpus, anchored_embedder, tmp_path):
    # Assemble once (dry run) to learn the exact prompt this corpus produces...
    first = analyze_term(
        "novelty",
        populated_corpus,
        dry_run=True,
        embedder=anchored_embedder,
        progress=lambda _msg: None,
    )
    user_prompt = open(first["prompt_file"], encoding="utf-8").read()
    system_prompt = load_system_prompt(populated_corpus)
    model = str(populated_corpus.get("llm.model"))

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    key = prompt_fingerprint(model, system_prompt, user_prompt)
    (fixtures / f"{key}.json").write_text(json.dumps({"response": LLM_BODY}), encoding="utf-8")

    # ...then run in mock mode: prompts must assemble byte-identically.
    result = analyze_term(
        "novelty",
        populated_corpus,
        mock=True,
        fixtures_dir=fixtures,
        embedder=anchored_embedder,
        progress=lambda _msg: None,
    )
    assert result["status"] == "ok"
    report = open(result["report_file"], encoding="utf-8").read()
    assert "<!-- generated:" in report
    assert 'Semantic Drift Report: "novelty"' in report
    assert "## Joint Clustering: where your passages sit (computed)" in report
    assert "## Computed Divergence Flags" in report
    assert "## Cluster–Register Audit (computed)" in report
    assert "## Quotation Grounding (computed)" in report
    assert "## Run Parameters" in report
    assert "terminological variants" in report  # distance-scale legend
    assert "Clustering mode: joint" in report
    assert "whole corpus (no acquisition-date window applied)" in report
    assert result["audit"]["literature"]["one_to_one"] == 1
    grounding = result["grounding"]
    assert grounding["n_verified"] <= grounding["n_quotes"]


def test_independent_mode_reproduces_the_papers_two_partitions(populated_corpus, anchored_embedder):
    """--independent clusters each corpus separately and cross-assigns; the
    prompt and report must say which computation produced the numbers."""
    from cda.config import load_config

    config = load_config(overrides={**populated_corpus.data, "clustering": {"mode": "independent"}})
    result = analyze_term(
        "novelty",
        config,
        dry_run=True,
        embedder=anchored_embedder,
        progress=lambda _msg: None,
    )
    assert result["clustering_mode"] == "independent"
    prompt_text = open(result["prompt_file"], encoding="utf-8").read()
    assert "Clustering mode: INDEPENDENT" in prompt_text
    assert "NOT comparable across the two blocks" in prompt_text
    assert "Joint cluster composition" not in prompt_text


def test_stability_check_does_not_change_the_analysis(
    populated_corpus, anchored_embedder, tmp_path
):
    """Turning the stability check on must not move a single reported number.

    It is a diagnostic: it re-clusters the same passages under shuffled
    orderings to see whether k recurs. It used to do that on the pipeline's
    own Passage objects, and since clustering assigns ``passage.cluster_id``
    in place, the live lists came back carrying the last shuffled trial's
    labels — which independent-mode ``compute_flags`` then read. The suite
    missed it because the only independent-mode end-to-end test runs
    dry_run=True, skipping both the check and the flags.
    """
    from cda.config import load_config

    def run(stability_enabled):
        config = load_config(
            overrides={
                **populated_corpus.data,
                "clustering": {
                    "mode": "independent",
                    "stability_check": {"enabled": stability_enabled, "trials": 3},
                },
            }
        )
        first = analyze_term(
            "novelty", config, dry_run=True, embedder=anchored_embedder, progress=lambda _m: None
        )
        user_prompt = open(first["prompt_file"], encoding="utf-8").read()
        fixtures = tmp_path / f"fixtures_{stability_enabled}"
        fixtures.mkdir()
        key = prompt_fingerprint(
            str(config.get("llm.model")), load_system_prompt(config), user_prompt
        )
        (fixtures / f"{key}.json").write_text(json.dumps({"response": LLM_BODY}), encoding="utf-8")
        return analyze_term(
            "novelty",
            config,
            mock=True,
            fixtures_dir=fixtures,
            embedder=anchored_embedder,
            progress=lambda _m: None,
            return_passages=True,
        )

    off, on = run(False), run(True)
    assert on["stability"]["enabled"] and not off["stability"]["enabled"]

    # The diagnostic ran, and nothing downstream of it moved.
    assert on["flags"] == off["flags"]
    assert on["researcher_cluster"] == off["researcher_cluster"]
    assert on["literature_cluster"] == off["literature_cluster"]
    for side in ("researcher", "literature"):
        assert [p.cluster_id for p in on["passages"][side]] == [
            p.cluster_id for p in off["passages"][side]
        ], f"{side} passages were relabelled by the stability check"


def test_system_prompt_is_persisted_by_both_dry_and_mock_runs(
    populated_corpus, anchored_embedder, tmp_path
):
    """Both modes persist {base}_system.md with the exact system prompt used.

    `cda verify` used to re-load the *currently configured* system prompt, so
    editing the prompt file between analyze and verify showed up as model
    disagreement. The persisted copy lets verify replay the run as it ran —
    which only works if every mode that writes a prompt writes both files.
    """
    expected = load_system_prompt(populated_corpus)

    dry = analyze_term(
        "novelty",
        populated_corpus,
        dry_run=True,
        embedder=anchored_embedder,
        progress=lambda _m: None,
    )
    assert Path(dry["system_file"]).read_text(encoding="utf-8") == expected
    # The pair shares one run basename, so verify can join them by name.
    assert Path(dry["system_file"]).name == Path(dry["prompt_file"]).name.replace(
        "_prompt.md", "_system.md"
    )

    user_prompt = Path(dry["prompt_file"]).read_text(encoding="utf-8")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    key = prompt_fingerprint(str(populated_corpus.get("llm.model")), expected, user_prompt)
    (fixtures / f"{key}.json").write_text(json.dumps({"response": LLM_BODY}), encoding="utf-8")
    mock = analyze_term(
        "novelty",
        populated_corpus,
        mock=True,
        fixtures_dir=fixtures,
        embedder=anchored_embedder,
        progress=lambda _m: None,
    )
    assert Path(mock["system_file"]).read_text(encoding="utf-8") == expected


def test_negative_clustering_seed_is_a_config_error_up_front(tmp_config, anchored_embedder):
    """clustering.seed: -1 used to pass validation and die deep inside the
    k-sweep as numpy's raw 'expected non-negative seed' ValueError. tmp_config
    has no ingested corpus: reaching FileNotFoundError here would mean the
    validation ran after the corpus was opened, i.e. too late."""
    from cda.config import ConfigError, load_config

    config = load_config(overrides={**tmp_config.data, "clustering": {"seed": -1}})
    with pytest.raises(ConfigError, match="clustering.seed"):
        analyze_term(
            "novelty", config, dry_run=True, embedder=anchored_embedder, progress=lambda _m: None
        )


def test_dry_run_prompt_printing_survives_a_non_utf8_console(monkeypatch):
    """On Windows, *redirected* stdout defaults to a legacy codepage; printing
    a prompt carrying arbitrary corpus text used to die in a UnicodeEncodeError
    rewrapped as a bare 'charmap codec' message — after all the expensive work.
    The console copy degrades instead and points at the UTF-8 prompt file."""
    import sys

    from cda.pipeline import _print_prompt_block

    class LegacyConsole:
        encoding = "ascii"

        def __init__(self):
            self.chunks = []

        def write(self, s):
            s.encode("ascii")  # raises UnicodeEncodeError exactly like cp1252
            self.chunks.append(s)
            return len(s)

        def flush(self):
            pass

    console = LegacyConsole()
    monkeypatch.setattr(sys, "stdout", console)
    _print_prompt_block("novelty — 참신성")  # em dash + Korean term
    printed = "".join(console.chunks)
    assert "novelty" in printed
    assert "?" in printed  # unencodable characters replaced, not fatal
    assert "authoritative" in printed  # the note names the real UTF-8 file


def test_unknown_clustering_mode_names_both_options(populated_corpus, anchored_embedder):
    from cda.config import load_config

    config = load_config(overrides={**populated_corpus.data, "clustering": {"mode": "pooled"}})
    with pytest.raises(ConfigError, match="clustering.mode must be"):
        analyze_term(
            "novelty", config, dry_run=True, embedder=anchored_embedder, progress=lambda _m: None
        )


def test_missing_corpus_error_points_to_ingest(tmp_config, anchored_embedder):
    with pytest.raises(FileNotFoundError, match="cda ingest"):
        analyze_term(
            "novelty",
            tmp_config,
            dry_run=True,
            embedder=anchored_embedder,
            progress=lambda _msg: None,
        )


def test_skip_below_minimum_researcher_passages(tmp_config, anchored_embedder):
    ingest_records(tmp_config, anchored_embedder, _literature_records(), progress=lambda _msg: None)
    result = analyze_term(
        "novelty",
        tmp_config,
        dry_run=True,
        embedder=anchored_embedder,
        progress=lambda _msg: None,
    )
    assert result["status"] == "skipped"
    assert "mywork" in result["reason"]


def test_ingest_is_idempotent(tmp_config, anchored_embedder):
    records = _literature_records()
    ingest_records(tmp_config, anchored_embedder, records, progress=lambda _msg: None)
    first_count = get_collection(tmp_config).count()
    ingest_records(tmp_config, anchored_embedder, records, progress=lambda _msg: None)
    assert get_collection(tmp_config).count() == first_count


def test_drafts_only_drops_corpus_mywork_passages(populated_corpus, anchored_embedder):
    """--drafts-only scopes the researcher corpus to the drafts: the mywork-
    tagged corpus chunks are dropped, the literature side is untouched."""
    from cda.pipeline import extract_passages

    collection = get_collection(populated_corpus)
    kw = {"embedder": anchored_embedder, "collection": collection, "progress": lambda _m: None}

    r_full, lit_full, _ = extract_passages("novelty", populated_corpus, **kw)
    r_only, lit_only, stats = extract_passages("novelty", populated_corpus, drafts_only=True, **kw)

    dropped = stats["corpus_researcher_passages_dropped"]
    assert dropped >= 1  # the Lee_2022 mywork corpus chunk(s) were dropped
    assert len(r_only) == len(r_full) - dropped  # only those were removed
    assert r_only and all(p.metadata.get("draft") for p in r_only)  # drafts remain
    assert len(lit_only) == len(lit_full)  # literature untouched


# --- clustering result must not depend on saturation-induced order -----


def test_saturation_selection_restores_canonical_order(anchored_embedder):
    """select_literature_by_saturation must hand clustering the same order
    regardless of whether the sweep ran, or whether it reordered candidates
    to rank them — otherwise the saturation setting silently changes the
    clustering result even when it selects every candidate (A10). Both axes
    are covered: `scope` (which governs the geometry, A12/A14) and `enabled`
    (which skips the sweep outright)."""
    from cda.pipeline import _passage_sort_key, select_literature_by_saturation
    from tests.conftest import make_passages

    passages = make_passages([f"alpha term appears in passage {i}" for i in range(30)])
    config = load_config(
        overrides={
            "saturation": {"start": 10, "step": 5, "max_passages": 100},
            "clustering": {"min_cluster_size": 2, "silhouette_subsample": 100},
        }
    )
    expected_order = sorted(passages, key=_passage_sort_key, reverse=True)

    for setting in ({"scope": "sample"}, {"scope": "prompt"}, {"enabled": False}):
        cfg = load_config(
            overrides={
                **config.data,
                "saturation": {**config.data["saturation"], **setting},
            }
        )
        selected, _info, _ran = select_literature_by_saturation(
            "alpha", list(passages), cfg, embedder=anchored_embedder
        )
        assert [p.chunk_id for p in selected] == [p.chunk_id for p in expected_order]


def test_saturation_on_and_off_cluster_identically_when_all_candidates_selected(
    anchored_embedder,
):
    """The regression this guards: before the A10 fix, disabling saturation on
    a pool small enough that saturation would have kept every candidate
    anyway still changed the k-means result, because the two paths handed
    clustering the same passages in different orders."""
    from cda.clustering.joint import cluster_joint
    from cda.pipeline import select_literature_by_saturation
    from tests.conftest import make_passages

    researcher = make_passages(
        [f"my alpha reading, case {i}" for i in range(6)], source="mine.docx"
    )
    literature = make_passages(
        [f"an alpha treatment, study {i}" for i in range(8)]
        + [f"a beta treatment, study {i}" for i in range(10)]
    )
    # start == the full candidate pool: the sweep runs (exercising the
    # term-similarity reordering) but has only one increment to evaluate, so
    # it never reaches saturation and keeps every candidate — exactly the
    # "N* == n_candidates either way" shape that exposed the bug in the demo.
    config = load_config(
        overrides={
            "saturation": {"start": len(literature), "step": 5, "max_passages": 100},
            "clustering": {"min_cluster_size": 2, "silhouette_subsample": 200},
        }
    )
    kwargs = dict(max_k=5, n_representatives=3, min_cluster_size=2, silhouette_subsample=200)

    results = {}
    for label, setting in (
        ("sample", {"scope": "sample"}),
        ("prompt", {"scope": "prompt"}),
        ("disabled", {"enabled": False}),
    ):
        cfg = load_config(
            overrides={
                **config.data,
                "saturation": {**config.data["saturation"], **setting},
            }
        )
        selected, _info, _ran = select_literature_by_saturation(
            "alpha", list(literature), cfg, embedder=anchored_embedder
        )
        assert len(selected) == len(literature)  # every path keeps every candidate
        joint = cluster_joint(list(researcher), selected, anchored_embedder, **kwargs)
        results[label] = (joint.n_clusters, joint.silhouette)

    assert results["sample"][0] == results["prompt"][0] == results["disabled"][0]
    # Same partition; the silhouette is a float summed in a different order, so
    # compare it as a number rather than as a bit pattern.
    assert results["prompt"][1] == pytest.approx(results["sample"][1], abs=1e-12)
    assert results["disabled"][1] == pytest.approx(results["sample"][1], abs=1e-12)


# --- built-in clustering stability check --------------------------


def test_stability_check_runs_by_default_and_is_in_the_result(populated_corpus, anchored_embedder):
    from cda.config import load_config

    config = load_config(
        overrides={**populated_corpus.data, "clustering": {"stability_check": {"trials": 3}}}
    )
    first = analyze_term(
        "novelty", config, dry_run=True, embedder=anchored_embedder, progress=lambda _m: None
    )
    user_prompt = open(first["prompt_file"], encoding="utf-8").read()
    system_prompt = load_system_prompt(config)
    model = str(config.get("llm.model"))
    fixtures = config.path("paths.output_dir") / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    key = prompt_fingerprint(model, system_prompt, user_prompt)
    (fixtures / f"{key}.json").write_text(json.dumps({"response": LLM_BODY}), encoding="utf-8")

    result = analyze_term(
        "novelty",
        config,
        mock=True,
        fixtures_dir=fixtures,
        embedder=anchored_embedder,
        progress=lambda _m: None,
    )
    assert result["stability"]["enabled"] is True
    assert result["stability"]["mode"] == "joint"
    assert result["stability"]["trials"] == 3
    assert 0.0 <= result["stability"]["k_match_rate"] <= 1.0

    report = open(result["report_file"], encoding="utf-8").read()
    assert "## Clustering Stability (computed)" in report
    # The rate is over the shuffled orderings only; trial 0 re-runs the
    # canonical order and is not evidence that reordering preserves k.
    assert result["stability"]["shuffled_trials"] == 2
    assert "matched" in report and "of 2 shuffled orderings" in report
    # Trial 0 re-clusters this run's own canonical order; the summary records
    # whether it landed on the run's own k, and here it must have.
    assert result["stability"]["canonical_reproduced"] is True


def test_stability_check_skipped_on_dry_run(populated_corpus, anchored_embedder):
    result = analyze_term(
        "novelty",
        populated_corpus,
        dry_run=True,
        embedder=anchored_embedder,
        progress=lambda _m: None,
    )
    # dry-run never reaches render_report, so there is nothing to assert on
    # the report text — the guarantee under test is that dry-run stays fast:
    # no stability key is threaded through at all.
    assert "stability" not in result


def test_stability_check_disabled_via_config(populated_corpus, anchored_embedder):
    from cda.config import load_config

    config = load_config(
        overrides={
            **populated_corpus.data,
            "clustering": {"stability_check": {"enabled": False}},
        }
    )
    first = analyze_term(
        "novelty", config, dry_run=True, embedder=anchored_embedder, progress=lambda _m: None
    )
    user_prompt = open(first["prompt_file"], encoding="utf-8").read()
    system_prompt = load_system_prompt(config)
    model = str(config.get("llm.model"))
    fixtures = config.path("paths.output_dir") / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    key = prompt_fingerprint(model, system_prompt, user_prompt)
    (fixtures / f"{key}.json").write_text(json.dumps({"response": LLM_BODY}), encoding="utf-8")

    result = analyze_term(
        "novelty",
        config,
        mock=True,
        fixtures_dir=fixtures,
        embedder=anchored_embedder,
        progress=lambda _m: None,
    )
    assert result["stability"]["enabled"] is False
    report = open(result["report_file"], encoding="utf-8").read()
    assert "disabled" in report


def test_stability_check_does_not_change_the_assembled_prompt(populated_corpus, anchored_embedder):
    """The stability check runs entirely on the clustering side, after the
    prompt-relevant summaries are already built — it must never perturb the
    prompt fixture replay depends on."""
    from cda.config import load_config

    enabled = load_config(
        overrides={**populated_corpus.data, "clustering": {"stability_check": {"enabled": True}}}
    )
    disabled = load_config(
        overrides={**populated_corpus.data, "clustering": {"stability_check": {"enabled": False}}}
    )
    with_check = analyze_term(
        "novelty", enabled, dry_run=True, embedder=anchored_embedder, progress=lambda _m: None
    )
    without_check = analyze_term(
        "novelty", disabled, dry_run=True, embedder=anchored_embedder, progress=lambda _m: None
    )
    assert open(with_check["prompt_file"]).read() == open(without_check["prompt_file"]).read()


def test_mock_without_fixtures_dir_fails_before_touching_the_corpus(tmp_config, anchored_embedder):
    """The provider is now validated up front: a missing fixtures directory
    (or, live, a missing API key) must not surface after minutes of
    extraction and clustering."""
    from cda.llm.base import LLMError

    # tmp_config has no ingested corpus: if the provider check came after
    # collection opening, this would raise FileNotFoundError instead.
    with pytest.raises(LLMError, match="fixtures"):
        analyze_term(
            "novelty",
            tmp_config,
            mock=True,
            fixtures_dir=None,
            embedder=anchored_embedder,
            progress=lambda _msg: None,
        )


def test_out_dir_pointing_at_a_file_fails_fast(tmp_config, anchored_embedder, tmp_path):
    """`--out-dir report.md` (an existing file) used to crash with a raw
    FileExistsError only AFTER the paid LLM call, losing the report text."""
    from cda.config import ConfigError, load_config

    blocker = tmp_path / "report.md"
    blocker.write_text("i am a file", encoding="utf-8")
    config = load_config(
        overrides={
            **tmp_config.data,
            "paths": {**tmp_config.data["paths"], "output_dir": str(blocker)},
        }
    )
    with pytest.raises(ConfigError, match="output directory|not an\n?.*existing file|Cannot use"):
        analyze_term(
            "novelty",
            config,
            dry_run=True,
            embedder=anchored_embedder,
            progress=lambda _msg: None,
        )


def test_failed_llm_call_still_leaves_the_prompt_on_disk(populated_corpus, anchored_embedder):
    """The prompt and diagnostics are persisted BEFORE the LLM call, so a
    rate limit or interrupt no longer discards the assembled prompt."""
    from cda.llm.base import FixtureMissError

    out_dir = populated_corpus.path("paths.output_dir")
    before = set(out_dir.glob("drift_*_prompt.md")) if out_dir.is_dir() else set()
    # An empty fixtures dir means the mock call itself fails — after the
    # prompt should already have been written.
    empty_fixtures = out_dir / "empty_fixtures"
    empty_fixtures.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FixtureMissError):
        analyze_term(
            "novelty",
            populated_corpus,
            mock=True,
            fixtures_dir=empty_fixtures,
            embedder=anchored_embedder,
            progress=lambda _msg: None,
        )
    after = set(out_dir.glob("drift_*_prompt.md"))
    assert len(after) == len(before) + 1


def test_relpath_keys_are_distinct_for_colliding_slugs():
    """`a/paper.pdf` vs `a_paper.pdf`, and any two fully non-ASCII filenames
    (논문.pdf, 리뷰.pdf), used to collapse to one chunk-id key; chroma's add()
    silently stores nothing for an existing id, so the second file never
    entered the corpus while the stats counted it as ingested."""
    from cda.corpus.ingest import _relpath_key

    assert _relpath_key("a/paper.pdf") != _relpath_key("a_paper.pdf")
    assert _relpath_key("논문.pdf") != _relpath_key("리뷰.pdf")
    assert _relpath_key("Paper.pdf") != _relpath_key("paper.pdf")
    # Deterministic across runs — incremental ingest state depends on it.
    assert _relpath_key("a/paper.pdf") == _relpath_key("a/paper.pdf")


def test_utf16_draft_loads_as_readable_text(tmp_path):
    """A UTF-16 draft used to pass errors='ignore' as NUL-riddled text that
    printed a successful word count but could never match the term regex."""
    from cda.corpus.drafts import load_drafts

    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "with_bom.txt").write_bytes("novelty appears here often".encode("utf-16"))
    (drafts / "bomless.txt").write_bytes("novelty appears here too".encode("utf-16-le"))
    texts = load_drafts(drafts, progress=lambda _msg: None)
    assert texts["with_bom.txt"] == "novelty appears here often"
    assert "novelty" in texts["bomless.txt"]
    assert "\x00" not in texts["bomless.txt"]


def test_bomless_utf16_be_draft_is_decoded_the_right_way_round(tmp_path):
    """Python's utf-16 codec assumes little-endian without a BOM, so a
    big-endian draft used to load as mojibake ('渀漀瘀攀氀琀礀') — no exception,
    a plausible word count, and no term ever matching."""
    from cda.corpus.drafts import load_drafts

    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "big_endian.txt").write_bytes("novelty appears here".encode("utf-16-be"))
    (drafts / "little_endian.txt").write_bytes("novelty appears there".encode("utf-16-le"))

    messages: list[str] = []
    texts = load_drafts(drafts, progress=messages.append)
    assert texts["big_endian.txt"] == "novelty appears here"
    assert texts["little_endian.txt"] == "novelty appears there"
    # A guess about a whole draft's encoding is never silent.
    assert any("utf-16-be" in m for m in messages)
    assert any("utf-16-le" in m for m in messages)
