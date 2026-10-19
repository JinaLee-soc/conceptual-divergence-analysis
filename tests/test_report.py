"""Direct unit tests for report rendering."""

import json
from datetime import datetime
from pathlib import Path

from cda.config import load_config
from cda.divergence.report import render_report, write_outputs
from cda.registers.audit import audit_report
from cda.types import ClusterSummary

NOW = datetime(2026, 1, 2, 3, 4, 5)


def summary(k, n, sil):
    return ClusterSummary(True, n, k, sil, "[x] Embedding-level cluster separation detected")


def build_report(saturation_reached=False):
    config = load_config()
    stats = {
        "chunks_total": 10,
        "chunks_with_term_window": 5,
        "researcher_passages": 4,
        "literature_passages": 6,
        "classification_reasons": {"zotero:mywork": 3, "default:literature": 6},
    }
    saturation = {
        "enabled": True,
        "ran": True,
        "n_candidates": 100,
        "n_selected": 57,
        "saturation_reached": saturation_reached,
        "ari_threshold": 0.8,
        "consecutive_increments": 2,
        "max_passages": 1200,
    }
    audit = audit_report("### Cluster A: x\n## 3. Divergence\nmaps 1:1 onto cluster 0.\n", 2, 3)
    return render_report(
        term="novelty",
        llm_body='# Semantic Drift Report: "novelty"\n\nBody text.',
        model="claude-sonnet-4-6",
        stats=stats,
        researcher=summary(2, 4, 0.111),
        literature=summary(3, 6, 0.222),
        saturation=saturation,
        flags={"available": False, "reason": "unit test"},
        audit=audit,
        grounding={"n_quotes": 4, "n_verified": 4, "unverified": []},
        config=config,
        now=NOW,
    )


def test_header_carries_run_metadata():
    report = build_report()
    assert "<!-- generated: 2026-01-02T03:04:05 | model: claude-sonnet-4-6" in report
    assert "chunks_total: 10" in report
    assert "researcher_clusters: k=2, sil=0.111, N=4" in report
    assert "literature_clusters: k=3, sil=0.222, N=6" in report


def test_computed_sections_present_in_order():
    report = build_report()
    positions = [
        report.index("# Semantic Drift Report"),
        report.index("## Computed Divergence Flags"),
        report.index("## Cluster–Register Audit (computed)"),
        report.index("## Quotation Grounding (computed)"),
        report.index("## Run Parameters"),
        report.index("### MyWork classification"),
    ]
    assert positions == sorted(positions)
    assert "*Flags unavailable: unit test.*" in report
    assert "| zotero:mywork | 3 |" in report
    assert "**4/4**" in report  # grounding counts rendered


def test_distance_scale_legend_uses_paper_labels():
    report = build_report()
    assert (
        "Semantic distance scale: 1 = terminological variants; 2 = overlapping usage; "
        "3 = adjacent usage; 4 = distinct registers; 5 = incommensurable." in report
    )


def test_saturation_not_reached_is_loudly_flagged():
    assert "**NOT reached: interpret with caution**" in build_report(saturation_reached=False)
    assert "saturation reached" in build_report(saturation_reached=True)


def test_saturation_off_says_so_rather_than_going_silent():
    """Disabling saturation entirely must still be stated in the report.

    A missing line would leave a reader unable to tell a run that clustered
    every retrieved passage from one that clustered a sample of them — the
    difference that makes two runs non-comparable.
    """
    report = render_report(
        term="novelty",
        llm_body='# Semantic Drift Report: "novelty"\n\nBody text.',
        model="claude-sonnet-4-6",
        stats={
            "chunks_total": 10,
            "chunks_with_term_window": 5,
            "researcher_passages": 4,
            "literature_passages": 6,
            "classification_reasons": {},
        },
        researcher=summary(2, 4, 0.111),
        literature=summary(3, 6, 0.222),
        saturation={"enabled": False, "ran": False, "n_candidates": 6, "n_selected": 6},
        flags={"available": False, "reason": "unit test"},
        audit=audit_report("", 2, 3),
        grounding={"n_quotes": 0, "n_verified": 0, "unverified": []},
        config=load_config(),
        now=NOW,
    )
    assert "Saturation selection: disabled. Full literature pool clustered (6 passages" in report


def test_saturation_scope_prompt_line_states_full_pool():
    """scope=prompt reports full-pool clustering, never the sample-scope
    wording (reached / below sweep start / caution)."""
    from cda.divergence.report import _saturation_line

    line = _saturation_line(
        {
            "enabled": True,
            "scope": "prompt",
            "ran": False,
            "n_candidates": 909,
            "n_selected": 909,
            "reason": "scope=prompt: full pool clustered; no subsampling applied",
        }
    )
    assert "Full literature pool clustered (909 passages" in line
    assert "This is the default." not in line  # a scope fact, not a claim about defaults
    for sample_scope_wording in ("reached", "below the sweep start", "caution"):
        assert sample_scope_wording not in line


def test_saturation_swept_but_not_converged_is_not_mislabeled_below_start():
    """A sweep that ran to the cap without converging uses every candidate
    (n_selected == n_candidates) — but it must NOT be described as 'below the
    sweep start': the pool did reach the start; the ARI simply never plateaued.
    """
    from cda.divergence.report import _saturation_line

    swept = {
        "enabled": True,
        "ran": True,
        "n_candidates": 104,
        "n_selected": 104,
        "saturation_reached": False,
        "ari_threshold": 0.8,
        "consecutive_increments": 2,
        "max_passages": 1200,
        "history": [{"n": 50}, {"n": 75}, {"n": 100}, {"n": 104}],
    }
    line = _saturation_line(swept)
    assert "below the sweep start" not in line
    assert "**NOT reached: interpret with caution**" in line
    assert "N*=104 of 104" in line


def test_saturation_pool_below_start_says_so():
    """No sweep ran (the pipeline records ran=False): below-start message."""
    from cda.divergence.report import _saturation_line

    skipped = {
        "enabled": True,
        "ran": False,
        "n_candidates": 40,
        "n_selected": 40,
        "saturation_reached": False,
    }
    assert "below the sweep start" in _saturation_line(skipped)


def test_write_outputs_files(tmp_path):
    written = write_outputs(
        out_dir=tmp_path,
        term="boundary work",
        report_text="report body",
        prompt_text="prompt body",
        diagnostics={"k_sweep": []},
        now=NOW,
    )
    prompt = Path(written["prompt_file"])
    assert prompt.name == "drift_boundary_work_20260102_030405_prompt.md"
    assert prompt.read_text() == "prompt body"
    # Report shares the run's second-precision timestamp with the prompt, so a
    # same-day rerun cannot overwrite it (and verify can pair the two).
    assert Path(written["report_file"]).name == "drift_boundary_work_20260102_030405.md"
    assert json.loads(Path(written["diagnostics_file"]).read_text()) == {"k_sweep": []}


def test_write_outputs_same_second_runs_get_collision_suffixes(tmp_path):
    """Two runs finishing in the same second must not overwrite each other."""
    first = write_outputs(
        out_dir=tmp_path,
        term="novelty",
        report_text="first report",
        prompt_text="first prompt",
        diagnostics={"run": 1},
        now=NOW,
    )
    second = write_outputs(
        out_dir=tmp_path,
        term="novelty",
        report_text="second report",
        prompt_text="second prompt",
        diagnostics={"run": 2},
        now=NOW,
    )
    assert Path(second["report_file"]).name == "drift_novelty_20260102_030405-1.md"
    assert Path(second["prompt_file"]).name == "drift_novelty_20260102_030405-1_prompt.md"
    # The first run's files survive untouched.
    assert Path(first["report_file"]).read_text() == "first report"
    assert Path(first["prompt_file"]).read_text() == "first prompt"


def test_dry_run_then_live_run_keeps_both_prompts(tmp_path):
    """A dry run and a live run in the same second must not share filenames.

    The dry run writes only a prompt; the live run's suffix check must still
    notice it and move on, instead of pairing its report with the dry-run
    prompt (or overwriting it).
    """
    dry = write_outputs(
        out_dir=tmp_path,
        term="novelty",
        report_text=None,
        prompt_text="dry prompt",
        diagnostics=None,
        now=NOW,
    )
    live = write_outputs(
        out_dir=tmp_path,
        term="novelty",
        report_text="live report",
        prompt_text="live prompt",
        diagnostics=None,
        now=NOW,
    )
    assert Path(dry["prompt_file"]).read_text() == "dry prompt"
    assert Path(live["prompt_file"]).name == "drift_novelty_20260102_030405-1_prompt.md"
    assert Path(live["report_file"]).name == "drift_novelty_20260102_030405-1.md"


def test_write_outputs_persists_the_system_prompt_beside_the_user_prompt(tmp_path):
    """{base}_system.md carries the exact system prompt used, so `cda verify`
    can re-judge under this run's instructions even after the configured
    prompt file changes."""
    written = write_outputs(
        out_dir=tmp_path,
        term="novelty",
        report_text=None,
        prompt_text="user prompt body",
        diagnostics=None,
        now=NOW,
        system_prompt_text="system prompt body",
    )
    system = Path(written["system_file"])
    assert system.name == "drift_novelty_20260102_030405_system.md"
    assert system.read_text(encoding="utf-8") == "system prompt body"

    # The post-call write joins the same base only with identical inputs.
    again = write_outputs(
        out_dir=tmp_path,
        term="novelty",
        report_text="report body",
        prompt_text="user prompt body",
        diagnostics=None,
        now=NOW,
        base=written["base"],
        system_prompt_text="system prompt body",
    )
    assert again["system_file"] == written["system_file"]
    assert system.read_text(encoding="utf-8") == "system prompt body"


def test_write_outputs_without_system_prompt_writes_no_system_file(tmp_path):
    written = write_outputs(
        out_dir=tmp_path,
        term="novelty",
        report_text=None,
        prompt_text="prompt only",
        diagnostics=None,
        now=NOW,
    )
    assert "system_file" not in written
    assert not list(tmp_path.glob("*_system.md"))


def test_dry_run_writes_no_report(tmp_path):
    written = write_outputs(
        out_dir=tmp_path,
        term="novelty",
        report_text=None,
        prompt_text="prompt only",
        diagnostics=None,
        now=NOW,
    )
    assert "report_file" not in written
    assert "diagnostics_file" not in written
    assert Path(written["prompt_file"]).exists()


# --- clustering stability section --------------------------------------


def test_stability_section_skipped_when_none():
    from cda.divergence.report import render_stability_markdown

    assert "skipped (dry run)" in render_stability_markdown(None)


def test_stability_section_disabled():
    from cda.divergence.report import render_stability_markdown

    text = render_stability_markdown({"enabled": False})
    assert "disabled" in text
    assert "clustering.stability_check.enabled" in text


def test_stability_section_joint_mode_low_match_rate():
    from cda.divergence.report import render_stability_markdown

    stability = {
        "enabled": True,
        "mode": "joint",
        "canonical_k": 11,
        "trials": 5,
        "k_match_rate": 0.2,
        "k_seen": [2, 4, 6, 9, 11],
        "silhouette_min": 0.083,
        "silhouette_max": 0.178,
    }
    text = render_stability_markdown(stability)
    assert "Joint partition" in text
    assert "k=11 matched 20% of 5" in text
    assert "2–11" in text
    assert "one plausible partition" in text  # low-match caveat present


def test_stability_section_joint_mode_full_agreement():
    from cda.divergence.report import render_stability_markdown

    stability = {
        "enabled": True,
        "mode": "joint",
        "canonical_k": 3,
        "trials": 5,
        "k_match_rate": 1.0,
        "k_seen": [3],
        "silhouette_min": 0.2,
        "silhouette_max": 0.21,
    }
    text = render_stability_markdown(stability)
    assert "matched 100% of 5" in text
    assert "robust" in text
    assert "one plausible partition" not in text  # high-match wording, not the caveat


def test_stability_section_independent_mode_reports_both_corpora():
    from cda.divergence.report import render_stability_markdown

    stability = {
        "enabled": True,
        "mode": "independent",
        "trials": 5,
        "researcher": {
            "canonical_k": 2,
            "trials": 5,
            "k_match_rate": 0.8,
            "k_seen": [2, 3],
            "silhouette_min": 0.1,
            "silhouette_max": 0.12,
        },
        "literature": {
            "canonical_k": 4,
            "trials": 5,
            "k_match_rate": 0.4,
            "k_seen": [2, 3, 4, 7],
            "silhouette_min": 0.09,
            "silhouette_max": 0.15,
        },
    }
    text = render_stability_markdown(stability)
    assert "Researcher corpus" in text and "k=2 matched 80%" in text
    assert "Literature corpus" in text and "k=4 matched 40%" in text


def test_render_report_always_includes_stability_section():
    report = build_report()
    assert "## Clustering Stability (computed)" in report
    assert "skipped (dry run)" in report  # build_report() doesn't pass stability


def test_write_text_atomic_leaves_no_partial_file(tmp_path, monkeypatch):
    """A crash mid-write must leave either the old content or the new one on
    disk — never a truncated fragment that verify or the GUI would read as a
    complete run."""
    import os

    from cda.outputs import write_text_atomic

    target = tmp_path / "drift_x.md"
    target.write_text("old complete report", encoding="utf-8")

    real_replace = os.replace

    def exploding_replace(src, dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", exploding_replace)
    try:
        write_text_atomic(target, "new report half-w")
    except OSError:
        pass
    monkeypatch.setattr(os, "replace", real_replace)
    # The old file is intact and no temp litter remains.
    assert target.read_text() == "old complete report"
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []  # the temp name is dot-prefixed


def _umask_default_mode() -> int:
    """The mode a plain ``open(path, "w")`` would give a new file here."""
    import os

    current = os.umask(0)
    os.umask(current)
    return 0o666 & ~current


def test_write_text_atomic_keeps_umask_permissions(tmp_path):
    """mkstemp makes 0600 files and os.replace carries that mode over, which
    would leave reports unreadable to the docker-compose host user."""
    from cda.outputs import write_text_atomic

    target = tmp_path / "drift_x.md"
    write_text_atomic(target, "report")
    assert target.stat().st_mode & 0o777 == _umask_default_mode()


def test_unique_run_basename_claims_the_name_before_anything_is_written(tmp_path):
    """Two processes starting an analysis in the same second (the GUI and the
    CLI, say) must not both be handed the same basename: the second one's
    rename would silently replace the first one's report."""
    from cda.outputs import unique_run_basename

    tails = (".md", "_prompt.md")
    first = unique_run_basename(tmp_path, "drift_novelty_20260730_120000", tails)
    # Nothing has been written yet — a check-then-act version hands out the
    # same name again here.
    second = unique_run_basename(tmp_path, "drift_novelty_20260730_120000", tails)
    third = unique_run_basename(tmp_path, "drift_novelty_20260730_120000", tails)
    assert first == "drift_novelty_20260730_120000"
    assert second == "drift_novelty_20260730_120000-1"
    assert third == "drift_novelty_20260730_120000-2"
    # The claim markers are hidden and never match what the GUI or verify glob.
    assert sorted(p.name for p in tmp_path.glob("drift_*")) == []
    assert len(list(tmp_path.glob(".drift_*.claim"))) == 3


def test_unique_run_basename_skips_a_taken_figure_name(tmp_path):
    """The passage map is written as ``{base}.png`` after the report, so a
    basename whose figure exists is not free either."""
    from cda.outputs import unique_run_basename

    (tmp_path / "drift_novelty_20260730_120000.png").write_bytes(b"")
    base = unique_run_basename(
        tmp_path, "drift_novelty_20260730_120000", (".md", "_prompt.md", ".png")
    )
    assert base == "drift_novelty_20260730_120000-1"
