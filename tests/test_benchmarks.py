"""The stability benchmark must count mappings with the core audit's parser,
not a raw merge/split word count (which miscounts negations and prose)."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import clustering_comparison  # noqa: E402
from stability_protocol import run_summary  # noqa: E402


def test_stability_run_summary_does_not_miscount_negated_mapping():
    # A raw "\bmerg\w*\b"/"\bsplit\w*\b" count would report merge=1, split=1 for
    # this line; the hardened parser reads it as the 1:1 declaration it is.
    report = (
        "## 3. Researcher vs. Literature Divergence\n"
        "### Literature Register L-I\n"
        "- **k-means/LLM mapping:** no splitting or merging required; maps 1:1.\n"
    )
    assert run_summary(report)["mapping"] == {"one_to_one": 1, "merge": 0, "split": 0}


def test_stability_run_summary_counts_a_real_merge():
    report = (
        "## 3. Researcher vs. Literature Divergence\n"
        "### Literature Register L-I\n"
        "- **k-means/LLM mapping:** merges clusters 0 and 1.\n"
    )
    assert run_summary(report)["mapping"] == {"one_to_one": 0, "merge": 1, "split": 0}


def test_algorithm_sweep_uses_supplied_fixed_sample_and_post_merge(monkeypatch):
    X = np.asarray([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99], [-0.1, 0.98]])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    evaluation = np.asarray([0, 1, 2, 3])
    seen = []
    real = clustering_comparison.silhouette_on_sample

    def recording_score(X_arg, labels, sample_indices):
        seen.append(tuple(sample_indices.tolist()))
        return real(X_arg, labels, sample_indices)

    monkeypatch.setattr(clustering_comparison, "silhouette_on_sample", recording_score)
    labels_by_k = {
        2: np.asarray([0, 0, 1, 1, 1]),
        3: np.asarray([0, 0, 1, 1, 2]),
    }
    labels, final_k, _score = clustering_comparison.sweep_best(
        labels_by_k.get, X, range(2, 4), evaluation, 2
    )
    assert labels is not None
    assert final_k == 2
    assert seen == [tuple(evaluation), tuple(evaluation)]


def test_algorithm_comparison_never_fits_standard_kmeans_on_raw_embeddings():
    source = (BENCH_DIR / "clustering_comparison.py").read_text(encoding="utf-8")
    assert "fit_predict(X_raw)" not in source
    assert "fit_predict(X_norm)" in source


# ---------------------------------------------------------------------------
# Structural guards on the benchmark scripts themselves. The scripts are
# argparse mains that need a live vector store, so these read the source
# (the same style as the repo's packaged-copy identity guards) rather than
# executing a multi-minute run.

ALL_SCRIPTS = sorted(p for p in BENCH_DIR.glob("*.py"))


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_benchmark_filenames_use_the_safe_term_slug(script: Path) -> None:
    """Output filenames must slug the term, never interpolate it raw.

    A term with a path separator or a Windows-forbidden character would
    otherwise run the full benchmark and then crash on write_text, losing
    the computed report entirely.
    """
    text = script.read_text(encoding="utf-8")
    assert "safe_term_slug" in text, f"{script.name} never imports safe_term_slug"
    # Raw `{args.term}` in the report *body* is fine (readers want the real
    # term); in a filename stem it is not.
    raw_in_filename = re.search(r"(?:benchmark|bootstrap|satdiag)\w*_\{args\.term\}", text)
    assert raw_in_filename is None, f"{script.name} builds a filename from the raw term"


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_benchmark_reports_are_printed_before_being_written(script: Path) -> None:
    """The report must reach stdout before write_text can raise.

    These scripts run for minutes (or spend live API calls); a disk or
    filename problem at the last step must not destroy the result.
    """
    text = script.read_text(encoding="utf-8")
    first_print = min(
        (
            text.index(marker)
            for marker in ("print(report", 'print("\\n" + report')
            if marker in text
        ),
        default=None,
    )
    first_write = min(
        (m.start() for m in re.finditer(r"write_text\(report", text)),
        default=None,
    )
    assert first_print is not None, f"{script.name} never prints its report"
    assert first_write is not None, f"{script.name} never writes its report"
    assert first_print < first_write, f"{script.name} writes its report before printing it"
