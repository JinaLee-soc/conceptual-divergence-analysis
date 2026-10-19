"""Regression tests for validation, diagnose,
emergence labelling, CLI error paths, GUI structure, and CI/Docker hygiene."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from cda.config import load_config
from cda.validation import metrics
from cda.validation.agents import suggest_edit
from cda.validation.curate import (
    _per_cluster_silhouettes,
    apply_pending_edit,
    apply_recut,
)
from cda.validation.prompt_loader import load_validation_prompts
from cda.validation.report import render_validation_report
from cda.validation.session import create_session, record_verdict, reveal_readings

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 18, 9, 0, 0)


def _three_cluster_session(tmp_path, *, params=None, readings=True, sides=("literature",)):
    """Clusters 0=alpha, 1=beta, 2=gamma (three passages each) on each side."""
    passages, partitions = {}, []
    for side in sides:
        assignments = {}
        for cluster, topic in ((0, "alpha"), (1, "beta"), (2, "gamma")):
            for i in range(3):
                chunk_id = f"{side}:{topic}::{i}"
                passages[chunk_id] = {
                    "text": f"{topic} usage of the term in {side}, case {i}",
                    "side": side,
                    "source": f"{topic}.abstract",
                }
                assignments[chunk_id] = cluster
        partitions.append(
            {
                "revision": 0,
                "side": side,
                "kind": "original",
                "assignments": assignments,
                "clusters": [
                    {
                        "cluster_id": c,
                        "n_passages": 3,
                        "silhouette": None,
                        "shown": sorted(k for k, v in assignments.items() if v == c),
                    }
                    for c in (0, 1, 2)
                ],
            }
        )
    reading_rows = (
        [
            {
                "side": side,
                "cluster_id": c,
                "revision": 0,
                "model": "m",
                "gloss": "g",
                "judgment": "artifact",
                "rationale": "r",
                "valid": True,
                "raw": "",
            }
            for side in sides
            for c in (0, 1, 2)
        ]
        if readings
        else []
    )
    if params is None:
        params = {"seed": 42, "readers": ["m"], "embedding_model": _configured_model()}
    return create_session(
        "novelty",
        params=params,
        sides={side: {"validated": True, "note": None} for side in sides},
        partitions=partitions,
        passages=passages,
        readings=reading_rows,
        output_dir=tmp_path,
        now=NOW,
    )


def _configured_model() -> str:
    return str(load_config().get("embedding.model"))


def _judge(session, clusters=(0, 1, 2), side="literature", verdict="coherent"):
    for cluster in clusters:
        record_verdict(session, side=side, cluster_id=cluster, revision=0, verdict=verdict, now=NOW)


def _pending_merge(session, clusters=(0, 1)):
    session["pending_edit"] = {
        "edit_id": len(session["edits"]) + 1,
        "kind": "merge",
        "side": "literature",
        "clusters": list(clusters),
        "revision": 0,
        "status": "pending",
        "agent_advice": {"recommendation": "merge", "argument": "same", "valid": True},
    }


class _AdviceProvider:
    model = "stub"

    def __init__(self):
        self.users = []

    def complete(self, *, system, user, max_tokens, temperature):
        self.users.append(user)
        return '{"recommendation": "merge", "argument": "same usage"}'


# --- 1. curation silhouettes ------------------------------------------------


def test_curation_silhouettes_match_sklearn_with_singleton_zero():
    from sklearn.metrics import silhouette_samples

    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 5))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = [0, 0, 0, 1, 1, 2]  # cluster 2 is a singleton
    scores = silhouette_samples(X, labels, metric="cosine")
    expected = {
        c: round(float(scores[np.array(labels) == c].mean()), 4) for c in sorted(set(labels))
    }
    assert _per_cluster_silhouettes(X, labels) == expected
    assert _per_cluster_silhouettes(X, labels)[2] == 0.0


def test_curation_silhouette_is_none_when_undefined():
    X = np.eye(3)
    assert _per_cluster_silhouettes(X, [0, 0, 0]) == {0: None}


# --- 2-4. diagnose ------------------------------------------------------------


def test_venue_concentration_keeps_unknown_venues_out_of_the_hhi():
    from cda.diagnose.report import venue_concentration

    works = [{"primary_location": {"source": None}} for _ in range(8)] + [
        {"primary_location": {"source": {"display_name": f"J{i}"}}} for i in range(2)
    ]
    stats = venue_concentration(works)
    assert stats["hhi"] == 0.5
    assert stats["n_unknown_venue"] == 8
    assert stats["n_works"] == 2
    assert all(venue != "(unknown venue)" for venue, _, _ in stats["top"])


class _Collection:
    def __init__(self, metadatas):
        self.metadatas = metadatas

    def get(self, include, limit, offset):
        return {"metadatas": [] if offset else self.metadatas}


def _gather(monkeypatch, tmp_path, *, metadatas, works, declared=None, baseline=None):
    import cda.diagnose.report as report

    monkeypatch.setattr(report, "works_by_dois", lambda dois, cache, progress: works)
    monkeypatch.setattr(
        report,
        "list_fields",
        lambda cache: [{"id": "https://openalex.org/fields/33", "display_name": "Social Sciences"}],
    )
    monkeypatch.setattr(report, "top_topics_for_field", lambda field_id, cache: baseline or [])
    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path)}})
    return report.gather_composition(
        config,
        declared_fields=declared,
        collection=_Collection(metadatas),
        progress=lambda _m: None,
    )


def test_unknown_declared_field_raises_naming_it_and_the_valid_choices(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _gather(
            monkeypatch,
            tmp_path,
            metadatas=[{"pdf_relpath": "a.pdf", "doi": "10.1/x"}],
            works={"10.1/x": {"doi": "https://doi.org/10.1/x"}},
            declared=["Social Science"],
        )
    message = str(caught.value)
    assert "'Social Science'" in message and "Social Sciences" in message


def test_two_pdfs_sharing_a_doi_are_accounted_for(monkeypatch, tmp_path):
    from cda.diagnose.report import render_diagnosis_markdown

    data = _gather(
        monkeypatch,
        tmp_path,
        metadatas=[
            {"pdf_relpath": "a.pdf", "doi": "10.1/x"},
            {"pdf_relpath": "b.pdf", "doi": "10.1/X"},
        ],
        works={"10.1/x": {"doi": "https://doi.org/10.1/x", "publication_year": 2020}},
    )
    line = next(
        row for row in render_diagnosis_markdown(data, NOW).splitlines() if "Works with DOIs" in row
    )
    assert "Works with DOIs: 2" in line
    assert "resolved via OpenAlex: 1" in line and "unresolvable: 0" in line
    assert "sharing a DOI with another work: 1" in line


def test_untouched_topics_count_secondary_topics_as_present(monkeypatch, tmp_path):
    from cda.diagnose.report import underrepresented_topics

    work = {
        "doi": "https://doi.org/10.1/x",
        "primary_topic": {"display_name": "Topic A"},
        "topics": [{"display_name": "Topic A"}, {"display_name": "Topic B"}],
    }
    baseline = [{"display_name": "Topic A"}, {"display_name": "Topic B"}, {"display_name": "C"}]
    data = _gather(
        monkeypatch,
        tmp_path,
        metadatas=[{"pdf_relpath": "a.pdf", "doi": "10.1/x"}],
        works={"10.1/x": work},
        declared=["social sciences"],
        baseline=baseline,
    )
    assert underrepresented_topics(baseline, data["corpus_topic_names"]) == ["C"]


def test_temporal_median_is_the_statistical_median():
    from cda.diagnose.report import temporal_profile

    profile = temporal_profile([{"publication_year": 2000}, {"publication_year": 2010}])
    assert profile["median"] == 2005
    assert "share_last_5y_of_span" not in profile


# --- 5. agreement coverage ------------------------------------------------------


def test_first_curation_edit_requires_a_verdict_on_every_original_cluster(
    tmp_path, anchored_embedder
):
    session_dir, session = _three_cluster_session(tmp_path)
    _judge(session, clusters=(0, 1))  # cluster 2 never judged
    _pending_merge(session)
    with pytest.raises(ValueError, match=r"literature cluster\(s\) 2"):
        apply_pending_edit(
            session,
            "apply-merge",
            load_config(),
            embedder=anchored_embedder,
            now=NOW,
            progress=lambda _m: None,
        )
    assert session["pending_edit"] is not None  # still decidable
    with pytest.raises(ValueError, match=r"literature cluster\(s\) 2"):
        suggest_edit(
            {**session, "pending_edit": None},
            kind="merge",
            side="literature",
            clusters=[0, 1],
            config=load_config(),
            embedder=anchored_embedder,
            prompts=load_validation_prompts(load_config()),
            provider=_AdviceProvider(),
            now=NOW,
            progress=lambda _m: None,
        )


def test_recut_requires_a_verdict_on_every_original_cluster(tmp_path, anchored_embedder):
    params = {
        "seed": 42,
        "embedding_model": _configured_model(),
        "clustering_mode": "joint",
        "clustering_method": "hierarchical",
    }
    _dir, session = _three_cluster_session(tmp_path, params=params)
    _judge(session, clusters=(0,))
    with pytest.raises(ValueError, match=r"literature cluster\(s\) 1, 2"):
        apply_recut(
            session, 2, load_config(), embedder=anchored_embedder, now=NOW, progress=lambda _m: None
        )


def test_gui_no_longer_promises_a_reading_refresh_that_does_not_exist():
    for path in (ROOT / "gui").glob("*.py"):
        assert "unless refreshed after an edit" not in path.read_text(encoding="utf-8")


# --- 6. reveal logging --------------------------------------------------------


def test_report_shows_no_reading_until_its_reveal_is_logged(tmp_path):
    _dir, session = _three_cluster_session(tmp_path)
    config = load_config()
    record_verdict(session, side="literature", cluster_id=0, revision=0, verdict="mixed", now=NOW)
    report = render_validation_report(session, config, NOW)
    row = next(line for line in report.splitlines() if line.startswith("| Cluster 0"))
    assert "artifact" not in row and "not yet revealed" in row
    assert "artifact" not in report.split("## Researcher vs agent agreement")[1]
    assert "withheld until the readings of cluster(s) 0 are revealed" in report

    reveal_readings(session, side="literature", cluster_id=0, revision=0, now=NOW)
    report = render_validation_report(session, config, NOW)
    row = next(line for line in report.splitlines() if line.startswith("| Cluster 0"))
    assert "artifact" in row
    assert "0% agreement over 1 cluster(s)" in report
    later = record_verdict(
        session, side="literature", cluster_id=0, revision=0, verdict="artifact", now=NOW
    )
    assert later["revised_after_reveal"] and not later["blind"]


def test_revision_advice_logs_a_reveal_and_flags_later_verdicts(tmp_path, anchored_embedder):
    _dir, session = _three_cluster_session(tmp_path)
    _judge(session)
    suggest_edit(
        session,
        kind="merge",
        side="literature",
        clusters=[0, 1],
        config=load_config(),
        embedder=anchored_embedder,
        prompts=load_validation_prompts(load_config()),
        provider=_AdviceProvider(),
        now=NOW,
        progress=lambda _m: None,
    )
    assert {(r["cluster_id"], r["via"]) for r in session["reveals"]} == {
        (0, "revision_advice"),
        (1, "revision_advice"),
    }
    later = record_verdict(
        session, side="literature", cluster_id=1, revision=0, verdict="mixed", now=NOW
    )
    assert later["revised_after_reveal"] and not later["blind"]


# --- 7. merge ids -----------------------------------------------------------------


def test_merge_suggestion_refuses_duplicate_cluster_ids(tmp_path, anchored_embedder):
    _dir, session = _three_cluster_session(tmp_path)
    _judge(session)
    provider = _AdviceProvider()
    with pytest.raises(ValueError, match="two different cluster ids"):
        suggest_edit(
            session,
            kind="merge",
            side="literature",
            clusters=[0, 0, 1],
            config=load_config(),
            embedder=anchored_embedder,
            prompts=load_validation_prompts(load_config()),
            provider=provider,
            now=NOW,
            progress=lambda _m: None,
        )
    assert provider.users == []


# --- 8. embedder identity -----------------------------------------------------------


@pytest.mark.parametrize(
    ("params_update", "overrides", "match"),
    [
        ({}, {"embedding": {"model": "some/other-model"}}, "embedding model"),
        ({"embedding_revision": "abc"}, {"embedding": {"revision": "def"}}, "embedding revision"),
        ({"embedding_model": None}, {}, "did not record its embedding model"),
    ],
)
def test_curation_refuses_a_different_embedder_than_the_session_recorded(
    tmp_path, anchored_embedder, params_update, overrides, match
):
    params = {"seed": 42, "embedding_model": _configured_model(), **params_update}
    if params["embedding_model"] is None:
        del params["embedding_model"]
    _dir, session = _three_cluster_session(tmp_path, params=params)
    _judge(session)
    _pending_merge(session)
    with pytest.raises(ValueError, match=match):
        apply_pending_edit(
            session,
            "apply-merge",
            load_config(overrides=overrides),
            embedder=anchored_embedder,
            now=NOW,
            progress=lambda _m: None,
        )
    assert len(session["partitions"]) == 1


# --- 9. agreement statistics ----------------------------------------------------


def test_agreement_uses_sklearn_kappa_and_a_named_percent_agreement():
    from sklearn.metrics import cohen_kappa_score

    assert metrics.cohen_kappa_score is cohen_kappa_score
    assert metrics.percent_agreement(["a", "b", "a"], ["a", "a", "a"]) == pytest.approx(2 / 3)
    assert metrics.percent_agreement([], []) is None
    a, b = (
        ["coherent", "mixed", "coherent", "artifact"],
        ["coherent", "coherent", "coherent", "artifact"],
    )
    assert metrics.cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


def test_a_reading_without_a_valid_flag_fails_loudly(tmp_path):
    _dir, session = _three_cluster_session(tmp_path)
    _judge(session)
    del session["readings"][0]["valid"]
    with pytest.raises(KeyError):
        metrics.verdict_agreement(session)


# --- 10. emergence clock labelling ------------------------------------------------


def _skipped_emergence():
    return {
        "coverage": {"n_literature": 5, "n_dated": 4, "n_undated": 1, "rate": 0.8},
        "params": {"match_threshold": 0.8},
        "status": "skipped",
        "reason": "too few",
    }


def test_filesystem_emergence_report_names_its_clock_and_remedy():
    from cda.divergence.emergence import render_emergence_report

    text = render_emergence_report(
        term="novelty",
        emergence=_skipped_emergence(),
        stats={},
        config=load_config(),
        date_source="filesystem",
        now=NOW,
    )
    assert 'Zotero\'s "Date Added"' not in text
    assert "--zotero-csv" not in text
    assert "arrival on disk" in text and "cda ingest --pdf-dir ... --reingest" in text


def test_emergence_chart_axis_names_the_filesystem_clock():
    sys.path.insert(0, str(ROOT / "gui"))
    pytest.importorskip("altair")
    import visuals

    emergence = {
        "status": "ok",
        "date_source": "filesystem",
        "final": {
            "clusters": [
                {
                    "final_cluster": 0,
                    "birth_n": 10,
                    "growth": [
                        {"n": 10, "as_of": "2019-03-18", "size": 6},
                        {"n": 20, "as_of": "2019-12-02", "size": 9},
                    ],
                }
            ]
        },
    }
    title = visuals.emergence_chart(emergence).to_dict()["layer"][0]["encoding"]["x"]["title"]
    assert "Zotero" not in title and "filesystem" in title


# --- 11. OpenAlex errors never carry the email ------------------------------------


def test_openalex_connection_error_does_not_leak_the_mailto(monkeypatch):
    import requests

    from cda.diagnose import openalex

    monkeypatch.setenv("OPENALEX_MAILTO", "private.person@example.org")

    def refuse(url, params, timeout):
        raise requests.ConnectionError(
            f"Max retries exceeded with url: /works?mailto=private.person%40example.org "
            f"({params['mailto']})"
        )

    monkeypatch.setattr(openalex.requests, "get", refuse)
    monkeypatch.setattr(openalex.time, "sleep", lambda _s: None)
    with pytest.raises(openalex.OpenAlexError) as caught:
        openalex.get_json("works", {"filter": "doi:10.1/x"}, max_retries=1)
    message = str(caught.value)
    assert "private.person" not in message and "example.org" not in message
    assert "mailto=<redacted>" in message


# --- 12. CLI error paths ------------------------------------------------------------


def test_validate_missing_session_dir_is_a_friendly_error(tmp_path, capsys):
    from cda.cli import main

    code = main(["validate", "--term", "novelty", "--session", str(tmp_path / "nope")])
    assert code == 1
    assert "No validation session directory at" in capsys.readouterr().err


def test_validate_locked_session_is_a_friendly_error(tmp_path, capsys, monkeypatch):
    from cda.cli import main

    monkeypatch.setattr("cda.validation.session._LOCK_TIMEOUT_SECONDS", 0.1)
    session_dir, _session = _three_cluster_session(tmp_path)
    (session_dir / "session.lock").write_text("pid 1\n", encoding="utf-8")
    code = main(["validate", "--term", "novelty", "--session", str(session_dir)])
    assert code == 1
    assert "session.lock" in capsys.readouterr().err


def test_cli_does_not_reimport_path_locally():
    source = (ROOT / "src" / "cda" / "cli.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s+from pathlib import Path", source, flags=re.M)


# --- 13. GUI ---------------------------------------------------------------------------


def test_pca_projection_is_sklearn_pca():
    sys.path.insert(0, str(ROOT / "gui"))
    pytest.importorskip("altair")
    import visuals
    from sklearn.decomposition import PCA

    X = np.random.default_rng(1).normal(size=(12, 6))
    coords, explained = visuals._pca_2d(X)
    pca = PCA(n_components=2, svd_solver="full")
    assert np.allclose(coords, pca.fit_transform(X))
    assert explained == pytest.approx(float(pca.explained_variance_ratio_.sum()))


def test_validate_tab_lives_in_its_own_module_with_one_session_import():
    app = (ROOT / "gui" / "app.py").read_text(encoding="utf-8")
    validate = (ROOT / "gui" / "validate.py").read_text(encoding="utf-8")
    assert "def render_validate_tab" in validate and "def render_validate_tab" not in app
    assert "cda.validation" not in app
    assert validate.count("from cda.validation.session import") == 1
    assert "_validation_partition" not in validate


def test_dendrogram_uses_the_pipeline_row_normalizer():
    validate = (ROOT / "gui" / "validate.py").read_text(encoding="utf-8")
    recut = validate.split("def _render_validation_recut")[1]
    assert "normalize_rows(" in recut and "np.where(norms == 0" not in recut


# --- 14. CI and Docker ------------------------------------------------------------------


def _workflow(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_record_fixtures_never_splices_inputs_into_scripts():
    workflow = _workflow("record-fixtures.yml")
    job = workflow["jobs"]["record"]
    assert job["env"]["TERM_INPUT"] == "${{ inputs.term }}"
    for step in job["steps"]:
        assert "${{" not in step.get("run", "")
    assert workflow["permissions"] == {"contents": "read"}
    assert job["permissions"] == {"contents": "write"}


def test_ci_is_read_only_and_tests_the_pinned_interpreter():
    workflow = _workflow("ci.yml")
    assert workflow["permissions"] == {"contents": "read"}
    matrix = workflow["jobs"]["test-offline"]["strategy"]["matrix"]["python-version"]
    assert "3.11" in matrix and "3.12" in matrix


def test_docker_image_matches_the_pinned_interpreter_and_model_revision():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile
    assert "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" in dockerfile  # pragma: allowlist secret
    assert "revision='${EMBEDDING_REVISION}'" in dockerfile


@pytest.mark.parametrize(("flag", "expected"), [([], False), (["--prune-missing"], True)])
def test_ingest_passes_prune_missing_through(tmp_path, monkeypatch, flag, expected):
    from cda.cli import main

    seen = {}
    monkeypatch.setattr("cda.corpus.embeddings.get_embedder", lambda config: None)
    monkeypatch.setattr(
        "cda.corpus.ingest.ingest_pdf_dir",
        lambda *args, **kwargs: seen.update(kwargs) or {},
    )
    code = main(["ingest", "--pdf-dir", str(tmp_path), "--corpus-dir", str(tmp_path / "c"), *flag])
    assert code == 0
    assert seen["prune_missing"] is expected
