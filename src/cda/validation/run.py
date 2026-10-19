"""``cda validate`` orchestration: the paper's clustering, then a session.

Runs the pipeline's exact Stage 1 and Stage 2 (same seeds, same saturation
subsampling), collects the independent agent cluster readings (mock-able:
the reading prompts are deterministic for a given corpus and config), and
creates the session the researcher's verdicts accumulate into. No register
assignment happens here; validation never calls the analysis LLM step.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from cda.clustering.spherical import normalize_rows
from cda.config import Config
from cda.corpus.embeddings import get_embedder
from cda.corpus.ingest import get_collection
from cda.pipeline import clustering_mode, extract_passages, map_divergence, resolution_params
from cda.types import Passage
from cda.validation.agents import estimate_cost, read_clusters, reader_provider, resolve_readers
from cda.validation.prompt_loader import load_validation_prompts
from cda.validation.report import render_validation_report, write_validation_outputs
from cda.validation.session import create_session

_CLUSTER_LABEL = re.compile(r"^Cluster (\d+)$")


def _cluster_of(passage: Passage) -> int | None:
    match = _CLUSTER_LABEL.match(str(passage.cluster_id))
    return int(match.group(1)) if match else None


def _shown_by_cluster(passages: list[Passage], n_shown: int) -> dict[int, list[str]]:
    """Centroid-nearest ``n_shown`` chunk ids per cluster (what both the
    researcher and the reader agent see)."""
    members: dict[int, list[Passage]] = {}
    for passage in passages:
        cluster = _cluster_of(passage)
        if cluster is not None and passage.embedding is not None:
            members.setdefault(cluster, []).append(passage)
    shown: dict[int, list[str]] = {}
    for cluster, group in sorted(members.items()):
        # Same guard the clustering path uses: a NaN row would not raise here,
        # it would quietly poison the centroid-nearest sort keys and hand the
        # researcher (and the reader agent) an arbitrary set of passages.
        X = normalize_rows(
            np.asarray([p.embedding for p in group], dtype=float),
            [f"{p.chunk_id} (cluster {cluster})" for p in group],
        )
        centroid = X.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / norm if norm > 0 else centroid
        ranked = sorted(
            range(len(group)), key=lambda i: (-float(X[i] @ centroid), group[i].chunk_id)
        )
        shown[cluster] = [group[i].chunk_id for i in ranked[:n_shown]]
    return shown


def run_validation(
    term: str,
    config: Config,
    *,
    mock: bool = False,
    fixtures_dir: Path | None = None,
    record_fixtures: bool = False,
    embedder=None,
    collection=None,
    providers: dict[str, object] | None = None,
    progress: Callable[[str], None] = print,
) -> dict:
    """Cluster exactly as ``cda analyze`` would, read, and open a session."""
    term = term.strip()
    if not term:
        # An empty term compiles to a regex matching every sentence: the run
        # would cluster the whole corpus (and, live, pay for readings of it)
        # filed under a meaningless slug.
        raise ValueError("--term is empty. Pass the term the analysis should validate.")

    # Construct every reader's provider BEFORE the heavy stages, for the same
    # reason analyze_term validates its provider up front: a missing API key
    # (ANTHROPIC_API_KEY, or OPENAI_API_KEY for an OpenAI-compatible second
    # reader) or a missing fixtures directory would otherwise surface only
    # inside read_clusters, after extraction and clustering had already spent
    # minutes on a real corpus — and, with a misconfigured reader 2+, after
    # reader 1's calls had already been paid for. Construction is cheap in every mode:
    # live providers only read the environment, mock providers only hold a
    # path. Tests that pass ``providers`` keep their stubs untouched.
    readers = resolve_readers(config)
    if providers is None:
        providers = {
            spec: reader_provider(
                spec, mock=mock, fixtures_dir=fixtures_dir, record=record_fixtures
            )
            for spec in readers
        }

    embedder = embedder or get_embedder(config)
    collection = collection if collection is not None else get_collection(config)

    researcher, literature, stats = extract_passages(
        term, config, embedder=embedder, collection=collection, progress=progress
    )
    # The 5th value is the joint partition (None under independent mode); the
    # session only needs the per-corpus views, whose cluster ids are the shared
    # ones under joint clustering.
    researcher_summary, literature_summary, literature, _saturation, _joint = map_divergence(
        term, researcher, literature, stats, config, embedder=embedder, progress=progress
    )

    n_shown = config.get_int("validation.passages_shown_per_cluster", 8, minimum=1)
    sides: dict[str, dict] = {}
    partitions: list[dict] = []
    passages_map: dict[str, dict] = {}
    shown_texts: dict[str, dict[int, list[str]]] = {}
    notes: list[str] = []

    for side, summary, pool in (
        ("researcher", researcher_summary, researcher),
        ("literature", literature_summary, literature),
    ):
        if not summary.used:
            sides[side] = {"validated": False, "note": summary.note}
            notes.append(f"{side} side not validated: {summary.note}")
            continue
        assignments: dict[str, int] = {}
        unclustered = 0
        for passage in pool:
            cluster = _cluster_of(passage)
            if cluster is None:
                unclustered += 1
                continue
            assignments[passage.chunk_id] = cluster
            passages_map[passage.chunk_id] = {
                "text": passage.text,
                "side": side,
                "source": passage.source,
            }
        shown = _shown_by_cluster(pool, n_shown)
        clusters = [
            {
                "cluster_id": cluster,
                "n_passages": sum(1 for c in assignments.values() if c == cluster),
                "silhouette": None,  # per-cluster silhouettes appear once edits recompute them
                "shown": shown.get(cluster, []),
            }
            for cluster in sorted(set(assignments.values()))
        ]
        note = None
        if unclustered:
            note = f"{unclustered} passage(s) had no embedding and are not part of any cluster"
            notes.append(f"{side}: {note}")
        sides[side] = {"validated": True, "note": note}
        partitions.append(
            {
                "revision": 0,
                "side": side,
                "kind": "original",
                "assignments": assignments,
                "clusters": clusters,
            }
        )
        shown_texts[side] = {
            cluster: [passages_map[chunk_id]["text"] for chunk_id in ids]
            for cluster, ids in shown.items()
        }

    if not partitions:
        raise ValueError(
            f"Neither corpus side produced a clustering to validate for '{term}'. "
            + " ".join(notes)
        )

    prompts = load_validation_prompts(config)
    n_clusters = sum(len(clusters) for clusters in shown_texts.values())
    n_calls = n_clusters * len(readers)
    approx_tokens = sum(
        len(text) // 4
        for clusters in shown_texts.values()
        for texts in clusters.values()
        for text in texts
    ) * len(readers)
    if not mock:
        progress(f"[validate] cluster readings: {estimate_cost(n_calls, approx_tokens)}")
    readings = read_clusters(
        term,
        shown_texts,
        prompts,
        config,
        mock=mock,
        fixtures_dir=fixtures_dir,
        record=record_fixtures,
        providers=providers,
        progress=progress,
    )

    now = datetime.now()
    session_dir, session = create_session(
        term,
        params={
            "seed": config.get_int("clustering.seed", 42, minimum=0),
            "passages_shown_per_cluster": n_shown,
            "readers": readers,
            # Recorded so re-renders of the report (and anyone reading the
            # audit trail) see the model that actually produced this run's
            # clustering, even if embedding.model changes later.
            "embedding_model": str(config.get("embedding.model")),
            "embedding_revision": config.get("embedding.revision", None),
            # Recorded so the session knows whether its revision-0 partition
            # is a cut of one shared tree — the precondition for the GUI's
            # "move the cutoff line" recut (see curate.apply_recut).
            "clustering_mode": clustering_mode(config),
            "clustering_method": resolution_params(config)["method"],
        },
        sides=sides,
        partitions=partitions,
        passages=passages_map,
        readings=readings,
        output_dir=config.path("paths.output_dir"),
        now=now,
    )
    report_text = render_validation_report(session, config, now)
    written = write_validation_outputs(session_dir, session, report_text, now)
    progress(f"[validate] session: {session_dir}")
    progress(
        "[validate] record your verdicts in the GUI's Validate tab "
        "(streamlit run gui/app.py); the report updates as you go"
    )
    return {
        "status": "ok",
        "session_dir": str(session_dir),
        "report_file": written["report_file"],
        "notes": notes,
    }
