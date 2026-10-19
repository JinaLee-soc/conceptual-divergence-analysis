"""The GUI's Validate tab: blind verdicts, reveals, curation, recut, endorsement.

All enforcement (blind-then-reveal, verdict completeness, embedder checks)
lives in cda.validation; this module renders it. Every handler that mutates
session.json holds ``session_lock`` across its whole reload→mutate→save cycle
(the lock is not reentrant, so helpers called inside it never take it again).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from shared import embedder_or_error, render_report_file, run_with_live_log

from cda.clustering.hierarchy import cut_height, linkage_matrix
from cda.clustering.spherical import normalize_rows
from cda.validation.agents import suggest_edit
from cda.validation.curate import apply_pending_edit, apply_recut, require_session_embedder
from cda.validation.prompt_loader import load_validation_prompts
from cda.validation.report import render_validation_report, write_validation_outputs
from cda.validation.run import run_validation
from cda.validation.session import (
    current_endorsement,
    current_revision,
    endorse,
    find_latest_session,
    has_verdict,
    is_revealed,
    load_session,
    partition_for,
    record_verdict,
    reveal_readings,
    save_session,
    session_lock,
)


def _validation_session():
    """The active validation session (dir, state) or (None, None).

    A session that stops loading is said out loud, once, and the stale
    reference is dropped: silently returning (None, None) would make a corrupt
    or deleted session look like no session at all, with nothing telling the
    researcher their verdict trail exists but cannot be read.
    """
    session_dir = st.session_state.get("validation_session_dir")
    if not session_dir:
        return None, None
    try:
        return Path(session_dir), load_session(Path(session_dir))
    except FileNotFoundError:
        st.info(f"The session at {session_dir} is no longer on disk; start or resume another.")
        st.session_state.pop("validation_session_dir", None)
        return None, None
    except ValueError as exc:
        # Corrupt record or incompatible schema — load_session's message names
        # the file and the recovery; the reference is kept so a restored file
        # is picked up on the next pass.
        st.error(str(exc))
        return None, None


def _refresh_validation_report(session_dir, session, config) -> None:
    """Save the session and re-render its report.

    Every caller holds ``session_lock`` around its whole load→mutate→save
    cycle (the lock is not reentrant, so it cannot be re-acquired here):
    saving outside the lock would reopen the two-tab lost-update window the
    lock exists to close.
    """
    save_session(session_dir, session)
    report = render_validation_report(session, config, datetime.now())
    write_validation_outputs(session_dir, session, report, datetime.now())


def _render_validation_cluster(session_dir, session, side, cluster, config) -> None:
    """One cluster's inspection block: passages, verdict, then the reveal.

    The agent reading is fetched only through session.reveal_readings, which
    refuses until a verdict exists; this widget cannot unblind the researcher.
    """
    revision = current_revision(session, side)
    cluster_id = cluster["cluster_id"]
    key = f"validate_{side}_{cluster_id}_{revision}"
    st.markdown(f"**Cluster {cluster_id}** ({cluster['n_passages']} passages)")
    for chunk_id in cluster["shown"]:
        st.markdown(f'- "{session["passages"][chunk_id]["text"]}"')
    hidden = cluster["n_passages"] - len(cluster["shown"])
    if hidden > 0:
        st.caption(f"({hidden} more passages in this cluster are not shown)")

    if not has_verdict(session, side, cluster_id, revision):
        others = [
            c["cluster_id"]
            for c in partition_for(session, side, revision)["clusters"]
            if c["cluster_id"] != cluster_id
        ]
        # "overlaps" points at another cluster; with a single-cluster
        # partition there is nothing to point at, and offering the option
        # would end in a commit-time crash from an empty selectbox.
        choices = (
            ["coherent", "mixed", "overlaps", "artifact"]
            if others
            else ["coherent", "mixed", "artifact"]
        )
        verdict = st.radio(
            "Your verdict (committed before you see the agent's reading)",
            choices,
            key=f"{key}_verdict",
            horizontal=True,
        )
        overlaps_with = None
        if verdict == "overlaps":
            overlaps_with = st.selectbox("Overlaps with cluster", others, key=f"{key}_overlap")
        notes = st.text_input("Notes (optional)", key=f"{key}_notes")
        if st.button("Commit verdict", key=f"{key}_commit"):
            # Reload under the lock: another tab may have committed since this
            # pass's snapshot.
            try:
                with session_lock(session_dir):
                    session = load_session(session_dir)
                    if not has_verdict(session, side, cluster_id, revision):
                        record_verdict(
                            session,
                            side=side,
                            cluster_id=cluster_id,
                            revision=revision,
                            verdict=verdict,
                            notes=notes,
                            overlaps_with=overlaps_with,
                            now=datetime.now(),
                        )
                        _refresh_validation_report(session_dir, session, config)
            except TimeoutError as exc:
                st.error(str(exc))
                return
            st.rerun()
        return

    # Only a first reveal changes the record, so only it is written.
    first_reveal = not is_revealed(session, side, cluster_id, revision)
    if first_reveal:
        try:
            with session_lock(session_dir):
                session = load_session(session_dir)
                readings = reveal_readings(
                    session, side=side, cluster_id=cluster_id, revision=revision, now=datetime.now()
                )
                _refresh_validation_report(session_dir, session, config)
        except TimeoutError as exc:
            st.error(str(exc))
            return
    else:
        readings = reveal_readings(
            session, side=side, cluster_id=cluster_id, revision=revision, now=datetime.now()
        )
    mine = [
        v
        for v in session["verdicts"]
        if v["side"] == side and v["cluster_id"] == cluster_id and v["revision"] == revision
    ]
    st.caption(
        f"Your verdict: {mine[0]['verdict']}"
        + (f" ({mine[0]['notes']})" if mine[0]["notes"] else "")
    )
    if not readings:
        st.info(
            "No agent reading for this cluster: agent readings exist only for "
            "the original clustering, so clusters created by a curation edit "
            "are judged without one."
        )
    for reading in readings:
        if not reading.get("valid"):
            st.warning(
                f"Agent reading from {reading['model']} was unparseable and is excluded from agreement."
            )
            continue
        st.markdown(
            f"Agent reading ({reading['model']}): **{reading['judgment']}**. "
            f"{reading['gloss']} ({reading['rationale']})"
        )
    if len(mine) == 1:
        revised = st.radio(
            "Revise your verdict after the reveal? (the blind original stays on record)",
            ["keep", "coherent", "mixed", "artifact"],
            key=f"{key}_revise",
            horizontal=True,
        )
        if revised != "keep" and st.button("Record revision", key=f"{key}_revise_commit"):
            try:
                with session_lock(session_dir):
                    session = load_session(session_dir)
                    record_verdict(
                        session,
                        side=side,
                        cluster_id=cluster_id,
                        revision=revision,
                        verdict=revised,
                        notes="revised after reveal",
                        now=datetime.now(),
                    )
                    _refresh_validation_report(session_dir, session, config)
            except TimeoutError as exc:
                st.error(str(exc))
                return
            st.rerun()


def render_validate_tab(config, mock_llm: bool, key_ok: bool) -> None:
    st.subheader("Validate the clustering (experimental)")
    st.write(
        "You are the validator: for each cluster you record a blind verdict "
        "(before seeing the agent's independent reading of the same passages), "
        "then optionally curate the structure with merges and splits. The "
        "verdict trail and your agreement with the agent readings are the "
        "validity evidence; curation never changes what `cda analyze` reports."
    )

    term = st.text_input("Term", placeholder="e.g. novelty", key="validate_term")
    demo_mode = st.checkbox(
        "Use the toy demo corpus (same corpus and session as `cda validate --demo`)",
        key="validate_demo",
    )
    if demo_mode:
        # Mirror the CLI's --demo: demo paths, demo fixtures, self-heal ingest.
        from cda.demo import demo_config, find_demo_dir

        try:
            demo_dir = find_demo_dir()
        except FileNotFoundError as exc:
            # Running the app from outside the cloned repository is the one
            # way this fails; say so instead of dumping the traceback.
            st.error(str(exc))
            return
        config = demo_config(demo_dir)
        fixtures_dir = demo_dir / "fixtures" / "validation"
    else:
        fixtures_dir = config.path("paths.output_dir") / "fixtures" / "validation"

    fixtures_recorded = fixtures_dir.is_dir() and any(fixtures_dir.glob("*.json"))
    if demo_mode and mock_llm and not fixtures_recorded:
        st.warning(
            "No recorded demo readings yet, so the mock toggle has nothing to "
            "replay. One-time setup: switch OFF the sidebar's mock toggle "
            "(enter your API key in the sidebar) and start the demo "
            "validation once. That live run costs a few cents, records the "
            "readings, and the demo replays offline forever after."
        )
    elif mock_llm and not fixtures_recorded:
        # The same refusal for an own-corpus session (the CLI checks this up
        # front too): without it, a mock validation started anyway and died
        # only after the heavy extraction and clustering stages.
        st.warning(
            "No recorded readings to replay for this corpus, so the mock "
            "toggle has nothing to work with. A live validation run records "
            "the readings first: switch OFF the sidebar's mock toggle (enter "
            "your API key in the sidebar), and the start button below then "
            "records every reading as it runs — this term replays offline "
            "for free afterwards."
        )

    start_col, resume_col = st.columns(2)
    with start_col:
        if mock_llm:
            start_label = "Start validation (replay recorded readings)"
        elif demo_mode and not fixtures_recorded:
            start_label = (
                "Start validation (live, a few cents; records the demo readings for offline replay)"
            )
        else:
            start_label = "Start validation (live readings, a few cents)"
        if st.button(start_label, disabled=not term, key="validate_start"):
            if mock_llm and not fixtures_recorded:
                # Refuse before any work starts: the warning above explains
                # the way out, and failing later (after clustering) would
                # spend minutes to reach the same dead end.
                st.error(
                    "Nothing to replay yet — record the readings with one "
                    "live run first: switch OFF the sidebar's mock toggle "
                    "(enter your API key in the sidebar) and press this "
                    "button again. After that live run, the mock toggle "
                    "replays the recorded readings for free."
                )
                return
            embedder = embedder_or_error(config)
            if embedder is None:
                return

            def _start(progress):
                if demo_mode:
                    from cda.demo import ensure_toy_corpus_ingested

                    ensure_toy_corpus_ingested(config, demo_dir, embedder)
                return run_validation(
                    term,
                    config,
                    mock=mock_llm,
                    fixtures_dir=fixtures_dir,
                    # Live runs record every completed reading as a replayable
                    # fixture, so an interrupted run (rate limit, closed tab)
                    # resumes without paying again — and the demo's first live
                    # run doubles as the recording that enables offline replay.
                    record_fixtures=not mock_llm,
                    embedder=embedder,
                    progress=progress,
                )

            result = run_with_live_log(_start)
            if result:
                st.session_state["validation_session_dir"] = result["session_dir"]
                st.rerun()
    with resume_col:
        if st.button(
            "Resume latest session for this term", disabled=not term, key="validate_resume"
        ):
            try:
                st.session_state["validation_session_dir"] = str(
                    find_latest_session(config.path("paths.output_dir"), term)
                )
                st.rerun()
            except FileNotFoundError as exc:
                st.error(str(exc))

    session_dir, session = _validation_session()
    if not session:
        return
    st.caption(f'Session: {session_dir} (term: "{session["term"]}")')

    # --- inspection queue -----------------------------------------------
    for side in ("researcher", "literature"):
        info = session["sides"].get(side)
        if not info:
            continue
        st.markdown(f"### {side.capitalize()} clusters")
        if not info.get("validated"):
            st.caption(f"Not validated: {info.get('note')}")
            continue
        revision = current_revision(session, side)
        partition = partition_for(session, side, revision)
        pending = [
            c
            for c in partition["clusters"]
            if not has_verdict(session, side, c["cluster_id"], revision)
        ]
        done = len(partition["clusters"]) - len(pending)
        st.caption(f"Verdicts: {done}/{len(partition['clusters'])} clusters (revision {revision})")
        for cluster in partition["clusters"]:
            with st.expander(
                f"Cluster {cluster['cluster_id']} "
                + (
                    "(verdict recorded)"
                    if has_verdict(session, side, cluster["cluster_id"], revision)
                    else "(needs your verdict)"
                ),
                expanded=not has_verdict(session, side, cluster["cluster_id"], revision),
            ):
                _render_validation_cluster(session_dir, session, side, cluster, config)

    _render_validation_curation(session_dir, session, config, mock_llm, key_ok)
    _render_validation_recut(session_dir, session, config)

    # --- endorse ----------------------------------------------------------
    st.markdown("### Endorse")
    all_done = all(
        has_verdict(session, side, c["cluster_id"], current_revision(session, side))
        for side in ("researcher", "literature")
        if session["sides"].get(side, {}).get("validated")
        for c in partition_for(session, side, current_revision(session, side))["clusters"]
    )
    endorsement = current_endorsement(session)
    if endorsement:
        st.success(f"Endorsed at {endorsement['at']}.")
    else:
        _render_endorsement_withdrawn_notice(session)
        note = st.text_input("Endorsement note (optional)", key="validate_endorse_note")
        if st.button(
            "Endorse this structure (finalizes the report)",
            disabled=not all_done,
            key="validate_endorse",
        ):
            # Locked reload→endorse→save: same lost-update reasoning as every
            # other mutation of session.json in this file.
            try:
                with session_lock(session_dir):
                    session = load_session(session_dir)  # another tab may have written
                    endorse(session, note=note, now=datetime.now())
                    _refresh_validation_report(session_dir, session, config)
            except TimeoutError as exc:
                st.error(str(exc))
                return
            st.rerun()
        if not all_done:
            st.caption("Endorsement unlocks once every cluster has a verdict.")

    report_path = session_dir / f"{session_dir.name}.md"
    if report_path.is_file():
        with st.expander("Validation report (updates as you work)"):
            render_report_file(report_path, key_prefix="validate_session")


def _render_endorsement_withdrawn_notice(session: dict) -> None:
    """Say why an earlier endorsement no longer stands, if one was retracted."""
    withdrawals = session.get("endorsement_withdrawals") or []
    stale = session.get("endorsement")
    if withdrawals:
        last = withdrawals[-1]
        st.warning(
            f"Your endorsement of {last['at']} was withdrawn: edit "
            f"{last['withdrawn_by_edit']} changed the structure (now revision(s) "
            f"{last['withdrawn_at_revisions']}). Judge the changed clusters, then "
            "endorse again."
        )
    elif stale:
        st.warning(
            f"The endorsement of {stale['at']} covers revisions {stale['revisions']}, "
            "but the structure has changed since. Judge the changed clusters, then "
            "endorse again."
        )


def _render_validation_curation(session_dir, session, config, mock_llm: bool, key_ok: bool) -> None:
    st.markdown("### Curate (live)")
    st.caption(
        "Suggest a merge or split; the revision agent argues and, for splits, "
        "proposes a semantic split beside the mechanical one. You decide. "
        "Edited clusters return to the inspection queue; agreement statistics "
        "always refer to the original clustering."
    )
    live_ok = key_ok and not mock_llm
    if not live_ok:
        st.caption(
            "Curation calls the API live (arbitrary edits cannot be "
            "pre-recorded): enter your API key in the sidebar and switch off "
            "the mock toggle to enable it."
        )

    pending = session.get("pending_edit")
    if pending:
        advice = pending.get("agent_advice") or {}
        st.info(
            f"Pending {pending['kind']} on {pending['side']} clusters "
            f"{pending['clusters']}. Agent recommends: "
            f"{advice.get('recommendation') or 'n/a'}. {advice.get('argument') or ''}"
        )
        if pending["kind"] == "split" and pending.get("split_proposals"):
            proposals = pending["split_proposals"]
            if proposals.get("semantic"):
                st.markdown(
                    f"- Semantic split: A = {proposals['semantic']['a_gloss']} "
                    f"| B = {proposals['semantic']['b_gloss']}"
                )
            else:
                st.markdown("- Semantic split: n/a (the agent's reply was unusable)")
            st.markdown("- Mechanical split: seeded k=2 k-means on the member embeddings")
        decisions = (
            ["reject", "apply-merge"]
            if pending["kind"] == "merge"
            else ["reject", "apply-semantic", "apply-mechanical"]
        )
        decision = st.radio("Decision", decisions, key="validate_decision", horizontal=True)
        if st.button("Apply decision", key="validate_decide"):
            embedder = embedder_or_error(config)
            if embedder is None:
                return
            # Locked reload→apply→save, like the verdict handlers: an edit
            # rewrites the partitions, so a verdict committed by another tab
            # mid-cycle would otherwise be erased wholesale.
            try:
                with session_lock(session_dir):
                    session = load_session(session_dir)  # another tab may have written
                    apply_pending_edit(
                        session,
                        decision,
                        config,
                        embedder=embedder,
                        session_dir=session_dir,
                        now=datetime.now(),
                        progress=lambda _m: None,
                    )
                    _refresh_validation_report(session_dir, session, config)
            except (TimeoutError, ValueError) as exc:
                st.error(str(exc))
                return
            st.rerun()
        return

    sides = [s for s, info in session["sides"].items() if info.get("validated")]
    if not sides:
        return
    side = st.selectbox("Side", sides, key="validate_curate_side")
    revision = current_revision(session, side)
    cluster_ids = [c["cluster_id"] for c in partition_for(session, side, revision)["clusters"]]
    kind = st.radio("Edit", ["merge", "split"], key="validate_curate_kind", horizontal=True)
    if kind == "merge":
        chosen = st.multiselect("Clusters to merge (pick two)", cluster_ids, key="validate_merge")
    else:
        chosen = [st.selectbox("Cluster to split", cluster_ids, key="validate_split")]
    ready = (kind == "merge" and len(chosen) == 2) or (kind == "split" and len(chosen) == 1)
    # Blind-then-reveal applies to the advice channel too (docs/validation_protocol.md):
    # the agent's argument is a reading of these clusters, so it stays locked
    # until the researcher's blind verdict on each of them is committed. The
    # data layer (suggest_edit) enforces this; the disabled button explains it.

    unjudged = [c for c in chosen if not has_verdict(session, side, int(c), revision)]
    if unjudged:
        st.caption(
            "Record your blind verdict on cluster(s) "
            f"{', '.join(str(c) for c in unjudged)} first — the agent's argument "
            "is a reading of those passages, and seeing it before your verdict "
            "would unblind it."
        )
    if st.button(
        "Ask the revision agent (live, a few cents)",
        disabled=not (ready and live_ok) or bool(unjudged),
        key="validate_suggest",
    ):
        # suggest_edit refuses this too; checking first keeps the refusal a
        # plain error rather than a failed live-call log.
        try:
            require_session_embedder(session, config)
        except ValueError as exc:
            st.error(str(exc))
            return
        embedder = embedder_or_error(config)
        if embedder is None:
            return
        prompts = load_validation_prompts(config)
        # The lock spans the live call deliberately: suggest_edit mutates the
        # loaded session (pending_edit), so the load and the save must be one
        # cycle — a verdict committed by another tab between them would be
        # erased by this save. Another tab that tries to write meanwhile gets
        # the lock's own message: let this finish and retry.
        try:
            with session_lock(session_dir):
                session = load_session(session_dir)  # another tab may have written
                result = run_with_live_log(
                    lambda progress: suggest_edit(
                        session,
                        kind=kind,
                        side=side,
                        clusters=[int(c) for c in chosen],
                        config=config,
                        embedder=embedder,
                        prompts=prompts,
                        now=datetime.now(),
                        progress=progress,
                    )
                )
                if result:
                    _refresh_validation_report(session_dir, session, config)
        except TimeoutError as exc:
            st.error(str(exc))
            return
        if result:
            st.rerun()


def _render_validation_recut(session_dir, session, config) -> None:
    """Move the dendrogram cutoff (joint hierarchical sessions only, LLM-free).

    The expander appears only when the session's revision-0 partition is a cut
    of one shared tree (clustering.mode: joint + clustering.method:
    hierarchical); any other cut of that same tree is then reachable by
    choosing a different cluster count. The tree is rebuilt on demand from the
    stored passage texts (deterministic), cached in session state so slider
    moves redraw instantly.
    """
    params = session.get("params", {})
    if (
        params.get("clustering_method") != "hierarchical"
        or params.get("clustering_mode") != "joint"
    ):
        return

    st.markdown("### Adjust resolution (no API call)")
    with st.expander("Move the cutoff line on the clustering dendrogram"):
        st.caption(
            "Your clusters are one cut of a single hierarchical tree over all "
            "passages. Choosing a different number of clusters moves the "
            "cutoff line: clusters whose membership is unchanged keep their "
            "verdicts; all others return to the inspection queue. Applies to "
            "both corpus sides at once. Free and local."
        )
        sides = [s for s, info in session["sides"].items() if info.get("validated")]
        n_passages = len(session["passages"])
        current_k = len(
            {
                cluster
                for side in sides
                for cluster in partition_for(session, side, current_revision(session, side))[
                    "assignments"
                ].values()
            }
        )
        max_k = min(40, n_passages - 1)
        k = st.slider(
            "Number of clusters (the cutoff level)",
            min_value=2,
            max_value=max_k,
            value=min(current_k, max_k),
            key="validate_recut_k",
        )

        cache_key = f"validate_recut_tree::{session_dir}"
        if cache_key not in st.session_state:
            if st.button("Compute the dendrogram", key="validate_recut_tree_btn"):
                embedder = embedder_or_error(config)
                if embedder is None:
                    return

                chunk_ids = sorted(session["passages"])
                with st.spinner(f"Embedding {len(chunk_ids)} passages and building the tree..."):
                    # The same guard apply_recut uses, so the tree drawn here
                    # is the tree a recut applies.
                    X = normalize_rows(
                        np.asarray(
                            embedder.embed([session["passages"][c]["text"] for c in chunk_ids]),
                            dtype=float,
                        ),
                        chunk_ids,
                    )
                    st.session_state[cache_key] = linkage_matrix(X)
                st.rerun()
        if cache_key in st.session_state:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from scipy.cluster.hierarchy import dendrogram

                Z = st.session_state[cache_key]
                height = cut_height(Z, k)
                fig, ax = plt.subplots(figsize=(8, 3))
                dendrogram(
                    Z,
                    ax=ax,
                    truncate_mode="lastp",
                    p=min(40, n_passages),
                    no_labels=True,
                    color_threshold=height,
                )
                ax.axhline(height, color="#c0392b", linestyle="--", linewidth=1.2)
                ax.text(
                    0.99,
                    height,
                    f" cut: k={k}",
                    color="#c0392b",
                    fontsize=8,
                    va="bottom",
                    ha="right",
                    transform=ax.get_yaxis_transform(),
                )
                ax.set_ylabel("cosine distance")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except ImportError:
                st.info(
                    "Install matplotlib to see the dendrogram (pip install "
                    "matplotlib). The recut itself works without it."
                )

        if st.button(f"Apply recut at k={k}", key="validate_recut_apply"):
            embedder = embedder_or_error(config)
            if embedder is None:
                return
            # Locked reload→recut→save: a recut rewrites both sides'
            # partitions, the widest write this tab performs, so a verdict
            # committed by another tab mid-cycle would otherwise vanish.
            try:
                with session_lock(session_dir):
                    session = load_session(session_dir)  # another tab may have written
                    apply_recut(
                        session,
                        k,
                        config,
                        embedder=embedder,
                        session_dir=session_dir,
                        now=datetime.now(),
                        progress=lambda _m: None,
                    )
                    _refresh_validation_report(session_dir, session, config)
            except (TimeoutError, ValueError) as exc:
                st.error(str(exc))
                return
            st.rerun()
