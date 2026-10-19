"""Result visualizations for the CDA GUI.

Pure presentation: every chart is built from the result dict `analyze_term`
returns (plus the clustered passages when requested) — nothing here touches
retrieval, clustering, prompts, or reports, so visuals can evolve freely
without invalidating recorded fixtures or the paper's pipeline.

Charts use Altair, which ships with Streamlit (no extra dependency). The
audience is non-CS researchers: each chart carries a one-line "how to read
this" caption.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Okabe–Ito colorblind-safe pair: researcher warm, literature cool.
CORPUS_COLORS = alt.Scale(domain=["researcher", "literature"], range=["#D55E00", "#0072B2"])


def _pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Project embeddings to 2D. Returns (coords, explained variance share).

    The full SVD solver is deterministic, including component signs.
    """
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(np.asarray(matrix, dtype=float))
    return coordinates, _explained_share(pca)


def _explained_share(pca: PCA) -> float:
    """Variance share of the fitted components; 0.0 when there is no variance.

    sklearn divides by the total variance, so identical embeddings yield NaN
    ratios, which a caption would print as "nan%".
    """
    if not np.sum(pca.explained_variance_) > 0:
        return 0.0
    return float(np.sum(pca.explained_variance_ratio_))


def _passages_frame(passages: dict) -> pd.DataFrame | None:
    """One row per embedded passage across both corpora, with 2D coordinates."""
    rows = []
    for corpus in ("researcher", "literature"):
        for passage in passages.get(corpus, []):
            if passage.embedding is None:
                continue
            rows.append(
                {
                    "corpus": corpus,
                    "cluster": passage.cluster_id,
                    "source": passage.source,
                    "year": passage.year,
                    "text": (passage.text[:140] + "…") if len(passage.text) > 140 else passage.text,
                    "embedding": passage.embedding,
                }
            )
    if len(rows) < 3:
        return None
    frame = pd.DataFrame(rows)
    coordinates, explained = _pca_2d(np.array(frame.pop("embedding").tolist()))
    frame["x"], frame["y"] = coordinates[:, 0], coordinates[:, 1]
    frame.attrs["explained"] = explained
    return frame


def passage_map_chart(frame: pd.DataFrame) -> alt.Chart:
    """Both corpora in one shared embedding plane — divergence is distance."""
    # mark_point (not mark_circle) so the shape channel actually renders:
    # circle vs square distinguishes the corpora even without color vision.
    return (
        alt.Chart(frame)
        .mark_point(filled=True, size=70, opacity=0.75)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("corpus:N", scale=CORPUS_COLORS, legend=alt.Legend(title=None)),
            shape=alt.Shape("corpus:N", legend=None),
            tooltip=["corpus:N", "cluster:N", "source:N", "year:N", "text:N"],
        )
        .properties(height=420)
        .interactive()
    )


def joint_cluster_map_chart(frame: pd.DataFrame) -> alt.Chart:
    """The same plane, coloured by shared cluster — where the researcher sits.

    Only meaningful under joint clustering, where a cluster id names one region
    holding passages from both corpora: colour is the shared cluster, shape
    (and outline) is the corpus, so the researcher's position *inside* each
    cluster is the thing you read off the chart.
    """
    # Literature drawn first and faint, the researcher's own passages on top in
    # solid black outline: with one colour per cluster, the corpora have to be
    # separated by weight, not hue.
    literature = (
        alt.Chart(frame)
        .transform_filter(alt.datum.corpus == "literature")
        .mark_point(filled=True, size=55, opacity=0.45)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("cluster:N", legend=alt.Legend(title="shared cluster")),
            shape=alt.value("circle"),
            tooltip=["cluster:N", "corpus:N", "source:N", "year:N", "text:N"],
        )
    )
    researcher = (
        alt.Chart(frame)
        .transform_filter(alt.datum.corpus == "researcher")
        .mark_point(filled=True, size=150, opacity=0.95, stroke="black", strokeWidth=1.2)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("cluster:N", legend=None),
            shape=alt.value("diamond"),
            tooltip=["cluster:N", "corpus:N", "source:N", "year:N", "text:N"],
        )
    )
    return (literature + researcher).properties(height=440).interactive()


def joint_composition_chart(joint) -> alt.Chart | None:
    """Per shared cluster: your passages vs the literature's, with the flag line.

    A normalized bar per cluster (researcher share of membership) with the
    concentration threshold as a dashed rule, ordered by that share — so the
    clusters the researcher dominates and the ones they never enter sit at
    opposite ends of the chart.
    """
    if joint is None or not getattr(joint, "used", False) or not joint.clusters:
        return None
    rows = []
    for entry in joint.clusters:
        label = f"Cluster {entry['cluster_id']}"
        rows.append(
            {
                "cluster": label,
                "corpus": "researcher",
                "passages": entry["n_researcher"],
                "share": entry["researcher_share"],
            }
        )
        rows.append(
            {
                "cluster": label,
                "corpus": "literature",
                "passages": entry["n_literature"],
                "share": 1.0 - entry["researcher_share"],
            }
        )
    frame = pd.DataFrame(rows)
    order = [
        f"Cluster {entry['cluster_id']}"
        for entry in sorted(joint.clusters, key=lambda e: -e["researcher_share"])
    ]
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(
                "passages:Q",
                stack="normalize",
                title="share of the cluster's passages",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("cluster:N", title=None, sort=order),
            color=alt.Color("corpus:N", scale=CORPUS_COLORS, legend=alt.Legend(title=None)),
            tooltip=["cluster:N", "corpus:N", "passages:Q", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=max(120, 26 * len(joint.clusters)))
    )


def cluster_size_chart(frame: pd.DataFrame) -> alt.Chart:
    """How the passages distribute over each corpus's k-means clusters."""
    counts = frame.groupby(["corpus", "cluster"], as_index=False).size()
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("size:Q", title="passages"),
            y=alt.Y("cluster:N", title=None),
            color=alt.Color("corpus:N", scale=CORPUS_COLORS, legend=None),
            row=alt.Row("corpus:N", title=None, header=alt.Header(labelFontWeight="bold")),
            tooltip=["corpus:N", "cluster:N", "size:Q"],
        )
        .properties(height=90)
    )


def saturation_chart(saturation: dict) -> alt.Chart | None:
    """The saturation sweep: ARI between consecutive increments vs sample size."""
    history = [
        {
            "n": step["n"],
            "ARI": step["ari_vs_prev"],
            "k": step.get("k"),
            "silhouette": step.get("silhouette"),
        }
        for step in saturation.get("history", [])
        if step.get("ari_vs_prev") is not None
    ]
    if not history:
        return None
    frame = pd.DataFrame(history)
    threshold = float(saturation.get("ari_threshold", 0.8))
    line = (
        alt.Chart(frame)
        .mark_line(point=True, color="#0072B2")
        .encode(
            x=alt.X("n:Q", title="literature passages sampled (N)"),
            y=alt.Y("ARI:Q", scale=alt.Scale(domain=[0, 1]), title="ARI vs previous increment"),
            tooltip=["n:Q", "ARI:Q", "k:Q", "silhouette:Q"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(strokeDash=[6, 4], color="#D55E00")
        .encode(y="threshold:Q")
    )
    return (line + rule).properties(height=280)


def flags_quadrant_chart(flags: dict) -> alt.Chart | None:
    """Every literature cluster on the two flag axes; thresholds as dashed rules.

    Top-left quadrant (small literature share, high researcher share) is where
    researcher-concentrated clusters live — the paper's >80% / <10% rule made
    visible.
    """
    if not flags.get("available"):
        return None
    frame = pd.DataFrame(flags["rows"])
    frame["flag"] = [
        "researcher-concentrated"
        if row["researcher_concentrated"]
        else ("underrepresented" if row["underrepresented_in_researcher_usage"] else "—")
        for _, row in frame.iterrows()
    ]
    thresholds = flags["thresholds"]
    points = (
        alt.Chart(frame)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X(
                "literature_share:Q",
                title="cluster's share of literature passages",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "researcher_share_of_combined:Q",
                title="researcher share of cluster membership",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            size=alt.Size("n_literature:Q", legend=None),
            color=alt.Color(
                "flag:N",
                scale=alt.Scale(
                    domain=["researcher-concentrated", "underrepresented", "—"],
                    range=["#D55E00", "#CC79A7", "#0072B2"],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                "cluster:N",
                "n_literature:Q",
                "n_researcher_assigned:Q",
                alt.Tooltip("literature_share:Q", format=".1%"),
                alt.Tooltip("researcher_share_of_combined:Q", format=".1%"),
                "flag:N",
            ],
        )
    )
    x_rule = (
        alt.Chart(pd.DataFrame({"v": [thresholds["literature_share_threshold"]]}))
        .mark_rule(strokeDash=[6, 4], color="#666")
        .encode(x="v:Q")
    )
    y_rule = (
        alt.Chart(pd.DataFrame({"v": [thresholds["researcher_share_threshold"]]}))
        .mark_rule(strokeDash=[6, 4], color="#666")
        .encode(y="v:Q")
    )
    return (points + x_rule + y_rule).properties(height=340)


EMERGENCE_AXIS_TITLES = {
    "zotero": "added to your library (Zotero 'Date Added')",
    "filesystem": "PDF arrived on disk (filesystem date)",
}


def emergence_chart(emergence: dict) -> alt.Chart | None:
    """Each final cluster's membership as the library grew; births marked.

    One line per final cluster over acquisition time (the run's date source);
    the large point is the snapshot where the cluster first matched the final
    structure (its birth).
    """
    if emergence.get("status") != "ok":
        return None
    clusters = (emergence.get("final") or {}).get("clusters", [])
    rows = [
        {
            "cluster": f"Cluster {cluster['final_cluster']}",
            "as_of": point["as_of"],
            "n": point["n"],
            "size": point["size"],
            "is_birth": point["n"] == cluster["birth_n"],
        }
        for cluster in clusters
        for point in cluster["growth"]
    ]
    if len({row["as_of"] for row in rows}) < 2:
        return None
    frame = pd.DataFrame(rows)
    base = alt.Chart(frame).encode(
        x=alt.X("as_of:T", title=EMERGENCE_AXIS_TITLES[emergence["date_source"]]),
        y=alt.Y("size:Q", title="passages in cluster"),
        color=alt.Color("cluster:N", legend=alt.Legend(title=None)),
        tooltip=["cluster:N", "as_of:T", "n:Q", "size:Q"],
    )
    lines = base.mark_line(point=True)
    births = base.transform_filter(alt.datum.is_birth).mark_point(
        filled=True, size=220, shape="diamond"
    )
    return (lines + births).properties(height=300)


def render_stability_note(st, stability: dict | None) -> None:
    """Surface the order-stability check beside the headline k.

    The reason this is not buried in the full report: k is the first number a
    reader takes away from the visual summary, and how much to trust it is
    exactly what the stability check answers. A shaky k is shown as a warning
    rather than a caption so it is not mistaken for a footnote.
    """
    if not stability or not stability.get("enabled"):
        return
    if stability.get("mode") == "joint":
        corpora = [("k", stability)]
    else:
        corpora = [
            ("researcher k", stability.get("researcher", {})),
            ("literature k", stability.get("literature", {})),
        ]

    for label, corpus in corpora:
        if not corpus:
            continue
        rate = corpus["k_match_rate"]
        # The denominator is the shuffled orderings, not every trial: trial 0
        # re-clusters the canonical order as a determinism check and is not
        # evidence that reordering preserves k. Reporting the total here would
        # contradict the written report, which uses the shuffled count.
        shuffled = corpus.get("shuffled_trials", max(corpus["trials"] - 1, 0))
        if rate is None:
            st.info(
                f"**{label}={corpus['canonical_k']}** — no shuffled orderings were "
                "run, so nothing here says whether that k survives a reordering."
            )
            continue
        message = (
            f"**{label}={corpus['canonical_k']}** recurred in {rate:.0%} of "
            f"{shuffled} independently shuffled passage orderings"
        )
        if rate < 1.0:
            seen = ", ".join(str(k) for k in corpus["k_seen"])
            st.warning(
                f"{message} (k values seen: {seen}). Read this k as one plausible "
                "partition among several, not the term's one true register count — "
                "usually a sign of weak cluster separation. The registers themselves "
                "can still be sound; it is their *number* that is soft.",
                icon=":material/warning:",
            )
        else:
            st.caption(f"{message} — the partition is robust, not just reproducible. ✓")


def grounding_note(grounding: dict | None) -> str:
    """The caption fragment summarizing quote-grounding — warnings included.

    This must agree with the markdown report's grounding section
    (render_grounding_markdown): quotation marks that fell outside any
    checkable pair enclose text that was NOT verified, so the caption may
    never show the ✓ over them — a "12/12 ✓" here while the report warns
    below it would be a false all-clear on the one number a reader trusts.
    """
    grounding = grounding or {}
    n_quotes = grounding.get("n_quotes", 0)
    n_verified = grounding.get("n_verified", 0)
    unpaired = grounding.get("unpaired_marks", 0)
    if n_quotes:
        note = f" · quotations grounded in the prompt: {n_verified}/{n_quotes}"
        if n_verified < n_quotes or unpaired:
            return note + " Warning: check the report's grounding section"
        return note + " ✓"
    if unpaired:
        # No checkable pairs at all, yet quotation marks are present: the
        # report warns about exactly this, so silence here would read as
        # "nothing quoted, nothing to check" — which is not what happened.
        return (
            " · Warning: quotation marks fell outside any checkable pair — "
            "check the report's grounding section"
        )
    return ""


def render_result_visuals(st, result: dict) -> None:
    """The visual summary shown after a successful run (demo or analyze)."""
    researcher = result.get("researcher_cluster", {})
    literature = result.get("literature_cluster", {})
    joint = result.get("joint")

    columns = st.columns(4)
    if joint is not None and getattr(joint, "used", False):
        occupied = sum(1 for entry in joint.clusters if entry["n_researcher"])
        columns[0].metric("Shared clusters (joint k)", joint.n_clusters)
        columns[1].metric("Joint silhouette", f"{joint.silhouette:.3f}")
        columns[2].metric("…containing your passages", f"{occupied} of {joint.n_clusters}")
        columns[3].metric(
            "Your passages / literature", f"{joint.n_researcher} / {joint.n_literature}"
        )
    else:
        columns[0].metric("Researcher registers (k)", researcher.get("k", "n/a"))
        columns[1].metric("Researcher silhouette", f"{researcher.get('silhouette', 0):.3f}")
        columns[2].metric("Literature registers (k)", literature.get("k", "n/a"))
        columns[3].metric("Literature silhouette", f"{literature.get('silhouette', 0):.3f}")
    st.caption(
        f"{researcher.get('n', '?')} researcher / {literature.get('n', '?')} literature "
        "passages. Low silhouettes are normal for sentence embeddings; see "
        f"docs/interpretation_guide.md.{grounding_note(result.get('grounding'))}"
    )
    render_stability_note(st, result.get("stability"))

    st.subheader("Passage map")
    frame = _passages_frame(result.get("passages", {}))
    if frame is None:
        st.info("Passage coordinates unavailable for this run.")
    else:
        st.altair_chart(passage_map_chart(frame), width="stretch")
        st.caption(
            "Every dot is one term-sentence, projected onto a shared 2D plane "
            f"(PCA, {frame.attrs['explained']:.0%} of variance). Orange = your "
            "writing, blue = literature. Separated orange regions are candidate "
            "divergent registers; hover any dot to read the sentence."
        )

    if joint is None or not getattr(joint, "used", False):
        return

    st.subheader("Where your passages sit in the shared clusters")
    if frame is not None:
        st.altair_chart(joint_cluster_map_chart(frame), width="stretch")
        st.caption(
            "The same plane, now coloured by shared cluster: your passages were "
            "clustered together with the literature, so each colour is one region "
            "both corpora can occupy. Large outlined diamonds are your sentences, "
            "faint circles the literature's — a colour with no diamonds is "
            "literature usage you have not taken up."
        )
    chart = joint_composition_chart(joint)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        threshold = (
            (result.get("flags") or {}).get("thresholds", {}).get("researcher_share_threshold")
        )
        threshold_note = (
            f" A cluster above {threshold:.0%} orange while holding little of the "
            "literature is flagged researcher-concentrated in the report."
            if isinstance(threshold, (int, float))
            else ""
        )
        st.caption(
            "Each bar is one shared cluster, split by whose passages fill it, "
            f"sorted by your share.{threshold_note}"
        )
