"""Passage map: the joint partition drawn in the pooled embedding space.

Every clustered passage is projected into one PCA plane fitted on both corpora
together. Literature passages are dots, the researcher's are stars. Under joint
clustering both take their colour from the shared cluster they landed in — so
where the researcher's passages sit *relative to the literature's mass* is
readable directly, which is the question the divergence flags answer
numerically. Under independent clustering the two partitions are unrelated, so
each corpus gets its own palette and its own labels (R0.., L0..).

Why a figure and not only the tables: a cluster that the flags call
"researcher-concentrated" can mean two very different things — the researcher
occupies a region the literature genuinely does not, or the literature simply
was not sampled there. Those look identical in a table and obvious in this
plot.

The plot is deterministic and LLM-free: it reads the embeddings and cluster ids
the clustering stage already produced, adds no randomness, and never changes
the analysis. `--dry-run` skips it along with the rest of stage 3.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA

from cda.outputs import create_temp, install_new

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cda.types import JointClustering, Passage

# Colour-blind-safe qualitative palette (Okabe-Ito, extended by repetition of
# shape rather than by inventing low-contrast hues). Joint mode only.
PALETTE = [
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#e69f00",
    "#56b4e9",
    "#8c564b",
    "#7f7f7f",
    "#117733",
    "#882255",
    "#44aa99",
    "#999933",
]
UNASSIGNED_COLOUR = "#bbbbbb"
_UNASSIGNED = "Cluster N/A"
_CLUSTER_ID = re.compile(r"Cluster (\d+)")

_MISSING_MATPLOTLIB = (
    "Drawing the passage map needs matplotlib, which is not installed.\n"
    "  install it:  pip install matplotlib\n"
    "  or rerun without --figure — the report and its tables do not need it."
)


def _cluster_index(passage: Passage) -> int | None:
    """'Cluster 3' -> 3; the unassigned sentinel 'Cluster N/A' -> None."""
    if passage.cluster_id == _UNASSIGNED:
        return None
    match = _CLUSTER_ID.fullmatch(str(passage.cluster_id))
    if match is None:
        raise ValueError(
            f"Passage {passage.chunk_id} has an unrecognized cluster id {passage.cluster_id!r}."
        )
    return int(match.group(1))


def _independent_palettes() -> tuple[list[str], list[str]]:
    """(literature, researcher) palettes for independent mode, from matplotlib's maps.

    The partitions are unrelated, so the corpora must never share a colour.
    Literature takes tab20's 18 non-grey entries (grey marks unassigned
    passages), its darker shade of every hue first, which covers
    literature_max_k (15). Researcher stars take Brewer's Dark2, whose 8 colours
    cover researcher_max_k (8) and stay far from tab20's hues as the
    literature dots are drawn (40% opacity).
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    tab20 = [to_hex(colour) for colour in colormaps["tab20"].colors]
    hues = [tab20[i : i + 2] for i in range(0, len(tab20), 2) if tab20[i] != "#7f7f7f"]
    literature = [dark for dark, _ in hues] + [light for _, light in hues]
    researcher = [to_hex(colour) for colour in colormaps["Dark2"].colors]
    return literature, researcher


def _colours(passages: list[Passage], palette: list[str]) -> list[str]:
    indices = [_cluster_index(p) for p in passages]
    return [UNASSIGNED_COLOUR if i is None else palette[i % len(palette)] for i in indices]


def _cluster_label_positions(passages: list[Passage], project) -> dict[int, tuple[float, float]]:
    """Projected centroid of each assigned cluster among ``passages``."""
    by_cluster: dict[int, list] = {}
    for passage in passages:
        index = _cluster_index(passage)
        if index is not None:
            by_cluster.setdefault(index, []).append(passage.embedding)
    return {
        cluster_id: tuple(project(embeddings).mean(axis=0))
        for cluster_id, embeddings in sorted(by_cluster.items())
    }


def _cluster_labels(
    researcher, literature, project, *, joint_mode: bool
) -> list[tuple[str, tuple]]:
    """(text, position) per cluster label.

    Joint mode: one 'C<id> (nR)' label per shared cluster at the pooled
    centroid. Independent mode: 'R<id>' and 'L<id>' labels per partition.
    """
    if not joint_mode:
        return [
            (f"R{cid}", xy) for cid, xy in _cluster_label_positions(researcher, project).items()
        ] + [(f"L{cid}", xy) for cid, xy in _cluster_label_positions(literature, project).items()]
    researcher_counts: dict[int, int] = {}
    for passage in researcher:
        index = _cluster_index(passage)
        if index is not None:
            researcher_counts[index] = researcher_counts.get(index, 0) + 1
    labels = []
    for cluster_id, xy in _cluster_label_positions(researcher + literature, project).items():
        n_researcher = researcher_counts.get(cluster_id, 0)
        labels.append((f"C{cluster_id}" + (f" ({n_researcher}R)" if n_researcher else ""), xy))
    return labels


def _pca_plane(matrix: np.ndarray):
    """Fit a 2D PCA plane and return (project, explained variance share).

    Uses the same scikit-learn PCA as the GUI's interactive map, so the
    exported figure and the on-screen chart show the same plane.
    """
    # scikit-learn divides by the total variance, which is zero (ratio nan)
    # when every embedding is identical; the plane then explains nothing.
    with np.errstate(divide="ignore", invalid="ignore"):
        pca = PCA(n_components=2, svd_solver="full").fit(np.asarray(matrix, dtype=float))

    def project(rows) -> np.ndarray:
        return pca.transform(np.asarray(rows, dtype=float))

    share = float(np.sum(pca.explained_variance_ratio_))
    return project, share if np.isfinite(share) else 0.0


def _embedded(passages: list[Passage]) -> list[Passage]:
    return [p for p in passages if p.embedding is not None]


def draw_passage_map(
    *,
    term: str,
    researcher_passages: list[Passage],
    literature_passages: list[Passage],
    joint: JointClustering | None,
    path: Path,
    dpi: int = 300,
    subtitle: str = "",
):
    """Write the passage map to ``path``. Returns the path, or None if there
    is nothing to draw (too few embedded passages)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # never needs a display
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised by the CLI path
        raise RuntimeError(_MISSING_MATPLOTLIB) from exc

    researcher = _embedded(researcher_passages)
    literature = _embedded(literature_passages)
    if len(researcher) + len(literature) < 3:
        return None

    pooled = researcher + literature
    project, explained = _pca_plane(np.array([p.embedding for p in pooled], dtype=float))

    joint_mode = joint is not None and joint.used
    if joint_mode:
        literature_palette = researcher_palette = PALETTE
    else:
        literature_palette, researcher_palette = _independent_palettes()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    if literature:
        xy = project([p.embedding for p in literature])
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=7,
            c=_colours(literature, literature_palette),
            alpha=0.40,
            linewidths=0.0,
            zorder=2,
        )
    if researcher:
        xy = project([p.embedding for p in researcher])
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            marker="*",
            s=150,
            c=_colours(researcher, researcher_palette),
            alpha=1.0,
            linewidths=0.5,
            edgecolors="#1a1a1a",
            zorder=4,
        )

    for label, position in _cluster_labels(researcher, literature, project, joint_mode=joint_mode):
        ax.annotate(
            label,
            position,
            fontsize=7,
            ha="center",
            va="center",
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": "#999999",
                "lw": 0.4,
                "alpha": 0.9,
            },
        )

    mode = "joint partition" if joint_mode else "per-corpus partitions"
    head = f'"{term}" — {len(researcher):,} researcher and {len(literature):,} literature passages'
    ax.set_title(f"{head}\n{subtitle or mode}", fontsize=9, loc="left", pad=7, linespacing=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    legend = (
        "colour = shared cluster; (nR) counts researcher passages."
        if joint_mode
        else "colours and labels are per corpus (R = researcher, L = literature partition; "
        "ids are not comparable across corpora)."
    )
    fig.text(
        0.01,
        0.015,
        f"● literature passage   ★ researcher passage   {legend}  PCA plane on all "
        f"{len(pooled):,} passages, {explained:.1%} of variance.",
        fontsize=6.4,
        color="#555555",
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    # Render beside the target and link it into place: a crash mid-savefig must
    # not leave a corrupt PNG under the final name, and an existing figure is
    # never replaced.
    handle, temporary = create_temp(path.parent, path.name)
    os.close(handle)
    try:
        fig.savefig(
            temporary,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            format=(path.suffix.lstrip(".") or "png"),
        )
        install_new(temporary, path)
    finally:
        plt.close(fig)
        os.unlink(temporary)
    return path
