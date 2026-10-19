"""Cluster emergence over acquisition time: ordering, coverage, births, matching."""

import random
from datetime import datetime, timedelta

import pytest

from cda.clustering.emergence import passage_date_added, track_emergence
from tests.conftest import make_passages

PARAMS = dict(
    start=8,
    step=4,
    match_threshold=0.8,
    max_k=4,
    n_restarts=3,
    max_iter=50,
    min_cluster_size=2,
    silhouette_subsample=200,
    seed=42,
)


def date_weekly(passages, first="2019-01-07 09:00:00"):
    """One passage a week, in list order."""
    base = datetime.strptime(first, "%Y-%m-%d %H:%M:%S")
    for i, passage in enumerate(passages):
        passage.metadata["date_added"] = (base + timedelta(days=7 * i)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )


def track(passages, embedder, **overrides):
    params = dict(PARAMS)
    params.update(overrides)
    return track_emergence(passages, embedder, **params)


def test_passage_date_added_normalizes_and_rejects():
    passages = make_passages(["a", "b", "c"])
    passages[0].metadata["date_added"] = "2021-03-14T09:22:11"  # raw ISO 'T' form
    passages[1].metadata["date_added"] = "not a date"
    assert passage_date_added(passages[0]) == "2021-03-14 09:22:11"
    assert passage_date_added(passages[1]) is None
    assert passage_date_added(passages[2]) is None  # no key at all


def test_input_order_never_matters(anchored_embedder):
    """(date_added, chunk_id) is a total order: shuffled input, identical result."""
    passages = make_passages(
        [f"alpha usage of the term, case {i}." for i in range(8)]
        + [f"beta reading of the term, case {i}." for i in range(8)]
    )
    date_weekly(passages)
    # Two passages share one timestamp — only chunk_id breaks the tie.
    passages[3].metadata["date_added"] = passages[2].metadata["date_added"]

    reference = track(list(passages), anchored_embedder)
    shuffled = list(passages)
    random.Random(7).shuffle(shuffled)
    assert track(shuffled, anchored_embedder) == reference
    # First snapshot's as-of date is the 8th-earliest acquisition, not an
    # artifact of input order.
    assert reference["snapshots"][0]["as_of"] == passages[7].metadata["date_added"][:10]


def test_coverage_reported_and_undated_excluded(anchored_embedder):
    dated = make_passages([f"alpha topic sentence {i}" for i in range(10)])
    date_weekly(dated)
    undated = make_passages([f"beta topic sentence {i}" for i in range(3)], source="nodate.pdf")
    result = track(dated + undated, anchored_embedder)

    assert result["coverage"] == {
        "n_literature": 13,
        "n_dated": 10,
        "n_undated": 3,
        "rate": round(10 / 13, 4),
    }
    assert all(record["n"] <= 10 for record in result["snapshots"])
    assert result["snapshots"][-1]["n"] == 10


def test_skipped_when_fewer_dated_than_start(fake_embedder):
    passages = make_passages([f"term sentence {i}" for i in range(5)])
    date_weekly(passages)
    result = track(passages, fake_embedder)
    assert result["status"] == "skipped"
    assert "emergence.start" in result["reason"]
    assert result["final"] is None


def test_birth_detected_for_late_arriving_cluster(anchored_embedder):
    """A theme acquired later is born later, with zero size before birth."""
    early = make_passages([f"alpha reading of the term, case {i}." for i in range(12)])
    late = make_passages(
        [f"beta reading of the term, case {i}." for i in range(12)], source="later.pdf"
    )
    date_weekly(early, first="2019-01-07 09:00:00")
    date_weekly(late, first="2021-01-04 09:00:00")
    result = track(early + late, anchored_embedder)

    assert result["status"] == "ok"
    assert result["final"]["k"] == 2
    by_text = {
        ("beta" if any("beta" in text for text in cluster["representatives"]) else "alpha"): cluster
        for cluster in result["final"]["clusters"]
    }
    assert set(by_text) == {"alpha", "beta"}
    # Alpha is fully present in the first snapshot (8 ≤ 12 alpha passages);
    # beta passages only start at position 13, so its cluster is born later.
    assert by_text["alpha"]["birth_n"] == 8
    assert by_text["beta"]["birth_n"] > 12
    for point in by_text["beta"]["growth"]:
        if point["n"] < by_text["beta"]["birth_n"]:
            assert point["size"] == 0


def test_matching_is_one_to_one_and_thresholded(anchored_embedder):
    passages = make_passages(
        [f"alpha case {i}" for i in range(10)] + [f"beta case {i}" for i in range(10)]
    )
    date_weekly(passages)

    result = track(passages, anchored_embedder)
    for record in result["snapshots"][:-1]:
        if record.get("status") != "ok":
            continue
        matched_finals = [m["final_cluster"] for m in record["matches"]]
        matched_snaps = [m["snapshot_cluster"] for m in record["matches"]]
        assert len(matched_finals) == len(set(matched_finals))
        assert len(matched_snaps) == len(set(matched_snaps))

    # An impossible threshold (> 1.0) kills every match: every final cluster
    # is only "born" at the final snapshot itself.
    unmatched = track(passages, anchored_embedder, match_threshold=1.01)
    final_n = unmatched["final"]["n"]
    for cluster in unmatched["final"]["clusters"]:
        assert cluster["birth_n"] == final_n
        assert cluster["persistence"] is None
        assert cluster["mean_match_cosine"] is None


def test_final_snapshot_covers_all_dated_passages(anchored_embedder):
    passages = make_passages(
        [f"alpha case {i}" for i in range(9)] + [f"beta case {i}" for i in range(9)]
    )
    date_weekly(passages)
    result = track(passages, anchored_embedder)
    final_record = result["snapshots"][-1]
    assert final_record["n"] == 18
    assert result["final"]["n"] == 18
    total = sum(cluster["n_passages"] for cluster in result["final"]["clusters"])
    assert total == final_record["n_with_embeddings"]


def test_track_emergence_is_deterministic(anchored_embedder):
    passages = make_passages(
        [f"alpha case {i}" for i in range(8)] + [f"beta case {i}" for i in range(8)]
    )
    date_weekly(passages)
    first = track(passages, anchored_embedder)
    # embeddings are hydrated in place; rebuild to prove a cold run agrees
    rebuilt = make_passages(
        [f"alpha case {i}" for i in range(8)] + [f"beta case {i}" for i in range(8)]
    )
    date_weekly(rebuilt)
    assert track(rebuilt, anchored_embedder) == first


def test_ari_chain_resets_on_no_partition(fake_embedder):
    """A snapshot with no valid partition resets the ARI comparison chain."""
    # Eight byte-identical texts embed to one point → no k gives 2 clusters →
    # the first snapshot has no partition; later snapshots add real structure.
    degenerate = make_passages(["identical sentence"] * 8)
    varied = make_passages([f"varied sentence number {i}" for i in range(8)], source="v.pdf")
    passages = degenerate + varied
    date_weekly(passages)
    result = track(passages, fake_embedder)

    assert result["snapshots"][0]["status"] == "no_partition"
    following = result["snapshots"][1]
    # Asserted, not guarded on: if this snapshot ever stops being "ok" the
    # test would otherwise pass vacuously, checking nothing about the reset.
    assert following.get("status") == "ok"
    assert following["ari_vs_prev"] is None


def test_track_emergence_rejects_meaningless_start_and_step(fake_embedder):
    """start=0 used to negative-index the newest date into the first row."""
    passages = make_passages([f"sentence {i}" for i in range(6)])
    date_weekly(passages)
    with pytest.raises(ValueError, match="emergence.start"):
        track(passages, fake_embedder, start=0)
    with pytest.raises(ValueError, match="emergence.step"):
        track(passages, fake_embedder, step=0)


def test_track_emergence_reads_the_requested_date_key(fake_embedder):
    """The timeline must use the same clock the window filter used.

    Passages dated only under the filesystem key have no Zotero timeline at
    all — and a filesystem-keyed run must order by the filesystem dates.
    """
    from cda.clustering.emergence import track_emergence

    passages = make_passages([f"sentence {i}" for i in range(6)])
    for i, passage in enumerate(passages):
        passage.metadata = {"date_added_fs": f"2021-01-{i + 1:02d} 09:00:00"}

    kwargs = dict(
        start=3,
        step=2,
        match_threshold=0.8,
        max_k=4,
        n_restarts=2,
        max_iter=20,
        min_cluster_size=2,
        silhouette_subsample=100,
        seed=42,
    )
    zotero_clock = track_emergence(passages, fake_embedder, **kwargs)
    assert zotero_clock["status"] == "skipped"  # no 'date_added' anywhere
    assert zotero_clock["coverage"]["n_dated"] == 0

    fs_clock = track_emergence(passages, fake_embedder, date_key="date_added_fs", **kwargs)
    assert fs_clock["coverage"]["n_dated"] == 6
    assert fs_clock["params"]["date_key"] == "date_added_fs"
    assert fs_clock["snapshots"][0]["as_of"] == "2021-01-03"
