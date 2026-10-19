"""The acquisition-date window: bound parsing, filtering, and pipeline wiring."""

import re

import pytest

from cda.config import load_config
from cda.corpus.timeframe import (
    describe_timeframe,
    describe_window,
    filter_by_date_added,
    parse_bound,
    timeframe_from_config,
)
from tests.conftest import make_passages


def _dated(pairs):
    """Passages carrying a ``date_added`` (pairs of (text, date))."""
    passages = make_passages([text for text, _ in pairs])
    for passage, (_text, date) in zip(passages, pairs, strict=True):
        if date is not None:
            passage.metadata = {"date_added": date}
    return passages


def test_parse_bound_pads_the_upper_bound_to_end_of_day():
    # "as of 1 May" must include everything added during 1 May.
    assert parse_bound("2026-05-01", end_of_day=True) == "2026-05-01 23:59:59"
    assert parse_bound("2026-05-01", end_of_day=False) == "2026-05-01 00:00:00"


def test_parse_bound_keeps_an_explicit_time():
    # An upper bound to the minute covers that whole minute.
    assert parse_bound("2026-05-01 14:30", end_of_day=True) == "2026-05-01 14:30:59"
    assert parse_bound("2026-05-01 14:30", end_of_day=False) == "2026-05-01 14:30:00"
    assert parse_bound("2026-05-01T14:30:15", end_of_day=True) == "2026-05-01 14:30:15"


def test_parse_bound_rejects_unreadable_dates():
    """A typo must fail loudly: a silently dropped filter would report a
    snapshot that was never taken."""
    for bad in ("last tuesday", "05/01/2026", "", "2026-13-45x"):
        with pytest.raises(ValueError, match="Could not read"):
            parse_bound(bad, end_of_day=False)


def test_window_inactive_by_default_and_passes_everything_through():
    config = load_config()
    window = timeframe_from_config(config)
    assert not window["active"]
    passages = _dated([("a", "2020-01-01 00:00:00"), ("b", None)])
    kept, counts = filter_by_date_added(passages, window)
    assert kept == passages
    assert counts["n_kept"] == counts["n_before"] == 2


def test_window_from_config_normalizes_both_bounds():
    config = load_config(
        overrides={"timeframe": {"added_from": "2019-01-01", "added_to": "2026-05-01"}}
    )
    window = timeframe_from_config(config)
    assert window["active"]
    assert window["added_from"] == "2019-01-01 00:00:00"
    assert window["added_to"] == "2026-05-01 23:59:59"


def test_inverted_bounds_are_an_error():
    config = load_config(
        overrides={"timeframe": {"added_from": "2026-05-01", "added_to": "2019-01-01"}}
    )
    with pytest.raises(ValueError, match="later than"):
        timeframe_from_config(config)


def test_upper_bound_keeps_the_whole_final_day():
    window = timeframe_from_config(load_config(overrides={"timeframe": {"added_to": "2026-05-01"}}))
    passages = _dated(
        [
            ("before", "2026-04-30 12:00:00"),
            ("on the cutoff, late", "2026-05-01 23:00:00"),
            ("after", "2026-05-02 00:30:00"),
        ]
    )
    kept, counts = filter_by_date_added(passages, window)
    assert [p.text for p in kept] == ["before", "on the cutoff, late"]
    assert counts["n_outside_window"] == 1


def test_both_bounds_isolate_one_period():
    window = timeframe_from_config(
        load_config(overrides={"timeframe": {"added_from": "2020-01-01", "added_to": "2020-12-31"}})
    )
    passages = _dated(
        [
            ("too early", "2019-12-31 23:00:00"),
            ("inside", "2020-06-15 09:00:00"),
            ("too late", "2021-01-01 00:00:00"),
        ]
    )
    kept, _counts = filter_by_date_added(passages, window)
    assert [p.text for p in kept] == ["inside"]


def test_undated_passages_are_dropped_and_counted():
    window = timeframe_from_config(load_config(overrides={"timeframe": {"added_to": "2026-05-01"}}))
    passages = _dated([("dated", "2020-01-01 00:00:00"), ("undated", None), ("blank", "")])
    kept, counts = filter_by_date_added(passages, window)
    assert [p.text for p in kept] == ["dated"]
    assert counts["n_undated"] == 2 and counts["n_undated_dropped"] == 2


def test_include_undated_keeps_them():
    window = timeframe_from_config(
        load_config(overrides={"timeframe": {"added_to": "2026-05-01", "include_undated": True}})
    )
    passages = _dated([("dated", "2020-01-01 00:00:00"), ("undated", None)])
    kept, counts = filter_by_date_added(passages, window)
    assert len(kept) == 2
    assert counts["n_undated"] == 1 and counts["n_undated_dropped"] == 0


def test_drafts_are_never_filtered():
    """Drafts are the manuscript under examination, not works acquired in a
    period — a snapshot of your reading is still a snapshot of YOUR reading."""
    window = timeframe_from_config(load_config(overrides={"timeframe": {"added_to": "2019-01-01"}}))
    draft, corpus = make_passages(["my draft sentence", "a corpus sentence"])
    draft.metadata = {"draft": True}
    corpus.metadata = {"date_added": "2025-06-01 00:00:00"}
    kept, counts = filter_by_date_added([draft, corpus], window)
    assert [p.text for p in kept] == ["my draft sentence"]
    assert counts["n_drafts_exempt"] == 1


def test_describe_window_reads_as_a_snapshot():
    window = timeframe_from_config(load_config(overrides={"timeframe": {"added_to": "2026-05-01"}}))
    text = describe_window(window)
    assert "on or before 2026-05-01" in text and "snapshot" in text
    assert "undated passages excluded" in text


def test_describe_timeframe_reports_counts_and_skips_empty_pools():
    window = timeframe_from_config(load_config(overrides={"timeframe": {"added_to": "2026-05-01"}}))
    _kept, literature_counts = filter_by_date_added(
        _dated([("in", "2020-01-01 00:00:00"), ("out", "2027-01-01 00:00:00")]), window
    )
    _empty, researcher_counts = filter_by_date_added([], window)
    text = describe_timeframe(
        {"window": window, "researcher": researcher_counts, "literature": literature_counts}
    )
    assert "literature: 1 of 2 passages kept" in text
    assert "researcher:" not in text  # nothing to report for an empty pool


def test_describe_timeframe_without_a_window():
    assert "whole corpus" in describe_timeframe({})


# --- pipeline wiring --------------------------------------------------------


def _records(dates):
    return [
        {
            "source": f"Author{i}_2020_Study_{i}.pdf",
            "abstract": (
                f"This study examines novelty in evaluation, case {i}. "
                f"Novelty recurs across the literature in variant {i}."
            ),
            "title": f"Study {i}",
            "year": "2020",
            "date_added": date,
        }
        for i, date in enumerate(dates)
    ]


def _config(tmp_path, **timeframe):
    return load_config(
        overrides={
            "paths": {
                "corpus_dir": str(tmp_path / "corpus"),
                "drafts_dir": str(tmp_path / "drafts"),
                "output_dir": str(tmp_path / "outputs"),
            },
            "timeframe": timeframe,
        }
    )


def test_extract_passages_applies_the_window_and_records_stats(tmp_path, fake_embedder):
    from cda.corpus.ingest import get_collection, ingest_records
    from cda.pipeline import extract_passages

    dates = ["2019-05-01 09:00:00"] * 4 + ["2026-06-01 09:00:00"] * 4
    config = _config(tmp_path)
    ingest_records(config, fake_embedder, _records(dates), progress=lambda _m: None)
    collection = get_collection(config)
    kwargs = {"embedder": fake_embedder, "collection": collection, "progress": lambda _m: None}

    _r_all, lit_all, stats_all = extract_passages("novelty", config, **kwargs)
    assert "timeframe" not in stats_all

    snapshot = _config(tmp_path, added_to="2026-05-01")
    _r, lit_window, stats = extract_passages("novelty", snapshot, **kwargs)
    assert 0 < len(lit_window) < len(lit_all)
    assert stats["timeframe"]["literature"]["n_kept"] == len(lit_window)
    assert stats["literature_passages"] == len(lit_window)
    assert all(p.metadata["date_added"] < "2026-05-01" for p in lit_window)


def test_window_that_empties_the_literature_says_which_case_it_is(tmp_path, fake_embedder):
    """An empty pool would render a report full of "clustering unavailable"
    notes; the window that caused it must be named instead."""
    from cda.corpus.ingest import get_collection, ingest_records
    from cda.pipeline import extract_passages

    config = _config(tmp_path)
    ingest_records(
        config, fake_embedder, _records(["2026-06-01 09:00:00"] * 4), progress=lambda _m: None
    )
    collection = get_collection(config)

    # Everything is dated, but outside the window.
    with pytest.raises(ValueError, match="excludes all"):
        extract_passages(
            "novelty",
            _config(tmp_path, added_to="2019-01-01"),
            embedder=fake_embedder,
            collection=collection,
            progress=lambda _m: None,
        )


def test_window_on_an_undated_corpus_points_at_reingest(tmp_path, fake_embedder):
    """The common trap: a corpus ingested before "Date Added" was recorded."""
    from cda.corpus.ingest import get_collection, ingest_records
    from cda.pipeline import extract_passages

    config = _config(tmp_path)
    records = [dict(record) for record in _records([""] * 4)]
    for record in records:
        record.pop("date_added")
    ingest_records(config, fake_embedder, records, progress=lambda _m: None)

    with pytest.raises(ValueError, match="--reingest"):
        extract_passages(
            "novelty",
            _config(tmp_path, added_to="2026-05-01"),
            embedder=fake_embedder,
            collection=get_collection(config),
            progress=lambda _m: None,
        )


# --- acquisition clocks -----------------------------------------------------


def test_date_source_selects_which_clock_the_window_reads():
    """The two clocks can disagree by years, so a run reads exactly one."""
    passages = make_passages(["a", "b"])
    # Added to Zotero this year, but the file has been on disk since 2019.
    passages[0].metadata = {
        "date_added": "2026-07-01 09:00:00",
        "date_added_fs": "2019-03-04 09:00:00",
    }
    # Both clocks agree here.
    passages[1].metadata = {
        "date_added": "2019-05-01 09:00:00",
        "date_added_fs": "2019-05-02 09:00:00",
    }

    zotero = timeframe_from_config(load_config(overrides={"timeframe": {"added_to": "2026-05-01"}}))
    kept, _counts = filter_by_date_added(passages, zotero)
    assert [p.text for p in kept] == ["b"]  # 'a' entered Zotero after the cutoff

    filesystem = timeframe_from_config(
        load_config(
            overrides={"timeframe": {"added_to": "2026-05-01", "date_source": "filesystem"}}
        )
    )
    kept, _counts = filter_by_date_added(passages, filesystem)
    assert [p.text for p in kept] == ["a", "b"]  # both files predate the cutoff


def test_default_clock_is_zotero():
    window = timeframe_from_config(load_config())
    assert window["date_source"] == "zotero"


def test_unknown_date_source_explains_both_options():
    with pytest.raises(ValueError, match="timeframe.date_source must be one of"):
        timeframe_from_config(load_config(overrides={"timeframe": {"date_source": "mtime"}}))


def test_describe_window_names_the_clock():
    """Two runs differing only in clock must not be confusable in a report."""
    zotero = timeframe_from_config(load_config(overrides={"timeframe": {"added_to": "2026-05-01"}}))
    assert 'Zotero "Date Added"' in describe_window(zotero)
    filesystem = timeframe_from_config(
        load_config(
            overrides={"timeframe": {"added_to": "2026-05-01", "date_source": "filesystem"}}
        )
    )
    assert "file arrival on disk" in describe_window(filesystem)


def test_ingest_records_the_filesystem_clock(tmp_path, fake_embedder):
    """Every ingested PDF carries date_added_fs, whether or not Zotero matched."""
    from cda.corpus.ingest import file_acquisition_date

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal")
    stamp = file_acquisition_date(pdf)
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", stamp)
