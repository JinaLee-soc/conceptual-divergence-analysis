"""Session state: blind-then-reveal enforcement, provenance flags, discovery."""

import os
from datetime import datetime

import pytest

from cda.validation.session import (
    create_session,
    endorse,
    find_latest_session,
    load_session,
    record_verdict,
    reveal_readings,
    save_session,
    session_lock,
)

NOW = datetime(2026, 7, 14, 10, 0, 0)


def make_session(tmp_path, term="novelty"):
    partitions = [
        {
            "revision": 0,
            "side": "literature",
            "kind": "original",
            "assignments": {"a::c0": 0, "b::c0": 0, "c::c0": 1, "d::c0": 1},
            "clusters": [
                {"cluster_id": 0, "n_passages": 2, "silhouette": 0.2, "shown": ["a::c0", "b::c0"]},
                {"cluster_id": 1, "n_passages": 2, "silhouette": 0.3, "shown": ["c::c0", "d::c0"]},
            ],
        }
    ]
    readings = [
        {
            "side": "literature",
            "cluster_id": 0,
            "revision": 0,
            "model": "m1",
            "gloss": "usage one",
            "judgment": "coherent",
            "rationale": "same usage",
            "valid": True,
        }
    ]
    return create_session(
        term,
        params={"seed": 42},
        sides={"literature": {"validated": True, "note": None}},
        partitions=partitions,
        passages={key: {"text": f"text {key}", "side": "literature"} for key in "abcd"},
        readings=readings,
        output_dir=tmp_path,
        now=NOW,
    )


def test_session_roundtrip_save_load_equal(tmp_path):
    session_dir, session = make_session(tmp_path)
    assert load_session(session_dir) == session
    record_verdict(
        session, side="literature", cluster_id=0, revision=0, verdict="coherent", now=NOW
    )
    save_session(session_dir, session)
    assert load_session(session_dir) == session


def test_reveal_refused_before_verdict(tmp_path):
    _dir, session = make_session(tmp_path)
    with pytest.raises(ValueError, match="commit a verdict before revealing"):
        reveal_readings(session, side="literature", cluster_id=0, revision=0, now=NOW)


def test_blind_and_revised_flags_are_derived(tmp_path):
    _dir, session = make_session(tmp_path)
    first = record_verdict(
        session, side="literature", cluster_id=0, revision=0, verdict="mixed", now=NOW
    )
    assert first["blind"] is True and first["revised_after_reveal"] is False

    readings = reveal_readings(session, side="literature", cluster_id=0, revision=0, now=NOW)
    assert len(readings) == 1 and readings[0]["judgment"] == "coherent"

    revised = record_verdict(
        session,
        side="literature",
        cluster_id=0,
        revision=0,
        verdict="coherent",
        notes="agent convinced me",
        now=NOW,
    )
    assert revised["blind"] is False and revised["revised_after_reveal"] is True
    # The blind original stays in the trail.
    assert [v["verdict"] for v in session["verdicts"]] == ["mixed", "coherent"]


def test_overlaps_verdict_requires_target(tmp_path):
    _dir, session = make_session(tmp_path)
    with pytest.raises(ValueError, match="overlaps_with"):
        record_verdict(
            session, side="literature", cluster_id=0, revision=0, verdict="overlaps", now=NOW
        )
    entry = record_verdict(
        session,
        side="literature",
        cluster_id=0,
        revision=0,
        verdict="overlaps",
        overlaps_with=1,
        now=NOW,
    )
    assert entry["overlaps_with"] == 1


def test_unknown_verdict_and_revision_rejected(tmp_path):
    _dir, session = make_session(tmp_path)
    with pytest.raises(ValueError, match="verdict must be one of"):
        record_verdict(
            session, side="literature", cluster_id=0, revision=0, verdict="fine", now=NOW
        )
    with pytest.raises(ValueError, match="no partition"):
        record_verdict(
            session, side="literature", cluster_id=0, revision=3, verdict="coherent", now=NOW
        )


def test_find_latest_session_exact_slug_and_timestamp_order(tmp_path):
    make_session(tmp_path, term="novelty")
    # A prefix-extending term must never be picked up for "novelty".
    later = datetime(2026, 7, 15, 9, 0, 0)
    other_dir = tmp_path / f"validation_novelty_claims_{later.strftime('%Y%m%d_%H%M%S')}"
    other_dir.mkdir()
    (other_dir / "session.json").write_text("{}", encoding="utf-8")

    found = find_latest_session(tmp_path, "novelty")
    assert found.name == "validation_novelty_20260714_100000"

    with pytest.raises(FileNotFoundError, match="cda validate --term"):
        find_latest_session(tmp_path, "consensus")


def test_same_second_sessions_get_distinct_dirs_and_the_finder_sees_them(tmp_path):
    """Two sessions in one second must neither merge nor hide from resume.

    create_session escapes a name collision with a -N suffix; the finder must
    recognize that name too, or "resume latest" would silently resume the
    older, unsuffixed session — stranding the newer verdict trail.
    """
    first_dir, _ = make_session(tmp_path)
    second_dir, _ = make_session(tmp_path)  # same term, same NOW → same name
    assert first_dir != second_dir
    assert second_dir.name == f"{first_dir.name}-1"
    assert find_latest_session(tmp_path, "novelty") == second_dir


@pytest.mark.parametrize(("wanted", "other"), [("C++", "C#"), ("social capital", "social-capital")])
def test_find_latest_session_checks_stored_term_for_a_shared_slug(tmp_path, wanted, other):
    first, _ = make_session(tmp_path, term=wanted)
    second, _ = make_session(tmp_path, term=other)
    assert second.name == first.name + "-1"
    assert find_latest_session(tmp_path, wanted) == first
    assert find_latest_session(tmp_path, other) == second


def test_find_latest_session_does_not_resume_only_a_colliding_term(tmp_path):
    make_session(tmp_path, term="C#")
    with pytest.raises(FileNotFoundError):
        find_latest_session(tmp_path, "C++")


def test_endorse_records_current_revisions(tmp_path):
    _dir, session = make_session(tmp_path)
    for cluster_id in (0, 1):
        record_verdict(
            session,
            side="literature",
            cluster_id=cluster_id,
            revision=0,
            verdict="coherent",
            now=NOW,
        )
    endorsement = endorse(session, note="looks right", now=NOW)
    assert endorsement["revisions"] == {"literature": 0}
    assert session["endorsement"]["note"] == "looks right"


def test_endorse_refuses_while_a_cluster_lacks_a_verdict(tmp_path):
    """An endorsement asserts every cluster was judged; the data layer is the
    enforcement, not the GUI's disabled button."""
    _dir, session = make_session(tmp_path)
    record_verdict(
        session,
        side="literature",
        cluster_id=0,
        revision=0,
        verdict="coherent",
        now=NOW,
    )
    with pytest.raises(ValueError, match="cannot endorse"):
        endorse(session, note="", now=NOW)


def test_record_verdict_refuses_a_ghost_cluster(tmp_path):
    """A verdict on a cluster id that does not exist would silently distort
    the agreement statistics."""
    _dir, session = make_session(tmp_path)
    with pytest.raises(ValueError, match="ghost cluster"):
        record_verdict(
            session,
            side="literature",
            cluster_id=99,
            revision=0,
            verdict="coherent",
            now=NOW,
        )
    with pytest.raises(ValueError, match="different existing cluster"):
        record_verdict(
            session,
            side="literature",
            cluster_id=0,
            revision=0,
            verdict="overlaps",
            overlaps_with=0,
            now=NOW,
        )


def test_load_session_names_a_corrupt_file(tmp_path):
    session_dir, _session = make_session(tmp_path)
    (session_dir / "session.json").write_text("{truncated", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_session(session_dir)


def test_report_withholds_readings_until_a_blind_verdict_exists(tmp_path):
    """Blind judgment must hold in the report, not only in the write path.

    ``run_validation`` writes the report when the session opens — before any
    verdict exists — and the CLI tells the researcher to keep it open while
    they work. A reading printed next to an empty verdict would show the
    model's answer to the one person whose independent judgment the agreement
    statistic is measuring.
    """
    from cda.config import load_config
    from cda.validation.report import render_validation_report

    _dir, session = make_session(tmp_path)
    config = load_config()

    text = render_validation_report(session, config, NOW)
    assert "withheld until you record a verdict" in text

    record_verdict(
        session,
        side="literature",
        cluster_id=0,
        verdict="coherent",
        revision=0,
        now=NOW,
    )
    text = render_validation_report(session, config, NOW)
    assert "withheld until you record a verdict" not in text
    assert "coherent" in text


def test_session_lock_creates_and_removes_the_lockfile(tmp_path):
    """The lock serializes load→mutate→save across tabs and processes; a
    normal exit — or an exception inside the block — must always release it,
    or every later save of the session would time out."""
    session_dir, session = make_session(tmp_path)
    lock_path = session_dir / "session.lock"
    with session_lock(session_dir):
        assert lock_path.is_file()
        # A full cycle works under the lock.
        session = load_session(session_dir)
        record_verdict(
            session, side="literature", cluster_id=0, revision=0, verdict="coherent", now=NOW
        )
        save_session(session_dir, session)
    assert not lock_path.exists()

    with pytest.raises(RuntimeError, match="sabotage"):
        with session_lock(session_dir):
            raise RuntimeError("sabotage")
    assert not lock_path.exists()


def test_session_lock_blocks_a_second_holder_until_released(tmp_path, monkeypatch):
    """A second acquisition while the lock is held waits, then times out with
    a message naming the lockfile and saying when deleting it is safe."""
    import cda.validation.session as session_mod

    session_dir, _session = make_session(tmp_path)
    monkeypatch.setattr(session_mod, "_LOCK_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(session_mod, "_LOCK_RETRY_SECONDS", 0.05)
    with session_lock(session_dir):
        with pytest.raises(TimeoutError) as excinfo:
            with session_lock(session_dir):
                pass  # pragma: no cover - acquisition must fail
    message = str(excinfo.value)
    assert "session.lock" in message
    assert "delete" in message
    # Released by the outer holder: acquiring again now succeeds immediately.
    with session_lock(session_dir):
        pass
    assert not (session_dir / "session.lock").exists()


def test_session_json_is_written_atomically(tmp_path, monkeypatch):
    """session.json IS the audit trail; a truncated write would erase it.

    A direct write_text truncates before writing, so a crash mid-write leaves
    a valid-looking but incomplete record. Writing beside the target and
    renaming means a reader sees the old record or the new one, never a
    partial one.
    """
    session_dir, session = make_session(tmp_path)
    original = load_session(session_dir)

    real_replace = os.replace

    def explode(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", explode)
    session["verdicts"].append({"sabotage": True})
    with pytest.raises(OSError):
        save_session(session_dir, session)

    # The record on disk is still the previous, complete one, and the failed
    # write's temp file was cleaned up rather than left beside it.
    monkeypatch.setattr(os, "replace", real_replace)
    assert load_session(session_dir) == original
    assert not list(session_dir.glob(".session.json.*"))
