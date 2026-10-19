"""Validation session state: one directory per session, one JSON of record.

A session captures everything a validation run produces or the researcher
decides: the partitions (revision 0 = the pipeline's original clustering;
later revisions from applied edits), the agent cluster readings, the
researcher's verdicts (with their blind/revised provenance), the curation
audit trail, and the endorsement.

The blind-then-reveal protocol is enforced HERE, not only in the GUI:
``reveal_readings`` refuses to reveal a cluster's readings until a verdict
for it exists, and ``record_verdict`` derives the ``blind`` flag from the
recorded reveal events. A UI cannot accidentally unblind the evidence.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from cda.extraction.windows import safe_term_slug, term_identity
from cda.outputs import write_text_atomic

SCHEMA = "cda-validation-session/1"
VERDICTS = ("coherent", "mixed", "overlaps", "artifact")

# How long session_lock waits for another writer before giving up, and how
# often it retries. A session save is a JSON dump of a few hundred KB at most,
# so a lock held for ten seconds means a crashed holder, not a slow one.
_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_RETRY_SECONDS = 0.1

# The optional -N suffix is the same-second collision escape create_session
# takes; the finder must recognize every name create_session can produce, or
# "resume latest" would silently resume an older session.
_SESSION_DIR = re.compile(r"^validation_(?P<slug>.+)_(?P<ts>\d{8}_\d{6})(?:-(?P<n>\d+))?$")


def create_session(
    term: str,
    *,
    params: dict,
    sides: dict,
    partitions: list[dict],
    passages: dict,
    readings: list[dict],
    output_dir: Path,
    now: datetime,
) -> tuple[Path, dict]:
    """Create the session directory and its initial state."""
    slug = safe_term_slug(term)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # The timestamp resolves to the second, so two sessions on the same term
    # started in the same second would land on one directory and overwrite
    # each other's verdicts. Claim the name exclusively and add a suffix if it
    # is taken: a session record must never be silently merged with another.
    base = Path(output_dir) / f"validation_{slug}_{timestamp}"
    for suffix in range(100):
        session_dir = base if suffix == 0 else Path(f"{base}-{suffix}")
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f"Could not find a free session directory beside {base}")
    session = {
        "schema": SCHEMA,
        "term": term,
        "created": now.isoformat(timespec="seconds"),
        "params": params,
        "sides": sides,
        "partitions": partitions,
        "passages": passages,
        "readings": readings,
        "verdicts": [],
        "reveals": [],
        "edits": [],
        "pending_edit": None,
        "endorsement": None,
    }
    save_session(session_dir, session)
    return session_dir, session


@contextmanager
def session_lock(session_dir: Path):
    """Serializes load→mutate→save on one session across tabs and processes.

    The atomic writer below guarantees no reader ever sees a *partial*
    session.json, but it cannot stop a lost update: two GUI tabs (or a tab
    and a CLI process) each load the record, each mutate their own copy, and
    the second save silently erases the first tab's blind verdicts from the
    audit trail. Holding this lock across the whole load→mutate→save cycle
    closes that window.

    Implementation: an O_EXCL lockfile (``session.lock``) beside
    session.json. O_CREAT|O_EXCL is atomic on every platform Python runs on
    and works across threads and processes alike, which is why it is used
    instead of fcntl (absent on Windows). NOT reentrant: acquire it once, at
    the outermost site that performs the full cycle — a nested acquisition
    would wait on its own lock until the timeout.

    On timeout the error names the lockfile and says when deleting it is
    safe, because the one way a lock outlives its holder is a crash mid-cycle
    (the ``finally`` below cannot run if the process is killed).
    """
    lock_path = Path(session_dir) / "session.lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Another save of this validation session appears to be in "
                    f"progress: {lock_path} still exists after "
                    f"{_LOCK_TIMEOUT_SECONDS:.0f}s. If another tab or process "
                    "is working on this session, let it finish and retry. If "
                    "none is (for example after a crash), the lock is stale: "
                    "delete that file and retry — session.json itself is "
                    "written atomically and is not damaged."
                ) from None
            time.sleep(_LOCK_RETRY_SECONDS)
    try:
        try:
            # Who holds the lock, for a human deciding whether it is stale.
            os.write(fd, f"pid {os.getpid()}\n".encode())
        finally:
            os.close(fd)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def save_session(session_dir: Path, session: dict) -> None:
    """Write the session record atomically.

    This file *is* the audit trail — the blind verdicts, the readings, and the
    order they were recorded in. A direct ``write_text`` truncates first, so a
    crash or a full disk mid-write leaves a valid-looking but truncated file
    and the evidence is gone. Writing beside it and renaming means a reader
    sees either the old record or the new one, never a partial one.
    """
    target = Path(session_dir) / "session.json"
    payload = json.dumps(session, indent=2, ensure_ascii=False, default=str)
    # The shared helper, not a local copy: it writes beside the target under a
    # unique mkstemp name (the GUI serves browser sessions as threads of one
    # process, so a shared per-pid name would let two threads interleave
    # writes and install a truncated record), and improvements to it — file
    # permissions, fsync before the rename — reach this writer too.
    write_text_atomic(target, payload)


def load_session(session_dir: Path) -> dict:
    path = Path(session_dir) / "session.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No validation session at {session_dir} (missing session.json). "
            "Start one with: cda validate --term <term>"
        )
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        # A record cut inside a multi-byte character fails at the read, before
        # any JSON parsing, so the branch below never sees it.
        raise ValueError(
            f"{path} is not readable as UTF-8 (byte {exc.start}). The session "
            "record was damaged after it was written; restore it from a backup "
            "or version control, or start a fresh session with: "
            "cda validate --term <term>"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON (line {exc.lineno}: {exc.msg}). The "
            "session record may have been hand-edited or truncated by a full "
            "disk; restore it from a backup or version control — the atomic "
            "writer never leaves partial files, so this did not come from an "
            "interrupted save."
        ) from exc
    if session.get("schema") != SCHEMA:
        raise ValueError(
            f"{path} has schema {session.get('schema')!r}, expected {SCHEMA!r}. "
            "It was written by an incompatible CDA version."
        )
    return session


def find_latest_session(output_dir: Path, term: str) -> Path:
    """Newest session dir whose stored term matches the requested term.

    Matches the full directory-name pattern and compares the slug exactly,
    then sorts by the parsed timestamp: a term whose slug extends another
    ("novelty_claims" vs "novelty") can never be picked up by mistake, and
    ordering never falls back to lexicographic filename comparison.
    """
    slug = safe_term_slug(term)
    root = Path(output_dir)
    candidates = []
    if root.is_dir():
        for path in root.iterdir():
            match = _SESSION_DIR.match(path.name)
            if match and match.group("slug") == slug and (path / "session.json").is_file():
                stored = load_session(path).get("term")
                if not isinstance(stored, str) or term_identity(stored) != term_identity(term):
                    continue
                # Within one second, a higher collision suffix means a later
                # session: create_session hands out -1, -2, … in order.
                candidates.append((match.group("ts"), int(match.group("n") or 0), path))
    if not candidates:
        raise FileNotFoundError(
            f"No validation session found for '{term}' under {root}. "
            "Start one with: cda validate --term " + term
        )
    return max(candidates, key=lambda entry: (entry[0], entry[1]))[2]


def current_revision(session: dict, side: str) -> int:
    return max(p["revision"] for p in session["partitions"] if p["side"] == side)


def partition_for(session: dict, side: str, revision: int) -> dict:
    for partition in session["partitions"]:
        if partition["side"] == side and partition["revision"] == revision:
            return partition
    raise ValueError(f"no partition for side={side} revision={revision}")


def has_verdict(session: dict, side: str, cluster_id: int, revision: int) -> bool:
    return any(
        v["side"] == side and v["cluster_id"] == cluster_id and v["revision"] == revision
        for v in session["verdicts"]
    )


def is_revealed(session: dict, side: str, cluster_id: int, revision: int) -> bool:
    """Whether this cluster's readings have already been revealed.

    Public so a caller can tell a first reveal (worth persisting) from a
    re-render of an already-revealed cluster (nothing changed, no write).
    """
    return any(
        r["side"] == side and r["cluster_id"] == cluster_id and r["revision"] == revision
        for r in session["reveals"]
    )


def log_reveal(
    session: dict, *, side: str, cluster_id: int, revision: int, via: str, now: datetime
) -> None:
    """Record that an agent's reading of this cluster was shown (idempotent).

    Every channel that displays an agent's view of a cluster's passages (its
    readings, or the revision agent's argument about it) logs here first, so
    any later verdict on that cluster is flagged ``revised_after_reveal``.
    """
    if not has_verdict(session, side, cluster_id, revision):
        raise ValueError(
            f"No verdict recorded yet for {side} cluster {cluster_id} "
            f"(revision {revision}); commit a verdict before revealing the "
            "agent's reading."
        )
    if not is_revealed(session, side, cluster_id, revision):
        session["reveals"].append(
            {
                "side": side,
                "cluster_id": cluster_id,
                "revision": revision,
                "via": via,
                "at": now.isoformat(timespec="seconds"),
            }
        )


def record_verdict(
    session: dict,
    *,
    side: str,
    cluster_id: int,
    revision: int,
    verdict: str,
    notes: str = "",
    overlaps_with: int | None = None,
    now: datetime,
) -> dict:
    """Append a verdict; its provenance flags are derived, not caller-supplied.

    The first verdict for a cluster, recorded before any reveal, is blind.
    Anything recorded after that cluster's readings were revealed is a
    post-reveal revision (the blind original stays in the trail).
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    if verdict == "overlaps" and overlaps_with is None:
        raise ValueError("an 'overlaps' verdict needs overlaps_with=<cluster id>")
    partition = partition_for(session, side, revision)  # validates side/revision
    known = {c["cluster_id"] for c in partition["clusters"]}
    if cluster_id not in known:
        raise ValueError(
            f"no {side} cluster {cluster_id} at revision {revision} "
            f"(clusters: {sorted(known)}) — a verdict on a ghost cluster would "
            "silently distort the agreement statistics"
        )
    if overlaps_with is not None and (overlaps_with not in known or overlaps_with == cluster_id):
        raise ValueError(
            f"overlaps_with must name a different existing cluster "
            f"(got {overlaps_with}; clusters: {sorted(known)})"
        )
    entry = {
        "side": side,
        "cluster_id": cluster_id,
        "revision": revision,
        "verdict": verdict,
        "notes": notes,
        "overlaps_with": overlaps_with,
        "blind": not is_revealed(session, side, cluster_id, revision),
        "revised_after_reveal": is_revealed(session, side, cluster_id, revision),
        "at": now.isoformat(timespec="seconds"),
    }
    session["verdicts"].append(entry)
    return entry


def reveal_readings(
    session: dict, *, side: str, cluster_id: int, revision: int, now: datetime
) -> list[dict]:
    """Return the cluster's readings; refuses until a verdict exists.

    This is the blind-then-reveal protocol enforced at the data layer: the
    agent's reading cannot reach any caller (GUI included) before the
    researcher has committed a verdict for that cluster.
    """
    log_reveal(
        session, side=side, cluster_id=cluster_id, revision=revision, via="readings", now=now
    )
    return [
        r
        for r in session["readings"]
        if r["side"] == side and r["cluster_id"] == cluster_id and r["revision"] == revision
    ]


def endorse(session: dict, *, note: str = "", now: datetime) -> dict:
    """The researcher endorses the current structure; the session is complete.

    Refuses while any cluster still lacks a verdict: an endorsement asserts
    that every cluster was judged, and the GUI's disabled button is a
    courtesy, not the enforcement — enforcement lives here, like the reveal
    gate, so no caller can produce an endorsement the record contradicts.
    """
    for side in ("researcher", "literature"):
        if not any(p["side"] == side for p in session["partitions"]):
            continue
        revision = current_revision(session, side)
        partition = partition_for(session, side, revision)
        missing = [
            c["cluster_id"]
            for c in partition["clusters"]
            if not has_verdict(session, side, c["cluster_id"], revision)
        ]
        if missing:
            raise ValueError(
                f"cannot endorse: {side} cluster(s) "
                f"{', '.join(str(c) for c in missing)} (revision {revision}) "
                "have no verdict yet"
            )
    session["endorsement"] = {
        "revisions": current_revisions(session),
        "note": note,
        "at": now.isoformat(timespec="seconds"),
    }
    return session["endorsement"]


def current_revisions(session: dict) -> dict[str, int]:
    """The latest partition revision of every side the session holds."""
    return {
        side: current_revision(session, side)
        for side in ("researcher", "literature")
        if any(p["side"] == side for p in session["partitions"])
    }


def withdraw_endorsement(session: dict, *, edit_id: int, now: datetime) -> dict | None:
    """Retract the endorsement after an edit changed the endorsed structure.

    Call AFTER the edit's new partition revision is appended. An endorsement
    asserts that the researcher judged the structure at specific revisions;
    once an applied merge, split, or recut moves a side past those revisions,
    the endorsement no longer describes the current clustering. The retracted
    record is kept in ``endorsement_withdrawals`` (the audit trail), never
    deleted. Returns the withdrawal entry, or None when nothing was endorsed.
    """
    endorsement = session.get("endorsement")
    if not endorsement:
        return None
    withdrawal = {
        **endorsement,
        "withdrawn_by_edit": edit_id,
        "withdrawn_at_revisions": current_revisions(session),
        "withdrawn_at": now.isoformat(timespec="seconds"),
    }
    session.setdefault("endorsement_withdrawals", []).append(withdrawal)
    session["endorsement"] = None
    return withdrawal


def current_endorsement(session: dict) -> dict | None:
    """The endorsement, only if it still covers the current revisions.

    Sessions recorded before edits withdrew endorsements can hold one whose
    revisions an applied edit has since moved past; such a record must not
    read as an endorsement of the structure now on screen.
    """
    endorsement = session.get("endorsement")
    if endorsement and endorsement.get("revisions") == current_revisions(session):
        return endorsement
    return None
