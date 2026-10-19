"""Zotero CSV export parsing and metadata resolution."""

import pytest

from cda.corpus.zotero import (
    ZoteroRecord,
    build_doi_index,
    build_filename_index,
    load_zotero_csv,
    match_by_content,
    match_by_filename,
    normalize_date_added,
    normalize_key,
    record_fields,
)


def metadata_for_pdf(name, index):
    record, _ = match_by_filename(name, index)
    return record_fields(record) if record is not None else {}


CSV_TEXT = """\
"Key","Item Type","Publication Year","Author","Title","DOI","File Attachments","Manual Tags","Automatic Tags"
"AB12","journalArticle","2021","Doe, J.","Evaluating Novelty in Peer Review","10.1000/nov21","/home/u/Zotero/storage/AB12/Doe_2021_Evaluating Novelty.pdf","peer-review; evaluation",""
"CD34","journalArticle","2019","Lee, J.","My Own Study of Consensus","10.1000/own19","/home/u/Zotero/storage/CD34/Lee_2019_My Own Study.pdf","#MyWork; consensus","auto-tag"
"EF56","note","","","A note without attachments","","","",""
"""


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "zotero_export.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


def test_load_and_index(csv_path):
    records = load_zotero_csv(csv_path)
    assert len(records) == 3
    index = build_filename_index(records)
    assert normalize_key("Doe_2021_Evaluating Novelty.pdf") in index
    assert len(index) == 2  # the note has no PDF attachment


def test_metadata_for_pdf(csv_path):
    index = build_filename_index(load_zotero_csv(csv_path))
    meta = metadata_for_pdf("Doe_2021_Evaluating Novelty.pdf", index)
    assert meta["title"] == "Evaluating Novelty in Peer Review"
    assert meta["year"] == "2021"
    assert meta["doi"] == "10.1000/nov21"
    assert "peer-review" in meta["zotero_tags"]
    # Unmatched files get no metadata rather than wrong metadata.
    assert metadata_for_pdf("Unknown_2020.pdf", index) == {}


def test_mywork_tag_visible_downstream(csv_path):
    index = build_filename_index(load_zotero_csv(csv_path))
    meta = metadata_for_pdf("Lee_2019_My Own Study.pdf", index)
    assert "#MyWork" in meta["zotero_tags"]

    from cda.extraction.mywork import mywork_classifier

    resolver = mywork_classifier({"priority": ["zotero_tags"]})
    is_researcher, reason = resolver("Lee_2019_My Own Study.pdf", meta)
    assert is_researcher
    assert reason == "zotero:mywork"


def test_date_added_parsed_and_normalized(tmp_path):
    """Zotero's Date Added reaches chunk metadata normalized — or not at all."""
    csv_text = (
        '"Title","Publication Year","DOI","File Attachments","Manual Tags","Date Added"\n'
        '"Dated Paper","2021","10.1/d","/z/storage/A/Dated Paper.pdf","","2021-03-14T09:22:11Z"\n'
        '"Undated Paper","2020","10.1/u","/z/storage/B/Undated Paper.pdf","","not a date"\n'
    )
    path = tmp_path / "export.csv"
    path.write_text(csv_text, encoding="utf-8")
    records = load_zotero_csv(path)
    expected = "2021-03-14 09:22:11"  # stored in UTC on every machine
    assert records[0].date_added == expected
    assert records[1].date_added == ""

    index = build_filename_index(records)
    assert metadata_for_pdf("Dated Paper.pdf", index)["date_added"] == expected
    # Empty-safe convention: no key beats a wrong/empty value.
    assert "date_added" not in metadata_for_pdf("Undated Paper.pdf", index)


def test_missing_export_has_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="Export Library"):
        load_zotero_csv(tmp_path / "nope.csv")


def test_same_basename_in_different_dirs_both_ingest(tmp_path, fake_embedder, monkeypatch):
    """Regression: recursive ingest must not skip a second `paper.pdf`.

    Two different files share a basename in different folders; identity is the
    relative path, so both land and their chunks don't collide.
    """
    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import get_collection, ingest_pdf_dir

    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: f"content of {p} " * 60)
    pdf_dir = tmp_path / "lib"
    (pdf_dir / "a").mkdir(parents=True)
    (pdf_dir / "b").mkdir(parents=True)
    (pdf_dir / "a" / "paper.pdf").write_text("x")
    (pdf_dir / "b" / "paper.pdf").write_text("y")

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    stats = ingest_pdf_dir(config, fake_embedder, pdf_dir, progress=lambda _m: None)
    assert (stats["pdfs_seen"], stats["pdfs_ingested"], stats["pdfs_skipped"]) == (2, 2, 0)

    got = get_collection(config).get(include=["metadatas"])
    assert {m["pdf_relpath"] for m in got["metadatas"]} == {"a/paper.pdf", "b/paper.pdf"}
    assert len(set(got["ids"])) == len(got["ids"])  # no chunk-id collision
    assert all(m["pdf_filename"] == "paper.pdf" for m in got["metadatas"])  # basename kept

    # Re-ingest is idempotent (both recognized as already present).
    again = ingest_pdf_dir(config, fake_embedder, pdf_dir, progress=lambda _m: None)
    assert again["pdfs_ingested"] == 0 and again["pdfs_skipped"] == 2


def test_colliding_basenames_do_not_inherit_ambiguous_zotero_metadata(
    tmp_path, fake_embedder, monkeypatch
):
    """Zotero matches by basename, so two `paper.pdf`s would resolve to the same
    record — its mywork tag included. The ambiguous match is skipped (a uniquely
    named file still resolves normally)."""
    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import get_collection, ingest_pdf_dir

    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: f"content of {p} " * 60)
    csv_text = (
        '"Key","Publication Year","Title","DOI","File Attachments","Manual Tags","Automatic Tags"\n'
        '"AB12","2021","Ambiguous Paper","10.1/x","/z/AB12/paper.pdf","#MyWork",""\n'
        '"CD34","2019","Distinct Paper","10.1/y","/z/CD34/lee_2019_distinct.pdf","#MyWork",""\n'
    )
    csv_path = tmp_path / "zotero.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    pdf_dir = tmp_path / "lib"
    (pdf_dir / "a").mkdir(parents=True)
    (pdf_dir / "b").mkdir(parents=True)
    (pdf_dir / "a" / "paper.pdf").write_text("x")
    (pdf_dir / "b" / "paper.pdf").write_text("y")
    (pdf_dir / "lee_2019_distinct.pdf").write_text("z")

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    ingest_pdf_dir(config, fake_embedder, pdf_dir, zotero_csv=csv_path, progress=lambda _m: None)

    by_relpath: dict[str, dict] = {}
    for meta in get_collection(config).get(include=["metadatas"])["metadatas"]:
        by_relpath.setdefault(meta["pdf_relpath"], meta)

    # Both colliding files ingested, but neither carries the ambiguous Zotero
    # title/tag — so neither can be mis-tagged mywork off the wrong record.
    for relpath in ("a/paper.pdf", "b/paper.pdf"):
        assert relpath in by_relpath
        assert "zotero_tags" not in by_relpath[relpath]
        assert "title" not in by_relpath[relpath]
    # The uniquely named file resolves to its record normally.
    distinct = by_relpath["lee_2019_distinct.pdf"]
    assert distinct["title"] == "Distinct Paper"
    assert "#MyWork" in distinct["zotero_tags"]


# --- content-based fallback matching -----------------------------------------


def _records():
    from cda.corpus.zotero import ZoteroRecord

    return [
        ZoteroRecord(
            title="Doing Gender in the Laboratory",
            year="2019",
            doi="10.1000/abc123",
            date_added="2019-04-01 09:00:00",
            tags=["gender"],
            attachment_keys=["doinggender2019"],
        ),
        ZoteroRecord(
            title="Measuring Novelty in Science with Word Embedding",
            year="2021",
            doi="10.1000/xyz789",
            date_added="2021-02-11 09:00:00",
        ),
    ]


def test_normalize_doi_strips_resolver_prefixes():
    from cda.corpus.zotero import normalize_doi

    for raw in (
        "10.1000/ABC123",
        "https://doi.org/10.1000/abc123",
        "http://dx.doi.org/10.1000/abc123",
        "doi: 10.1000/abc123",
        "10.1000/abc123.",
    ):
        assert normalize_doi(raw) == "10.1000/abc123"


def test_match_by_content_prefers_doi():
    """DOI is an identifier; a title prefix is only strong evidence."""
    from cda.corpus.zotero import build_doi_index, build_title_index, match_by_content

    records = _records()
    text = "Some front matter\nhttps://doi.org/10.1000/xyz789\nAn unrelated heading"
    record, how = match_by_content(text, build_doi_index(records), build_title_index(records))
    assert how == "doi"
    assert record.year == "2021"


def test_match_by_content_falls_back_to_title():
    from cda.corpus.zotero import build_doi_index, build_title_index, match_by_content

    records = _records()
    # No DOI in the text, and the title is broken across lines as PDFs do.
    text = "Measuring Novelty in\nScience with Word Embedding\n\nAbstract. Novelty is a core value."
    record, how = match_by_content(text, build_doi_index(records), build_title_index(records))
    assert how == "title"
    assert record.doi == "10.1000/xyz789"


def test_match_by_content_attaches_nothing_when_two_titles_appear():
    """The head naming two library titles is ambiguous — attach nothing.

    A paper's opening pages quote other papers' titles (references, an
    epigraph). Resolving to the first index hit would attach an arbitrary
    record's tags and acquisition date — and a `mywork` tag would silently
    move the file to the researcher's side of the analysis. Same rule as
    the DOI branch directly above.
    """
    from cda.corpus.zotero import ZoteroRecord, build_doi_index, build_title_index, match_by_content

    records = [
        ZoteroRecord(
            title="Measuring Novelty in Science with Word Embedding",
            year="2021",
            doi="",
            date_added="2021-02-11 09:00:00",
        ),
        ZoteroRecord(
            title="Consensus Formation in Fragmented Research Communities",
            year="2010",
            doi="",
            date_added="2010-06-01 09:00:00",
            tags=["mywork"],
        ),
    ]
    # This file IS the novelty paper, but its first page cites the other
    # library title in full — the cited record must not win by index order.
    text = (
        "Measuring Novelty in Science with Word Embedding\n\n"
        "Abstract. We extend Consensus Formation in Fragmented Research "
        "Communities with an embedding-based measure."
    )
    record, how = match_by_content(text, build_doi_index(records), build_title_index(records))
    assert record is None and how == "ambiguous_title"


def test_match_by_content_returns_nothing_when_unknown():
    from cda.corpus.zotero import build_doi_index, build_title_index, match_by_content

    records = _records()
    record, how = match_by_content(
        "A paper about something else entirely",
        build_doi_index(records),
        build_title_index(records),
    )
    assert record is None and how == ""


def test_title_index_drops_ambiguous_prefixes():
    """Two different works sharing a title prefix resolve to neither: a wrong
    acquisition date is worse than a missing one, because a missing one is
    reported and a wrong one is not."""
    from cda.corpus.zotero import ZoteroRecord, build_title_index

    shared = "The Very Long Shared Opening Of Two Different Papers "
    records = [
        ZoteroRecord(title=shared + "About Gender", year="2019", doi="10.1/a"),
        ZoteroRecord(title=shared + "About Novelty", year="2020", doi="10.1/b"),
    ]
    assert list(build_title_index(records).values()) == [None]  # ambiguous key
    # ...but two records for the SAME work collapse harmlessly.
    same = [
        ZoteroRecord(title=shared + "About Gender", year="2019", doi="10.1/a"),
        ZoteroRecord(title=shared + "About Gender", year="2019", doi="10.1/a"),
    ]
    assert list(build_title_index(same).values()) == [same[0]]


def test_title_index_keys_short_titles_by_their_whole_title():
    """A title shorter than the prefix is indexed whole, so a short library
    title quoted on another work's first page is seen (and makes it ambiguous)."""
    from cda.corpus.zotero import ZoteroRecord, build_title_index

    record = ZoteroRecord(title="Short Title", year="2020", doi="")
    assert build_title_index([record]) == {"shorttitle": record}


def test_ambiguous_doi_head_attaches_nothing():
    """A paper's first pages name its own DOI and its references' DOIs.

    Taking the first one that happens to be in the library attaches a cited
    work's record — tags and acquisition date included — and a mywork tag
    would move the file to the researcher's side of the analysis. Absent
    metadata is safer than confident wrong metadata.
    """
    index = build_doi_index(
        [
            ZoteroRecord(title="My Paper", year="2021", doi="10.1000/mine"),
            ZoteroRecord(title="Cited Paper", year="2019", doi="10.1000/theirs"),
        ]
    )

    both = "Intro text. doi:10.1000/theirs is cited here. See also 10.1000/mine."
    assert match_by_content(both, index, {}) == (None, "ambiguous_doi")

    only_one = "This paper, doi:10.1000/mine, argues that..."
    record, how = match_by_content(only_one, index, {})
    assert how == "doi" and record.title == "My Paper"


def test_a_single_known_doi_among_unknown_ones_still_matches():
    """Only DOIs the library knows count; unknown ones are not ambiguity."""
    index = build_doi_index([ZoteroRecord(title="My Paper", year="2021", doi="10.1000/mine")])
    text = "doi:10.9999/stranger and doi:10.1000/mine and doi:10.8888/other"
    record, how = match_by_content(text, index, {})
    assert how == "doi" and record.title == "My Paper"


def test_zoned_dates_are_converted_to_utc():
    """Every stored acquisition time is UTC, independent of the machine's zone."""
    assert normalize_date_added("2026-05-01T00:00:00Z") == "2026-05-01 00:00:00"
    assert normalize_date_added("2026-05-01T05:00:00+05:00") == "2026-05-01 00:00:00"
    assert normalize_date_added("2026-05-01T00:00:00-0500") == "2026-05-01 05:00:00"


def test_naive_and_bare_dates_are_taken_as_utc():
    """Zotero writes naive Date Added values in UTC; they are kept as written."""
    assert normalize_date_added("2026-05-01 13:45:00") == "2026-05-01 13:45:00"
    assert normalize_date_added("2026-05-01") == "2026-05-01 00:00:00"
    assert normalize_date_added("2026-05-01 13:45") == "2026-05-01 13:45:00"
    assert normalize_date_added("not a date") == ""


# --- ingest selection and resume ---------------------------------------------


def test_select_pdfs_is_case_insensitive_and_strips_trailing_slash(tmp_path):
    """Publisher downloads are routinely named .PDF, and shell tab completion
    appends / to a directory — neither should silently change the corpus."""
    from cda.corpus.ingest import select_pdfs

    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "B.PDF").write_text("x")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "report.pdf").write_text("x")

    kept, excluded = select_pdfs(tmp_path, recursive=True, exclude=["outputs/"])
    assert sorted(p.name for p in kept) == ["B.PDF", "a.pdf"]
    assert excluded == 1


def test_ingest_refuses_an_empty_pdf_dir(tmp_path, fake_embedder):
    """Zero PDFs exiting as success reads as a completed ingest of nothing."""
    from cda.config import load_config
    from cda.corpus.ingest import ingest_pdf_dir

    (tmp_path / "lib").mkdir()
    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    with pytest.raises(FileNotFoundError, match="Nothing to ingest"):
        ingest_pdf_dir(config, fake_embedder, tmp_path / "lib", progress=lambda _m: None)


def test_interrupted_ingest_is_finished_on_the_next_run(tmp_path, fake_embedder, monkeypatch):
    """An interrupted run leaves a file truncated mid-document; the next run
    must re-ingest it whole rather than freeze the truncation as complete."""
    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import get_collection, ingest_pdf_dir

    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: "word " * 1600)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "paper.pdf").write_text("x")
    config = load_config(
        overrides={
            "paths": {"corpus_dir": str(tmp_path / "corpus")},
            "corpus": {"chunk_size_words": 100, "chunk_overlap_words": 0},
        }
    )

    # Simulate the interruption: keep only the first chunk of a 16-chunk doc.
    ingest_pdf_dir(config, fake_embedder, lib, progress=lambda _m: None)
    collection = get_collection(config)
    got = collection.get(include=["metadatas"])
    total = int(got["metadatas"][0]["total_chunks"])
    assert total > 1
    doomed = [i for i in got["ids"] if not i.endswith("::c0")]
    collection.delete(ids=doomed)
    assert len(collection.get()["ids"]) == 1

    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=lambda _m: None)
    assert stats["pdfs_ingested"] == 1  # re-ingested, not skipped
    assert len(get_collection(config).get()["ids"]) == total


def test_non_zotero_csv_is_rejected_with_guidance(tmp_path):
    """A Paperpile/Mendeley/EndNote export used to load as N empty records —
    no error, all MyWork matching silently dead, and the eventual failure
    message pointed at drafts/config instead of this file."""
    csv_text = (
        '"Title","Authors","Journal","Publication year","Item type","DOI"\n'
        '"Some Paper","Doe J","Nature","2021","Journal Article","10.1/x"\n'
    )
    path = tmp_path / "paperpile.csv"
    path.write_text(csv_text, encoding="utf-8")
    with pytest.raises(ValueError, match="Zotero"):
        load_zotero_csv(path)


def test_headerless_csv_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("just,some,values\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not look like a Zotero CSV"):
        load_zotero_csv(path)


@pytest.mark.parametrize(
    "value", ["2026-02-30 10:00:00Z", "2026-02-30", "2026-05-01 24:00:00Z", "2026-13-01 10:00"]
)
def test_impossible_dates_are_rejected(value):
    with pytest.raises(ValueError, match="not a real date"):
        normalize_date_added(value)


def test_oversized_note_field_does_not_crash(tmp_path):
    """Zotero exports can carry >128 KB Notes columns, over csv's default
    field limit."""
    import csv as csv_module

    big_note = "x" * 200_000
    csv_text = f'"Title","Item Type","Notes"\n"Paper","journalArticle","{big_note}"\n'
    path = tmp_path / "big.csv"
    path.write_text(csv_text, encoding="utf-8")
    before = csv_module.field_size_limit()
    records = load_zotero_csv(path)
    assert records[0].title == "Paper"
    # The limit is process-global; raising it permanently would change csv
    # behavior for any program that merely imports cda.
    assert csv_module.field_size_limit() == before


# --- chunk-id integrity -------------------------------------------------------


def test_no_text_file_does_not_leave_stale_chunks_behind(tmp_path, fake_embedder, monkeypatch):
    """ "this file will not appear in any analysis" has to be true.

    A run interrupted mid-file leaves truncated chunks; if the PDF then
    becomes unreadable (Dropbox eviction, corruption), the message used to be
    printed while those chunks stayed in the corpus forever — appearing in
    every analysis, with metadata still claiming the full total_chunks.
    """
    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import get_collection, ingest_pdf_dir

    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "paper.pdf").write_text("x")
    config = load_config(
        overrides={
            "paths": {"corpus_dir": str(tmp_path / "corpus")},
            "corpus": {"chunk_size_words": 100, "chunk_overlap_words": 0},
        }
    )

    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: "word " * 1600)
    ingest_pdf_dir(config, fake_embedder, lib, progress=lambda _m: None)
    collection = get_collection(config)
    doomed = [i for i in collection.get()["ids"] if not i.endswith("::c0")]
    collection.delete(ids=doomed)  # simulate the interruption

    # Now the file yields nothing at all.
    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: "")
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    assert stats["pdfs_no_text"] == 1
    assert get_collection(config).get()["ids"] == []
    assert any("will not appear in any analysis" in m for m in messages)
    assert any("removed" in m for m in messages)


def test_a_chunk_id_that_does_not_land_is_reported(tmp_path, fake_embedder):
    """Chroma's add() is a silent no-op for an existing id — it neither
    overwrites nor raises — so a chunk-id collision would discard a source's
    text while the stats counted it as ingested. Verify the batch landed."""
    from cda.config import load_config
    from cda.corpus.ingest import _add_chunks, get_collection

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    collection = get_collection(config, create=True)
    meta = {"pdf_filename": "a.pdf", "pdf_relpath": "a.pdf", "total_chunks": 2}
    _add_chunks(collection, fake_embedder, "same_key", ["one", "two"], meta)

    # A second source that happens to derive the same key: its chunks would
    # vanish silently.
    other = {"pdf_filename": "b.pdf", "pdf_relpath": "b.pdf", "total_chunks": 2}
    with pytest.raises(RuntimeError, match="b.pdf"):
        _add_chunks(collection, fake_embedder, "same_key", ["three", "four"], other)


def test_record_sources_that_share_a_chunk_key_are_refused(tmp_path, fake_embedder):
    """Records keep the historical non-injective key (demo chunk ids must stay
    byte-identical for the recorded fixtures), so a collision is refused up
    front instead of silently dropping one record."""
    from cda.config import load_config
    from cda.corpus.ingest import ingest_records

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    records = [
        {"source": "paper_1.txt", "abstract": "novelty " * 40},
        {"source": "paper-1.txt", "abstract": "consensus " * 40},
    ]
    with pytest.raises(ValueError, match="paper-1.txt, paper_1.txt"):
        ingest_records(config, fake_embedder, records, progress=lambda _m: None)


def test_the_committed_toy_corpus_has_no_key_collisions():
    """The demo must keep ingesting: the guard above is only useful if the
    shipped corpus passes it."""
    import json
    from collections import Counter
    from pathlib import Path

    from cda.corpus.zotero import normalize_key

    jsonl = Path(__file__).resolve().parents[1] / "demo" / "toy_corpus" / "abstracts.jsonl"
    sources = [
        json.loads(line)["source"] for line in jsonl.read_text().splitlines() if line.strip()
    ]
    assert len(sources) == len(set(sources))
    colliding = [key for key, n in Counter(normalize_key(s) for s in sources).items() if n > 1]
    assert colliding == []


def test_unicode_spelling_of_a_filename_does_not_duplicate_the_document(
    tmp_path, fake_embedder, monkeypatch
):
    """macOS and Linux spell 'café.pdf' differently (NFC vs NFD), and a
    Dropbox-synced folder flips between them. On the raw string the same file
    looks absent, so it would be ingested a second time under a second
    spelling — every chunk duplicated, with no warning."""
    import unicodedata

    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import get_collection, ingest_pdf_dir

    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: "content " * 60)
    nfc = unicodedata.normalize("NFC", "café.pdf")
    nfd = unicodedata.normalize("NFD", "café.pdf")
    assert nfc != nfd

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / nfc).write_text("x")
    ingest_pdf_dir(config, fake_embedder, lib, progress=lambda _m: None)
    stored = get_collection(config).get(include=["metadatas"])
    n_chunks = len(stored["ids"])
    assert {m["pdf_relpath"] for m in stored["metadatas"]} == {nfc}

    # The same file now presents the other spelling (rename, not a new file).
    (lib / nfc).rename(lib / nfd)
    if not (lib / nfd).exists():  # pragma: no cover - macOS normalizes on write
        pytest.skip("this filesystem normalizes filenames itself")
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    assert stats["pdfs_ingested"] == 0 and stats["pdfs_skipped"] == 1
    after = get_collection(config).get(include=["metadatas"])
    assert len(after["ids"]) == n_chunks  # not doubled
    assert {m["pdf_relpath"] for m in after["metadatas"]} == {nfc}


def test_normalize_key_folds_accents_and_guards_the_empty_key():
    """NFC and NFD spellings of one name must produce one key, and a name
    with no ASCII skeleton must produce NO key — an empty key in the index
    would join every non-ASCII file to whichever record claimed it first,
    wrong title, DOI, acquisition date, and mywork tag included."""
    import unicodedata

    from cda.corpus.zotero import build_filename_index

    nfc = unicodedata.normalize("NFC", "café.pdf")
    nfd = unicodedata.normalize("NFD", "café.pdf")
    assert nfc != nfd
    assert normalize_key(nfc) == normalize_key(nfd) == "cafe"

    assert normalize_key("논문.pdf") == ""
    record = ZoteroRecord(
        title="한국어 논문",
        year="2021",
        doi="10.1/k",
        tags=["mywork"],
        attachment_keys=[normalize_key("논문.pdf")],
    )
    # The empty key is not indexed...
    assert build_filename_index([record]) == {}
    # ...and never looked up, even against an index that carries one.
    assert metadata_for_pdf("리뷰.pdf", {"": record}) == {}


def test_fully_non_ascii_filenames_still_content_match(tmp_path, fake_embedder, monkeypatch):
    """Two Korean-named PDFs both normalize to the empty key. That used to
    count as a basename collision, which suppressed content matching for both
    — so neither could resolve its own record even though each names its own
    DOI on page one."""
    import unicodedata

    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import get_collection, ingest_pdf_dir

    def fake_extract(path):
        name = unicodedata.normalize("NFC", path.name)
        if name == "논문.pdf":
            return "This paper, doi:10.1000/kor1, argues about novelty. " * 20
        if name == "리뷰.pdf":
            return "This review, doi:10.1000/kor2, surveys consensus. " * 20
        return "Plain content about novelty with no DOI anywhere. " * 20

    monkeypatch.setattr(ing, "extract_pdf_text", fake_extract)
    csv_text = (
        '"Title","Item Type","Publication Year","DOI","File Attachments","Manual Tags"\n'
        '"Korean Paper One","journalArticle","2021","10.1000/kor1","/z/A/논문.pdf","#MyWork"\n'
        '"Korean Paper Two","journalArticle","2019","10.1000/kor2","/z/B/리뷰.pdf",""\n'
        '"Distinct Paper","journalArticle","2020","10.1000/plain","/z/C/distinct.pdf",""\n'
    )
    csv_path = tmp_path / "zotero.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in ("논문.pdf", "리뷰.pdf", "distinct.pdf"):
        (lib / name).write_text(name)  # distinct bytes: identical files are one source

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    stats = ingest_pdf_dir(
        config, fake_embedder, lib, zotero_csv=csv_path, progress=lambda _m: None
    )
    assert stats["zotero_by_doi"] == 2  # each Korean file resolved via its own DOI
    assert stats["zotero_by_filename"] == 1  # the ASCII name still matches by filename
    assert stats["zotero_unmatched"] == 0

    by_title = {}
    for meta in get_collection(config).get(include=["metadatas"])["metadatas"]:
        by_title[meta.get("title")] = meta
    # Each file carries its OWN record: the mywork tag stays on paper one only.
    assert "#MyWork" in by_title["Korean Paper One"]["zotero_tags"]
    assert "zotero_tags" not in by_title["Korean Paper Two"]
    assert by_title["Korean Paper Two"]["doi"] == "10.1000/kor2"


# --- chunking floor -----------------------------------------------------------


def test_tiny_chunk_size_still_windows_the_whole_document():
    """Regression: with size < the 20-word tail floor, every window after the
    first was 'too short' and chunking silently kept chunk 0 only — a direct
    caller with size=10 over 10k words got 1 chunk instead of ~1000."""
    from cda.corpus.ingest import chunk_words

    chunks = chunk_words("word " * 10_000, size=10, overlap=0)
    assert len(chunks) == 1000
    assert all(len(chunk.split()) == 10 for chunk in chunks)


def test_chunk_size_below_the_tail_floor_is_refused_by_the_pipeline(tmp_path, fake_embedder):
    """The config floor: a size the chunker cannot honor must fail with an
    explanation, not silently truncate every document to its first window."""
    from cda.config import ConfigError, load_config
    from cda.corpus.ingest import ingest_records

    config = load_config(
        overrides={
            "paths": {"corpus_dir": str(tmp_path / "corpus")},
            "corpus": {"chunk_size_words": 10, "chunk_overlap_words": 0},
        }
    )
    with pytest.raises(ConfigError, match="chunk_size_words"):
        ingest_records(
            config,
            fake_embedder,
            [{"source": "a.txt", "abstract": "word " * 100}],
            progress=lambda _m: None,
        )


# --- the per-store ingest lock ------------------------------------------------


def test_a_held_ingest_lock_is_waited_on_then_reported(tmp_path, fake_embedder, monkeypatch):
    """Two first-run ingests of one fresh store both saw every record absent
    and both added; Chroma silently kept the existing rows and the landed-count
    check raised a corpus-corruption-shaped error for a transient race. The
    lockfile serializes them; a lock that never clears must name itself."""
    import cda.corpus.ingest as ing
    from cda.config import load_config

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / ".ingest.lock").write_text("12345\n")
    monkeypatch.setattr(ing, "_INGEST_LOCK_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(ing, "_INGEST_LOCK_POLL_SECONDS", 0.01)

    config = load_config(overrides={"paths": {"corpus_dir": str(corpus_dir)}})
    messages: list[str] = []
    with pytest.raises(RuntimeError, match="ingest.lock"):
        ing.ingest_records(
            config,
            fake_embedder,
            [{"source": "a.txt", "abstract": "word " * 100}],
            progress=messages.append,
        )
    assert any("another ingest" in m.lower() for m in messages)
    # The stale lock is not cleaned up by the waiter: it belongs to the holder.
    assert (corpus_dir / ".ingest.lock").exists()


def test_the_ingest_lock_is_released_after_a_run(tmp_path, fake_embedder):
    from cda.config import load_config
    from cda.corpus.ingest import ingest_records

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    stats = ingest_records(
        config,
        fake_embedder,
        [{"source": "a.txt", "abstract": "word " * 100}],
        progress=lambda _m: None,
    )
    assert stats["records_ingested"] == 1
    assert not (tmp_path / "corpus" / ".ingest.lock").exists()


# --- stored-text normalization ------------------------------------------------


def test_extracted_pdf_text_is_stored_nfc_normalized(tmp_path, fake_embedder, monkeypatch):
    """PDF extractors emit accents however the file stored them (often NFD);
    an NFC term then misses in $contains, regex, and find. Stored text is NFC
    by contract (the query side normalizes symmetrically)."""
    import unicodedata

    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import get_collection, ingest_pdf_dir

    nfd_text = unicodedata.normalize("NFD", "café society is examined here at length ") * 10
    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: nfd_text)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "paper.pdf").write_text("x")
    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    ingest_pdf_dir(config, fake_embedder, lib, progress=lambda _m: None)

    nfc_term = unicodedata.normalize("NFC", "café")
    for document in get_collection(config).get(include=["documents"])["documents"]:
        assert unicodedata.is_normalized("NFC", document)
        assert nfc_term in document


def test_draft_text_is_nfc_normalized(tmp_path):
    """The drafts side of the same contract (see the test above)."""
    import unicodedata

    from cda.corpus.drafts import load_drafts

    nfd_text = unicodedata.normalize("NFD", "café society draft text")
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "a.txt").write_text(nfd_text, encoding="utf-8")
    texts = load_drafts(drafts, progress=lambda _m: None)
    assert texts["a.txt"] == unicodedata.normalize("NFC", nfd_text)
    assert unicodedata.is_normalized("NFC", texts["a.txt"])


# --- drafts that contribute nothing are announced -----------------------------


def test_unsupported_and_empty_drafts_are_announced(tmp_path):
    """A .doc draft skipped in silence looks exactly like a draft that was
    counted; the researcher only finds out when 'their' passages are missing.
    Stray non-document files must stay quiet so real notes aren't buried."""
    from cda.corpus.drafts import load_drafts

    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "old_draft.doc").write_text("x")
    (drafts / "empty.txt").write_text("   ")
    (drafts / ".DS_Store").write_bytes(b"junk")

    messages: list[str] = []
    texts = load_drafts(drafts, progress=messages.append)
    assert texts == {}
    assert any("old_draft.doc" in m and ".docx" in m for m in messages)
    assert any("empty.txt" in m and "no extractable text" in m for m in messages)
    assert not any("DS_Store" in m for m in messages)


def test_a_corpus_ingested_under_the_old_spelling_is_not_duplicated(
    tmp_path, fake_embedder, monkeypatch
):
    """Stores built before NFC normalization hold whatever spelling the disk
    gave. The old-spelling chunks are removed and the file is re-ingested once
    under the normalized name — a one-time re-add, never a silent duplicate."""
    import unicodedata

    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import _add_chunks, get_collection, ingest_pdf_dir

    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: "content " * 60)
    nfd = unicodedata.normalize("NFD", "café.pdf")
    nfc = unicodedata.normalize("NFC", "café.pdf")

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / nfd).write_text("x")
    if next(lib.iterdir()).name != nfd:  # pragma: no cover - some filesystems normalize
        pytest.skip("this filesystem normalizes filenames itself")

    # Pre-fix store: chunks written under the un-normalized relpath.
    collection = get_collection(config, create=True)
    _add_chunks(
        collection,
        fake_embedder,
        "legacy_key",
        ["one", "two"],
        {"pdf_filename": nfd, "pdf_relpath": nfd, "total_chunks": 2},
    )

    messages: list[str] = []
    ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    got = get_collection(config).get(include=["metadatas"])
    assert {m["pdf_relpath"] for m in got["metadatas"]} == {nfc}
    assert any("Unicode spelling" in m for m in messages)


def test_nfd_chunks_are_purged_even_when_the_disk_presents_nfc(
    tmp_path, fake_embedder, monkeypatch
):
    """Sibling case (a) of the migration above: the pre-fix store holds NFD
    chunks and the disk now presents NFC. Here disk spelling == normalized
    spelling, so the old cleanup (which only probed when they differed) never
    looked — the file re-ingested under NFC while the NFD chunks stayed, and
    every chunk was silently duplicated."""
    import unicodedata

    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import _add_chunks, get_collection, ingest_pdf_dir

    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: "content " * 60)
    nfc = unicodedata.normalize("NFC", "café.pdf")
    nfd = unicodedata.normalize("NFD", "café.pdf")

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / nfc).write_text("x")
    if next(lib.iterdir()).name != nfc:  # pragma: no cover - some filesystems normalize
        pytest.skip("this filesystem normalizes filenames itself")

    # Pre-fix store: chunks written under the decomposed spelling.
    collection = get_collection(config, create=True)
    _add_chunks(
        collection,
        fake_embedder,
        "legacy_key",
        ["one", "two"],
        {"pdf_filename": nfd, "pdf_relpath": nfd, "total_chunks": 2},
    )

    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    assert stats["pdfs_ingested"] == 1
    got = get_collection(config).get(include=["metadatas"])
    assert {m["pdf_relpath"] for m in got["metadatas"]} == {nfc}
    assert len(got["ids"]) == 1  # 60 words → one chunk; the NFD pair is gone
    assert any("Unicode spelling" in m for m in messages)


def test_both_spellings_stored_leave_exactly_one_document(tmp_path, fake_embedder, monkeypatch):
    """Sibling case (b): both spellings already stored (two pre-fix machines
    syncing one Dropbox corpus does this). The skip path used to never look,
    and --reingest deleted only the NFC spelling — the duplicate survived
    both. Every visit now purges the alternate spelling."""
    import unicodedata

    import cda.corpus.ingest as ing
    from cda.config import load_config
    from cda.corpus.ingest import _add_chunks, get_collection, ingest_pdf_dir

    monkeypatch.setattr(ing, "extract_pdf_text", lambda p: "content " * 60)
    nfc = unicodedata.normalize("NFC", "café.pdf")
    nfd = unicodedata.normalize("NFD", "café.pdf")

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / nfc).write_text("x")
    if next(lib.iterdir()).name != nfc:  # pragma: no cover - some filesystems normalize
        pytest.skip("this filesystem normalizes filenames itself")

    from cda.extraction.windows import SENTENCE_SEGMENTER
    from cda.provenance import file_digest

    collection = get_collection(config, create=True)
    current = {
        "source_file_sha256": file_digest(lib / nfc),
        "sentence_segmenter": SENTENCE_SEGMENTER,
        "chunk_size_words": config.get("corpus.chunk_size_words"),
        "chunk_overlap_words": config.get("corpus.chunk_overlap_words"),
    }
    for spelling, key in ((nfc, "nfc_key"), (nfd, "nfd_key")):
        _add_chunks(
            collection,
            fake_embedder,
            key,
            ["one"],
            {"pdf_filename": spelling, "pdf_relpath": spelling, "total_chunks": 1, **current},
            contexts=[(0, "one", [3])],
        )

    # The skip path: the NFC copy is complete, so the file is skipped — but
    # the NFD duplicate must still be removed.
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    assert stats["pdfs_ingested"] == 0 and stats["pdfs_skipped"] == 1
    got = get_collection(config).get(include=["metadatas"])
    assert {m["pdf_relpath"] for m in got["metadatas"]} == {nfc}
    assert len(got["ids"]) == 1  # not two documents' worth
    assert any("Unicode spelling" in m for m in messages)

    # The --reingest path: re-seed the duplicate and re-ingest; both spellings
    # must be replaced by exactly one freshly ingested document.
    _add_chunks(
        collection,
        fake_embedder,
        "nfd_key",
        ["one"],
        {"pdf_filename": nfd, "pdf_relpath": nfd, "total_chunks": 1},
    )
    stats = ingest_pdf_dir(config, fake_embedder, lib, reingest=True, progress=lambda _m: None)
    assert stats["pdfs_ingested"] == 1
    got = get_collection(config).get(include=["metadatas"])
    assert {m["pdf_relpath"] for m in got["metadatas"]} == {nfc}
    assert len(got["ids"]) == 1
