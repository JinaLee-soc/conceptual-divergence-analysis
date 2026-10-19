# Building a researcher-curated corpus from an existing Zotero library

This is the setup guide referenced in the paper: existing Zotero library +
PDF directory → a working local CDA corpus, including the drafts directory
and MyWork tagging. Time: ~15 minutes of your attention plus unattended
indexing (a few minutes per hundred PDFs).

## What you need

- Your PDFs in a directory (Zotero's storage folder works as-is — the
  ingester searches recursively).
- Zotero (any recent version). Optional but recommended: it supplies
  title/year/DOI metadata and the `mywork` tag.
- This repository installed (`pip install -e .` or Docker).

## Step 1 — Tag your own papers in Zotero

In Zotero, add the tag **`mywork`** to every item *you authored* that is in
your library (your published papers, accepted manuscripts). This is the
"Zotero tags" signal of MyWork resolution: text from these PDFs will be
classified as researcher-authored rather than literature.

Spelling is forgiving (`mywork`, `MyWork`, `#my-work` all match); the tag
just has to be on the item.

## Step 2 — Export the library as CSV

Zotero → **File → Export Library…** → Format: **CSV** → save (e.g.
`zotero_export.csv`). That's the entire Zotero integration: CDA never
connects to Zotero directly, needs no Zotero API key, and works with any
Zotero version. The export is read once at ingest; you re-export only when
your library changes materially.

## Step 3 — Ingest

```bash
cda ingest --pdf-dir ~/Zotero/storage --zotero-csv ~/zotero_export.csv
```

What this does, per PDF: extract text (PyMuPDF) → split into sentences
(pysbd) and overlapping word windows (500-word windows, 100-word overlap —
the default that produced the paper's corpus; configurable in `corpus.*`)
→ embed locally with `all-MiniLM-L6-v2` (pinned to one model snapshot by
`embedding.revision`) → store in a persistent ChromaDB at `./corpus/chroma`
with the Zotero metadata (title, year, DOI, tags, and the "Date Added"
acquisition timestamp, stored in UTC, which powers the experimental
`cda emergence` timeline) attached to every chunk.

Re-running is cheap: a file is skipped when it is unchanged. Ingest
re-ingests a file whose content changed (file hash) and files stored by an
older CDA version that used a different sentence segmenter. A new Zotero CSV
export updates each file's Zotero fields in place, without re-extracting or
re-embedding. Files stored for this `--pdf-dir` that are no longer on disk,
or that now fall under `--exclude` or `--no-recursive`, are listed and kept
in the corpus; add `--prune-missing` to remove them. `--reingest` forces a refresh
of every file. A refreshed file is rebuilt completely before its stored
chunks are replaced, so a failed refresh keeps the stored version; an
encrypted or unreadable PDF is listed and skipped. Without `--zotero-csv`,
a refreshed file keeps the Zotero metadata already stored for it. A file
stored under another name (a rename, including a letter-case change)
replaces the old name, and a second copy of the same file elsewhere is
skipped with a warning. Sources stored before library roots were recorded
are never removed by `--prune-missing`. Corpora built before the `date_added` field existed need one
`--reingest` pass with the CSV for `cda emergence` to see acquisition dates.

Zotero metadata is attached only when the match is unambiguous. Titles are
searched only in a PDF's opening text, and a title shorter than 40 letters
can make a match ambiguous but never attaches a record on its own. If a PDF's
filename, DOI or title is shared by more than one Zotero item, that key
attaches nothing; ingest prints which file was affected and why (also
stored as `zotero_unmatched_reason`). When the filename is the ambiguous
key, the DOI and title found in the PDF are still tried. Rename the file or
merge the duplicate Zotero items to resolve it.

One placement rule matters: the ChromaDB store is a live database (SQLite
plus HNSW index files), so keep it on a local, non-synced disk. A cloud-sync
client (Dropbox, OneDrive, iCloud) rewriting those files mid-ingest can leave
a torn store or a stale snapshot. Syncing the repository clone itself is
fine — it is the store that must not sync. If your clone lives in a synced
folder, set `paths.corpus_dir` in your config to a local path outside it.

Nothing here uses the network: extraction, embedding, and storage are all
local. The GUI's **Ingest** tab does the same thing with a progress display.
(Running the GUI in Docker? The container sees only the mounted `corpus/`,
`drafts/`, and `outputs/` folders — put PDFs under `corpus/` and ingest from
that path; any other host path is invisible inside the container.)

## Step 4 — The drafts directory

Create `./drafts/` and put your working manuscripts there (.docx, .md, .txt,
.pdf; subfolders fine). Drafts are read at analysis time, never ingested
into the corpus, and are **always** classified as your own writing — this is
the "direct draft inclusion" signal, and it is what makes CDA useful *before*
a paper is published.

Each draft is read in full, with no size or page limit. For a `.docx` file,
CDA reads the body text (including tables and text boxes), footnotes and
endnotes; headers, footers and comments are skipped. A draft that cannot be
read stops the analysis with a message naming the file, and so do two
drafts with identical text (every sentence would otherwise count twice).

## Step 5 — (Optional) strengthen MyWork resolution

The four metadata signals are tried in configurable priority order
(`mywork.priority` in your config). Two are worth filling in:

```yaml
# my-config.yaml (start from config/config.example.yaml)
mywork:
  # working papers whose file names match no published title
  # (matched against the file name and its stem, ignoring case):
  filename_patterns: ["*_draft*", "myproject_*"]
  # fuzzy title matching against your published papers' file names:
  publication_titles:
    - "The Exact Title of My First Paper"
    - "The Exact Title of My Second Paper"
```

Title matching counts shared content words between a title and a file name.
Titles in any script are read word by word, with accents folded on both
sides. English stop words (scikit-learn's list) and Latin-script words under
three letters do not count, so a title must have at least `title_match_min_overlap` (default
3) content words; a shorter one is refused when the config loads, since it
could never match. Use a `filename_patterns` entry for such papers.

Every report includes a *MyWork classification* table showing which signal
classified how many retrieved chunks — check it after your first run;
misattributed chunks are usually a missing tag or pattern.

## Step 6 — Verify

```bash
cda analyze --term <a term you know you use> --dry-run
```

The dry run prints the assembled prompt: you should see your draft file
names under RESEARCHER CLUSTER STRUCTURE and literature sources under
LITERATURE PASSAGES. The GUI's **Setup** tab shows corpus chunk counts.

## Step 7 — Pre-flight: diagnose the corpus you built

Before your first analysis, run the composition diagnostic once and skim it:

```bash
cda diagnose --fields "Social Sciences"    # free; needs OPENALEX_MAILTO in .env
```

The report shows venue concentration, field/topic coverage, the temporal
profile, and the large topics of your declared fields that the corpus does
not touch. This is the recommended quality check on the baseline itself:
every divergence CDA reports is *relative to this corpus*, so you want its
structure to be a deliberate curation choice rather than an accident.
Personalization is the method's strength — the diagnostic keeps it from
quietly becoming fragmentation. Re-run it after major library changes;
responses are cached locally, so re-runs are instant and offline.

## Without Zotero

Skip steps 1–2 and ingest the PDF directory alone:

```bash
cda ingest --pdf-dir ~/my_papers
```

You lose title/year/DOI metadata and the `mywork` tag signal — so mark your
own papers via `mywork.filename_patterns` / `publication_titles` instead
(step 5), or rely entirely on the drafts directory. `cda diagnose` needs
DOIs, so the Zotero path is required for the corpus-composition report.

## Corpus size and scope

CDA is built for researcher-scale curated corpora — hundreds to a few
thousand papers that *you selected* (the paper's demonstration corpora hold
roughly 3,000 papers each). The corpus should reflect the literatures you
actually write against: that selectivity is not a bias to correct, it is the
method's reference point. See [faq.md](faq.md) on size, and run
`cda diagnose` to see your corpus's structure.

### Which files get ingested

`cda ingest` walks sub-folders by default. That is usually right, and
occasionally wrong in a way that is easy to miss: a curated library folder
often also holds exported reports, tool output, or book scans, and a
recursive walk takes those too. They then count as literature in every
analysis you run, without anything appearing to go wrong.

Two ways to bound it:

```
cda ingest --pdf-dir ~/Library --no-recursive          # only the top level
cda ingest --pdf-dir ~/Library --exclude outputs --exclude notes
```

A pattern that matches a directory name skips that whole sub-tree; a pattern
with wildcards (`--exclude '*.draft.pdf'`) is matched against each file's path
relative to `--pdf-dir`. Both are also settable as `corpus.recursive` and
`corpus.exclude` in your config, in which case the flags add to them. The run
reports how many files each rule skipped, so you can check the count against
what you expected before trusting the corpus.


### Changes to stored inputs

Ordinary ingestion refreshes changed chunk settings as well as source text.
For JSONL ingestion, changes to year, title, DOI, tags, authorship or acquisition
metadata also trigger refresh, including removal of a previously supplied field.
Older chunks without the recorded settings are refreshed once.

A corpus cannot reuse the same relative PDF path from two different PDF roots.
Ingestion detects that collision before modifying stored sources. Use separate
corpus directories or arrange the libraries under one common root so their
relative paths differ. Duplicate JSONL source IDs are also rejected before writes.

If every PDF has been removed from an existing library directory, `--prune-missing`
can still remove its missing stored sources. Without that explicit option, an
empty selection remains an error and stored sources are preserved.
