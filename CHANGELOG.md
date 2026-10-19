# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Lock, exclusion and lookup repairs

- `cda verify` no longer fails because an older run for the same term has a
  damaged saved prompt. Only the first 4 KB of each same-slug prompt is read
  to identify its term (a leading BOM is tolerated); unreadable prompts are
  skipped, and the newest intact run is verified. When none remains, the
  error lists the skipped runs with a diagnosis. Advice commands quote the
  term (`--term 'C#'`).
- The GUI's "Resume latest session for this term" skips session directories
  whose `session.json` cannot be loaded, resumes the newest intact session
  with a matching stored term, and shows an error instead of a traceback
  when none remains.
- `cda ingest --exclude` could be defeated by a symlink elsewhere in the
  library, and `--prune-missing` then renamed the excluded source to the link
  route. Each physical file is now judged once by its canonical route (its
  real path under `--pdf-dir`, or the first route in sorted order for a folder
  outside it) and stored under it; sources stored under a link route are
  renamed to their real path. Every physical folder is walked once.
- A `.ingest.lock` left by a crashed or killed run no longer stalls every
  corpus operation for the full timeout: a lock whose holder process no longer
  exists is removed and acquisition retried at once. A genuine timeout raises
  `CorpusLockError`, which the CLI reports as an error without a traceback.
- The demo creates the toy-corpus store under the corpus lock, so concurrent
  first visitors cannot corrupt a fresh store.
- `cda demo` and the GUI demo buttons stream progress and explain a lock
  timeout (without host paths on the public demo).
- `cda validate --session DIR --term X` compares terms the way result lookup
  does, so `AI` resumes a session saved as `ai`; a mismatch names the stored
  term.
- With `retrieval.keyword_max_chunks` set, every source needing re-ingestion
  is listed at once.
### Corpus access and result lookup

- Analysis and corpus diagnosis wait for ingestion to finish and hold the same
  store lock across all retrieval pages. Direct store paths and symlink aliases
  share that lock. Clustering and model calls run after it is released.
- Result verification and validation-session resume compare the saved target
  term, so terms sharing a filename slug (such as `C++` and `C#`) stay separate.
  Existing report and session filenames remain readable.
- A configured keyword retrieval cap counts only chunks containing a matching
  sentence; substring false positives no longer consume the cap.
- Validation checks its output directory before calling any reader model.
- PDF exclusions are applied before aliases are deduplicated, preserving files
  reachable through an included path while stopping symlink cycles.
- Fixed cluster counts reject fractional and nonfinite values instead of
  truncating them or failing with a Python conversion error.

### Review fixes

**Re-ingest required.** The sentence segmenter is now `pysbd-0.3.4+blocks-v3`
(CJK full stops end sentences); your usual `cda ingest` refreshes existing
corpora once.

- `--exclude`, `--no-recursive` and MyWork `filename_patterns` match file and
  folder names in either Unicode form (NFC or NFD), so excluded folders are no
  longer ingested and then pruned.
- Sources stored before library roots were recorded are adopted when found at
  the same path, or replaced when renamed, instead of blocking analysis; the
  analysis error names every legacy source.
- Phrase terms match any run of whitespace, including non-breaking spaces.
- Context windows over a chunk overlap are merged instead of stopping analysis.
- A folder that cannot be read is reported and never pruned; the ingest
  finishes. Hidden and AppleDouble (`._`) files in the drafts folder are
  skipped with a note.
- Running heads and page numbers are removed from PDF drafts, so a .docx and
  its PDF export are refused as near-duplicates.
- Generic library titles ("Introduction", "Editorial" and similar) are not used
  for Zotero title matching; Windows attachment paths in a CSV are read.
- Date bounds accept one-digit hours and refuse AM/PM or time-zone names.
- docx list markers are removed; typed HTML in prose is kept. Symlinked
  sub-folders under `--pdf-dir` are followed. Japanese voicing marks are kept
  in MyWork titles. The ingest summary counts sources unchanged by a Zotero
  export.
- Quotation grounding treats straight quotes touching a letter of any script,
  or spaced inner quotes, as ambiguous nesting; reads ". . ." as an elision,
  source names containing "]", blockquote continuations and a final ";";
  ellipses no longer count as words.
- The cluster–register audit reads more negations ("cannot", "never",
  "is not observed", "merge count 0") and chooses register-table columns by
  header.
- `cda verify` reads every distance rating on a line and never a cluster id.
- `figure.*`, `distance_scale` and `llm.temperature` (0–1 for Anthropic, 0–2
  for `openai:` models) are validated before retrieval; theme entries with an
  empty tag or description are refused.
- The mock demo prefers a fixture recorded under the configured model and
  falls back to the committed demo responses; the report names the model
  replayed. The record-fixtures workflow commits new recordings and checks
  that they replay.
- Run files are written on filesystems without hard links.
- Embedding provenance resolves short model names; run comparison checks the
  model id, revision and weight, configuration and tokenizer file hashes.
- Passage-map palettes cover the configured maximum k with distinct colours.
- Silhouettes are scored on a sample that includes every cluster, so a pinned
  k no longer fails and no candidate is dropped by sampling.
- When no valid partition is found, the stability section and report say so
  instead of reporting robust structure; silhouettes show "n/a".
- The order-stability benchmark refuses pinned-k and hierarchical configs; the
  bootstrap benchmark clusters as `cda analyze` does and checks its inputs; the
  saturation diagnostics use the configured ranking. Flag thresholds refuse
  booleans and non-whole counts.
- Applying a curation edit or recut withdraws an endorsement; the report and
  GUI show the withdrawal.
- GUI alerts are distinguishable without colour (border style and a text
  label); the public demo never shows host paths.
- `cda diagnose --demo` builds the toy corpus first; CI also tests Python 3.13.

### GUI display repairs

- The public demo's installation command enters the cloned directory.
- Alert icons and border shades use the monochrome interface palette.
- Generated metadata comments are hidden in the report view; the downloaded
  report retains its complete provenance header.

### Source preservation and literal phrase repairs

**Re-ingest required.** The sentence segmenter is now `pysbd-0.3.4+blocks-v2`.
Run your usual `cda ingest` once; the toy demo refreshes its corpus automatically.

- Blank lines and the 10,000-character block target no longer split a sentence.
- Literal phrases are found across chunk edges even with zero overlap. Candidate
  chunks without the full phrase do not consume the retrieval cap.
- Handled storage-write failures restore the previous source's text, metadata,
  and embeddings. Metadata-only refreshes also roll back failed updates; failed
  recovery stops ingestion and reports that the corpus needs repair. This does
  not provide transaction guarantees across process or machine crashes.
- Duplicate-file handling removes obsolete text, checks a live copy's current
  bytes, and preserves the selected copy when another copy is excluded and pruned.
- Reusing a Zotero CSV rechecks changed filename ambiguity and repairs fields
  consistently across all chunks.
- PDF and supplied-record inputs cannot overwrite each other's source identity.
- File-read failures during hashing keep the previous stored source and allow
  ingestion to continue with the other files. Directory access errors stop
  pruning instead of treating inaccessible files as missing.

### Corpus formats and report validation

- Sentence segmentation uses bounded blocks where sentence boundaries allow,
  reducing runtime for long documents.
- A new Zotero CSV export updates Zotero fields in place instead of
  re-ingesting every file (`metadata_updated` in the ingest summary).
- Stored files that now fall under `--exclude` or `--no-recursive` are listed
  and removed with `--prune-missing`.
- Date Added values with fractional seconds or a space before the time zone
  keep their zone; non-ISO values are left undated with a counted warning.
  An upper bound given to the minute includes the whole minute.
- MyWork titles in non-Latin scripts and with accented letters tokenize
  correctly; the three-letter minimum applies to Latin-script words only.
- arXiv identifiers in file names are no longer read as years.
- Symbol-font characters in .docx drafts are decoded; drafts sharing at least
  90% of their sentences (each with three or more) are refused, naming both.
- Quotation grounding accepts footnote digits, underscores and non-Latin text
  next to a quote mark, does not report inner quotes of a verified quotation
  as unpaired, and retries a quotation whose final . ? ! sits inside the mark.
- `themes` entries are validated before retrieval; `clustering.silhouette_subsample`
  must be at least 100.
- Independent-mode figures use separate colour sets with enough colours for
  the default maximum k; a figure of identical embeddings reports 0% variance.
- A k whose merged cluster has no mean direction is skipped in the silhouette
  sweep instead of stopping it; collapsed k values are recorded in the sweep
  log; flag thresholds outside [0, 1] are refused by name.
- The public demo turns Streamlit usage statistics off in `.streamlit/config.toml`,
  and its footer states the live setting.
- The revision agent checks the session's embedding model before any paid call.
- Error, warning, info and success messages have distinct alert elements;
  the unstable-k warning includes a visible marker.
- Validation reports print missing silhouettes as "n/a".
- `--overwrite-fixtures` without `--record-fixtures` is refused.
- The `python-dateutil` dependency is pinned.

### Data-integrity repairs

**Re-ingest required.** Chunks now store their ingest-time sentence
boundaries; corpora built earlier are refreshed by your usual `cda ingest`
command, and analysis refuses chunks without them until then.

- Analysis uses the sentence boundaries recorded at ingest instead of
  re-segmenting chunk text, so overlapping chunks no longer yield shifted,
  duplicated or conflicting sentences.
- Text ending in spaced dots (". . .") or a detached "?!" no longer stops
  ingest or draft loading; the punctuation stays with the preceding sentence.
- A refreshed source is fully rebuilt before its stored chunks are replaced;
  a failure keeps the stored version.
- Encrypted or otherwise unreadable PDFs are listed as unreadable and skipped;
  ingest continues and a previously stored version is kept.
- Re-ingesting without `--zotero-csv` keeps each source's stored Zotero
  metadata.
- The same file is stored once: a renamed file (including a letter-case
  rename) replaces its old name, and a second live copy is skipped with a
  warning. Missing-file checks compare names case-exactly.
- `--prune-missing` never removes sources whose library root was not
  recorded; they are listed as "root unknown; not pruned".
- Zotero titles are matched only in a document's opening text; more than one
  matching library title attaches nothing, and a short title alone never
  attaches a record.
- The cluster–register audit no longer ends a negation at a hyphenated word
  ("no k-means merge" is not a merge), reads negated lists and "merges: none",
  accepts cluster ranges and Oxford-comma lists, pairs a stated divergence
  rate with the corpus named in the same clause, and notes statements it
  cannot pair or reversed ranges.
- Unquoted YAML dates in a configuration are read as text instead of stopping
  the run at the manifest step.
- Run manifests record distributions without a name instead of failing;
  package comparison uses normalized names and reports duplicate installs.
- OpenAI-compatible replies that finish with `finish_reason` None, "eos",
  "eos_token", "end_turn" or "stop_sequence" are accepted.
- An embedding model loaded from `SENTENCE_TRANSFORMERS_HOME` is resolved
  there; an unresolvable snapshot is recorded as unresolved.

### Input and retrieval validation

- Refuse duplicate retrieval IDs and repeated pages instead of overwriting hits;
  reject invalid paging limits before contacting the database.
- Refresh JSONL record metadata and changed PDF/record chunk settings. Validate
  stored chunk counts and positions before reusing a source.
- Reject duplicate record source IDs before ingestion and reject collisions
  between PDF roots before changing stored sources.
- Allow `--prune-missing` when the final PDF has been removed from an existing
  library directory; nonexistent directories still fail.
- Apply the same Unicode normalization and duplicate-draft checks to supplied
  text as to file-loaded drafts.
- Exclude physical `pdf_root` locations from input/sample hashes so moving
  identical source data does not falsely change the analysis fingerprint.
  Bibliographic metadata and relative source identities remain part of the hash.


### 2026-10-18

**Re-ingest required.** Sentence segmentation changed, so every existing
corpus must be re-ingested once: rerun your usual
`cda ingest --pdf-dir ... [--zotero-csv ...]` command. Ingest detects the
old chunks and rewrites them; analysis refuses chunks segmented by another
segmenter until you do. The toy demo store refreshes itself the same way.

#### Added

- `cda ingest --prune-missing` removes stored sources whose PDF is no longer
  under `--pdf-dir`. Without it, such sources are listed and kept.
- `cda analyze --overwrite-fixtures`: with `--record-fixtures`, replaces an
  existing replayable fixture for the same prompt.
- `python -m cda.provenance RECEIPT --compare REFERENCE --strict-environment`
  also requires identical platform, Python build, package versions and
  embedding device.
- Every run saves the model's raw reply as `*_llm_response.txt` before any
  post-processing; `cda verify` compares against it when present.
- Dependencies: `pysbd==0.3.4` and `docx2python==3.7.1`.

#### Changed

- Sentences are split with pysbd 0.3.4, which handles quotations,
  "et al.", "Fig." and "U.S." correctly. Each chunk records the segmenter
  (`sentence_segmenter`).
- Re-ingest refreshes a source when its PDF changed (file hash), the Zotero
  CSV changed (hash), a JSONL record's text changed, or the sentence
  segmenter changed. Unchanged sources are still skipped.
- Corpus retrieval uses a case-insensitive Chroma `$regex` match on the
  literal term, so every letter case is found; whole-word extraction follows.
- Drafts are read in full (no size or page cap). `.docx` drafts are read with
  docx2python: body text including tables and text boxes, plus footnotes and
  endnotes; headers, footers and comments are skipped.
- MyWork classification is a set of plain functions (`mywork_classifier`,
  `load_mywork_rules`, `classify_source`). `filename_patterns` match the file
  name or its stem. Title matching ignores English stop words
  (scikit-learn's list) and words under three letters.
- All acquisition dates (Zotero "Date Added" and filesystem dates) are stored
  in UTC, and `--added-from` / `--added-to` are read as UTC dates.
- The embedding model is pinned to a snapshot:
  `embedding.revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The run
  manifest records the resolved snapshot and its file hashes.
- Spherical k-means is seeded with scikit-learn's k-means++
  (`sklearn.cluster.kmeans_plusplus`); each (seed, k, restart) draws from its
  own random stream. The test suite checks the result against scikit-learn's
  `KMeans`. The toy demo's selected k changes as a result (novelty: k=2).
- Fixed-k and automatically selected partitions are scored on the same
  silhouette evaluation sample. An undefined silhouette is recorded as null
  instead of -1. When `max_k` or a fixed k is capped at n-1, diagnostics
  record `k_clipped_from` and `k_clipped_to`. Final cluster labels are always
  numbered 0..k-1.
- Divergence flags compare exact count ratios against the thresholds;
  shares are rounded for display only.
- The cluster–register audit accepts cluster lists such as
  "Clusters 0 and 3" and "C2", notes register-table rows it dropped or found
  twice, limits a negation to its own clause, and cross-checks the per-corpus
  divergence rate the model states. The rate's denominator is the number of
  declared registers.
- Quotation grounding matches whole words, handles `[...]` elisions, and
  lists quote pairs with ambiguous nesting as "ambiguous nesting, not
  checked"; they count as unverified.
- Run manifests and receipts use schema version 2. The comparison's config
  hash excludes `paths.*`, and platform, Python, package or device
  differences are reported as environment differences rather than failing
  the comparison (see `--strict-environment`).
- The independent-mode passage map uses separate palettes and labels for
  the two partitions (R0.., L0..).
- Validation: the first curation edit or recut requires a blind verdict on
  every cluster of the original clustering. Readings and per-reader agreement
  appear in the report only after their reveal is logged, and asking the
  revision agent for advice logs a reveal. Curation refuses to run when the
  configured embedding model or revision differs from the session's.
- Validation curation silhouettes use scikit-learn's `silhouette_samples`
  (a singleton scores 0).
- `cda diagnose`: venue shares and the Herfindahl–Hirschman index are
  computed over works with a known venue, and works without one are counted
  separately; the counts line includes works sharing a DOI; the "untouched
  areas" check considers all of a work's topics; the median year is the
  statistical median.
- GUI: the Validate tab lives in `gui/validate.py`, shared helpers in
  `gui/shared.py`; the interactive passage map uses scikit-learn PCA.
- CI runs with a read-only token and tests on Python 3.11 and 3.12; the
  record-fixtures workflow passes its input through an environment variable.
  The Docker image is based on `python:3.11-slim` and bakes in the pinned
  embedding revision.
- Demo: new `novelty` and `consensus` responses
  (`c093845b1deb6ec8.json`, `d31386a610efdb61.json`) written by Claude in a
  Claude Code session against the current prompts. They are not API
  recordings or manuscript results. The synthetic validation reading
  fixtures were regenerated (four files).

#### Fixed

- `retrieval.window_mode: context` no longer fails where two overlapping
  chunks give the same sentence different neighbors.
- Small `corpus.chunk_overlap_words` values no longer drop the end of a
  document; every word lands in at least one chunk.
- Year inference from file names and metadata requires digit boundaries, so
  a number such as `120195` no longer yields a year.
- A filename, DOI or title (its first 40 normalized characters) shared by
  distinct Zotero records attaches no metadata; the reason is recorded (`zotero_unmatched_reason`) and
  printed. When the filename is ambiguous, DOI and title matching are still
  tried.
- Impossible dates (in the Zotero CSV or in a timeframe bound) are rejected.
- Identical drafts are refused, and a draft that cannot be parsed stops the
  run instead of being skipped.
- A MyWork title that can never reach `title_match_min_overlap`, and invalid
  numeric MyWork settings, are config errors at load time.
- Reports of runs with saturation disabled state the true literature count,
  and no longer say "This is the default."
- Live providers refuse replies that did not finish normally (for example a
  refusal), not only replies cut off at `llm.max_tokens`.
- Fixtures recorded for `openai:` model specs are keyed with the provider
  prefix, so they never collide with an Anthropic fixture for the same name.
- `cda analyze --record-fixtures` refuses to replace a replayable fixture
  unless `--overwrite-fixtures` is given.
- The bootstrap benchmark computes Hennig's Jaccard on the points both
  clusterings share and shows "n/a" for clusters that were never scored; the
  order-stability benchmark's share table excludes the canonical trial.
- The emergence report and chart name the filesystem clock when
  `--date-source filesystem` is used.
- OpenAlex error messages never include the `OPENALEX_MAILTO` address.
- `cda validate --session` with a missing directory, or a session locked by
  another process, gives a one-line error.
- A mistyped `cda diagnose --fields` name is an error that lists the valid
  field names.

#### Removed

- `python-docx` (replaced by docx2python).
- The `MyWorkResolver` class (replaced by the functions above).

### 2026-10-18

#### Changed

- The `novelty` and `consensus` demo responses are now written by Claude in a
  Claude Code session (replay model `claude-code-session`) rather than
  recorded from API calls; reports name this provenance.
- **Re-ingestion required for older corpora:** repeated sentences are kept as
  distinct occurrences at their source positions, and only chunk overlap is
  deduplicated (and counted).

#### Added

- Saved analyses record code, input, sample, configuration, environment and
  model provenance, plus output integrity checksums, with an exact comparator
  for the deterministic stages.

#### Fixed

- Missing, mismatched or unusable passage and term embeddings now fail instead
  of silently shrinking the sample or switching the ranking rule.
- Cosine silhouette and ARI are computed with scikit-learn, including for
  degenerate vectors; spherical k-means itself is unchanged.
- Reusing a run basename with conflicting content fails before any output is
  written.

### 2026-08-18

#### Fixed

- Silhouette-based selection of k evaluates every candidate on one fixed,
  seeded passage sample, scores singletons as zero, and applies the
  minimum-size merge before scoring, so the reported k and silhouette describe
  the same partition.
- The algorithm-comparison benchmark (spherical k-means, k-means,
  agglomerative) fits all methods on the same normalized matrix, with the same
  small-cluster rule and the same silhouette sample.

### Earlier changes

#### Added

- Every analysis writes a passage map (`<report>.png`): all clustered passages
  in one PCA plane, literature as dots and the researcher's as stars, coloured
  by shared cluster. Configurable via `figure.enabled` and `figure.dpi`;
  shown in the GUI with a download button; skipped by `--dry-run`. Adds
  `matplotlib` as a dependency.
- Every report includes a "Clustering Stability (computed)" section: the same
  passages are re-clustered under shuffled orderings to show whether k
  recurs. Configured by `clustering.stability_check`; no LLM calls; never
  changes the prompt; skipped by `--dry-run` and when `clustering.k` or
  `clustering.method` is overridden.
- Resolution controls `clustering.k` (`auto` or a fixed number) and
  `clustering.method` (`kmeans` or deterministic `hierarchical`); defaults
  unchanged.
- `saturation.scope` (`sample` or `prompt`); reports and prompts state which
  scope produced them.
- `cda ingest --no-recursive` and repeatable `--exclude PATTERN` (also
  `corpus.recursive` and `corpus.exclude`); the run reports how many files
  each rule skipped.
- Zotero metadata matching falls back to the DOI and then the title found in
  the PDF when filenames do not match; ambiguous matches attach nothing, and
  each match records how it was made (`zotero_match`).
- Corpus timeframe: `--added-from` / `--added-to` restrict an analysis to works
  added to the library in a period (by acquisition date, not publication
  year), for `cda analyze`, `cda emergence`, `cda validate`, `cda diagnose`
  and the GUI. Undated passages are dropped and counted unless
  `--include-undated` is given; drafts are never filtered. Off by default.
- A second acquisition clock, `date_added_fs` (when the file arrived on disk),
  selectable with `timeframe.date_source` / `--date-source`; reports name the
  clock used.
- Ingest stores Zotero's "Date Added" as `date_added`. Older corpora need one
  `cda ingest --pdf-dir PATH --zotero-csv EXPORT.csv --reingest` pass for
  timeframe and emergence features.
- `llm.model` accepts a `provider:model` spec (for example
  `openai:gpt-4o` with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`); a bare
  model name still means Anthropic. Unknown provider prefixes are refused.
- `cda validate --term X` and a GUI Validate tab (experimental): record a blind
  verdict per cluster before an LLM's reading is revealed, then see agreement
  (percent and Cohen's kappa) and a full verdict and curation trail. Merge and
  split curation stays inside the session.
- Dendrogram recut in the GUI Validate tab for joint hierarchical runs:
  re-cut the same tree at a chosen k locally; unchanged clusters keep their
  verdicts.
- `cda emergence --term X` (experimental, no LLM calls): tracks when each
  literature cluster appeared as the library grew, with a matching GUI panel.
- Public demonstration mode for the GUI (`CDA_DEMO_MODE=1`): only the
  replayed toy-corpus analysis and a dry-run preview of pasted text, with no
  uploads, API spend, or external calls.
- The GUI sidebar accepts the Anthropic API key, held only in browser-session
  memory.
- `stability_protocol.py --save-replies` saves each run's raw reply beside the
  summary (off by default).
- Benchmarks: `clustering_order_stability.py` (detailed order-stability
  report), `bootstrap_stability.py` (stability under resampling of the
  personal library), and `saturation_diagnostics.py` (probes explaining a
  saturation-sweep outcome).

#### Changed

- Clustering is joint by default: both corpora are pooled into one spherical
  k-means partition, and the report shows where the researcher's passages sit
  in each shared cluster. This departs from the method in the paper;
  `cda analyze --independent` (or `clustering.mode: independent`) reproduces
  it. New keys `clustering.mode` and `clustering.joint_max_k` (15); system
  prompt v3; the GUI adds a shared-cluster map and a composition chart.
- The default `saturation.scope` is `prompt`: the full retrieved literature
  pool is clustered, because subsampling could produce a cluster that only
  looked researcher-specific. Set `saturation.scope: sample` for the paper's
  procedure.
- The order-stability check runs 30 shuffled orderings by default
  (`clustering.stability_check.trials: 31`, up from 5). This adds local
  clustering time on large corpora; lower it while iterating.
- The clustering-comparison benchmark pools both corpora to match the default
  joint analysis; `--per-corpus` restores the separate comparison.
- `cda analyze` defaults `--fixtures-dir` to `<output_dir>/fixtures` instead of
  `demo/fixtures`, so fixtures quoting a real corpus stay out of the repository.

#### Removed

- `--reanalyze-curated` and its GUI expander: a curated partition can never be
  replayed offline, and a report built on hand-edited clusters would read as
  canonical.

#### Fixed

- Clustering results no longer depend on passage order: saturation selection
  returns passages in one canonical order.
- The order-stability check no longer alters the partition it measures, and
  its rate counts only shuffled orderings, so it can report zero; counts in
  the console, report and benchmark now agree.
- Filenames with accents or non-Latin characters no longer match the wrong
  Zotero record; corpus text and search terms are normalized to NFC.
- An ambiguous DOI or title match no longer attaches another paper's
  metadata.
- Zotero acquisition dates keep their timezone. Re-ingest with `--reingest`
  if you rely on the Zotero clock.
- `cda emergence --date-source filesystem` orders the timeline by the same
  clock as the filter; `emergence.start` and `step` below 1 are refused.
- The Docker GUI binds to 127.0.0.1 instead of every network interface.
- The Docker image works for Linux users whose uid is not 1000 (`CDA_UID` and
  `CDA_GID`), and its bind-mount directories ship with the repository.
- Fresh installs on Python 3.11 work again (`scipy` pinned to 1.17.1).
- Live API failures raise an `LLMError` naming likely causes instead of a raw
  SDK traceback.
- The API key, output directory and fixtures directory are checked before
  heavy work; the prompt and diagnostics are saved before the LLM call; a
  reply cut off at `llm.max_tokens` is refused rather than saved as complete.
- `cda validate` builds every reader's provider before extraction, so a
  missing key fails in seconds.
- Interrupted live validation no longer re-bills: completed readings are
  recorded and replayed on rerun, and truncated readings replay as such.
- Validation readings stay hidden until a blind verdict is recorded, and the
  revision agent's advice is gated the same way.
- `cda validate --session DIR` reopens that session instead of starting a new
  one.
- Validation sessions are safe under concurrency: `session.json` is written
  atomically under a lock, "Resume latest" finds same-second sessions, and two
  tabs no longer overwrite each other's verdicts.
- The session layer refuses verdicts for unknown clusters, invalid
  `overlaps_with` values, premature endorsement, one-sided splits, and
  same-cluster merges; corrupt session files are reported with a recovery
  path.
- Fixtures recorded under a `provider:model` spec replay under `--mock-llm`.
- `cda verify --mock-llm` is reachable: `--record-fixtures` on a live verify
  records the second model's reply.
- `cda verify` compares against the system prompt the run actually used,
  matches the term exactly, and picks the latest run by timestamp.
- Result files are written atomically, non-Latin terms get distinct
  filenames, and runs started in the same second no longer collide.
- Config mistakes fail early: misspelled keys get a did-you-mean hint, null
  values mean "use the default", and numeric, list and boolean settings are
  validated before any heavy work.
- Keyword retrieval matches case variants of the term and terms with non-word
  edges such as `C++`.
- Ingest no longer loses data: chunk ids are unique per source path, scanned
  PDFs with no text are named, UTF-16 drafts decode correctly, non-Zotero CSV
  exports are refused with instructions, and stored chunks are verified.
- An interrupted ingest re-ingests partial files on the next run; concurrent
  ingests wait for each other; `corpus.chunk_size_words` below 20 is refused;
  file selection is case-insensitive; an ingest that finds no PDFs is an
  error.
- Unreadable or empty drafts are reported with what to do.
- NaN or zero embeddings raise errors naming the affected chunks.
- Joint partitions with one empty side report flags as unavailable instead
  of spurious findings.
- `cda diagnose` normalizes DOI prefixes and resolver URLs, skips malformed
  DOIs by name, and explains a locked cache; without `OPENALEX_MAILTO` it
  prints a one-line error.
- The quotation-grounding check reports quote marks it could not pair instead
  of an all-clear.
- `--demo` refuses path flags; `--dry-run --record-fixtures` and
  `--session --record-fixtures` are refused; `~` in path flags is expanded;
  replay warns when the current token cap is lower than at recording; a
  negative `clustering.seed` is a config error.
- Error messages that suggested impossible fixes now give ones that work.
- GUI: the Analyze and Validate tabs refuse mock runs with no fixtures before
  heavy work; live runs record replayable fixtures; paid results survive
  their download button and land where the Reports tab lists them; blank path
  fields are refused by name; a sidebar-entered key reaches every live path.
- GUI: missing reports, corrupted stores and failed model downloads show
  explanations instead of tracebacks; the launcher waits for the server.
- GUI on a shared host: session directories stay private and are swept six
  hours after last use; the paste box is capped at 20,000 characters on the
  server; clearing the OpenAlex email forgets it; the retention and network
  notices match actual behavior.
- A demo fixture that cannot replay (for example across CPU architectures)
  explains why.
- `cda validate --demo --mock-llm` runs offline again.
- An interrupted first `cda demo` run completes its toy-corpus store on the
  next run.
- Documentation: the 1,200-passage saturation cap bounds local clustering
  work, not prompt length or cost; `prompt_file: null` means v3; the
  cluster–register audit is described as negation-aware.
- README and quickstart create a virtual environment first, include the
  Windows activation line, and no longer close the shell when Python is
  missing.
- CI runs the offline GUI tests instead of skipping them.

### Initial public release

#### Added

- The Conceptual Divergence Analysis (CDA) pipeline accompanying the paper
  (see `CITATION.cff`): passage extraction with MyWork resolution, spherical
  k-means divergence mapping with saturation-based subsampling, LLM register
  assignment, divergence flags, and timestamped Markdown reports.
- `--mock-llm` fixture mode, an offline test suite, a toy-corpus demo, a
  Streamlit GUI, Docker deployment, OpenAlex corpus diagnostics, and
  clustering and stability benchmarks.
- A GUI visual summary after each run: passage map, cluster sizes, saturation
  sweep, and divergence-flag quadrant.
- A quotation-grounding check in every report that verifies quoted passages
  verbatim against the prompt.
- GUI drag-and-drop for PDFs, a Zotero CSV, and drafts.
- "Try it on your own text": analyze pasted paragraphs against the toy corpus,
  as a free dry run or a priced live run.
- `cda verify` leads with an agreement or discrepancy banner and returns a
  verdict programmatically.
- Documentation recommends corpus diagnosis before a first analysis and
  states scaling limits.
