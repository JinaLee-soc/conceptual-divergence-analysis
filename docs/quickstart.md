# Quickstart

From zero to a rendered divergence report, then to your first real analysis.
Every command here is copy-pasteable; nothing in steps 1-3 needs an API key.
Network is needed for package installation and, outside Docker, the first
embedding-model download.

## 1. Clone and install (≈2–5 minutes)

```bash
git clone https://github.com/JinaLee-soc/conceptual-divergence-analysis cda
cd cda
PYTHON=$(command -v python3.13 || command -v python3.12 || command -v python3.11) \
  || echo "Install Python 3.11, 3.12, or 3.13 first"
"$PYTHON" -m venv .venv && source .venv/bin/activate
pip install -e .
```

(On Windows: `py -3.12 -m venv .venv`, then `.venv\Scripts\activate` instead
of the `source` line.)

(Prefer containers? `docker compose up`, open http://localhost:8501, click
**Try the demo**, and skip to step 3's reading guidance. Two Docker-specific
notes. The container sees only the mounted `corpus/`, `drafts/`, and
`outputs/` folders — so when you later ingest your own PDFs from the GUI,
put them under `corpus/` first; any other host path is invisible inside the
container and fails with "PDF directory not found". And on Linux, if writing
one of those folders fails with a permission error, the container is writing
as uid 1000 and your account is not; start it as yourself instead:
`CDA_UID=$(id -u) CDA_GID=$(id -g) docker compose up`.)

## 2. Run the demo (≈1 minute; first run downloads the ~90 MB embedding model)

Prefer a GUI on macOS? Double-click `CDA GUI Demo.command` in the repository
folder. It creates `.venv` if needed, installs `.[gui]`, opens the local
Streamlit app in your browser, and lets you click **Try the demo**.

```bash
cda demo --term novelty --mock-llm
```

What happens: the committed toy corpus (144 CC0 abstracts + 4 fictional
researcher drafts) is indexed into a local vector store, the full pipeline
runs — literal term retrieval, MyWork resolution, spherical k-means,
register assignment — and the LLM step replays a committed
demonstration response (written by Claude in a Claude Code session, not an
API recording), so this is free and makes no Anthropic API call. The one-time model
download may print a Hugging Face warning about unauthenticated requests —
that is normal; no account is needed. If you need to run
with no network after the embedding model is cached, prefix the command with
`HF_HUB_OFFLINE=1`.

The command prints the report path
(`demo/_runtime/outputs/drift_novelty_<date>.md`). Open it.

## 3. Read the report

- **Summary** — how many senses, and whether researcher usage diverges.
- **Polysemy Map** — the researcher's usage registers with example sentences.
- **Researcher vs. Literature Divergence** — the heart of CDA: which
  literature registers exist and where the researcher's usage sits.
- **Computed Divergence Flags** — the >80%/<10% concentration table.
- **Quotation Grounding** — every passage the LLM quotes, verified verbatim
  against the prompt it was given (the computed hallucination guard).
- **Clustering Stability** — how often the same k recurs across 30 shuffled
  orderings of the same passages.
- **Run Parameters** — everything needed to reproduce the run.

Your report should match [../demo/expected_reports/](../demo/expected_reports/)
apart from the timestamp. The demo divergence is designed to be visible: the
fictional researcher treats *novelty* as "audience repair," while the
literature treats it as a property of ideas. A second example,
`cda demo --term consensus --mock-llm`, can assemble a slightly different
prompt on another CPU architecture; if it reports a missing fixture, record
your own once with a live run, `cda demo --term consensus --record-fixtures`
(needs `ANTHROPIC_API_KEY`, ~$0.10–0.15). Later mock runs replay that
recording, made under your configured `llm.model`, before the committed
`claude-code-session` fixture, and the report header names the model
replayed (see [demo/fixtures/README.md](../demo/fixtures/README.md)). Or
just use the novelty demo above.

Prefer clicking to typing from a terminal? `streamlit run gui/app.py`
(install with `pip install -e ".[gui]"`) gives the same demo as a button,
drag-and-drop ingestion, result charts — and a "**try it on your own text**"
box where you paste a few paragraphs and see them analyzed against the toy
literature (free prompt preview; live run shows its cost first).

## 4. See exactly what would be sent (still free)

```bash
cda demo --term novelty --dry-run
```

This assembles and prints the complete system + user prompt without sending
anything — the same inspect-before-send workflow you will use on your own
data (`cda analyze --term X --dry-run`).

## 5. Your first real analysis

1. Build a corpus from your own library —
   follow [corpus_setup_zotero.md](corpus_setup_zotero.md):

   ```bash
   cda ingest --pdf-dir ~/my_papers --zotero-csv ~/zotero_export.csv
   ```

   One placement rule: keep the resulting store (`corpus/chroma`) on a
   local, non-synced disk — a cloud-sync client (Dropbox, OneDrive, iCloud)
   rewriting its database files mid-ingest can corrupt it. Syncing the
   repository clone is fine; it is the store that must not sync (details in
   the setup guide).

2. **Pre-flight: see what you built.** Before your first analysis, run the
   corpus-composition diagnostic and skim the report:

   ```bash
   cda diagnose --fields "Social Sciences"   # free; needs OPENALEX_MAILTO in .env
   ```

   It shows venue concentration, field coverage, temporal profile, and the
   large areas of your declared fields the corpus doesn't touch — so your
   curation is a deliberate choice, not an accident, before it becomes the
   baseline every divergence is measured against. (Curation itself is the
   method's reference point, not a defect — see the report's own framing.)

3. Put a working draft or two in `./drafts/` (.docx, .md, .txt, .pdf). Each
   is read in full; a file that cannot be read, or two files with identical
   text, stop the run with a message naming the file.

4. Add your API key: `cp .env.example .env`, fill in `ANTHROPIC_API_KEY`
   (https://console.anthropic.com/). Per-term cost is ~$0.10–0.15.

5. Preview, then run:

   ```bash
   cda analyze --term novelty --dry-run   # inspect the prompt, free
   cda analyze --term novelty             # live run, ~2–3 minutes
   ```

   Your passages and the literature's are clustered together, and the report
   shows where yours sit in each shared cluster. Two flags are worth knowing:

   ```bash
   cda analyze --term novelty --independent          # legacy alternative: cluster each side separately
   cda analyze --term novelty --added-to 2026-05-01  # your library as it stood that day (UTC)
   ```

   If a snapshot looks wrong, check the clock behind it: a rebuilt Zotero
   library stamps every item with the import date, so add
   `--date-source filesystem` to cut on when the PDFs arrived on disk instead.

6. Reports accumulate in `./outputs/` — re-run after revising a manuscript
   and compare (that's the longitudinal design). For a comparison where only
   your manuscript changed, pin the corpus with the same `--added-to` on both
   runs; otherwise the literature side moved too.

Next: [interpretation_guide.md](interpretation_guide.md) before drawing
conclusions, and [privacy_and_cost.md](privacy_and_cost.md) for what leaves
your machine (short version: passages inside prompts, nothing else).


### Updating an existing corpus

After updating CDA, run your usual `cda ingest --pdf-dir ... --zotero-csv ...`
command once. Sentences are split with pysbd 0.3.4 in blocks ending at detected sentence
boundaries (`pysbd-0.3.4+blocks-v3`). Blank lines and block-size limits do not
split a sentence, and the CJK full stops 。！？． end a sentence without a
following space. Every chunk records the segmenter and the boundaries it
produced; chunks from an older version are detected and re-ingested, and
analysis stops with an actionable error until they are. The same command also
refreshes any source whose PDF changed and updates Zotero fields in place when
you supply a new CSV export; unchanged sources are skipped. Keep the same
corpus directory and source PDF root. Sources stored for that root whose PDF
is no longer on disk, or that now fall under `--exclude` or `--no-recursive`,
are listed and kept; add `--prune-missing` to remove them. `--reingest` forces rebuilding every
source. The embedding windows remain 500 words with 100-word overlap;
analysis reads the original complete sentence across their boundaries. A
multiword literal term is also found when it crosses a chunk edge, including
with zero overlap; the full phrase must occur in the stored sentence context.

Retrieval failures stop analysis before any model call. No partial result is
reported as a completed retrieval. Per-sample theme tags come from that
sample's source metadata; missing associations are not filled from the global
theme list.
