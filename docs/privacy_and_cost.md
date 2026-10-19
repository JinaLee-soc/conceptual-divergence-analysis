# Privacy and cost

CDA's design philosophy is local, researcher-curated infrastructure. This
page states exactly what leaves your machine, what never does, and what a
run costs. The GUI footer carries the same promise; a test enforces it.

## What leaves your machine

| Data | Destination | When |
|---|---|---|
| Extracted term-sentences (researcher + literature representatives), theme tags, cluster statistics — i.e. the assembled analysis prompt | Configured model provider | live analysis, demo, verification, validation/curation, and stability benchmarks |
| DOIs, topic/field queries, your `OPENALEX_MAILTO` email | OpenAlex API | `cda diagnose` and `demo/build_toy_corpus.py` only (the email is removed from any error message CDA prints) |

## The hosted demo (`CDA_DEMO_MODE=1`)

The public demonstration and the tool you install are deliberately not the
same thing. CDA analyzes curated literature corpora and unpublished drafts —
the material that should not be uploaded to someone else's server — so
the hosted page is a *demonstration*, never an analysis service.

Setting `CDA_DEMO_MODE=1` allows toy fixture replay and pasted-text prompt
previews. Pasting text sends it to the host's server memory. The public preview
does not write that text, its prompt, report, or diagnostics to files and does
not send it to an analysis API. Session memory still contains the input; use a
local install for confidential drafts. The paste limit is 20,000 characters.
Corpus ingestion, path inputs, API keys, live analysis, live validation, and
OpenAlex diagnosis are unavailable in public demo mode.

Mock and dry-run modes do not call analysis APIs. An uncached embedding model
can still require a download; `HF_HUB_OFFLINE=1` prevents that network access.
Live validation, curation, verification and benchmarks also send their prompts
to the configured reader/provider, not only the main analysis call.

Two things to understand about the prompt:

- It contains verbatim sentences from your corpus and drafts — including
  **source file names**, and draft file names are often working-paper
  titles. If that is sensitive, rename draft files before analyzing.
- You never have to send it blind: `--dry-run` prints the entire prompt
  first, and every live run saves the exact prompt beside the report
  (`drift_<term>_<timestamp>_prompt.md`). See the
  [interpretation guide](interpretation_guide.md#the-inspect-before-send-workflow).

Anthropic's API handles the prompt under its commercial data terms; consult
your institution's policy if manuscripts are under embargo-like constraints
— or run `--dry-run`, review, and decide per term.

## Full confidentiality: running the analysis model locally

The table above is the honest boundary of the default configuration: your
PDFs, vector store, and reports never leave, but the **extracted
term-sentences do**, inside the analysis prompt, whenever you run live
against the Anthropic API. For most academic literature that is an acceptable
trade. For an unpublished manuscript under embargo, an IRB-constrained
corpus, or an industry collaboration under NDA, it may not be.

For those cases the analysis model is swappable, so that *nothing* leaves the
machine at any stage. `llm.model` accepts a `provider:model` spec pointing at
any OpenAI-compatible endpoint, which includes local servers such as Ollama,
LM Studio, and vLLM:

```yaml
llm:
  model: "openai:llama3.1:70b"   # served locally
```

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1   # e.g. Ollama
export OPENAI_API_KEY=unused                       # required but ignored locally
```

With a cached embedding model and a local endpoint, those stages run on your
hardware. Configure validation and verifier reader models locally too; model
downloads and `cda diagnose` (OpenAlex) remain separate network operations. The register labels
will differ from the paper's, since they come from a different model; the
deterministic parts of the pipeline (retrieval, clustering, the flags, the
audit) are unaffected.

## What never leaves

- Your **PDFs** — text extraction and chunking are local (PyMuPDF).
- The **vector store** — ChromaDB runs embedded, on disk, in
  `<corpus_dir>/chroma`; its anonymized telemetry is explicitly disabled in
  code.
- The **embedding model runs locally** — `all-MiniLM-L6-v2` is downloaded
  once from Hugging Face (~90 MB) and computes on your machine; passages are
  never sent to an embedding API.
- **Reports, prompts, diagnostics, raw model replies, run manifests** —
  written to your output directory only.
- **GUI drag-and-drop and pasted text** — a browser "upload" here is the
  browser handing the file to the local CDA process on the same machine;
  dropped PDFs/CSVs/drafts are saved under your corpus/drafts directories.
  Pasted text in the try-your-own-text box leaves the machine only if you
  press the clearly-priced *live* button (inside the analysis prompt, like
  any live run — preview it first with the free dry-run button).
- Your **Zotero library** — read once from a CSV export on disk; CDA never
  connects to Zotero or its API.
- **Recorded fixtures** (`--record-fixtures`; the GUI's live Analyze,
  Validate, and demo runs record automatically so a later mock run can
  replay them) — written locally: `<output_dir>/fixtures` for `cda analyze`
  (your git-ignored output directory) and `demo/fixtures` for the demo. A
  fixture is keyed by the configured model and the exact prompt, so a
  recording never touches the committed `claude-code-session` demo
  fixtures. When a replayable fixture already exists for the same model and
  prompt, the command-line `--record-fixtures` refuses before the paid call
  unless you add `--overwrite-fixtures`; the GUI's live Analyze and demo
  buttons replace it (a deliberate live rerun supersedes your earlier
  recording). A mock demo replays the recording made under your configured
  `llm.model` when one exists and otherwise the committed
  `claude-code-session` fixture; the report header names the model whose
  response was replayed. A fixture contains the model's reply, which quotes
  corpus passages — treat fixtures from a real corpus like reports: keep
  them out of public repositories.

## Cost and runtime

| Operation | Network | Cost | Time |
|---|---|---|---|
| `cda demo --term … --mock-llm` | none | $0 | seconds (+ one-time ~90 MB model download) |
| `cda ingest` | none | $0 | minutes per few hundred PDFs (local embedding) |
| `cda analyze --term …` (live) | Anthropic | **~$0.10–0.15 per term** | **~2–3 minutes** |
| `cda analyze --term … --dry-run` | none | $0 | seconds–minute |
| `cda diagnose` | OpenAlex (free API) | $0 | seconds; responses cached in SQLite, re-runs are offline |
| `cda verify --term …` (live) | Anthropic or configured endpoint | ≈ one analysis | ~1–2 minutes |
| `cda emergence --term …` | none | $0 | ~1 minute (local clustering only) |
| `cda validate --term …` (live readings) | Anthropic or configured endpoint | **a few cents** (one short call per cluster per reader) | ~1 minute |
| validation curation (per suggestion, live) | Anthropic or configured endpoint | a few cents (1–2 short calls) | seconds |
| `benchmarks/stability_protocol.py` (live, 5 runs) | Anthropic | ~$0.50–0.75 | ~10 minutes |
| `benchmarks/clustering_comparison.py` | none | $0 | ~1 minute |

Per-term cost stays in this range even on large corpora, and not because of
the saturation cap. The prompt carries **cluster summaries**, not passages:
each cluster contributes its size and its five centroid-nearest
representatives, so prompt length is bounded by the number of clusters
(`clustering.joint_max_k`, default 15) rather than by the size of the
retrieved pool. Saturation subsampling (`scope: sample`, not the default) bounds the *local*
clustering work, not the API bill; with it off, a larger retrieved set means a
longer local k-means sweep, while the LLM call costs about the same.

## Keys and configuration

Two ways to supply your API key, neither of which can end up in version
control:

- **GUI (recommended):** enter the key in the sidebar's password field. It
  is kept only in that browser session's memory — never written to disk,
  exposed to the API client only while a run is executing and scrubbed
  immediately after — and it disappears when you close the tab (or press
  "Forget the key now").
- **Environment variable** (needed for the CLI): `ANTHROPIC_API_KEY`, via
  your shell or a git-ignored `.env` file (see
  [.env.example](../.env.example)).

Keys are never read from config files, and the secret scan in pre-commit and
CI guards the repository side.
