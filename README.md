# Conceptual Divergence Analysis (CDA)

[![CI](https://github.com/JinaLee-soc/conceptual-divergence-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/JinaLee-soc/conceptual-divergence-analysis/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python ≥3.11](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](pyproject.toml)

CDA compares how you use theoretical terms in your manuscripts with how those
terms are used in your curated literature corpus. It retrieves every passage
where a term appears literally and distinguishes your writing from the literature.
It then clusters both sets of passages together with spherical k-means and
identifies where your passages fall within each cluster. An LLM names the usage
*registers* and rates their semantic distance. A timestamped Markdown report
flags where your usage diverges from disciplinary convention before peer review.

This repository contains the full pipeline described in the accompanying paper,
including every reported parameter, prompt, and measurement choice, with pinned
versions. Two settings define the comparison:

- **Joint clustering is the default.** Each cluster represents a region of
  meaning that both your writing and the literature can occupy. A cluster containing no
  passages from your writing identifies a literature register you have not taken
  up. This is the joint partition described in the paper.
  `cda analyze --independent` remains a legacy alternative; each report names its mode.
- **`--added-to 2026-05-01`** analyzes the library as it stood on that date,
  using Zotero's "Date Added". This preserves the comparison as you add readings.

Run manifests and exact comparisons are documented in [Reproducibility](docs/reproducibility.md).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current literal-only workflow.
The [workflow figure](docs/workflow_schema.svg) shows an earlier version of the pipeline; it predates the v5 literal-only extraction.

## Try the web demo (no installation needed)

**[Open the CDA web demo](https://jinalee-soc.github.io/conceptual-divergence-analysis/)**

Choose **novelty** or **consensus**, then press **Play** or select a stage to
explore corpus counts, shared clusters, supporting passages, and the saved report.
Download the example reports as Markdown.

This walkthrough presents fixed toy results; it does not run a new analysis or
accept uploads. No API key or installation is required. To run the pipeline,
follow the local demo instructions below. See [web demo documentation](docs/github-pages-demo.md)
for source provenance and build instructions.

## Run the local demo (no API key needed)

On macOS, double-click [CDA GUI Demo.command](CDA%20GUI%20Demo.command) from
this folder to open the Streamlit GUI demo in your browser. The launcher
creates a local `.venv` on first run with Python 3.11, 3.12, or 3.13,
installs the GUI extra, and opens the GUI with the toy-corpus demo button
described below.

```bash
git clone https://github.com/JinaLee-soc/conceptual-divergence-analysis cda && cd cda
PYTHON=$(command -v python3.13 || command -v python3.12 || command -v python3.11) \
  || echo "Install Python 3.11, 3.12, or 3.13 first"
"$PYTHON" -m venv .venv && source .venv/bin/activate
pip install -e .
cda demo --term novelty --mock-llm
```

The local demo runs the entire pipeline, including retrieval, MyWork resolution,
clustering, register assignment, divergence flags, and quotation grounding. It
uses a committed CC0 toy corpus of 144 OpenAlex abstracts and four explicitly
fictional researcher drafts. The demo replays a labeled Claude-authored
demonstration response, so it is free and does not call an LLM.

The first non-Docker run downloads the embedding model once. A Hugging Face
warning about unauthenticated requests is normal; no account is needed. After
the model is cached, set `HF_HUB_OFFLINE=1` for fully disconnected demo runs.
Compare your output with [demo/expected_reports/](demo/expected_reports/).
The GUI also lets you paste a few paragraphs of your own writing and analyze
them against the toy literature without setting up a corpus.

## Install

| Path | Command | Notes |
|---|---|---|
| **Docker** (GUI, one command) | `docker compose up` → http://localhost:8501 | embedding model included; your corpus/drafts/reports are stored in mounted folders |
| **pip** | `pip install "git+https://github.com/JinaLee-soc/conceptual-divergence-analysis"` | CLI only; add `[gui]` extra for Streamlit |
| **source** | `git clone … && pip install -e ".[dev]"` | Python 3.11-3.13; `pytest` runs fully offline |

Live analysis requires an Anthropic API key (copy [.env.example](.env.example)
to `.env`). A single-term run takes ~2–3 minutes and costs ~$0.10–0.15.

**Docker on Linux.** The container writes the mounted `corpus/`, `drafts/`,
and `outputs/` folders as uid 1000, which is the first user account on a
typical personal machine but not on a shared or corporate one. If a step
fails with a permission error on one of those folders, run the container with your user and group IDs:

```bash
CDA_UID=$(id -u) CDA_GID=$(id -g) docker compose up
```

Docker Desktop on macOS and Windows translates ownership automatically. This
step is needed only on Linux, though it is harmless elsewhere.

**Docker and your own files.** The container sees only the mounted `corpus/`,
`drafts/`, and `outputs/` folders. A host path typed into the GUI's Ingest
tab (for example, `/home/me/papers`) is unavailable inside the container unless
it is mounted. An unmounted path returns "PDF directory not found" even when
the folder exists on your machine. Copy the PDFs under `corpus/` first (or add the folder as another
line under `volumes:` in [docker-compose.yml](docker-compose.yml)) and ingest
from that path. The GUI is reachable at http://localhost:8501 from this
machine only. The port is deliberately published on localhost because anyone
who can reach the GUI could run live analyses on your API key.

## Analyze your own research

1. **Build your corpus** — [docs/corpus_setup_zotero.md](docs/corpus_setup_zotero.md):
   existing Zotero library + PDF folder → local vector store
   (`cda ingest --pdf-dir … --zotero-csv …`). A PDF folder also works without Zotero.
   Keep the store (`corpus/chroma`) on a local, non-synced disk: a cloud-sync
   client (Dropbox, OneDrive, iCloud) rewriting its database files mid-ingest
   can corrupt it. The repository clone can remain in a synced folder if
   `paths.corpus_dir` points to a local, non-synced location.
   When upgrading from an earlier version, rerun `cda ingest` once.
   Sentence segmentation changed, and analysis refuses chunks from the old
   segmenter ([details](docs/quickstart.md#updating-an-existing-corpus)).
2. **Check corpus coverage** — `cda diagnose --fields "Social Sciences"`
   (free) shows venue concentration, field coverage, and untouched areas.
   These describe the baseline against which CDA measures divergence.
3. **Add your drafts** — put working papers (.docx/.md/.txt/.pdf) in
   `./drafts`; they are read in full and always treated as your own writing.
4. **Analyze** — `cda analyze --term novelty`, or use the GUI. Inspect any
   prompt before sending with `--dry-run`.
5. **Read the report** — [docs/interpretation_guide.md](docs/interpretation_guide.md)
   explains silhouettes, register boundaries, the divergence flags, and when
   to overrule the LLM.

## Documentation

| | |
|---|---|
| [Glossary](docs/glossary.md) | registers, validation, verification, and other CDA terms |
| [Quickstart](docs/quickstart.md) | clone → demo → first real analysis |
| [Corpus setup with Zotero](docs/corpus_setup_zotero.md) | the setup guide from the paper |
| [Interpretation guide](docs/interpretation_guide.md) | reading silhouettes, registers, flags; human-in-the-loop practices |
| [Privacy & cost](docs/privacy_and_cost.md) | what leaves your machine and what remains local |
| [Workflow schema](docs/workflow_schema.svg) | earlier workflow figure (predates v5) ([Mermaid source](docs/workflow_schema.mmd)) |
| [Tool card](docs/tool_card.md) | intended use, scope conditions, limitations, what CDA is *not* |
| [FAQ](docs/faq.md) | Zotero-free use, corpus size, saturation, offline mode |
| [Benchmarks](benchmarks/README.md) | clustering comparison & run-to-run stability protocols |
| [Architecture](ARCHITECTURE.md) | data flow, packages, and connections across the complete pipeline |

## For adopters

The paper treats independent adoption as its generalizability test. If you use
CDA in your own research program, please document the experience, including
failures, in an [adoption report](.github/ISSUE_TEMPLATE/adoption_report.md).

## Citation & license

Code: Apache-2.0 ([LICENSE](LICENSE)). Documentation and prompts: CC BY 4.0.
Toy corpus: CC0. Cite via [CITATION.cff](CITATION.cff).
