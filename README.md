# Conceptual Divergence Analysis (CDA)

[![CI](https://github.com/JinaLee-soc/conceptual-divergence-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/JinaLee-soc/conceptual-divergence-analysis/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python ≥3.11](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](pyproject.toml)

CDA compares how **you** use key theoretical terms in your own manuscripts
against how the same terms are used across **your curated literature
corpus**. It retrieves every passage where a term appears literally, separates your writing from the literature, clusters the two
together with spherical k-means and marks where your passages sit in each
resulting cluster, has an LLM name the usage *registers* and rate their
semantic distance, and writes a timestamped markdown report that flags where
your usage diverges from disciplinary convention — before a reviewer does.
This repository is **the full pipeline** described in the accompanying paper,
not configuration templates: every reported parameter, prompt, and measurement
choice ships here, pinned.

Two knobs are worth knowing before your first real run:

- **Joint clustering is the default**, so a cluster number names one region of
  meaning that both your writing and the literature can occupy — and a cluster
  with none of your passages is a literature register you have not taken up.
  This is the joint partition the paper describes.
  `cda analyze --independent` remains a legacy alternative; each report names its mode.
- **`--added-to 2026-05-01`** analyzes your library as it stood on that date
  (Zotero's "Date Added"), so a result stays reproducible after you keep
  reading.

Run manifests and exact comparisons are documented in [Reproducibility](docs/reproducibility.md).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current literal-only workflow.
The [workflow figure](docs/workflow_schema.svg) shows an earlier version of the pipeline; it predates the v5 literal-only extraction.

## Try it in two minutes (no API key needed)

On macOS, double-click [CDA GUI Demo.command](CDA%20GUI%20Demo.command) from
this folder to open the Streamlit GUI demo in your browser. The launcher
creates a local `.venv` on first run with Python 3.11, 3.12, or 3.13,
installs the GUI extra, and then starts the same toy-corpus demo button
described below.

```bash
git clone https://github.com/JinaLee-soc/conceptual-divergence-analysis cda && cd cda
PYTHON=$(command -v python3.13 || command -v python3.12 || command -v python3.11) \
  || echo "Install Python 3.11, 3.12, or 3.13 first"
"$PYTHON" -m venv .venv && source .venv/bin/activate
pip install -e .
cda demo --term novelty --mock-llm
```

The demo runs the **entire pipeline** — retrieval, MyWork resolution,
clustering, register assignment, divergence flags, quotation grounding — on a
committed CC0 toy corpus (144 OpenAlex abstracts +
four clearly-fictional researcher drafts), replaying an explicitly labeled Claude-authored demonstration response
so it is free and does not call an LLM. The first non-Docker run downloads the
embedding model once (a Hugging Face warning about unauthenticated requests
is normal — no account is needed); after it is cached, set `HF_HUB_OFFLINE=1`
for fully disconnected demo runs. Compare your output against
[demo/expected_reports/](demo/expected_reports/). In the GUI you can also
**paste a few paragraphs of your own writing** and see them analyzed against
the toy literature — no corpus setup needed.

## Install

| Path | Command | Notes |
|---|---|---|
| **Docker** (GUI, one command) | `docker compose up` → http://localhost:8501 | embedding model baked in; your corpus/drafts/reports live in mounted folders |
| **pip** | `pip install "git+https://github.com/JinaLee-soc/conceptual-divergence-analysis"` | CLI only; add `[gui]` extra for Streamlit |
| **source** | `git clone … && pip install -e ".[dev]"` | Python 3.11-3.13; `pytest` runs fully offline |

Live analysis needs an Anthropic API key (copy [.env.example](.env.example)
to `.env`). A single-term run takes ~2–3 minutes and costs ~$0.10–0.15.

**Docker on Linux.** The container writes the mounted `corpus/`, `drafts/`,
and `outputs/` folders as uid 1000, which is the first user account on a
typical personal machine but not on a shared or corporate one. If a step
fails with a permission error on one of those folders, start it as yourself:

```bash
CDA_UID=$(id -u) CDA_GID=$(id -g) docker compose up
```

Docker Desktop on macOS and Windows translates ownership for you, so this is
a Linux-only step (harmless elsewhere).

**Docker and your own files.** The container sees only the mounted `corpus/`,
`drafts/`, and `outputs/` folders. A host path typed into the GUI's Ingest
tab (say, `/home/me/papers`) does not exist inside the container, so it fails
with "PDF directory not found" even though the folder is right there on your
machine. Copy the PDFs under `corpus/` first (or add the folder as another
line under `volumes:` in [docker-compose.yml](docker-compose.yml)) and ingest
from that path. The GUI is reachable at http://localhost:8501 from this
machine only — the port is deliberately published on localhost, because
anyone who can reach it could run live analyses on your API key.

## Use it on your own research

1. **Build your corpus** — [docs/corpus_setup_zotero.md](docs/corpus_setup_zotero.md):
   existing Zotero library + PDF folder → local vector store
   (`cda ingest --pdf-dir … --zotero-csv …`). No Zotero? A PDF folder works.
   Keep the store (`corpus/chroma`) on a local, non-synced disk: a cloud-sync
   client (Dropbox, OneDrive, iCloud) rewriting its database files mid-ingest
   can corrupt it. Syncing the repository clone is fine — if the clone lives
   in a synced folder, just point `paths.corpus_dir` somewhere local.
   Upgrading from an earlier version? Rerun your `cda ingest` command once:
   sentence segmentation changed, and analysis refuses chunks from the old
   segmenter ([details](docs/quickstart.md#updating-an-existing-corpus)).
2. **Pre-flight the corpus** — `cda diagnose --fields "Social Sciences"`
   (free) shows venue concentration, field coverage, and untouched areas, so
   you know the baseline your divergences will be measured against.
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
| [Privacy & cost](docs/privacy_and_cost.md) | exactly what leaves your machine, and what never does |
| [Workflow schema](docs/workflow_schema.svg) | earlier workflow figure (predates v5) ([Mermaid source](docs/workflow_schema.mmd)) |
| [Tool card](docs/tool_card.md) | intended use, scope conditions, limitations, what CDA is *not* |
| [FAQ](docs/faq.md) | Zotero-free use, corpus size, saturation, offline mode |
| [Benchmarks](benchmarks/README.md) | clustering comparison & run-to-run stability protocols |
| [Architecture](ARCHITECTURE.md) | how the complete pipeline is composed — data flow, every package, the seams |

## For adopters

The paper treats independent adoption as its generalizability test. If you
run CDA on your own research program, we would genuinely like to hear how it
went — including failures: please file an
[adoption report](.github/ISSUE_TEMPLATE/adoption_report.md).

## Citation & license

Code: Apache-2.0 ([LICENSE](LICENSE)). Documentation and prompts: CC BY 4.0.
Toy corpus: CC0. Cite via [CITATION.cff](CITATION.cff).
