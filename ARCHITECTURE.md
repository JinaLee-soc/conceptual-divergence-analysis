# CDA — Architecture

How the complete pipeline is composed. This document maps the whole
repository: the design principles it obeys, the end-to-end data flow through
the paper's three stages, every source package and its responsibilities, and
the supporting systems (configuration, the mock/live LLM seam, the demo and
fixtures, the GUI, Docker, diagnostics, benchmarks, and tests).

Audience: anyone who wants to read, extend, or verify the code — reviewers,
adopters, and contributors. For *using* CDA, start at
[docs/quickstart.md](docs/quickstart.md); for *why each parameter has its
value*, see [config/config.example.yaml](config/config.example.yaml).

The installed code lives in `src/cda/`, the Streamlit interface in `gui/`,
and the tests in `tests/`. See the [glossary](docs/glossary.md) for CDA terms.

---

## 1. Design principles (what every file obeys)

These are enforced, not aspirational — each has a mechanism behind it.

1. **The paper's method is the default.** The pipeline implements the
   method the paper describes, and every reported parameter is a config
   default with the paper's value, documented in
   `config/config.example.yaml`.
2. **Two execution modes everywhere.** Every command runs `--mock-llm`
   (fixture replay: zero network, zero cost, used by tests and CI) and live
   (`ANTHROPIC_API_KEY`). All model access funnels through one seam,
   `src/cda/llm/`.
3. **No real corpus data, no secrets.** The only committed corpus is CC0
   (`demo/toy_corpus/`); API keys come from the environment only; a strict
   `.gitignore` and `detect-secrets` guard the boundary.
4. **Boring, readable code.** Short functions, docstrings that explain *why*,
   error messages that state the next step. No clever abstractions, no
   framework magic — plain functions passing dataclasses.
5. **Determinism where possible, honesty where not.** Everything seedable is
   seeded (default 42); the only nondeterministic step is the live LLM call,
   and reports say so.

Two structural consequences worth internalizing before reading the code:

- **The pipeline is a pipeline of plain functions**, not a class hierarchy.
  `analyze_term()` in `src/cda/pipeline.py` calls stage functions in order,
  passing `Passage` and `ClusterSummary` dataclasses between them. There is
  no framework, no dependency injection container, no plugin registry.
- **Embeddings and the LLM are the only two "services," and both are
  injectable.** An `Embedder` protocol and an `LLMProvider` protocol are the
  seams that make the offline test suite and the mock demo possible.

---

## 2. Repository layout

```
conceptual-divergence-analysis/
├── src/cda/                    # the installed package (console script: cda)
│   ├── __init__.py             # __version__
│   ├── types.py                # Passage, ClusterSummary dataclasses
│   ├── config.py               # layered YAML config loader
│   ├── defaults.yaml           # packaged copy of config/defaults.yaml (drift-guarded)
│   ├── prompts/                # packaged copy of prompts/ (drift-guarded)
│   ├── llm/                    # the provider seam
│   │   ├── base.py             #   LLMProvider protocol + prompt_fingerprint
│   │   ├── providers.py        #   AnthropicProvider, OpenAICompatibleProvider, get_provider
│   │   └── mock.py             #   MockProvider (replay), RecordingProvider (capture)
│   ├── corpus/                 # STAGE 0: build & read the corpus
│   │   ├── embeddings.py       #   Embedder protocol + SentenceTransformerEmbedder (pinned snapshot)
│   │   ├── zotero.py           #   Zotero CSV export → per-PDF metadata
│   │   ├── ingest.py           #   PDF dir / JSONL records → ChromaDB; get_collection
│   │   ├── timeframe.py        #   acquisition-date window ("Date Added" snapshot)
│   │   └── drafts.py           #   load .docx/.md/.txt/.pdf drafts at analysis time
│   ├── extraction/             # STAGE 1: passages
│   │   ├── windows.py          #   term-sentence extraction, pysbd sentence segmentation
│   │   ├── retrieval.py        #   literal ($regex) retrieval + build_passages
│   │   ├── mywork.py           #   mywork_classifier (researcher vs literature)
│   │   └── draft_passages.py   #   literal draft term-sentences
│   ├── clustering/             # STAGE 2: divergence mapping
│   │   ├── spherical.py        #   spherical k-means, silhouette k-selection, ARI, merge
│   │   ├── hierarchy.py        #   OPTIONAL: average-linkage tree + movable cut
│   │   ├── joint.py            #   DEFAULT: one partition over both corpora + views
│   │   ├── order_stability.py  #   shuffle-and-recluster: how robust is this run's k?
│   │   ├── saturation.py       #   term-anchored ranking + ARI-plateau subsampling
│   │   └── emergence.py        #   EXPERIMENTAL cluster emergence over acquisition time
│   ├── registers/              # STAGE 3a: register assignment
│   │   ├── prompt_loader.py    #   load the versioned system prompt
│   │   ├── assemble.py         #   build the user prompt (appendix-conformant)
│   │   ├── audit.py            #   deterministic match/split/merge audit
│   │   ├── grounding.py        #   quotation-grounding check (anti-hallucination)
│   │   └── verify.py           #   EXPERIMENTAL multi-model verification
│   ├── divergence/             # STAGE 3b: analysis outputs
│   │   ├── flags.py            #   concentration flags (joint membership | cross-assignment)
│   │   ├── report.py           #   timestamped markdown report rendering
│   │   ├── figure.py           #   passage map: both corpora in one PCA plane (PNG)
│   │   └── emergence.py        #   EXPERIMENTAL: cda emergence orchestration + report
│   ├── validation/             # EXPERIMENTAL: researcher-led clustering validation
│   │   ├── prompt_loader.py    #   named-block loader for validation_tasks_v1.md
│   │   ├── session.py          #   session state; blind-then-reveal enforced here
│   │   ├── agents.py           #   reader + revision agent roles
│   │   ├── curate.py           #   merge/split application (session-only)
│   │   ├── metrics.py          #   verdict agreement (Cohen's kappa), curation deltas
│   │   ├── report.py           #   the validation report
│   │   └── run.py              #   cda validate orchestration
│   ├── diagnose/               # corpus-composition diagnostic
│   │   ├── openalex.py         #   polite-pool client + SQLite cache
│   │   └── report.py           #   venue/field/temporal composition report
│   ├── pipeline.py             # THE ORCHESTRATOR: analyze_term()
│   ├── demo.py                 # cda demo: toy corpus → full pipeline
│   └── cli.py                  # argparse: ingest|analyze|diagnose|emergence|validate|verify|benchmark|demo
├── gui/
│   ├── app.py                  # Streamlit app (tabs + demo button)
│   ├── validate.py             # the Validate tab
│   ├── shared.py               # helpers shared by app.py and validate.py
│   └── visuals.py              # Altair result charts (presentation only)
├── prompts/register_assignment_v5.md   # two-analysis prompt + schematic template (v1–v4 frozen alongside)
├── config/{defaults,config.example}.yaml
├── demo/{build_toy_corpus.py, toy_corpus/, fixtures/, expected_reports/}
├── benchmarks/{clustering_comparison,clustering_order_stability,stability_protocol,
│               bootstrap_stability,saturation_diagnostics}.py + README.md
├── tests/                      # offline-by-default test suite (see §10)
├── docs/                       # user + reference docs
├── Dockerfile, docker-compose.yml, .github/workflows/ci.yml
└── pyproject.toml              # pinned deps, entry point, tool config
```

---

## 3. End-to-end data flow

The whole system is one call — `analyze_term(term, config, ...)` in
`src/cda/pipeline.py` — composed of the paper's three stages. Here is the
exact chain for a single `cda analyze --term novelty` run.

```
cli.main()  →  cli._cmd_analyze()  →  load_config()  →  pipeline.analyze_term()
│
├─ up front: validated_settings(config), output dir check, and (unless
│  --dry-run) get_provider(...)              # a typo or missing key fails before heavy work
│
├─ STAGE 1 — passage extraction  (extraction/)
│   get_collection(config)                     # open the ChromaDB corpus
│   keyword_hits(collection, term)             # case-insensitive $regex query (literal)
│   mywork_classifier(config.section("mywork")) # → classify(file, meta): researcher vs literature
│   build_passages(term, hits, classify, ...)  # → term-sentences as Passage objects
│   load_drafts(drafts_dir) + extract_draft_passages(...)   # drafts = researcher, always
│   passages sorted with a chunk_id tiebreaker # determinism: prompt bytes ≠ DB order
│
├─ STAGE 2 — divergence mapping  (clustering/)
│   rank_literature(literature, term, ...)     # term-embedding anchor ranking
│   saturation_select(ranked, ...)             # optional scope: sample only
│   ── joint mode (default) ──
│   cluster_joint(researcher, literature, ...) # ONE spherical k-means over both
│   corpus_views(joint)                        # → two ClusterSummary views, shared ids
│   ── independent mode (--independent, legacy alternative) ──
│   cluster_passages(researcher, ...)          # spherical k-means + silhouette k
│   cluster_passages(literature, ...)          # a second, separate partition
│   → two ClusterSummary objects (size + 5 centroid-nearest reps per cluster)
│
└─ STAGE 3 — register assignment + divergence analysis
    load_system_prompt(config)                 # registers/prompt_loader
    build_user_prompt(term, summaries, ...)    # registers/assemble (appendix format)
    ── dry-run? print both prompts, write prompt file, STOP ──
    provider.complete(system, user, ...)       # the ONE model call
    write_outputs(..., llm_response_text=...)  # raw reply saved before post-processing
    compute_flags(researcher, literature, ...) # divergence/flags (>80%/<10%)
    audit_report(llm_text, k_r, k_l)           # registers/audit (match/split/merge)
    check_quotations(llm_text, user_prompt)    # registers/grounding (quotes verified)
    render_report(...)                         # divergence/report
    write_outputs(...)                         # drift_<term>_<timestamp>.md (+ prompt, diagnostics, manifest, receipt)
```

Two things make this robust:

- **`--dry-run` never constructs a provider** — it
  assembles and prints the full system + user prompt and writes the prompt
  file, sending nothing. This is the human-in-the-loop review mechanism.
- **Deterministic passage ordering** (the `chunk_id` tiebreaker sort) means
  the same corpus + config always assembles a byte-identical prompt, which is
  what makes fixture replay by prompt hash possible (see §7).

---

## 4. Package reference

### `types.py` — the records that flow between stages
- **`Passage`** — one term-sentence: `chunk_id, source, year, text,
  similarity, metadata, cluster_id, embedding`. `embedding` is filled lazily
  by the clustering stage; `cluster_id` is assigned during clustering.
- **`ClusterSummary`** — one corpus's clustering result: `used, n_total,
  n_clusters, silhouette, note, clusters[], diagnostics`. The `clusters` list
  holds per-cluster size, the centroid-nearest representative texts, their
  sources, and role labels. Under joint clustering it is a *view* of the shared
  partition, so its cluster ids are the joint ones.
- **`JointClustering`** — the shared partition (joint mode only): `used,
  n_clusters, silhouette, note, n_researcher, n_literature, clusters[],
  diagnostics`, where each cluster entry carries its composition and
  corpus-marked representatives. `silhouette` is measured on the pooled
  passages, so it describes the joint partition and not either corpus alone.

Everything else in the pipeline is a function taking and returning these.

### `config.py` — layered configuration
`load_config(path, overrides)` merges three layers, lowest to highest:
packaged `defaults.yaml` → user YAML file → environment
(`CDA_VERIFICATION_MODEL`) → explicit overrides (CLI flags). `Config.get(
"section.key")` reads dotted paths and raises `ConfigError` with a pointer to
`config.example.yaml` on a miss. Secrets never appear here — API keys are
environment-only. See §6 for the packaged-copy drift guard.

### `llm/` — the provider seam (the key abstraction)
One protocol, `LLMProvider`, with a single method:
`complete(system, user, max_tokens, temperature) -> str`. Nothing else about a
provider is visible to the pipeline, which is what makes mock replay and
multi-model verification possible.

- **`base.py`** — the `LLMProvider` protocol and `prompt_fingerprint(model,
  system, user)`: a 16-hex sha256 of the assembled prompt. This fingerprint
  is the fixture key.
- **`providers.py`** — `AnthropicProvider` (live, pinned to `llm.model`, key
  from env, loud error if missing that points to `--mock-llm`);
  `OpenAICompatibleProvider` (EXPERIMENTAL, for `openai:<model>` endpoints);
  `resolve_model_spec("provider:model")`, which both the verification hook and
  `get_provider(config, mock, fixtures_dir)` use — so `llm.model` takes the
  same spec form and a bare name still means Anthropic.
- **`mock.py`** — `MockProvider` replays `demo/fixtures/<fingerprint>.json`;
  on a cache miss it fails **loudly** with the expected key and the two ways
  forward (never returns a wrong-but-plausible answer). `RecordingProvider`
  wraps a live provider and saves its response for replay. The committed toy
  fixtures are authored demonstration responses; their origin is documented
  in [demo/fixtures/README.md](demo/fixtures/README.md).

### `corpus/` — building and reading the corpus (Stage 0)
- **`embeddings.py`** — the `Embedder` protocol (`embed(texts) ->
  list[vectors]`) and `SentenceTransformerEmbedder` (lazy-loads
  `all-MiniLM-L6-v2` at the snapshot pinned by `embedding.revision`). Its
  `provenance()` records the resolved snapshot and hashes its cached files
  for the run manifest, without a network request. The protocol is why tests
  can inject a deterministic fake embedder and never download the model.
- **`zotero.py`** — `load_zotero_csv()` parses a Zotero **library CSV export**
  (no live Zotero connection, no API key); `build_filename_index()` +
  `match_by_filename()` + `record_fields()` resolve title/year/DOI/tags — and the "Date Added"
  acquisition timestamp (`date_added`, feeding `cda emergence`) — per PDF;
  the `mywork` tag is detected here. Zotero metadata is
  resolved at *ingest* time into chunk metadata, so analysis needs no Zotero.
  Filename matching is the primary join, but it fails
  wholesale on libraries whose PDFs were renamed by a download pipeline, so
  `match_by_content()` adds two fallbacks that read identity out of the
  document itself: its **DOI** (tried first), then its **title prefix**. A
  filename, DOI or title key shared by distinct Zotero records maps to no
  record, so nothing is attached for it; when the filename is ambiguous, the
  DOI and title fallbacks are still tried. Every resolved record records
  *how* it matched, in `zotero_match`; an unmatched file records why, in
  `zotero_unmatched_reason`. "Date Added" values are normalized to UTC, and
  an impossible date in the export is an error naming the row.
- **`timeframe.py`** — the acquisition-date window. `timeframe_from_config`
  normalizes `timeframe.added_from` / `added_to` into UTC bounds (stored
  acquisition times are UTC; a bare upper bound is padded to end-of-day, and
  an unparseable or impossible bound *raises* — a silently ignored filter
  would report a snapshot that was never taken);
  `filter_by_date_added` keeps the passages acquired inside it, returning counts
  of what fell outside and what carried no date. Undated passages are dropped
  and counted, drafts are exempt by construction (Stage 1 applies the filter
  before drafts join the pool). This is what makes a result re-obtainable later:
  `--added-to 2026-05-01` analyzes the library as it stood that day.
  `timeframe.date_source` picks which of the two clocks the window reads
  (`zotero` — the paper's — or `filesystem`); they answer different questions
  and can disagree by years, so the choice is explicit and every report names
  the clock it used.
- **`ingest.py`** — `get_collection(config)` opens (or creates) the persistent
  ChromaDB collection with cosine space and telemetry disabled;
  `file_acquisition_date()` records the **second acquisition clock**
  (`date_added_fs`, when the file landed on disk, in UTC) for every ingested PDF, which
  is the only clock available for a corpus with no Zotero export — or one whose
  Zotero library was rebuilt, collapsing every Zotero date onto the rebuild;
  `ingest_pdf_dir()` chunks PDFs into 500/100 word windows (every word lands
  in at least one window), embeds, and stores with Zotero metadata;
  `ingest_records()` does the same for JSONL abstract records (the demo
  path). Complete sources are reused when their content, chunk settings and
  sentence segmenter match. Zotero-only changes update metadata in place.
  Replacements are prepared before writing; handled write failures restore
  the previous rows, and failed recovery stops ingestion. This does not
  provide transaction guarantees across process or machine crashes.
  Missing or excluded sources are listed and kept, or removed with
  `--prune-missing`; directory access errors do not count as absence.
- **`drafts.py`** — `load_drafts(dir)` reads .docx/.md/.txt/.pdf at analysis
  time (never ingested into the corpus), each in full. `.docx` files are read
  with docx2python: body text including tables and text boxes, footnotes and
  endnotes, but not headers, footers or comments. A draft that cannot be
  parsed, or two drafts with identical text, stop the run with a
  `ConfigError`. Drafts are always researcher-authored.

### `extraction/` — passages (Stage 1)
- **`windows.py`** — the unit of analysis. `split_sentences` (pysbd 0.3.4,
  English; boundaries fall only between words and no text is dropped; the
  segmenter is recorded as `SENTENCE_SEGMENTER`), `term_regex` (whole-word),
  `extract_term_occurrences` (every term-sentence occurrence, with optional
  context expansion; no default truncation), `window_settings`, `infer_year`
  (a 19xx/20xx year bounded by non-digits), `safe_term_slug`.
- **`retrieval.py`** — `literal_filter` (a case-insensitive Chroma `$regex`
  on the escaped term, so every letter case matches), `keyword_hits`
  (uncapped by default), and `build_passages(term, hits, classify, config)`.
  Multiword terms use one literal word to find candidate chunks, then require
  the full phrase in the stored sentence context before applying the cap.
  This finds phrases across chunk edges. `build_passages` extracts term-sentences
  from every hit, applies the MyWork classifier, and returns `(researcher, literature, stats)` where `stats`
  carries the per-signal classification counts the report surfaces.
- **`mywork.py`** — plain functions: `load_mywork_rules` validates the
  `mywork` config section, `classify_source` tries four signals in
  configurable priority order (filename patterns → metadata flags → Zotero
  tags → profile titles; first match wins and the reason is logged), and
  `mywork_classifier` binds the two into `classify(filename, metadata)`.
  Filename patterns match the file name or its stem; title matching ignores
  scikit-learn's English stop words and words under three letters, and a
  publication title with too few content words to ever reach
  `title_match_min_overlap` is a config error. Drafts are the fifth signal,
  handled as a separate always-researcher channel.
- **`draft_passages.py`** — literal term-sentences from drafts; no fallback when the term is absent.

Ingestion retains the original full-sentence context for each word chunk.
Retrieval extracts from that context, never from an indexed fragment.
Indexed chunks without it, or segmented by a different segmenter than the
current one, are refused at analysis; ordinary ingestion refreshes those
sources. When two overlapping chunks yield the same sentence occurrence, it
is counted once, keeping the window that contains the other. A failed
retrieval page aborts the run without returning partial hits. Cluster summaries carry observed source tags
for each representative, separately from run-level researcher theme context.

### `clustering/` — divergence mapping (Stage 2)
The mathematical core uses NumPy for the spherical k-means iterations and
scikit-learn for k-means++ seeding (`sklearn.cluster.kmeans_plusplus`),
cosine silhouette and adjusted Rand index.

Two modes share all of that machinery, selected by `clustering.mode`:
**joint** (the default) pools both corpora into one partition; **independent**
(`--independent`) partitions each corpus separately as a legacy alternative. The mode changes only *which matrix* the k-means runs on — the
sweep, seeds, merging, and representative selection are identical, and every
report and prompt states which mode produced it. Joint is the default because
only a shared partition lets one cluster number name the same region of
meaning on both sides.

- **`spherical.py`** — `kmeans_labels` (spherical k-means: L2-normalized,
  cosine-as-dot-product, k-means++ seeding, centroids renormalized after
  every update; restart r for a given k draws from its own stream
  `SeedSequence([seed, k, r])`, so no two (k, restart) pairs share random
  numbers; labels are numbered 0..k-1), `silhouette` (cosine; `None` when
  undefined, never -1), `merge_small_clusters`, `adjusted_rand_index`
  (scikit-learn), `clip_k` (caps a requested k or max_k at n-1 and records
  `k_clipped_from` / `k_clipped_to`), `select_k_by_silhouette` (the k=2..max
  sweep: merge undersized clusters for each candidate, then compare the final
  partitions on one fixed seeded evaluation sample), and `cluster_passages`
  (the top-level entry: hydrate embeddings → normalize → select the
  post-merge partition → pick the 5 centroid-nearest representatives →
  `ClusterSummary`). Singleton sample silhouettes follow the standard
  definition and contribute zero.
- **`divergence/figure.py`** — `draw_passage_map` writes the passage map
  (`<report>.png`): both corpora's clustered passages in one PCA plane
  (numpy SVD), literature as dots and the researcher's as stars. Under joint
  clustering both are coloured by shared cluster; under independent
  clustering each corpus has its own palette and labels (R0.., L0..), since
  the two partitions' ids are unrelated. Deterministic, LLM-free, and
  additive — the pipeline calls it after the report is already on disk and
  swallows any failure.
- **`hierarchy.py`** — the optional resolution machinery.
  `linkage_matrix` builds one deterministic average-linkage tree (cosine);
  `cut_tree_at_k` / `cut_height` turn a chosen cluster count into a cut of
  that tree; `partition_matrix` is the dispatcher `cluster_passages` and
  `cluster_joint` call only when `clustering.method` / `clustering.k` depart
  from the paper's default (the default path never imports this module, or
  scipy). Cuts are pure — no small-cluster merging — so coarse cuts are exact
  unions of fine cuts, which is what lets `cda validate` re-cut the same tree
  at a different resolution (`validation/curate.py::apply_recut`). A fixed k
  (either method) is scored on the same silhouette evaluation sample the
  automatic sweep uses.
- **`joint.py`** — the default mode. `cluster_joint` pools the researcher's
  passages and the literature passages (the full retrieved pool by default;
  the saturation-selected subsample only under `saturation.scope: sample`),
  runs the same
  seeded sweep over the pooled matrix, and returns a `JointClustering`: per
  shared cluster, its composition (`n_researcher` / `n_literature`, the
  researcher's share of membership, the cluster's share of all literature) and
  the centroid-nearest representatives *marked* with the corpus each came from.
  `corpus_views` slices it into the two `ClusterSummary` objects everything
  downstream already consumes — keeping the appendix's prompt format, but with
  **shared cluster ids**, so the same number names the same region on both
  sides and a cluster absent from one side is a finding rather than a
  coincidence of labeling. `composition_note` renders the compact
  `C0 3/41, C1 0/22, …` line the prompt carries.
- **`order_stability.py`** — reproducibility is not robustness: the
  same seeded order always gives the same k, but a *different* order of the
  identical passages can still land on a different one when a term's
  separation is weak. `run_trials` re-clusters the same passage set (trial 0
  = the caller's exact order; trials 1..N-1 = independent shuffles, mode-aware
  — joint mode tracks one k, independent mode tracks the researcher and
  literature corpora separately since they can drift apart) and `summarize`
  reduces that to a match-rate-against-canonical dict. No dependency on
  `cda.pipeline` or `cda.config` (every knob is passed in), so it is shared —
  without a circular import — by `pipeline.py`'s small built-in check (§4,
  `analyze_term`) and by `benchmarks/clustering_order_stability.py`'s deeper,
  standalone one.
- **`saturation.py`** — `rank_literature` ranks candidates by cosine
  similarity to the target term's embedding (the external anchor);
  `saturation_select` grows the sample in increments, computes ARI between
  consecutive increments on the overlapping prefix, and declares saturation
  at the *earlier* of two consecutive increments ≥ 0.8 (conservative), capped
  at 1,200, flagging when the cap is hit without saturation.
- **`emergence.py`** — EXPERIMENTAL, post-paper. `track_emergence` orders the
  literature passages by Zotero's "Date Added" (the researcher's acquisition
  trajectory — never publication year), clusters cumulative count-based
  snapshots with the same seeded k-means (reusing the saturation sweep's
  prefix-ARI stability measure), and greedily matches each snapshot's
  clusters one-to-one to the final clustering by centroid cosine
  (`emergence.match_threshold`) — yielding per final cluster a birth point,
  a growth curve, and a persistence score. Passages without a `date_added`
  are excluded and counted, never guessed.

### `registers/` — register assignment (Stage 3a)
- **`prompt_loader.py`** — loads the system prompt (the first fenced block of
  `prompts/register_assignment_v5.md`) — never inlined in code, so the prompt
  that shaped a report is always versioned and inspectable.
- **`assemble.py`** — `build_user_prompt` assembles exactly the payload the
  paper's appendix documents: target term, researcher theme-tag context,
  extraction notes (with the saturation line), and both cluster structures
  with their 5 centroid-nearest representatives.
- **`audit.py`** — `audit_report` computes the match/split/merge divergence
  rate from the report text by a documented heuristic (labeled as such;
  reports n/a rather than fabricating a number). It reads only the structured
  per-register mapping declarations or a "clusters absorbed" register table
  (cluster lists such as "Clusters 0 and 3" or "C2" are accepted; dropped or
  duplicate table rows are noted). A negation applies only within its own
  clause. The rate is (merges + splits) / declared registers, and the
  per-corpus rate the model itself states is used only as a cross-check —
  a disagreement is noted, never adopted.
- **`grounding.py`** — the computed anti-hallucination guard:
  `check_quotations` verifies every ≥6-word quotation in the LLM body
  verbatim (typography-normalized; quotes spliced with `...`, `…` or `[...]`
  checked fragment by fragment, in order, on whole-word boundaries, within
  one passage) against the saved prompt. A quote pair whose nesting is
  ambiguous (a mark touching a word, as when a quoted passage contains its
  own quotes) is listed as "ambiguous nesting, not checked" and counts as
  unverified. The score and any unverified spans render as the report's
  "Quotation Grounding (computed)" section — a quote the model was never
  shown cannot be evidence, and unverified spans are exactly where a human
  should open the saved `_prompt.md`.
- **`verify.py`** — EXPERIMENTAL. `run_verify` re-runs register assignment on
  the *identical* saved prompt with a second model and writes an agreement
  table, led by an unmissable `STRUCTURAL AGREEMENT` / `⚠️ DISCREPANCY`
  banner (`verification_verdict`, also returned programmatically).
  Disagreement is surfaced, not adjudicated.

### `divergence/` — analysis outputs (Stage 3b)
- **`flags.py`** — the paper's flagging (>80% researcher share of a cluster
  holding <10% of the literature → researcher-concentrated; ≈0 researcher
  passages → underrepresented), from either of two bases.
  `compute_joint_flags` (joint mode) counts the shared cluster's actual
  members — §3.3's wording taken literally. `compute_flags` (independent mode)
  first cross-assigns each researcher passage to its nearest literature-cluster
  centroid, because independently-clustered corpora share no membership to
  count. Both decide each flag on exact count ratios (shares are rounded only
  for display), return the same shape, tagged with `basis`, and
  `render_flags_markdown` names the basis in the table's opening sentence.
- **`report.py`** — `render_report` assembles the final markdown (HTML-comment
  metadata header + the LLM body + a Clustering Stability section (`render_
  stability_markdown`, unconditional — always states the order-stability
  match rate, even a reassuring 100%) + computed flags + audit + quotation
  grounding + run parameters + MyWork classification table + distance-scale
  legend); `write_outputs` writes `drift_<term>_<timestamp>.md` and, beside
  it, the assembled `_prompt.md`, the `_system.md` prompt, the
  `_cluster_diagnostics.json`, the model's raw reply (`_llm_response.txt`,
  saved before any post-processing), and the run's `_manifest.json` and
  `_receipt.json` — all sharing one basename, so a rerun never overwrites an
  earlier report and a directory of reports supports longitudinal comparison.
  A run with saturation disabled states the full literature count it
  clustered.
- **`emergence.py`** — EXPERIMENTAL, post-paper, LLM-free. `run_emergence`
  reuses the pipeline's Stage-1 `extract_passages`, hands the full literature
  pool (no saturation subsampling — emergence describes the library, not a
  prompt budget) to `clustering/emergence.py`, and writes
  `emergence_<term>_<timestamp>.md` + `_diagnostics.json` in the drift-report
  naming convention. The report and the GUI chart name the clock the
  timeline uses (Zotero or filesystem). No provider is ever constructed, so
  the command is identical with and without an API key.

### `pipeline.py` — the orchestrator
`analyze_term(term, config, *, mock, dry_run, record_fixtures,
overwrite_fixtures, fixtures_dir, embedder, collection, return_passages,
drafts_only, progress)`. This is the
whole method in one readable function: the three stages in sequence, the
dry-run short-circuit, the provider call, and the result dict.
`return_passages=True` additionally returns the clustered passages for the
GUI's visualizations — an additive option that changes nothing about the
analysis.

`validated_settings(config)` runs at the very top of `analyze_term` and
validates every typo-prone setting — clustering mode and resolution, the
timeframe window, retrieval windows, flag thresholds, saturation and LLM
settings — so a mistake fails before the corpus is opened and the embedding
model loaded. The output directory and (outside `--dry-run`) the provider
are checked there too. With `--record-fixtures`, an existing replayable
fixture for the same prompt is refused before the paid call unless
`overwrite_fixtures` is set. `map_divergence` returns the joint
partition as a fifth value (`None` in independent mode), which is what the
flags, the report's composition section, and the GUI's charts read.

Right after Stage 2, `run_stability_check` (LLM-free; skipped for `--dry-run`,
which stays the fast/free preview path) re-clusters the same, already-selected
passages under 30 shuffled orderings (`clustering.stability_check`,
on by default) and returns a match-rate summary — passed to `render_report`,
the run's `_cluster_diagnostics.json`, and the result dict as `stability`. It
runs strictly after the summaries `build_user_prompt` reads are already built,
so it can never perturb the assembled prompt.

### `validation/` — researcher-led clustering validation (EXPERIMENTAL, post-paper)
The researcher is the validator; agents assist (see
docs/validation_protocol.md). `run.py` reuses the
pipeline's exact Stage 1 + Stage 2 (`extract_passages` + `map_divergence`,
same seeds and saturation), collects one independent *cluster reading* per
cluster per configured reader (`agents.read_clusters`; blind prompts: term +
passages only), and opens a session (`session.py`). The researcher records a
verdict per cluster (coherent / mixed / overlaps / artifact) BEFORE the
reading is revealed; `session.reveal_readings` enforces that order at the
data layer, and every channel that shows an agent's view of a cluster (its
readings, or the revision agent's advice about it) logs a reveal first, so a
later verdict on that cluster is marked as revised after reveal. The
validation report prints a reading, and a per-reader agreement statistic, only once the
corresponding reveals are logged. `metrics.py` computes researcher-vs-reader
agreement (percent + Cohen's kappa via scikit-learn, over clusters, small-n
disclosed, blind verdicts only). `curate.py` applies the researcher's
merge/split decisions and dendrogram recuts (the revision agent argues and
proposes a semantic split beside the seeded mechanical k=2 split), carrying
untouched clusters' verdicts forward. The first edit or recut requires a
blind verdict on every cluster of the original clustering, and curation
refuses to re-embed when the configured embedding model or revision differs
from the session's. Per-cluster silhouettes come from scikit-learn's
`silhouette_samples` (a singleton scores 0). The edit stays
inside the session directory and is never re-analyzed: a curated partition
records what the researcher judged, and running register assignment on a
structure they just shaped would produce a report that reads as canonical
while resting on a hand-adjusted partition. The
interactive loop lives in the GUI's Validate tab; initial cluster readings
are mock-able (recorded fixtures), curation calls are live-only.

### `cli.py` — the command surface
argparse with eight subcommands: `ingest | analyze | diagnose | emergence |
validate | verify | benchmark | demo`. Each `_cmd_*` loads config (folding
CLI flags in as highest-precedence overrides) and calls into the package.
`main()` loads `.env` first (so keys work without shell exports) and turns
the package's typed exceptions (including a missing or locked validation
session) into friendly one-line errors with exit code 1.

### `demo.py` — the zero-setup path
`run_demo` locates `demo/`, builds a demo config (paper defaults + demo paths
+ the fictional researcher's theme tags), ingests the committed toy-corpus
JSONL into a local store on first run, then calls `analyze_term`. This is the
same pipeline as `cda analyze` — the demo is not a mock of the pipeline, only
the LLM step is mocked. A `drafts_dir` override swaps the researcher side:
the GUI's "try it on your own text" mode writes pasted paragraphs into a
runtime drafts directory and analyzes *them* against the toy literature
corpus (free dry-run preview; live runs show the cost and require a key —
recorded fixtures are prompt-keyed and cannot cover arbitrary user text).

### `diagnose/` — the corpus-composition diagnostic
- **`openalex.py`** — a minimal polite-pool client (`OPENALEX_MAILTO`,
  backoff, batched DOI resolution; the address is removed from every error
  message) with a SQLite read-through cache so re-runs cost nothing; `reconstruct_abstract` rebuilds text from OpenAlex's
  inverted index (also used by the toy-corpus builder).
- **`report.py`** — resolves the corpus's DOIs against OpenAlex and reports
  venue concentration (top-N share + Herfindahl index, both over works with a
  known venue; works without one are counted separately), field/topic
  coverage, declared-field baseline ("structurally untouched areas", checked
  against all of each work's topics; a `--fields` name OpenAlex does not know
  is an error listing the valid ones), and temporal profile (statistical
  median year). Counts report works that share a DOI with another work. Framed reflexively — visibility for informed curation, never a
  defect score.

### `benchmarks/` — clustering and stability validation
Standalone scripts, kept out of the installed package so the pipeline stays
lean. `clustering_comparison.py` runs spherical vs standard k-means vs
agglomerative on the identical L2-normalized passage matrix, with one fixed
silhouette evaluation sample and the same minimum-cluster rule for every
candidate and algorithm (silhouette + pairwise ARI + LLM-register agreement);
`clustering_order_stability.py` is the deep-look sibling
of the check `cda analyze` already runs by default (§4) — a configurable
trial count (`--trials`, default 20), the full per-trial table with shares
computed over the shuffled orderings only, mode-aware (independent mode
reports the researcher and literature corpora's stability separately),
built on the shared `clustering/order_stability.py` so the two never drift
apart; `bootstrap_stability.py` resamples sources and scores each reference
cluster by Hennig's Jaccard on the points both clusterings share ("n/a" for a
cluster never scored); `saturation_diagnostics.py` probes why the saturation
rule fired where it did; `stability_protocol.py` runs N identical-prompt
register assignments and reports structural equivalence + wording variation. See
[benchmarks/README.md](benchmarks/README.md) for
pointing them at a real corpus.

---

## 5. The versioned prompt

`prompts/register_assignment_v5.md` contains the active system prompt and a
schematic user template. v5 asks for two analyses (a polysemy map and the
researcher-vs-literature divergence; no temporal-drift analysis), requires an
explicit mapping declaration for both corpora, and attaches observed source
tags to individual representatives. Versions v1–v4 remain in `prompts/` unchanged; any
behavioral change gets a new version file.

One prompt version covers both clustering modes: it documents how to read each,
and the per-run EXTRACTION NOTES say which one applies. The appendix's
user-prompt template is untouched by the mode; the mode line, the joint
composition counts, and the corpus-timeframe line are extraction notes, the same
conditional slot the appendix already uses for the saturation line.

---

## 6. Configuration system

- `config/defaults.yaml` carries the paper's value for **every** reported
  parameter. `config/config.example.yaml` is the same tree fully commented,
  each parameter annotated with its role in the method.
- A **packaged copy** lives at `src/cda/defaults.yaml` (and
  `src/cda/prompts/`) so the installed CLI works from any directory. A test
  (`test_config.py`) asserts the repo copy and packaged copy are byte-
  identical — the drift guard.
- Precedence: packaged defaults → user `--config` file → environment →
  CLI flags. Nothing seedable is left unseeded; k-selection method and
  L2-normalization are fixed-by-design (documented in the config comments)
  because changing them would break comparability with the paper.

---

## 7. Determinism and the mock/live seam

Determinism is what makes offline testing and the mock demo possible:

1. All randomness is seeded (k-means init/restarts, the single silhouette
   evaluation sample reused across a k sweep, representative selection) from
   `clustering.seed` (42).
2. Passages are sorted with a total-order `chunk_id` tiebreaker before
   clustering, so the assembled prompt never depends on database return order.
   `select_literature_by_saturation()` restores this exact order before
   returning, even when its internal sweep re-ranked candidates by term
   similarity to decide *which* to keep — k-means++ seeding draws initial
   centroids by array position, so without this,
   `saturation.enabled` could silently change the clustering *result* even
   when it changed no *passages*.
3. Therefore the same corpus + config assembles a **byte-identical prompt**,
   whose `prompt_fingerprint` names a fixture file.
4. `MockProvider` replays that fixture; `RecordingProvider` captures it on a
   live run. The committed `demo/fixtures/*.json` (novelty, consensus) let
   the demo, tests, and CI run offline; the current v5 responses were written
   by Claude in a Claude Code session rather than recorded from the API (see
   [demo/fixtures/README.md](demo/fixtures/README.md)).

The only nondeterministic component is the live LLM call (temperature 0.0,
still not bit-stable) — and reports state this. Everything computed *from*
the LLM's reply (the flags, the match/split/merge audit, the quotation-
grounding check) is deterministic post-processing: given the same reply and
prompt, the same numbers come out, which is why the mock demo reproduces the
committed expected reports byte-for-byte.

The stability check (§4, `order_stability.py`) is a partial, deliberate
exception worth naming precisely: `run_trials`' shuffles use Python's
`random.Random(seed)`, entirely separate from the numpy generators the
k-means seeding above uses, so a `stability_check.trials`/`seed_base` change
can never perturb the canonical clustering, the assembled prompt, or fixture
replay — only the "Clustering Stability" section's own numbers, which are
themselves fully deterministic given the same trial count and seed base
(`test_stability_check_does_not_change_the_assembled_prompt` asserts this
directly).

---

## 8. GUI composition

Four files, strictly presentation over the same functions the CLI uses:

- **`gui/app.py`** — Streamlit app. Sidebar (paths, mock toggle defaulting on
  without a key). A prominent "Try the demo" button plus a "try it on your
  own text" expander (paste paragraphs → analyzed as the researcher side
  against the toy literature; free dry-run preview or a live run with the
  cost shown). Tabs: Setup, Ingest (with drag-and-drop for PDFs + a Zotero
  CSV), Analyze (drafts drop-zone, a dry-run prompt-preview toggle, an
  "Analysis options" expander holding the joint/independent clustering toggle
  and the corpus-timeframe window — with a live preview of the window it will
  apply — plus the experimental cluster-emergence expander, an LLM-free timeline
  of when each literature cluster coalesced), Validate (rendered by
  `gui/validate.py`: blind verdicts on each cluster, then reveal and
  agreement — see `src/cda/validation/`), Reports
  (drift and emergence), Diagnose. On a successful run it shows a "Visual summary" tab
  beside "Full report," with the quotation-grounding score in the caption.
  Dropped files are saved by the local process — nothing leaves the machine.
  A footer states the network scope (Anthropic + OpenAlex only), enforced by
  a test.
- **`gui/validate.py`** — the Validate tab: verdicts, reveals, curation,
  dendrogram recut, endorsement. All enforcement lives in
  `cda.validation`; this module renders it.
- **`gui/shared.py`** — helpers both tab modules use (demo-mode check,
  cached embedder, session API key and OpenAlex email handling, report
  rendering, live progress log).
- **`gui/visuals.py`** — Altair charts built purely from the result dict
  (`return_passages=True`): a shared 2D PCA passage map (scikit-learn PCA,
  full SVD solver; researcher vs literature), cluster-size bars, the saturation ARI sweep, and the
  divergence-flags quadrant. Under joint clustering it adds the two charts that
  answer "where do my passages sit": the same PCA plane recoloured by *shared
  cluster* (literature faint and small, the researcher's own sentences as large
  outlined diamonds, so their position inside each cluster is what you read),
  and a normalized composition bar per shared cluster sorted by researcher
  share, which puts the clusters the researcher dominates and the ones they
  never enter at opposite ends. Altair ships with Streamlit, so no new
  dependency; nothing here touches the pipeline, so charts can evolve without
  invalidating fixtures.

---

## 9. Docker composition

`Dockerfile` — `python:3.11-slim`, non-root user, CPU-only torch (keeps the
image ~1.5 GB), pinned deps, and the embedding model **baked into the image**
at the same revision `embedding.revision` pins, so the container demo needs
no downloads. `docker-compose.yml` — one command
(`docker compose up`) mounts `./corpus ./drafts ./outputs`, keeps ChromaDB in
the corpus volume, publishes port 8501 on 127.0.0.1 only (the GUI can spend
the API key, so it is not offered to the local network), and passes the three
env vars through. The
`.dockerignore` keeps user data and secrets out of the image.

---

## 10. Testing architecture

The test suite is offline by default: `pytest` excludes the `integration`
marker via `pyproject.toml`. A separate integration tier uses the real
embedding model:

- **Offline tier** — unit tests for each module using a deterministic **fake
  embedder** (hash-based or topic-anchored, in `conftest.py`), so no model
  download and no network. Includes an end-to-end test that builds a tiny
  ChromaDB, runs a dry run to learn the assembled prompt, writes a fixture
  from it, and then runs in mock mode — proving prompt assembly is byte-stable
  and the whole chain works.
- **Integration tier** (`pytest -m integration`) — loads the real MiniLM model
  (384-dim check, semantic-neighbor sanity, determinism) and clicks the real
  demo button in Streamlit's `AppTest` runtime.
- Structural guards: packaged-copy identity (config + prompt), the
  example-covers-defaults check, and the network-scope footer assertion.

CI (`.github/workflows/ci.yml`, read-only token permissions) runs five jobs:
lint, format and secret checks; separate offline tests on Python 3.11 and
3.12; both `novelty` and `consensus` mock demos; and a Docker job that builds
the image, runs both demos inside the container, and checks GUI health.

---

## 11. Relation to the paper

The current version uses literal term-sentences, a joint full-pool
partition, two LLM analyses, and the same mapping-based register audit on both
sides. Optional sample saturation reports its selected fraction and whether it
stopped early; representativeness is left to the researcher. Quotation checks
require ordered fragments within one supplied passage and the cited source.
The current version does not claim to reproduce the manuscript's numerical
results.

---

## 12. Extension points

The seams are designed so common extensions touch one place:

| To… | Change | Nothing else moves because… |
|---|---|---|
| add a second LLM (e.g. verification) | a config block (`llm.verification_model: openai:<model>`) | providers go through `get_provider` / `resolve_model_spec` |
| run the analysis on a non-Anthropic model | `llm.model: openai:<model>` (+ `OPENAI_API_KEY`) | `get_provider` resolves the same `provider:model` spec |
| swap the embedding model | `embedding.model` and `embedding.revision` (+ re-ingest) | everything takes the `Embedder` protocol |
| change a parameter | `config/defaults.yaml` (+ packaged copy) | all magic numbers are config-driven |
| revise the prompt | a new `prompts/register_assignment_v5.md` | the loader reads the versioned file |
| add a chart | `gui/visuals.py` | visuals read only the result dict |
| switch clustering mode | `clustering.mode` (or `--independent`) | both modes return the same two `ClusterSummary` objects |
| pin the corpus to a date | `timeframe.added_to` (or `--added-to`) | the window is a Stage-1 filter; nothing downstream knows |
| change the stability check's trial count | `clustering.stability_check.trials` | `order_stability.py` takes trials as a plain argument, no other call site to touch |
| add a report section | `divergence/report.py` | rendering is isolated from analysis |
| add a validation reader model | `validation.readers` in config | readers go through `resolve_model_spec` |
| pin or change cluster resolution | `clustering.k` / `clustering.method` in config | both default to the paper's method |

The invariant behind all of these: **analysis logic, presentation, and I/O
are separate**, and the two external services (embeddings, LLM) are behind
protocols. That is what keeps a pipeline this faithful to a paper also this
easy to read.

Persisted analyses also write immutable run manifests and completion receipts.
See [reproducibility](docs/reproducibility.md) for their boundary and comparison command.
