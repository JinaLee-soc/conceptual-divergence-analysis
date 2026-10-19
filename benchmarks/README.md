# Benchmarks

Standalone scripts supply quantitative validation of the clustering.
The stability protocol operationalizes the run-to-run consistency check
described in the paper (§3.3); the clustering comparison and order-stability
scripts are additional evidence (neither is itself
described in the paper). All run on the committed toy corpus out of the box,
and on any real corpus via a config file.

| Script | Question it answers | Cost |
|---|---|---|
| `clustering_comparison.py` | Does the register structure depend on the choice of spherical k-means? (vs standard k-means, vs agglomerative; silhouette, pairwise ARI, agreement with LLM registers) | $0 — local embeddings only |
| `clustering_order_stability.py` | How much does a term's chosen k depend on the arbitrary order passages happen to be in? (N independent shuffles of the identical, already-selected passage set vs. the one canonical run `cda analyze` actually produces; shares and the most common k are computed over the shuffles only, excluding the canonical trial) | $0 — local embeddings only, no LLM call |
| `stability_protocol.py` | Is LLM register assignment stable across independent runs? (5 runs on an identical prompt; structural equivalence + wording variation) | live: ~$0.10–0.15 × runs; mock: $0 (harness check only — replaying one fixture N times has zero variance by construction) |
| `bootstrap_stability.py` | Would the same clusters appear had the library been a slightly different sample of papers? (source-level subsampling, B replicates; per-cluster Hennig Jaccard computed on the passages both clusterings share, with stability bands and "n/a" for a cluster never scored; k distribution, ARI vs the reference on shared passages) | $0 — local embeddings only; minutes to ~an hour depending on pool size and B |
| `saturation_diagnostics.py` | Why did the saturation rule fire (or not) where it did? Four probes: same-data ARI across seeds at fixed k, anchor vs random vs reversed orderings, a threshold x consecutive rule grid over one no-stop sweep, and drop-m source perturbations | $0 — local embeddings only; ~20-30 min per term |

`clustering_order_stability.py` measures how much the selected k changes when
the same passages are reordered. CDA sorts passages before clustering so a
run can be reproduced within its recorded environment. The benchmark tests
sensitivity to other orderings: weakly separated clusters may yield several
values of k. Read the canonical run's k against that distribution.

## On the toy corpus

```bash
# once, to create the analysis the benchmarks reuse:
cda demo --term novelty --mock-llm

python benchmarks/clustering_comparison.py --demo --term novelty
python benchmarks/clustering_order_stability.py --demo --term novelty --trials 20
python benchmarks/stability_protocol.py --demo --term novelty --mock-llm   # harness check
python benchmarks/stability_protocol.py --demo --term novelty              # real protocol (live, ~$0.60)
```

Results land in `demo/_runtime/outputs/benchmark_*.md`.

## On a real corpus (generating numbers for a manuscript)

1. Create a config for your corpus (start from `config/config.example.yaml`):

   ```yaml
   # my-config.yaml
   paths:
     corpus_dir: ~/path/to/your/chromadb   # a dir containing chroma.sqlite3
                                           # is used directly; otherwise CDA
                                           # expects <corpus_dir>/chroma
     drafts_dir: ~/path/to/your/drafts
     output_dir: ./outputs
   corpus:
     collection: your_collection_name      # must exist in that store
   mywork:
     filename_patterns: ["*your_draft_pattern*"]
     publication_titles: ["Exact Titles of Your Own Indexed Papers"]
   ```

   The store must have been embedded with the pinned model
   (`sentence-transformers/all-MiniLM-L6-v2`) and chunks must carry a
   `pdf_filename` metadata field — both true for corpora built by
   `cda ingest`.

2. Run one live analysis per term you want to benchmark (this creates the
   saved prompt + report the benchmarks reuse):

   ```bash
   cda analyze --term novelty --config my-config.yaml        # ~$0.10–0.15
   ```

3. Run the benchmarks against the same config:

   ```bash
   python benchmarks/clustering_comparison.py --config my-config.yaml --term novelty
   python benchmarks/clustering_order_stability.py --config my-config.yaml --term novelty --trials 30
   python benchmarks/stability_protocol.py    --config my-config.yaml --term novelty --runs 5
   ```

4. Numbers for a manuscript come straight from the generated tables:
   - clustering comparison → pooled chosen k, cosine silhouettes, and the
     pairwise ARI matrix (report ARI of each alternative vs the spherical
     solution). All methods fit the same normalized matrix and every k is
     scored on the same fixed silhouette sample after the same small-cluster
     rule; pass `--per-corpus` only for an independent-mode analysis;
   - clustering order-stability → how many distinct k values appeared across
     N shuffles and where the canonical run sits among them — cite this
     alongside a reported k for any term whose silhouette is low, as a
     robustness disclosure;
   - stability protocol → the structural-equivalence verdict across runs and
     the mean pairwise label-vocabulary overlap (structure stable, wording
     varies).

   The reports are timestamped in your output directory; cite the run date
   and the `cda` version from the report header.

**Privacy note:** `clustering_comparison.py` and `clustering_order_stability.py`
never call any API — both embed locally only. `stability_protocol.py` in live
mode sends the saved analysis prompt (which contains corpus passages) to the
Anthropic API, the same as a normal `cda analyze` run.
