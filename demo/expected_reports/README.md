# Expected demo reports

Reference outputs of `cda demo --term novelty --mock-llm` and
`cda demo --term consensus --mock-llm` against the committed toy corpus.

| File | What it is |
|---|---|
| `drift_novelty_v5_20261018.md`, `drift_consensus_v5_20261018.md` | current reports (v5 prompt, literal-only extraction, joint clustering) |

## Provenance of the current reports

The register-assignment responses in the two v5 reports were written by
Claude in a Claude Code session after reading the exact v5 toy prompts
(fixture model `claude-code-session`, see [../fixtures/](../fixtures/)); each
report names this provenance at the top. They are software examples, not
Anthropic API recordings, researcher-endorsed findings, or manuscript results.
Everything in the computed sections — clustering, flags, audit, grounding — is
deterministic post-processing of those responses and the toy corpus.

## What to look at first

The "Joint Clustering: where your passages sit" table is the shortest route
into what CDA found, because both corpora share one partition.

- **novelty** (joint k=2, silhouette 0.137; 38 researcher and 56 literature
  passages): Cluster 0 holds 32 researcher and 34 literature passages,
  Cluster 1 holds 6 and 22. Every shared cluster contains passages from both
  sides, and no cluster is flagged; the divergence the report describes is in
  how the two sides use the term inside shared clusters.
- **consensus** (joint k=3, silhouette 0.103; 33 researcher and 103 literature
  passages): Cluster 1 (18 literature passages, the climate-consensus
  register) contains none of the researcher's passages and is flagged
  **underrepresented** — divergence here reads as absence from a literature
  register rather than concentration in a researcher-exclusive one.

Neither k should be read as the one true register count for its term. The
"Clustering Stability" section states how often k recurs across 30 shuffled
orderings of the same passages: 33% for novelty and 37% for consensus.
`benchmarks/clustering_order_stability.py --demo --term novelty --trials 100`
runs the same check with a full per-trial table.

Quotation grounding verifies 10/10 quotations for each term verbatim against
individual prompt passages.

Under `--independent` the reports would instead show two separate partitions
with non-comparable cluster numbers, and the flags would come from
cross-assignment. Each report's Run Parameters section names the mode that
produced it.

## Checking your own demo run

Apart from the `generated:` timestamp in the header comment (and the
`_HHMMSS` suffix your run adds to the filename), your
`demo/_runtime/outputs/drift_<term>_<timestamp>.md` should match the v5 files
here. Prompt fingerprints can differ across CPU architectures (see
[../fixtures/README.md](../fixtures/README.md)).

These reports are shareable because they quote only CC0 OpenAlex abstracts
and the fictional synthetic drafts — this is exactly why the demo uses a toy
corpus instead of a real library (real CDA reports quote copyrighted
literature verbatim and must stay on your machine).
