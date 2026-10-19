# Run records and exact comparison

Every persisted `cda analyze` / `cda demo` run writes two run-record files
alongside its report (schema version 2):

- `*_manifest.json`: prepared **before** the LLM call. It identifies the
  installed source files and Git commit/dirty state, effective configuration,
  resolved paths, Python/platform and all installed distribution versions,
  the embedding model's configured and resolved snapshot revision with its
  cached file hashes, retrieved input fingerprints, ordered extracted/final
  samples, vector hashes, cluster assignments, and sample-flow counts.
- `*_receipt.json`: written **after** successful output generation (or successful
  dry-run preparation). It binds the prompts, diagnostics, manifest and, for
  a completed analysis, the report, the raw model reply
  (`*_llm_response.txt`, saved before any post-processing) and optional
  figure by SHA-256. A manifest without a receipt is an
  unfinished run.

No source passages or credentials are copied into these records. The normal
prompt/report files still contain selected passages. Store all run files together
and preserve the private source corpus separately. Identical retries of the same
artifact are accepted; different content under the same run basename is refused.
Normal new runs reserve a new basename automatically.

## Reproduction boundary

The analyzed input boundary is the **retrieved derived corpus records plus loaded
or supplied draft texts**, before date filtering and sample selection. It is not a
census of an entire library and does not certify that every raw PDF was retrieved.
New ingestion also records original PDF, extracted text and supplied Zotero CSV
hashes; JSONL records carry extracted-text hashes, and every chunk records the
sentence segmenter that produced it. Stores lacking sentence positions, or
segmented by another segmenter, must be re-ingested once (`cda ingest`
refreshes those sources; analysis refuses them until then). Ingest also
refreshes a source whose PDF, Zotero CSV or record text hash changed.
Keep the derived database and original sources; hashes cannot reconstruct them.

Every genuine term-bearing sentence occurrence is retained, including identical
sentences at different positions or in different drafts. Only duplicate retrieval
of the same source/sentence position through overlapping chunks is removed; its
count is recorded. Two drafts with identical full text are refused rather than
counted twice. Direct unindexed input records are independent source units.
Saturation selection remains a
separate explicit sample rule, with ratios/early-stop facts left for researcher
interpretation.

The package inventory is an **environment record, not a portable binary lockfile**.
Recreate the recorded versions in an isolated environment and preserve the
model cache. `embedding.revision` pins the model snapshot by default
(`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`); the manifest records it as
`configured_revision` and the snapshot actually loaded as `resolved_revision`.
Local model files are hashed too. Custom embedders are explicitly marked; they
do not acquire an invented model identity. Different hardware/backends may
produce different floating-point values.

## Verify and compare

```bash
python -m cda.provenance outputs/drift_TERM_RUN_receipt.json
python -m cda.provenance outputs/drift_TERM_RERUN_receipt.json \
  --compare outputs/drift_TERM_REFERENCE_receipt.json
```

The first command checks artifact integrity, including missing or changed files.
The second first verifies both receipts, then requires **exact equality**, with
no numerical tolerance, for the effective config (excluding `paths.*`),
recorded inputs, ordered samples, embeddings, cluster assignments, model
identity, installed CDA source digest, diagnostics and both prompts. Both runs
must use the same manifest schema version. Input and output locations may
differ; the automatic unique basename separates runs. The physical `pdf_root`
field is excluded from record/sample fingerprints, while source-relative names,
text and bibliographic metadata remain included.

Differences in platform, Python build, package versions or embedding device
are reported under `environment_differences` but do not fail the comparison.
Package names are compared in normalized form (PEP 503), and several installed
versions of one package are reported under `duplicate_packages`.
Add `--strict-environment` to require those to match as well:

```bash
python -m cda.provenance outputs/drift_TERM_RERUN_receipt.json \
  --compare outputs/drift_TERM_REFERENCE_receipt.json --strict-environment
```

Timestamps, output basenames, `paths.*`, generated figure bytes and LLM prose
are excluded from deterministic-stage comparison. Each run's own receipt still checks those
saved artifact bytes. Live LLM outputs are not guaranteed identical. A mock replay
checks software execution, not a fresh model inference; the committed Claude-authored
toy illustrations remain explicitly labeled as illustrations.

Neither command establishes scientific validity, representative sampling,
researcher adoption or reproduction of manuscript results. A final research
packet additionally needs researcher-adopted reference exports and its own scope
manifest. Run records supply execution evidence without asserting those
separate outcomes.
