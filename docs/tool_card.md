# CDA tool card

One page on what this tool is for, when it works, and when it does not.

## Intended use

CDA is a **reflective auditing instrument for a single researcher's
conceptual vocabulary**. It compares how you use a theoretical term across
your own manuscripts against how your curated literature corpus uses it, and
renders the comparison as named usage registers, semantic distances, and
divergence flags — so that unacknowledged polysemy or an idiosyncratic usage
register is visible to you before it is visible to a reviewer. It is
designed for longitudinal use: run, revise, re-run.

## Scope conditions (when results are meaningful)

- **A curated corpus you can vouch for.** The literature side of every
  comparison is *your* corpus — CDA measures divergence from *your reading
  program*, not from "the literature" in the abstract. Hundreds to a few
  thousand papers is the designed scale; run `cda diagnose` to see the
  corpus's structure.
- **Terms that carry theoretical weight.** Substantive concepts with
  register variation (*novelty*, *consensus*, *legitimacy*) work; generic or
  compound-heavy words (*control*, *model*) mostly cluster by collocation.
- **Enough researcher material.** A handful of drafts or tagged papers;
  below ~10 researcher passages the "researcher registers" are one
  document's phrasing.
- **English-language corpora** (the pinned embedding model and prompt are
  English-tuned).

## Known limitations

- **Register boundaries are approximate.** Silhouettes on single-sentence
  embeddings are low (≈0.1–0.3 typically); clusters have meaningful cores
  and fuzzy edges. See the [interpretation guide](interpretation_guide.md).
- **The LLM step is nondeterministic and can mislabel.** Structure is stable
  across runs (see the stability benchmark); wording varies; labels are
  proposals backed by quoted evidence you should check — and every report's
  **Quotation Grounding** section verifies each quoted passage against the
  prompt automatically, listing anything it couldn't confirm.
- **Saturation subsampling can under-sample minority registers**
  (term-anchored ranking; "premature saturation" in the guide). Not the
  default scope, so this applies only under `saturation.scope: sample`.
- **The audit and verification tables are heuristic parses** of the report —
  cross-checks, not ground truth (they read only structured mapping
  declarations and report n/a, rather than guess, when a report states its
  mapping in prose alone).
- **One embedding model, pinned** to a single snapshot
  (`embedding.revision`). Divergences finer than `all-MiniLM-L6-v2`'s
  resolution are invisible to stage 2.

## Scaling limits

CDA is deliberately an **individual-scale** instrument, and its design
choices stop making sense beyond that scale:

- **Corpus**: designed for researcher-curated libraries of hundreds to a few
  thousand papers. Ingestion and retrieval will *run* on much larger
  collections, but the analysis premise — that the corpus encodes one
  researcher's audience — dissolves as the collection approaches a field
  archive.
- **Per-term prompt budget**: the prompt carries cluster summaries — each
  cluster's size plus its five centroid-nearest representatives — so what one
  analysis *reads* is bounded by the cluster count (default max 15), not by
  the corpus. Per-term API cost therefore stays flat (~$0.10–0.15) regardless
  of corpus size. Local clustering time does grow with the retrieved pool:
  saturation subsampling (`saturation.scope: sample`, not the default) is the
  knob that bounds it, at the cost of clustering a sample rather than the
  whole retrieved set.
- **Not field mapping**: for reconstructing the structure or evolution of
  whole fields, use field-scale science-mapping tools (see "Related tools"
  in the [FAQ](faq.md)); CDA answers a different question — one researcher's
  position relative to their curated literature, term by term.

## Failure modes to expect

Compound-form terms producing collocation clusters; premature or unreached
saturation on very common terms; thin researcher pools; drafts whose file
names leak working titles into prompts (rename before analyzing if that
matters). Each of these is visible in the report itself — N counts,
silhouettes, and saturation status are printed in every header.

## What CDA is **not**

- **Not a plagiarism or similarity checker.** It never asks whether your
  text resembles someone else's — it asks whether your *usage of a concept*
  occupies the same register.
- **Not a recommendation engine.** It does not tell you what to read or
  cite; `cda diagnose` shows corpus structure, not reading lists.
- **Not a substitute for reading.** Registers summarize passages you are
  expected to check; every report quotes its evidence with sources so you
  can.
- **Not an arbiter of correct usage.** Divergence is a finding, not a fault
  — sometimes the divergent register is the contribution.
- **Not a hosted service.** Local-first by design; the only network calls
  are the LLM API (live runs) and OpenAlex (diagnostics).
