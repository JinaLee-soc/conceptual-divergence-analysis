# Toy corpus (demo data)

Everything in this directory is safe to redistribute; nothing here contains
copyrighted full text.

## Contents and provenance

- `abstracts.jsonl` — ~120–150 works fetched from the
  [OpenAlex](https://openalex.org) API by `demo/build_toy_corpus.py`:
  metadata (title, year, venue, DOI, authors) plus abstracts reconstructed
  from OpenAlex's `abstract_inverted_index`. OpenAlex releases this data as
  **CC0**. Email and postal contact details in one source record have been
  replaced with `[contact details removed]`; its provenance field records
  this edit. Author names and bibliographic identifiers are retained. The works are drawn from three searches adjacent to the sociology
  of science / science studies, ranked by citations, English-language
  articles with substantive abstracts only.
- `drafts/` — four **synthetic researcher drafts**, written for this demo by
  the tool's authors. The observatory, standards committee, people, findings,
  and citations in them are fictional; each file says so in a header. They
  use the terms *novelty* and *consensus* in a deliberately distinctive
  register so that the demo produces a visible, interpretable divergence
  against the literature corpus. **CC0.**

## Why a toy corpus?

Real CDA corpora are researcher-curated libraries of copyrighted PDFs, and
real CDA reports quote literature passages verbatim — neither can be
published. The toy corpus substitutes CC0 abstracts and fictional drafts so
that reviewers and adopters can run the *full* pipeline end-to-end
(`cda demo --term novelty --mock-llm`) and inspect a complete, shareable
report before pointing the tool at their own library.

## Regenerating

```bash
# needs OPENALEX_MAILTO in .env (polite pool); free, three API requests
python demo/build_toy_corpus.py
```

Regeneration fetches the original OpenAlex text and can restore contact details.
Review regenerated records for email and postal contact details before publishing.
Note that a changed corpus invalidates `demo/fixtures/` (fixtures are keyed
by prompt content) — re-record them after regenerating.
