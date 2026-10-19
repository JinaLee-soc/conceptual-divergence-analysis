# FAQ

**Can I use it without Zotero?**
Yes. `cda ingest --pdf-dir ~/my_papers` builds a corpus from a PDF folder
alone. You lose the metadata Zotero supplies (title/year/DOI and the
`mywork` tag), so mark your own papers via `mywork.filename_patterns` or
`publication_titles` in the config, or rely on the drafts directory —
drafts are always yours. `cda diagnose` is the one command that needs DOIs,
which in practice means the Zotero CSV path.
→ [corpus_setup_zotero.md](corpus_setup_zotero.md)

**Does my library leave my machine?**
No. PDFs, the vector store, embeddings, and reports are all local. What
leaves: the assembled analysis prompt (extracted sentences + file names) to
the Anthropic API on *live* runs only, and DOIs/queries to OpenAlex for
diagnostics. Mock mode, dry runs, tests, and ingestion contact nothing, and
you can print any prompt before it is sent (`--dry-run`).
→ [privacy_and_cost.md](privacy_and_cost.md)

**What corpus size is enough?**
CDA is built for researcher-scale curation: hundreds to a few thousand
papers (the paper's demonstration corpora are ~3,000 each; the committed
demo runs on 144 abstracts). What matters more than size is that the corpus
reflects the literatures you actually write against, and that the target
term appears often enough — below ~30 literature passages for a term,
register structure gets unstable, and the report's N counts will show it.
Run `cda diagnose` once **before your first analysis** (the recommended
pre-flight): it makes the corpus's structure visible so the baseline your
divergences are measured against is a deliberate choice.

**What if saturation is never reached?**
Only applies if you asked for the paper's subsample (`saturation.scope:
sample`). The default scope is `prompt`, which clusters every retrieved
literature passage and says so in the report. Under `sample`, if saturation is
not reached the report says so explicitly and proceeds with the capped sample (1,200 passages, the paper's
value) — some of the paper's own reported runs carry this flag. Read register
counts as provisional for that term; options are raising
`saturation.max_passages`, or returning to the default `prompt` scope for
that term and accepting a longer local clustering sweep.
→ [interpretation_guide.md](interpretation_guide.md)

**Do I need an API key just to try it?**
No. `cda demo --term novelty --mock-llm` replays a committed demonstration
response (written by Claude in a Claude Code session, not recorded from the
API; see [demo/fixtures/README.md](../demo/fixtures/README.md)), so there is
no Anthropic API call and no LLM cost. The first
non-Docker run may download the embedding model; after it is cached, use
`HF_HUB_OFFLINE=1 cda demo --term novelty --mock-llm` for disconnected runs.
A key (and ~$0.10-0.15 per term) is needed only for live analyses of your own
corpus.

**Can I use a different LLM?**
Yes, and there is one case where you should. The default is pinned to
`claude-sonnet-4-6`, the model the paper's results were produced with;
changing it changes the instrument, so the register labels will differ from
the published ones while the deterministic stages (retrieval, clustering,
flags) stay identical.

`llm.model` takes either a bare Anthropic model name or a `provider:model`
spec. `openai:MODEL` routes to any OpenAI-compatible endpoint, and setting
`OPENAI_BASE_URL` points that at a model running on your own machine (Ollama,
LM Studio, vLLM). That is the case where you should switch: under a local
model no stage of the pipeline makes an external call, so drafts under
embargo, IRB constraint, or NDA never leave your hardware. See
`docs/privacy_and_cost.md` for what does leave on a default run.

An unrecognized prefix is refused rather than guessed at, so a typo fails
immediately instead of being sent to Anthropic as a model name.

To compare two models rather than replace one, use the experimental
verification hook (`llm.verification_model`), which flags disagreements.

**Can I use a different embedding model?**
`embedding.model` is configurable but pinned by default for the same
fidelity reason, and `embedding.revision` pins the exact model snapshot
(`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), so every run embeds with the
same weights. Changing either requires re-ingesting the corpus (stored
vectors must match the query model). Comparisons with the paper's numbers
only hold on the pinned model.

**The mock demo says "fixture not found." What happened?**
Fixtures replay responses for byte-identical prompts. If you changed config
values, the prompt file, or the toy corpus, the fingerprint no longer
matches — the error message lists the expected key and the two ways forward
(run the unmodified demo, or record your own fixtures on a live run with
`--record-fixtures`).

**How does CDA relate to other research-infrastructure tools?**
CDA is the *individual-level, term-register* instrument in a growing
ecosystem. For collective/field-scale analysis of term networks and their
evolution, see [GARGANTEXT](https://gargantext.org/en/) (phylomemy
reconstruction of knowledge dynamics); for AI support across the whole
research lifecycle, see the
[TIB AIssistant](https://www.tib.eu/en/research-development/demonstrators/project/tib-aissistant)
platform. CDA is narrower on purpose: it compares *your* usage of specific
theoretical terms against *your* curated corpus and tells you where the two
diverge.

**Are my passages and the literature clustered together or separately?**
Together, by default: one spherical k-means partition over both, so a cluster
number names the same region of meaning on both sides and the report can show
where your passages sit inside each shared cluster — including the clusters
holding none of your passages (literature registers you have not taken up) and
the ones holding no literature (usage of yours the corpus does not share). The
paper's earlier published runs clustered the two sides *separately* and located
your passages by cross-assignment to the nearest literature centroid; `cda
analyze --independent` does exactly that, and every report's Run Parameters line names
the mode that produced it. The k-means is identical either way — same seeds, same
silhouette-selected k, same five centroid-nearest representatives — only the
matrix it runs on differs. Joint is the default because only a shared
partition makes cluster numbers comparable across the two sides.

**Can I analyze my library as it was at some earlier date?**
Yes: `cda analyze --term novelty --added-to 2026-05-01` restricts the corpus to
works you had added by the end of that day (UTC), using Zotero's "Date Added"
(when *you* acquired the work, never its publication year). All stored
acquisition dates are UTC, so a window selects the same works on any machine;
an impossible date such as `2026-02-30` is an error. `--added-from` opens the
other end, so a pair of bounds isolates one period of reading. This is what makes
a result reproducible after you keep reading — and what lets you compare two runs
where only your manuscript changed. It works on `cda emergence` and
`cda validate` too.

Passages whose source has no "Date Added" cannot be placed relative to a cutoff,
so they are dropped and counted; `--include-undated` keeps them, accepting that
the window is then advisory. If a window drops *everything*, the usual cause is a
corpus ingested before the timestamp was stored — the error says so and one
`cda ingest --pdf-dir PATH --zotero-csv EXPORT.csv --reingest` pass fixes it. Your drafts are never
filtered: they are the manuscript under examination, not works you acquired in a
period.

**My Zotero library was rebuilt, so every "Date Added" is the same month. Can I
still take a snapshot?**
Yes — use the other clock: `--date-source filesystem` reads when each PDF
arrived on disk instead. This is the common failure mode, and it is silent
unless you look: a re-imported library reports itself as entirely new, so a
window cuts on the import date rather than on your reading. Check before
trusting a Zotero-clock window:

```bash
cda emergence --term novelty
```

If the emergence timeline is one vertical cliff instead of a trajectory, your
Zotero dates record a rebuild. The filesystem clock tracks the download rather
than the reading, so neither is authoritative — but the reports name which one
produced them, and running both and comparing is a legitimate way to see how
much the answer depends on the choice.

**Zotero matched almost none of my PDFs. Why?**
Filename matching only works if your files still carry the names Zotero gave
them; a download pipeline that renames files defeats it entirely. Ingest falls
back to the DOI printed in the document and then to its title, and the ingest
summary reports how many matched each way:

```
[zotero] matched 0 by filename, 812 by DOI, 1583 by title; 980 unmatched, 12 of them ambiguous (unmatched files keep the filesystem acquisition date only)
```

"Ambiguous" files matched a filename, DOI or title that more than one Zotero
item shares; CDA attaches nothing rather than guess, and prints each such
file with the reason. Renaming the file or merging the duplicate Zotero items
fixes it.

Unmatched files still get the filesystem clock, so they are analyzed — they just
carry no Zotero title, tags, or "Date Added" (which also means no `mywork` tag,
so check your MyWork settings if your own papers are landing in the literature).

**What does `cda emergence` do, and why does it want a re-ingest?**
An experimental, LLM-free view of the *process*: literature passages for a
term are ordered by when you added each work to your library (Zotero's
"Date Added"), then cumulative snapshots are clustered so you can see when
each usage cluster first coalesced — your reading trajectory, not the
field's publication history. It needs the acquisition timestamp in chunk
metadata, which ingest only started storing with this feature: on an older
corpus run `cda ingest --pdf-dir PATH --zotero-csv EXPORT.csv --reingest` once (for the demo
store, delete `demo/_runtime` and rerun). Passages without a date are
excluded and counted, never guessed. Try it free:
`cda emergence --term novelty --demo`.

**How do I validate the clustering?**
`cda validate --term X` (or the GUI's Validate tab) opens a researcher-led
validation session: you record a blind verdict per cluster before seeing an
LLM agent's independent reading of the same passages, and the report gives
your agreement with the agent (percent + Cohen's kappa over clusters). You
can also curate (merge/split with agent advice); see the interpretation
guide for why curation never counts as evidence. The initial demo readings
replay offline once fixtures are recorded; curation calls are live.

**Does curating clusters change my drift report?**
No. Curated partitions live only inside the validation session. If you
apply a merge or split, the edit is recorded in the session and stays there.
There is deliberately no way to re-run the analysis on a curated partition:
a report built from clusters you just adjusted would read as canonical while
resting on your own edits. The drift report from `cda analyze` is untouched.

**Why markdown reports instead of a database or dashboard?**
Reports are the scholarly artifact: timestamped, diffable, quotable, and
annotatable in any editor. Longitudinal comparison is deliberately just
"keep the files" (plus the Reports tab in the GUI for browsing).

**Something else?**
Open an issue — and if you ran CDA on your own research program, the
[adoption report template](../.github/ISSUE_TEMPLATE/adoption_report.md)
is the most useful thing you can send us.
