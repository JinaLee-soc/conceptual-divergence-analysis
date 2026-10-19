# Interpretation guide

CDA's output is material for a judgment you make, not a verdict. This guide
covers what the numbers mean at this scale, what failure looks like, and
where the human belongs in the loop.

## Reading the shared clusters

By default your passages and the literature's are clustered **together**, so a
cluster is one region of meaning that either side can occupy, and the report's
"Joint Clustering: where your passages sit" table is where most of the reading
happens. Three patterns are worth naming:

- **A cluster with passages from both sides** is shared usage. Compare your
  representative sentence against the literature's *in that cluster* — if they
  read as the same operative definition, your usage is conventional there,
  whatever the cluster's label says.
- **A cluster with none of your passages** is a literature register you have
  not taken up. That is a fact about your writing, not a gap in the data; it may
  be a deliberate choice, an unread corner of your own library, or a reviewer's
  first question. The report lists these explicitly.
- **A cluster with no literature passages** is usage of yours with no
  counterpart in the corpus. This is the strongest divergence signal CDA can
  produce — and also the one most worth checking against the corpus's
  composition (`cda diagnose`) before concluding anything about the field.

The **researcher share** column is the same quantity the >80% / <10% flags
threshold: a small cluster that is almost entirely yours is where a private
sense of the term most likely lives.

One number to read carefully: the **silhouette is computed on the pooled
passages**, so it describes the joint partition, not your side or the
literature's side separately. The per-corpus blocks report it for reference, and
say so.

If you need results comparable with the accompanying paper's runs, use
`cda analyze --independent`: the two corpora are clustered separately and your
passages are located by cross-assignment to the nearest literature centroid.
Cluster numbers are then per-corpus labels and must not be compared across the
two sides. Every report's Run Parameters line names the mode it used.

## Silhouette scores in this setting

The silhouette (−1…1) measures how separated the usage clusters are in
embedding space. It is undefined for a partition with fewer than two
clusters (or one passage per cluster); the run's diagnostics then record it
as null rather than as a score. Values here are computed with cosine distance over
sentence embeddings of *single sentences* — short, noisy texts — so absolute
values run far lower than textbook examples:

- **≳ 0.3** — unusually crisp separation; registers will likely be obvious.
- **~0.1–0.3** — typical for real term-sentence sets (the demo's own joint
  partitions land at 0.137 for `novelty` and 0.103 for `consensus`). Cluster
  *cores* are meaningful; boundaries are approximate.
- **≲ 0.05** — treat the partition as a reading aid only; the term's usages
  may form a continuum rather than registers.

**Why register boundaries are approximate at low silhouettes:** k-means
assigns every passage to exactly one cluster even when many passages sit
between centroids. The representative passages you see are the five
*centroid-nearest* — the most typical members — so the summaries are more
coherent than the full clusters. Read the LLM's registers as descriptions of
cluster cores, not as partitions with sharp edges.

**Choosing a resolution yourself.** On real corpora the silhouette curve
across k is often nearly flat (differences of ~0.01 between neighboring k),
which means the "optimal" k is close to arbitrary: the data constrain the
family of reasonable partitions, not one number. When that is your
situation, treat granularity as an interpretive choice and make it
explicitly: pin `clustering.k`, or switch `clustering.method: hierarchical`
and inspect the tree at several cuts — under the hierarchical method a
coarser cut is always an exact merge of finer clusters, and `cda validate`
lets you move the cutoff line on the dendrogram after inspecting the
clusters. Findings that persist across nearby resolutions are robust;
findings visible at only one resolution are fragile. Whatever resolution
you settle on, the per-cluster evidence standard is unchanged: blind
verdicts (`cda validate`) and stability under resampling, not the
silhouette argmax.

## Reading the "Clustering Stability" section

Every report carries a **Clustering Stability (computed)** section, stating
something like *"k=2 matched 33% of 30 shuffled orderings."* This answers a question the silhouette alone cannot: not "how
separated are these clusters," but "if the identical passages had happened to
arrive in a different order, would clustering still have found this many
registers?"

The two numbers are not the same thing, and both matter:

- **Reproducibility** (guaranteed, always): the same corpus and config
  produce the exact same k every time you run `cda analyze` — a report is
  never a coin flip you happened to see once.
- **Robustness** (what this section measures): whether that k is *typical*
  for the term's actual embedding geometry, or a fluke of which specific
  points the seeded k-means++ initialization happened to draw as starting
  centroids. A low
  silhouette (above) is usually *why* a term is unstable — weak separation
  means nearby orderings can plausibly disagree about how many registers
  there are — but the stability check tells you directly, rather than
  leaving you to infer it from the silhouette alone.

A **100% match rate** is a genuinely reassuring result: it means the
partition is not just reproducible but structurally settled. A **low match
rate** (anything under ~60% of the 30 default orderings is worth noticing)
is not a defect in the run — it is the honest finding that this
term's register *count* is soft, even though every individual register the
LLM describes may still be a real, well-evidenced reading of the passages in
it. Treat k in that case as "this many, plausibly, give or take" rather than
a precise count you'd defend to a reviewer. For a fuller picture, run
`benchmarks/clustering_order_stability.py --term <term> --trials 100`
(still no LLM call, still free) — it reports the same match rate over more
orderings, plus the full distribution of k values seen across them.

If you'd rather not see this every run, `clustering.stability_check.enabled:
false` turns it off; the clustering itself is unaffected either way.

## What failure looks like

- **Generic or compound-form-heavy terms** (e.g. *control*): the term
  appears in control variable, control group, social control, quality
  control… The clusters then capture phrase-type, not theoretical register,
  and the LLM will dutifully name registers that are really collocations.
  Symptoms: registers whose labels are compound phrases; distance ratings
  of 1–2 everywhere. Remedy: analyze the specific compounds you actually
  care about, or accept that this term is not a good CDA target.
- **Premature saturation** (only under `saturation.scope: sample`, which is
  not the default): the saturation sweep ranks candidates by similarity to the
  *term's* embedding, so early increments over-sample passages closest to the
  term's dominant sense. If the structure of that dominant sense stabilizes
  quickly, saturation can be declared before minority registers enter the
  sample. Symptoms: saturation reached at a small N* while the candidate pool
  is large; a literature register you know exists is missing. Remedy: re-run
  with a higher `saturation.start`, or turn selection off again
  (`saturation.scope: prompt`, the default) for that term and accept a longer
  local clustering sweep.
- **Saturation not reached** (same condition): the report says so explicitly
  (and the paper's own runs include such cases). It means cluster structure
  was still moving at the cap — the register inventory may be unstable; treat
  counts as provisional. With saturation off the report reads "Saturation
  selection: disabled" followed by the number of literature passages
  clustered; every retrieved passage was clustered, so this caution does
  not apply.
- **Thin researcher pools** (< ~10 passages): one draft's phrasing dominates
  and "researcher registers" may just be that draft's sections. The report
  header shows N — check it before reading further. Under joint clustering a
  thin pool also means your passages may all land in one shared cluster, which
  reads as "conventional usage" when it is really "too little of your writing to
  say."
- **A window that changes the answer**: if you set `--added-from` /
  `--added-to`, the corpus is a *snapshot*, and clusters can appear or vanish
  purely because works entered your library later. That is the point of the
  feature — but compare a windowed run against an unwindowed one before reading
  a difference as a finding about the field. The report states the window and how
  many passages fell outside it.
- **A clock that isn't measuring what you think**: a window is only as
  meaningful as the timestamp behind it. Zotero's "Date Added" records when the
  *record* entered your library, so a rebuilt or re-imported library reports
  itself as entirely new and a cutoff then slices on the import, not on your
  reading. `--date-source filesystem` reads when each PDF arrived on disk
  instead, which survives a rebuild but tracks the download rather than the
  read. Neither is authoritative. If a snapshot's result surprises you, run it
  under both clocks: where they agree, the finding is about your corpus; where
  they diverge, it is about your metadata. Reports always name the clock used.

## When and how to overrule the LLM

The registers are proposals. Everything needed to check them is in the
report: cluster sizes, silhouettes, and the exact representative passages
with sources. Overrule when:

- two "registers" read as the same operative definition in different
  vocabulary (the distance rating should have been 1–2);
- a register's representatives visibly span two definitions (the LLM should
  have split; the audit section may show merge/split declarations);
- a label is catchier than the evidence (LLMs name confidently — check the
  quoted sentences, not the label).

Reports are plain markdown files you own: annotate them, or re-run —
same corpus and config reproduce the same clusters (only LLM wording varies;
see the stability benchmark). The **Cluster–Register Audit** table recomputes
the k-means↔register divergence rate from the report text; its counts are a
documented heuristic that reads only the structured per-register
`**k-means/LLM mapping:**` declarations (or a register table) in section 3 —
negation-aware within each clause, ignoring prose mentions, and reporting n/a
when no structured mapping is present. The rate is merges plus splits over
the declared registers. Register-table rows it could not use, or found
twice, are listed in the audit notes, as are reversed cluster ranges and
stated rates that cannot be paired unambiguously with one corpus. Treat the LLM's prose as the primary
record and the audit as an independent cross-check: when the rate the LLM
states for a corpus differs from the recomputed one, the audit says so.

Every report also carries a **Quotation Grounding** section: each passage
the LLM quotes (≥6 words) is checked verbatim against the exact prompt it
was given, and the score (e.g. `14/14 verified`) is printed with any
unverified spans listed. Fragments are matched on whole words, and a quote
spliced with `...` or `[...]` must match fragment by fragment, in order,
within one passage. A quote whose marks nest ambiguously (a Latin letter
touching the outside of a quote mark, as when a quoted passage contains its
own quotes) is listed as "ambiguous nesting, not checked" and counts as
unverified; footnote digits, underscores and non-Latin text next to a mark
do not count. A final . ? or ! inside the closing mark is retried without it. This is the
computed hallucination guard: a quote the model didn't receive cannot be
evidence. *Unverified* does not
automatically mean *fabricated* — models paraphrase and splice — but an
unverified quotation is precisely where you should open the saved
`_prompt.md` file and compare before relying on the claim. If unverified
quotations recur, prefer the passage texts in the prompt file over the
report's quotations.

## The inspect-before-send workflow

The pipeline's human-in-the-loop mechanism is that **no prompt needs to be a
surprise**:

```bash
cda analyze --term novelty --dry-run     # print everything, send nothing
```

(GUI: the *Prompt preview only* toggle.) Every live run also saves the exact
user prompt as `drift_<term>_<timestamp>_prompt.md` beside the report, so
what was sent is always auditable afterwards. If any passage in the preview
should not go to an external API, remove the source from the corpus or
drafts directory and re-run.

## Multi-model verification (experimental)

With `llm.verification_model` set, `cda verify --term X` re-runs register
assignment on the byte-identical prompt with a second model and writes an
agreement table. Read it structurally: matching register *counts* and
mappings across models is meaningful support; label wording always differs;
distance-rating deltas of 1 are ordinary ordinal noise, deltas ≥ 2 deserve a
human look at both reports. Disagreement is information about model
sensitivity, not an error signal — which is why the hook ships as
experimental and the primary report remains authoritative.

## Longitudinal reading

Reports are timestamped and accumulate. The designed use is before/after:
run a term, revise the manuscript, re-run, and compare the researcher
registers and flags. Single-run findings are hypotheses; the divergence
flags mark where to look, never what to conclude.

## Validating the clustering yourself (experimental)

`cda validate --term X` plus the GUI's Validate tab put YOU in the
validator's seat: for each cluster you read the passages and record a
verdict (coherent / mixed / overlaps / artifact) before an LLM agent's
independent reading of the same passages is revealed. Four things to keep
straight:

- **Why verdict-before-reveal matters.** If you saw the agent's reading
  first, your judgment would be anchored by it, and "you agreed with the
  agent" would mean nothing. The session software refuses to reveal a
  reading until your verdict is committed, so the agreement statistics are
  a real independence check. Post-reveal revisions are recorded but never
  counted.
- **Reading kappa at small n.** Agreement is computed over clusters, and a
  term rarely yields more than a handful, so Cohen's kappa comes with wide
  uncertainty. Read it together with the raw percent agreement and the
  verdict table; treat it as evidence of direction, not a precise
  measurement. Configuring a second reader model (`validation.readers`)
  strengthens the check.
- **Curation is repair, not evidence.** Before the first merge, split or
  recut, every cluster of the original clustering needs a blind verdict.
  Asking the revision agent about clusters counts as revealing them, like
  showing their readings. Merges and splits you apply produce
  a curated partition inside the session. The agreement numbers always
  refer to the original clustering: agreement with a structure you just
  edited into shape would be circular. A curated partition therefore lives
  inside the session folder and is never re-analyzed.
- **The verdict trail is the primary artifact.** The report documents what
  you judged, what the agent read blind, where you disagreed, what you
  changed and why. That documented trail, not any single number, is what a
  reviewer can evaluate.

## Reading the emergence report (experimental)

`cda emergence --term X` asks a different longitudinal question: as you
expanded your library, when did each literature cluster for the term
coalesce? Four things to keep straight when reading it:

- **The axis is acquisition, not publication.** The timeline is Zotero's
  "Date Added" — your reading trajectory — or, with `--date-source
  filesystem`, when each PDF arrived on disk; the report and chart name the
  clock used. A cluster "born in 2021" entered
  *your library* then; the field may have written it decades earlier.
- **Birth** is the first snapshot whose clustering already contained a
  cluster matching the final one (centroid cosine ≥ the configured
  threshold). Late birth can mean a genuinely new reading you picked up —
  or simply that too few of its passages had arrived to form a cluster.
- **Persistence** is the share of post-birth snapshots where the cluster
  kept its match. Low persistence means the cluster flickered in and out as
  the library grew — treat its birth date as soft.
- **Undated passages are excluded, and the coverage line says how many.**
  Below ~80% coverage the timeline describes a subsample of your library;
  re-ingest with your Zotero CSV before reading too much into it.

Like the divergence flags, emergence output marks where to look — a
late-born cluster is a prompt to ask what you were reading and writing at
the time, not a finding in itself.
