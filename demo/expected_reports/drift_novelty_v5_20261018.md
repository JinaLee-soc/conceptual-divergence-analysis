<!-- generated: 2026-10-18T19:13:37.649158 | model: claude-code-session | cda: v1.0.0 -->
<!-- chunks_total: 38, chunks_with_term_window: 38, researcher_passages: 38, literature_passages: 56 -->
<!-- clustering_mode: joint, joint_k=2, joint_silhouette=0.137 -->
<!-- researcher_clusters: k=2, sil=0.137, N=38 | literature_clusters: k=2, sil=0.137, N=56 -->


**Demo provenance:** Claude-authored demonstration response written in a Claude Code session; not an Anthropic API call.

# CDA Report: "novelty"

## Summary

The researcher uses novelty in two registers: novelty as a status an audience confers after repairing its expectations, and novelty as the advertised or counted quantity the drafts criticize. This is structured polysemy, although the first register also holds the drafts' statements of the property view they reject. Against the literature, the researcher's central sense has no counterpart: the literature treats novelty as an epistemic property or value of findings and as a criterion that evaluation and measurement assess.

## 1. Polysemy Map

### Cluster A: Novelty as audience-conferred status

- **k-means/LLM mapping:** 1:1 with embedding cluster 0
- **Operative definition:** Novelty is the visible repair of an audience's expectations, a status ratified after the fact rather than an attribute a finding carries.
- **Associated themes:** #science-as-maintenance, #evaluation-infrastructures (run-level tags; no passage-level tags were supplied)
- **Example usages:**
  1. "Novelty, on the account developed here, is the visible repair of a competent audience's expectations, performed with materials the claiming group has strategically prepared." - *draft_01_novelty_as_audience_repair.md*
  2. "Novelty, on our account, is the controlled tearing of expectations; consensus is the controlled sealing of them." - *draft_04_revision_memo.md*
- **Semantic distance from other clusters:** A–B: 3 (adjacent usage)

### Cluster B: Novelty as advertised or counted quantity

- **k-means/LLM mapping:** 1:1 with embedding cluster 1
- **Operative definition:** Novelty is what fields advertise and what published counts register, which the drafts treat as advertising rather than as the accomplishment itself.
- **Associated themes:** #evaluation-infrastructures (run-level tag; no passage-level tags were supplied)
- **Example usages:**
  1. "Where the archives are thin, the defensible move is to say so, and to treat published novelty counts and consensus percentages as measures of advertising and of machinery uptime, respectively — which is what they are." - *draft_03_measuring_seams_methods_note.md*
  2. "A field's intellectual metabolism is visible in how it spends these two budgets, and the pathological cases line up neatly: fields that advertise novelty faster than audiences can repair it, and fields that declare consensus faster than the machinery can carry the displaced risk." - *draft_04_revision_memo.md*
- **Semantic distance from other clusters:** B–A: 3 (adjacent usage)

### Cross-Cluster Disambiguation Flags

- A–B (3): give the counted quantity its own label, such as "novelty claims" or "novelty scores", so that "novelty" alone always means the conferred status.
- Within Cluster A: the position the drafts reject sits beside their own definition, as in "Under the property view, novelty exists at submission time and reviewers detect it." The drafts mark it, so this is structured rather than undifferentiated polysemy, but each restatement of the property view should be labeled as the view under critique.

## 2. Researcher vs. Literature Divergence

### Literature Register I: Novelty as an epistemic property or value

- **k-means/LLM mapping:** 1:1 with embedding cluster 0
- **Operative definition:** Novelty is a value or property of scientific results, such as unexpectedness relative to prior experience, whose definition is debated.
- **Representative evidence:** "Various forms of novelty in science are alike in being unexpected on the basis of prior experience." - *2013_explaining_science_s_success_understanding_how_scientific_kn.abstract*

### Literature Register II: Novelty as an evaluated research quality

- **k-means/LLM mapping:** 1:1 with embedding cluster 1
- **Operative definition:** Novelty is an attribute of good research that reviewers, publishers, and measures assess.
- **Representative evidence:** "Also publishers believe post-publication evaluation of novelty, significance and relevance remains problematic." - *2017_let_the_community_decide_the_vision_and_reality_of_soundness.abstract*

### Divergence

- Cluster 0 holds 32 of the 38 researcher passages and 34 of the 56 literature passages. Both sides define novelty here, but in opposite directions. The literature locates novelty in the result, as in "Novelty is a core value in science, and a reliable measurement of novelty is crucial." The drafts locate it in the audience that ratifies the result. The closest literature passage is a theory of authorial novelty ("We review some of this literature and elaborate constituents of a theory of authorial novelty."), which still places novelty with the author rather than the audience.
- Cluster 1 holds 6 researcher passages and 22 literature passages. The literature treats novelty as a quality that evaluation and measurement can capture ("This study proposes a new approach of measuring the novelty of scientific articles based on both citation data and text data."), whereas the drafts treat the same counts as measures of advertising.

No cluster meets the researcher-concentration rule, and both shared clusters contain passages from each corpus. The divergence lies in the meaning assigned within shared regions, not in where the researcher's passages sit.

k-means/LLM divergence rate: researcher 0/2 (two 1:1 mappings); literature 0/2 (two 1:1 mappings).

## 3. Actionable Recommendations

- draft_01_novelty_as_audience_repair.md: define novelty as a conferred status at first use, and label the property view each time the introduction restates it.
- draft_03_measuring_seams_methods_note.md and draft_04_revision_memo.md: refer to the counted quantity as "novelty claims" or "novelty scores", never as novelty without qualification.
- Suggested parenthetical for readers from the measurement literature: novelty (here, a status an audience confers after repairing its expectations, not an attribute of the result).
- Consider splitting the term into "ratified novelty" (the accomplishment) and "claimed novelty" (the assertion and the counts that register it).

<!-- cda: end of LLM body; computed sections follow -->

---



## Clustering Stability (computed)

The identical, already-selected passage set was re-clustered under independently shuffled orderings (no LLM call, no cost) to check whether this run's k is a stable answer or an artifact of which order the passages happened to be in. The count is `clustering.stability_check.trials`; `benchmarks/clustering_order_stability.py` runs the same check with a full per-trial table.

**Joint partition**: k=2 matched 33% of 30 shuffled orderings (k ranged 2–12; silhouette 0.095–0.166) — treat this run's k as one plausible partition among several nearby ones, not the term's one true register count.


## Joint Clustering: where your passages sit (computed)

One spherical k-means partition over 38 of your passages and 56 literature passages pooled (k=2, silhouette=0.137 on the pooled passages). Each row is a shared cluster: the same region of embedding space, seen from both sides.

| Shared cluster | your passages | literature | your share | your most typical passage |
|---|---:|---:|---:|---|
| Cluster 0 | 32 | 34 | 48.5% | "Second, novelty is rationed." |
| Cluster 1 | 6 | 22 | 21.4% | "<!-- SYNTHETIC DOCUMENT — fictional researcher draft written for the CDA demo. The datasets, coders, all people, findings, and citations below are in…" |

- Every shared cluster contains passages from both sides.


## Computed Divergence Flags

Researcher and literature passages were clustered **together**, so each row counts the researcher passages that are actually members of that shared cluster. A cluster is **researcher-concentrated** when the researcher share of its combined membership exceeds 80% while the cluster holds under 10% of literature passages; clusters with ≤0 researcher members are **underrepresented in researcher usage**.

| Shared cluster | n lit | lit share | n researcher members | researcher share | flags |
|---|---:|---:|---:|---:|---|
| Cluster 0 | 34 | 60.7% | 32 | 48.5% | — |
| Cluster 1 | 22 | 39.3% | 6 | 21.4% | — |

*Flags are descriptive aids — they mark where researcher and literature usage diverge structurally, not whether that divergence is a problem.*


## Cluster–Register Audit (computed)

| Corpus | k-means clusters | LLM registers / declarations | Divergence rate |
|---|---:|---|---:|
| Researcher | 2 | 2 registers | 0.0 |
| Literature | 2 | 2×1:1, 0×merge, 0×split | 0.0 |

*Computed from the report text by a documented heuristic (see `cda.registers.audit`); the LLM's own prose statement of the mapping appears in the literature-divergence section above.*


## Quotation Grounding (computed)

**10/10** quoted passages verified verbatim against individual prompt passages (typography-normalized; cited filenames checked when supplied).

*Every checked quotation occurs in one representative passage in the material it was shown; short unattributed labels are excluded.*


## Run Parameters

- Model: `claude-code-session` | Embedding: `sentence-transformers/all-MiniLM-L6-v2` | Seed: 42 | Representatives/cluster: 5
- Clustering mode: joint (researcher + literature pooled into one partition, k=2)
- Saturation scope: prompt. Full literature pool clustered (56 passages, no subsampling); the prompt is bounded by clusters x representatives.
- Corpus timeframe: whole corpus (no acquisition-date window applied).
- Semantic distance scale: 1 = terminological variants; 2 = overlapping usage; 3 = adjacent usage; 4 = distinct registers; 5 = incommensurable.

### MyWork classification

*Counts are retrieved chunks; every term-sentence from a chunk inherits the chunk's signal.*

| MyWork signal | chunks |
|---|---:|
| default:literature | 38 |
