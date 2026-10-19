<!-- generated: 2026-10-18T19:13:46.039672 | model: claude-code-session | cda: v1.0.0 -->
<!-- chunks_total: 82, chunks_with_term_window: 82, researcher_passages: 33, literature_passages: 103 -->
<!-- clustering_mode: joint, joint_k=3, joint_silhouette=0.103 -->
<!-- researcher_clusters: k=2, sil=0.103, N=33 | literature_clusters: k=3, sil=0.103, N=103 -->


**Demo provenance:** Claude-authored demonstration response written in a Claude Code session; not an Anthropic API call.

# CDA Report: "consensus"

## Summary

The researcher uses consensus in two registers: consensus as closure machinery that redistributes risk, which carries 30 of 33 passages, and consensus as a field-level quantity that surveys measure or fields declare, which the drafts name in order to criticize. This is structured polysemy with one dominant sense. Against the literature, the researcher shares a region with work on consensus as negotiated agreement but does not take up the climate-consensus register at all.

## 1. Polysemy Map

### Cluster A: Consensus as closure machinery

- **k-means/LLM mapping:** 1:1 with embedding cluster 0
- **Operative definition:** Consensus is an institutional operation that closes debate and redistributes risk, distinct from any underlying distribution of belief.
- **Associated themes:** #science-as-maintenance, #evaluation-infrastructures (run-level tags; no passage-level tags were supplied)
- **Example usages:**
  1. "Treating consensus as machinery rather than as agreement reorders the analytical questions." - *draft_02_consensus_as_closure_machinery.md*
  2. "This account also explains a pattern that agreement-based theories treat as pathological: the durability of consensus after its evidential basis erodes." - *draft_02_consensus_as_closure_machinery.md*
- **Semantic distance from other clusters:** A–B: 3 (adjacent usage)

### Cluster B: Consensus as a measured or declared field quantity

- **k-means/LLM mapping:** 1:1 with embedding cluster 2
- **Operative definition:** Consensus is a property of a field that surveys estimate or that fields announce, the operationalization the drafts argue presupposes the agreement view.
- **Associated themes:** #evaluation-infrastructures (run-level tag; no passage-level tags were supplied)
- **Example usages:**
  1. "The translation matters because both concepts are routinely operationalized through proxies that presuppose the theories I am arguing against: novelty as improbable recombination in reference lists, consensus as the modal answer in a survey of experts." - *draft_03_measuring_seams_methods_note.md*
  2. "A field's intellectual metabolism is visible in how it spends these two budgets, and the pathological cases line up neatly: fields that advertise novelty faster than audiences can repair it, and fields that declare consensus faster than the machinery can carry the displaced risk." - *draft_04_revision_memo.md*
- **Semantic distance from other clusters:** B–A: 3 (adjacent usage)

### Cross-Cluster Disambiguation Flags

- A–B (3): the drafts use the bare word both for the machinery and for the belief distribution it is contrasted with, as in "Survey instruments that poll belief distributions will find consensus where the machinery has simply not been needed". Reserve "consensus" for the machinery sense and call the surveyed quantity "expert agreement".

## 2. Researcher vs. Literature Divergence

### Literature Register I: Consensus as negotiated and measured agreement in science

- **k-means/LLM mapping:** 1:1 with embedding cluster 0
- **Operative definition:** Consensus is agreement that emerges from negotiation among scientists and varies in degree across fields.
- **Representative evidence:** "An increasingly precise scientific consensus gradually emerges from a process of negotiation, in which those participants who are members of the scientific elite exert most influence." - *1975_problem_areas_and_research_networks_in_science.abstract*

### Literature Register II: Scientific consensus on climate change

- **k-means/LLM mapping:** 1:1 with embedding cluster 1
- **Operative definition:** Consensus is the asserted or contested agreement among scientists about climate change, debated in public settings.
- **Representative evidence:** "There is a high degree of consensus amongst scientists about the basic science of global warming." - *2005_climate_change_menance_or_myth.abstract*

### Literature Register III: Consensus as the level or content of disciplinary agreement

- **k-means/LLM mapping:** 1:1 with embedding cluster 2
- **Operative definition:** Consensus names how far a discipline agrees on theory and method, or the prevailing view an author argues against.
- **Representative evidence:** "Without such unity of standards, consensus about scientific theories is supposedly unintelligible." - *1989_dominance_and_the_disunity_of_method_solving_the_problems_of.abstract*

### Divergence

- Cluster 0 holds 30 of the 33 researcher passages and 39 of the 103 literature passages. The researcher works beside the literature that studies how consensus forms, and the closest passages share the drafts' attention to process: "I focus on how, in this real world situation, concerns with credibility, authority, and expertise shaped the process by which this group negotiated the competing virtues of reaching consensus versus reporting accurately the nature and degree of disagreement among them." The divergence is in what consensus is. This literature treats it as agreement that should follow evidence ("Evidence is the most reliable measure, and consensus should change according to the evidence."), while the drafts treat it as machinery that can outlast its evidential basis.
- Cluster 2 holds 3 researcher passages and 46 literature passages. The literature measures consensus as a level of disciplinary agreement; the researcher's passages here name that measurement as a proxy to be rejected.
- Cluster 1 holds no researcher passages and 18 literature passages. The climate-consensus register is absent from the drafts, although draft_02 makes claims about what the public receives as scientific consensus, which is the domain this register addresses.

No cluster meets the researcher-concentration rule; the largest researcher share is 30 of 69 passages in cluster 0.

k-means/LLM divergence rate: researcher 0/2 (two 1:1 mappings); literature 0/3 (three 1:1 mappings).

## 3. Actionable Recommendations

- draft_02_consensus_as_closure_machinery.md: define consensus at first use as closure machinery and keep the agreement sense under a separate label.
- draft_02_consensus_as_closure_machinery.md: the claim about public reception of scientific consensus places the draft in the climate-consensus register; either engage that literature or limit the claim to the committee studied.
- draft_03_measuring_seams_methods_note.md: call the survey quantity "expert agreement" so that the proxy and the concept do not share a word.
- Suggested parenthetical for readers from the negotiation literature: consensus (here, the institutional operation that closes debate, not the agreement it announces).
- Consider splitting the term into "consensus machinery" and "expert agreement".

<!-- cda: end of LLM body; computed sections follow -->

---



## Clustering Stability (computed)

The identical, already-selected passage set was re-clustered under independently shuffled orderings (no LLM call, no cost) to check whether this run's k is a stable answer or an artifact of which order the passages happened to be in. The count is `clustering.stability_check.trials`; `benchmarks/clustering_order_stability.py` runs the same check with a full per-trial table.

**Joint partition**: k=3 matched 37% of 30 shuffled orderings (k ranged 3–10; silhouette 0.102–0.113) — treat this run's k as one plausible partition among several nearby ones, not the term's one true register count.


## Joint Clustering: where your passages sit (computed)

One spherical k-means partition over 33 of your passages and 103 literature passages pooled (k=3, silhouette=0.103 on the pooled passages). Each row is a shared cluster: the same region of embedding space, seen from both sides.

| Shared cluster | your passages | literature | your share | your most typical passage |
|---|---:|---:|---:|---|
| Cluster 0 | 30 | 39 | 43.5% | "Treating consensus as machinery rather than as agreement reorders the analytical questions." |
| Cluster 1 | 0 | 18 | 0.0% | — (none of your passages here) |
| Cluster 2 | 3 | 46 | 6.1% | "A field's intellectual metabolism is visible in how it spends these two budgets, and the pathological cases line up neatly: fields that advertise nov…" |

- Clusters with **none of your passages**: Cluster 1 — literature usage you have not taken up.


## Computed Divergence Flags

Researcher and literature passages were clustered **together**, so each row counts the researcher passages that are actually members of that shared cluster. A cluster is **researcher-concentrated** when the researcher share of its combined membership exceeds 80% while the cluster holds under 10% of literature passages; clusters with ≤0 researcher members are **underrepresented in researcher usage**.

| Shared cluster | n lit | lit share | n researcher members | researcher share | flags |
|---|---:|---:|---:|---:|---|
| Cluster 0 | 39 | 37.9% | 30 | 43.5% | — |
| Cluster 1 | 18 | 17.5% | 0 | 0.0% | underrepresented |
| Cluster 2 | 46 | 44.7% | 3 | 6.1% | — |

*Flags are descriptive aids — they mark where researcher and literature usage diverge structurally, not whether that divergence is a problem.*


## Cluster–Register Audit (computed)

| Corpus | k-means clusters | LLM registers / declarations | Divergence rate |
|---|---:|---|---:|
| Researcher | 2 | 2 registers | 0.0 |
| Literature | 3 | 3×1:1, 0×merge, 0×split | 0.0 |

*Computed from the report text by a documented heuristic (see `cda.registers.audit`); the LLM's own prose statement of the mapping appears in the literature-divergence section above.*


## Quotation Grounding (computed)

**10/10** quoted passages verified verbatim against individual prompt passages (typography-normalized; cited filenames checked when supplied).

*Every checked quotation occurs in one representative passage in the material it was shown; short unattributed labels are excluded.*


## Run Parameters

- Model: `claude-code-session` | Embedding: `sentence-transformers/all-MiniLM-L6-v2` | Seed: 42 | Representatives/cluster: 5
- Clustering mode: joint (researcher + literature pooled into one partition, k=3)
- Saturation scope: prompt. Full literature pool clustered (103 passages, no subsampling); the prompt is bounded by clusters x representatives.
- Corpus timeframe: whole corpus (no acquisition-date window applied).
- Semantic distance scale: 1 = terminological variants; 2 = overlapping usage; 3 = adjacent usage; 4 = distinct registers; 5 = incommensurable.

### MyWork classification

*Counts are retrieved chunks; every term-sentence from a chunk inherits the chunk's signal.*

| MyWork signal | chunks |
|---|---:|
| default:literature | 82 |
