# CDA — Glossary

Definitions used throughout CDA. See [Architecture](../ARCHITECTURE.md) for
implementation details and [Validation protocol](validation_protocol.md)
for the researcher-led validation rules.

- **Register** — one distinguishable way a theoretical term is used; what the
  clustering + LLM assignment stage names and describes.
- **Drift report** — the timestamped markdown output of `cda analyze`: the
  canonical comparison of the researcher's usage registers against the
  literature's.
- **Validation** — the researcher's assessment of whether the clustering's
  structure is real, recorded as blind per-cluster verdicts and compared with
  independent agent cluster readings (`cda validate`). Distinct from
  *verification*.
- **Verification** — re-running register assignment on the identical saved
  prompt with a second model and comparing structure (`cda verify`). Distinct
  from *validation*.
- **Verdict** — the researcher's judgment of one cluster: `coherent`,
  `mixed` (mixes two or more usages), `overlaps` (duplicates another
  cluster), or `artifact` (no recognizable shared usage).
- **Blind-then-reveal protocol** — the verdict is committed before the agent's
  reading of the same cluster is revealed; post-reveal revisions are recorded
  separately and never enter agreement statistics.
- **Cluster reading** — an LLM agent's independent description of one
  cluster's passages (gloss, judgment, rationale), produced without seeing
  cluster labels, sources, or anyone's verdicts.
- **Reader** — a model that produces cluster readings.
- **Revision agent** — the model that argues for or against a researcher's
  proposed merge/split and proposes the semantic split. Separate role from
  the readers.
- **Curated partition** — a clustering edited by the researcher's applied
  merges/splits inside a validation session. Never feeds back into
  `cda analyze`.
- **Endorsement** — the researcher's closing act of a validation session:
  accepting the (possibly curated) structure as her considered reading. An
  edit applied afterwards withdraws it (the report keeps the withdrawal);
  endorse again once the changed clusters are judged.
- **Emergence** — how the literature clusters for a term coalesced over the
  researcher's library-acquisition timeline (`cda emergence`); acquisition
  time, not publication time.
- **MyWork resolution** — the signals that classify a corpus chunk as
  researcher-authored versus literature.
- **Saturation** — the incremental subsampling that caps the literature pool
  once cluster structure stabilizes (ARI plateau).
