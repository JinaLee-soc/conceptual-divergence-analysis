# Validation protocol

## Evidence and independence

The researcher assesses each cluster before seeing the agents' independent
readings. Reader agents provide a second opinion; they do not replace human
judgment. Agreement statistics compare the researcher's blind verdicts with
agent readings for the original clustering. Report the number of eligible
clusters alongside agreement statistics, especially for small samples.

## Blind judgment and reveal

- Record a verdict before revealing a cluster's reading or requesting revision
  advice about it. These requirements are enforced in the session data layer.
- Revision advice is logged as a reveal, just like showing a reading.
- A verdict recorded after a reveal is marked `revised_after_reveal` and is
  excluded from agreement statistics.
- The validation report displays readings and per-reader agreement statistics
  only after the corresponding reveals have been logged.

## Curation

Before the first merge, split, or recut, record a blind verdict for every
cluster of the original clustering on the relevant sides. This preserves the
original judgment record before an edit can retire a cluster.

Curation records the researcher's judgments inside the validation session.
Untouched clusters retain their verdicts and provenance; affected clusters
require new judgments. Cluster silhouettes are recomputed to document the
quality of the edited partition.

An applied merge, split, or recut after endorsement withdraws the
endorsement: the session records the withdrawal (with the revision it
happened at) in its audit trail, the report shows it instead of "Endorsed",
and the GUI offers Endorse again once the changed clusters have verdicts.
Recuts are available in the GUI's Validate tab only.

Agreement statistics always use the original clustering. Curated partitions
are not fed into `cda analyze`, re-analyzed, or emitted as canonical CDA reports.
The original analysis outputs remain unchanged.
