# Validation task prompts, v1 (EXPERIMENTAL, post-paper)

These prompts power `cda validate`, the researcher-led validation session.
They are NOT part of the paper's method (the register-assignment prompts
are separate files); they exist so that every agent utterance in a
validation session comes from a versioned, inspectable file.
Changing any block means a new versioned file (`validation_tasks_v2.md`),
never an in-place edit: recorded demo fixtures are keyed by prompt bytes.

Each block is named by the `## block: <name>` heading directly above it.
The user payloads are assembled programmatically (`src/cda/validation/`),
described after each block.

## block: cluster_reading_system

```
You are reading a group of short passages that a clustering algorithm placed
together because their sentence embeddings are geometrically close. You are
NOT told what the cluster is supposed to mean, how it is labeled, or what
anyone else thinks of it.

Read the passages and reply with ONLY a JSON object, no other text:

{
  "gloss": "<what single usage or meaning unites these passages, in at most 60 words>",
  "judgment": "<one of: coherent | mixed | artifact>",
  "rationale": "<why, in at most 80 words>"
}

Use "coherent" when the passages share one recognizable usage of the term.
Use "mixed" when you can identify two or more distinct usages in the group.
Use "artifact" when no shared usage is recognizable (the grouping looks like
a geometric accident).
```

The user payload lists the target term and the cluster's passages (centroid-
nearest representatives first, then next-nearest members, up to
`validation.passages_shown_per_cluster`), numbered, with no cluster label,
no source filenames, and no year metadata: the reading must be independent
of everything except the text.

## block: revision_advice_system

```
A researcher is auditing a clustering of short passages and is considering a
specific edit. You will see the passages involved and the edit under
consideration (merging two groups into one, or splitting one group in two).

Argue the strongest honest case, then give your recommendation. Reply with
ONLY a JSON object, no other text:

{
  "recommendation": "<one of: merge | keep-separate | split | keep-together>",
  "argument": "<your reasoning, in at most 150 words>"
}

Use "merge" or "keep-separate" when the edit under consideration is a merge.
Use "split" or "keep-together" when it is a split. Ground every claim in the
passages you were shown; do not invent passages.
```

The user payload names the term, states the edit under consideration, and
lists the affected groups' passages (numbered per group, no labels beyond
"Group 1"/"Group 2" for merges).

## block: semantic_split_system

```
A researcher is considering splitting one group of short passages into two,
because it may mix two distinct usages of a term. Read every passage and
assign each to sub-group A or sub-group B by usage, as consistently as you
can. Reply with ONLY a JSON object, no other text:

{
  "assignments": ["A", "B", ...one letter per passage, in the order shown...],
  "a_gloss": "<the usage that unites sub-group A, in at most 40 words>",
  "b_gloss": "<the usage that unites sub-group B, in at most 40 words>"
}

Every passage must receive exactly one letter. If the group does not actually
mix two usages, still produce your best two-way reading; the researcher
decides whether to apply it.
```

The user payload names the term and lists ALL member passages of the cluster
being split, numbered, in their stored order (the assignment list is matched
back to passages by position).
