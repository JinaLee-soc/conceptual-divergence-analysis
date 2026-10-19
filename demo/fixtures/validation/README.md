# Validation reading fixtures

Fixtures replayed by `cda validate --term novelty --demo --mock-llm`, keyed
by prompt fingerprint (sha256 of model + system + user prompt), one JSON per
(reader model, cluster).

## Current contents: SYNTHETIC placeholders

The `*.json` files here are **hand-written illustrative readings, not
recorded model responses** (each carries `"synthetic": true` and a `_note`
saying so). They are software test illustrations, not evidence about
register validity. The current four files (two researcher and two
literature clusters, one reader) answer the prompts of the current demo
clustering. They exist so the demo validation flow — blind verdicts, reveal,
agreement, report — can be tried fully offline.

There is one file per (reader model, cluster), so the count follows the demo
corpus's clustering rather than being fixed. Regenerate them with
`python demo/build_validation_fixtures.py`, which writes them from the toy
corpus with no network call; do that whenever the toy corpus, the clustering
defaults, or the reading prompt changes, since each fixture is keyed by a
fingerprint over the prompt it answers.

Consequences:

- Fine for demonstrating the process and testing the UI offline.
- **Not evidence.** Do not cite agreement numbers produced against these
  readings, and do not commit them as if they were recordings.

## Replacing them with real recordings

Run the demo validation live ONCE (a few cents, ~4 short calls; needs
`ANTHROPIC_API_KEY`):

```bash
rm demo/fixtures/validation/*.json
cda validate --term novelty --demo        # live demo runs record automatically
```

or use the GUI's Validate tab with "Use the toy demo corpus" checked and the
mock toggle off. Then commit the recorded JSONs (their `_note` says
"Cached LLM response...", not SYNTHETIC).

Notes for recording:

- Record on the same architecture as the drift fixtures (see ../README.md):
  embedding-dependent passage selection can differ across BLAS builds, and
  `novelty` is the cross-platform-stable demo term.
- Any change to `prompts/validation_tasks_v1.md`, the reading payload
  assembly, `validation.passages_shown_per_cluster`, the pinned model, or
  the toy corpus invalidates every fingerprint here at once; re-record all
  files together. Prompt changes mean a new versioned file
  (`validation_tasks_v2.md`), never an in-place edit.
- After recording, verify offline replay:
  `rm -rf demo/_runtime && cda validate --term novelty --demo --mock-llm`.
