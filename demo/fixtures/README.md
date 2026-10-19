# LLM fixtures (for --mock-llm)

Each `<fingerprint>.json` file is a cached register-assignment response for one
exact demo prompt, replayed by `MockProvider` so that
`cda demo --term … --mock-llm`, the test suite, and CI run with zero network
and zero cost.

## Current files

| File | Term | Prompt | Origin |
|---|---|---|---|
| `c093845b1deb6ec8.json` | novelty | v5 | written by Claude (claude-opus-5-5) |
| `d31386a610efdb61.json` | consensus | v5 | written by Claude (claude-opus-5-5) |

The two v5 files answer the exact prompts the current pipeline assembles for
the toy corpus. Their responses were written by Claude in a Claude Code
session after reading those prompts, without an Anthropic API call; the
fixture model is `claude-code-session`, each file's `provenance` field says
so, and mock reports name this provenance. They are **not recorded API
responses, researcher-endorsed findings, or manuscript results**. Live runs
always use the configured provider.

## How replay works

- The file name is `sha256(model + system prompt + user prompt)[:16]`. The
  same corpus, config, and prompt version reproduce the same fingerprint, and
  any change to those inputs makes replay fail loudly with a cache-miss error
  (by design) until a new fixture is recorded.
- `cda demo --mock-llm` first looks for a fixture recorded under the
  configured `llm.model` (one you recorded on this machine, below) and
  otherwise replays the committed `claude-code-session` fixture. The report
  header names the model whose response was replayed; only the
  `claude-code-session` replay carries the demo-provenance line.
- The prompt embeds the clustering, which can differ between CPU
  architectures, so the committed fixtures may miss on another machine. A
  recording made there fixes replay for that machine.
- CI replays both terms, `novelty` and `consensus`, on linux/x64 with these
  same committed files: the `demo-mock` job runs `cda demo --mock-llm` for each
  term, and the `docker-smoke` job repeats both runs inside the Docker image.
- All responses concern **CC0 toy-corpus abstracts and fictional synthetic
  drafts** — they contain no copyrighted text and are safe to redistribute.

## Recording your own

Requires `ANTHROPIC_API_KEY` (≈$0.10–0.15 per term). A recording is keyed by
your configured `llm.model`, so it is a new file beside the committed
`claude-code-session` fixtures and never replaces them. If a replayable
recording already exists for the same model and prompt, `--record-fixtures`
refuses before the paid call; add `--overwrite-fixtures` to replace it. (The
GUI's demo button with the mock toggle off records the same way and replaces
your earlier recording.)

```bash
cda demo --term novelty --record-fixtures
cda demo --term consensus --record-fixtures
```

The manual `record-fixtures` GitHub workflow does the same on linux/x64,
checks that a mock run replays the new recording (the report header must
name the configured model), and commits any new fixture file.
