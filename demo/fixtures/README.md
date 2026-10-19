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
- CI replays both terms, `novelty` and `consensus`, on linux/x64 with these
  same committed files: the `demo-mock` job runs `cda demo --mock-llm` for each
  term, and the `docker-smoke` job repeats both runs inside the Docker image.
- All responses concern **CC0 toy-corpus abstracts and fictional synthetic
  drafts** — they contain no copyrighted text and are safe to redistribute.

## Recording your own

Requires `ANTHROPIC_API_KEY` (≈$0.10–0.15 per term). Recording never
replaces a fixture that already replays for the same prompt; to replace the
committed file for a term, add `--overwrite-fixtures` to the commands below:

```bash
cda demo --term novelty --record-fixtures
cda demo --term consensus --record-fixtures
```
