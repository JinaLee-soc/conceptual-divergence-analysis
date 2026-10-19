#!/usr/bin/env python3
"""Run-to-run stability of LLM register assignment (the paper's protocol).

Sends the *identical* saved prompt of a previous CDA analysis to the model N
times (default 5) with independent API calls, then reports structural
equivalence (register counts, 1:1/merge/split declarations) and wording
variation (pairwise label overlap) across runs — matching the paper's
inter-run consistency check ("cluster structures were equivalent across
runs, with only minor variation in wording").

Usage:
    # meaningful protocol (live; ~$0.10–0.15 per run → ~$0.60 for 5):
    python benchmarks/stability_protocol.py --demo --term novelty

    # harness check only (replays one committed fixture N times; variance is
    # trivially zero by construction — says nothing about the model):
    python benchmarks/stability_protocol.py --demo --term novelty --mock-llm

See benchmarks/README.md for running against a real corpus.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from cda.config import load_config
from cda.extraction.windows import safe_term_slug, tokenize
from cda.llm.mock import MockProvider
from cda.llm.providers import AnthropicProvider
from cda.registers.audit import literature_mapping_counts
from cda.registers.prompt_loader import load_system_prompt
from cda.registers.verify import extract_register_structure, find_latest_run


def run_summary(report_text: str) -> dict:
    structure = extract_register_structure(report_text)
    return {
        "n_registers": len(structure["researcher_register_labels"]),
        "labels": structure["researcher_register_labels"],
        # Reuse the core audit's parser (negation-aware, table-aware) so the
        # stability protocol counts 1:1/merge/split exactly as the report it
        # audits — not raw "merg"/"split" word hits, which would miscount
        # "no splitting or merging required" as one merge and one split.
        "mapping": literature_mapping_counts(report_text),
    }


def label_overlap(labels_a: list[str], labels_b: list[str]) -> float:
    """Jaccard overlap of label vocabularies — the wording-variation measure."""
    tokens_a = set().union(*(tokenize(label) for label in labels_a)) if labels_a else set()
    tokens_b = set().union(*(tokenize(label) for label in labels_b)) if labels_b else set()
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", default="novelty")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--demo", action="store_true", help="Use the toy corpus outputs")
    parser.add_argument("--config", help="Config file for a real corpus")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--fixtures-dir", default="demo/fixtures")
    parser.add_argument(
        "--save-replies",
        action="store_true",
        help=(
            "Write each run's raw reply beside the summary. Without this the "
            "replies are parsed and discarded, so a run that reports "
            "'structure did not reproduce' cannot afterwards be asked *which* "
            "structures it saw without paying for the calls again."
        ),
    )
    args = parser.parse_args()

    if args.demo:
        from cda.demo import demo_config, find_demo_dir

        config = demo_config(find_demo_dir())
    else:
        config = load_config(args.config)

    prompt_path, report_path = find_latest_run(config.path("paths.output_dir"), args.term)
    user_prompt = prompt_path.read_text(encoding="utf-8")
    # Same rule as `cda verify`: prefer the system prompt the run itself
    # saved. The configured prompt file can have changed since the analysis,
    # and variance measured against different instructions would not be
    # run-to-run variance.
    system_path = report_path.with_name(f"{report_path.stem}_system.md")
    if system_path.is_file():
        system_prompt = system_path.read_text(encoding="utf-8")
    else:
        system_prompt = load_system_prompt(config)
        print(
            "[stability] note: this run predates system-prompt persistence "
            f"(no {system_path.name}), so the currently configured system "
            "prompt is assumed to be the one the run used."
        )
    model = str(config.get("llm.model"))

    if args.mock_llm:
        provider = MockProvider(fixtures_dir=Path(args.fixtures_dir), model=model)
        print(
            "[stability] MOCK MODE: replaying one committed fixture "
            f"{args.runs}×. Run-to-run variance is trivially zero by "
            "construction — this exercises the harness only. Drop --mock-llm "
            "for the real protocol."
        )
    else:
        provider = AnthropicProvider(model=model)
        print(
            f"[stability] live protocol: {args.runs} independent calls to "
            f"{model} (~$0.10–0.15 each)."
        )

    summaries = []
    replies = []
    for run_index in range(args.runs):
        print(f"[stability] run {run_index + 1}/{args.runs} …")
        response = provider.complete(
            system=system_prompt,
            user=user_prompt,
            max_tokens=int(config.get("llm.max_tokens")),
            temperature=float(config.get("llm.temperature")),
        )
        summaries.append(run_summary(response))
        replies.append(response)

    counts = [summary["n_registers"] for summary in summaries]
    mappings = [tuple(sorted(summary["mapping"].items())) for summary in summaries]
    structurally_equivalent = len(set(counts)) == 1 and len(set(mappings)) == 1
    overlaps = [label_overlap(a["labels"], b["labels"]) for a, b in combinations(summaries, 2)]
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0

    lines = [
        f'# Register-assignment stability — term: "{args.term}"',
        "",
        f"Model: `{model}` · runs: {args.runs} · mode: "
        f"{'MOCK (harness check only — variance trivially zero)' if args.mock_llm else 'live'} · "
        f"prompt: `{prompt_path.name}`",
        "",
        "| Run | researcher registers | 1:1 | merge | split | labels |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for i, summary in enumerate(summaries, 1):
        mapping = summary["mapping"]
        lines.append(
            f"| {i} | {summary['n_registers']} | {mapping['one_to_one']} | "
            f"{mapping['merge']} | {mapping['split']} | {'; '.join(summary['labels'])} |"
        )
    lines += [
        "",
        f"**Structural equivalence across runs:** {'✅ yes' if structurally_equivalent else '⚠️ NO'} "
        "(identical register counts and mapping declarations).",
        f"**Wording variation:** mean pairwise label-vocabulary overlap "
        f"{mean_overlap:.2f} (1.0 = identical wording; the paper's protocol "
        "expects equivalent structure with sub-1.0 overlap).",
    ]
    report = "\n".join(lines) + "\n"
    # Print before writing: N live API calls' worth of result should survive
    # a filename or disk problem at the very last step.
    try:
        print(report)
    except UnicodeEncodeError:
        # Windows consoles often use a legacy codepage that cannot encode
        # corpus text. Degrade the console copy rather than die before the
        # report reaches disk (the file below is UTF-8 and unaffected).
        import sys

        encoding = sys.stdout.encoding or "utf-8"
        print(report.encode(encoding, errors="replace").decode(encoding))

    out_dir = config.path("paths.output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"benchmark_stability_{safe_term_slug(args.term)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    out_path.write_text(report, encoding="utf-8")
    print(f"[stability] written to {out_path}")
    if args.save_replies:
        for i, reply in enumerate(replies, 1):
            reply_path = out_path.with_name(f"{out_path.stem}_run{i}.md")
            reply_path.write_text(reply, encoding="utf-8")
        print(f"[stability] {len(replies)} raw replies written beside it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
