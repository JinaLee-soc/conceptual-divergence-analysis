"""Regenerate the SYNTHETIC validation reading fixtures for the demo.

`cda validate --term novelty --demo --mock-llm` replays one fixture per
(reader model, cluster), keyed by prompt fingerprint. This script writes those
fixtures without any network call, so the offline demo works from a fresh
clone — the repo's rule is that every command runs under `--mock-llm`.

The readings it writes are **hand-shaped illustrations, not model output**.
Each carries `"synthetic": true` so nothing downstream can mistake them for
evidence; `demo/fixtures/validation/README.md` says the same.

Run from the repo root:  python demo/build_validation_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cda.corpus.embeddings import get_embedder
from cda.corpus.ingest import get_collection
from cda.demo import demo_config, ensure_toy_corpus_ingested, find_demo_dir
from cda.llm.base import prompt_fingerprint
from cda.pipeline import extract_passages, map_divergence
from cda.validation.agents import reading_user_prompt, resolve_readers, spec_model
from cda.validation.prompt_loader import load_validation_prompts
from cda.validation.run import _shown_by_cluster

TERM = "novelty"

# Cycled across clusters so the demo shows every branch of the agreement
# table: matches, mismatches, and a cluster the reader calls an artifact.
JUDGMENTS = [
    ("coherent", "One consistent use of the term across the passages."),
    ("coherent", "The passages share an operative definition."),
    ("mixed", "Two distinguishable uses sit in this cluster."),
    ("artifact", "The passages share phrasing rather than meaning."),
]


def main() -> int:
    demo_dir = find_demo_dir()
    config = demo_config(demo_dir)
    embedder = get_embedder(config)
    ensure_toy_corpus_ingested(config, demo_dir, embedder)

    collection = get_collection(config)
    researcher, literature, stats = extract_passages(
        TERM, config, embedder=embedder, collection=collection, progress=lambda _m: None
    )
    researcher_summary, literature_summary, literature, _sat, _joint = map_divergence(
        TERM,
        researcher,
        literature,
        stats,
        config,
        embedder=embedder,
        progress=lambda _m: None,
    )

    n_shown = int(config.get("validation.passages_shown_per_cluster", 8))
    shown_texts: dict[str, dict[int, list[str]]] = {}
    for side, summary, pool in (
        ("researcher", researcher_summary, researcher),
        ("literature", literature_summary, literature),
    ):
        if not summary.used:
            continue
        by_id = {p.chunk_id: p for p in pool}
        shown_texts[side] = {
            cluster: [by_id[chunk_id].text for chunk_id in ids]
            for cluster, ids in _shown_by_cluster(pool, n_shown).items()
        }

    prompts = load_validation_prompts(config)
    out_dir = Path(__file__).resolve().parent / "fixtures" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for spec in resolve_readers(config):
        for side in sorted(shown_texts):
            for index, cluster_id in enumerate(sorted(shown_texts[side])):
                judgment, rationale = JUDGMENTS[index % len(JUDGMENTS)]
                key = prompt_fingerprint(
                    spec_model(spec),
                    prompts["cluster_reading_system"],
                    reading_user_prompt(TERM, shown_texts[side][cluster_id]),
                )
                payload = {
                    "response": json.dumps(
                        {
                            "gloss": f"Illustrative reading of {side} cluster {cluster_id}.",
                            "judgment": judgment,
                            "rationale": rationale,
                        }
                    ),
                    "synthetic": True,
                    "_note": (
                        "Hand-written illustrative reading, NOT a recorded model "
                        "response. Do not cite agreement numbers computed against it. "
                        "See demo/fixtures/validation/README.md."
                    ),
                }
                if (out_dir / f"{key}.json").exists():
                    continue  # Preserve the response already bound to this exact prompt.
                (out_dir / f"{key}.json").write_text(
                    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                )
                written += 1
    print(f"[ok] wrote {written} synthetic validation fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
