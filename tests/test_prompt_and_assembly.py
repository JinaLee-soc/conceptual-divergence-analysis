"""The versioned prompt file and user-prompt assembly.

Guards the paper-facing invariants: the system prompt carries the §3.3
distance-scale anchors, and the assembled user prompt matches the appendix
template's structure.
"""

from cda.config import load_config
from cda.registers.assemble import build_user_prompt, format_cluster_block
from cda.registers.prompt_loader import load_system_prompt
from cda.types import ClusterSummary


def summary(k=2, n=10, sil=0.42):
    clusters = [
        {
            "cluster_id": i,
            "n_passages": n // k,
            "representatives": [f"rep text {i}-{j}" for j in range(2)],
            "sources": [f"source_{i}_{j}.pdf" for j in range(2)],
            "rep_roles": ["top1", "top2"],
        }
        for i in range(k)
    ]
    return ClusterSummary(
        True, n, k, sil, "[x] Embedding-level cluster separation detected", clusters
    )


def test_system_prompt_carries_published_scale_anchors():
    config = load_config()
    system = load_system_prompt(config)
    scale = config.section("distance_scale")
    for label in scale.values():
        assert str(label) in system, f"scale anchor '{label}' missing from system prompt"
    for required in ("POLYSEMY DETECTION", "LITERATURE DIVERGENCE"):
        assert required in system
    assert "TEMPORAL DRIFT" not in system
    assert "Temporal Trajectory" not in system
    # The stale wording from the pre-release pipeline must not resurface.
    assert "randomly sampled passages" not in system


def test_cluster_block_matches_appendix_shape():
    block = format_cluster_block(summary())
    lines = block.splitlines()
    assert lines[0].startswith("Total passages clustered: 10 | k=2 | silhouette=0.420")
    assert "--- Cluster 0 (n=5) ---" in block
    assert 'Sample 1 [top1] [source_0_0.pdf]: "rep text 0-0"' in block
    # The appendix template has no per-block "Note:" line (provenance note 4).
    assert "Note:" not in block


def test_user_prompt_section_order_and_saturation_note():
    config = load_config(
        overrides={"themes": [{"tag": "evaluation-theory", "description": "how work is judged"}]}
    )
    saturation = {
        "enabled": True,
        "ranking": "term",
        "n_candidates": 500,
        "n_selected": 250,
        "saturation_reached": True,
        "ari_threshold": 0.8,
        "consecutive_increments": 2,
    }
    prompt = build_user_prompt("novelty", summary(), summary(k=3, n=30), saturation, config)

    sections = [
        'TARGET TERM: "novelty"',
        "RESEARCHER CONTEXT:",
        "EXTRACTION NOTES:",
        "RESEARCHER CLUSTER STRUCTURE (pre-computed from all researcher passages):",
        "LITERATURE PASSAGES (cluster summary):",
        "Please perform two analyses",
    ]
    positions = [prompt.index(s) for s in sections]
    assert positions == sorted(positions), "appendix section order violated"

    assert "#evaluation-theory — how work is judged" in prompt
    assert "saturation-selected N*=250 from 500 candidates" in prompt
    assert "external anchor, independent of researcher and literature registers" in prompt
    assert "Saturation reached at ARI >= 0.8 for 2 consecutive increments" in prompt


def test_no_saturation_note_when_selection_not_applied():
    config = load_config()
    saturation = {"enabled": True, "n_candidates": 30, "n_selected": 30}
    prompt = build_user_prompt("novelty", summary(), summary(), saturation, config)
    assert "all 30 retrieved literature passages; no subsampling" in prompt
    assert "(no theme tags provided)" in prompt
