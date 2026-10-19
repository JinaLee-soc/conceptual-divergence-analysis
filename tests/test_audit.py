"""Deterministic cluster–register audit parsing."""

from cda.registers.audit import audit_report, render_audit_markdown

SAMPLE_REPORT = """# Semantic Drift Report: "novelty"

## Summary
Two researcher senses detected.

## 1. Polysemy Map

### Cluster A: novelty-as-recombination
- **k-means/LLM mapping:** merges embedding clusters 0 and 1.
- **Operative definition:** one sentence.

### Cluster B: novelty-as-deviation
- **k-means/LLM mapping:** 1:1 with embedding cluster 2.
- **Operative definition:** another sentence.

## 2. Temporal Trajectory
Insufficient temporal data.

## 3. Researcher vs. Literature Divergence

### Literature Register L-I
- **k-means/LLM mapping:** maps 1:1 onto embedding cluster 0.

### Literature Register L-II
- **k-means/LLM mapping:** merges embedding clusters 1 and 2.

### Literature Register L-III
- **k-means/LLM mapping:** splits embedding cluster 3 into two senses.

## 4. Actionable Recommendations
- none
"""


def test_audit_counts_and_rates():
    audit = audit_report(SAMPLE_REPORT, k_researcher=3, k_literature=4)
    assert audit["researcher"]["llm_registers"] == 2
    assert audit["researcher"]["divergence_rate"] == 0.5
    assert audit["literature"]["one_to_one"] == 1
    assert audit["literature"]["merge"] == 1
    assert audit["literature"]["split"] == 1
    assert audit["literature"]["divergence_rate"] == round(2 / 3, 3)
    assert audit["notes"] == []


def test_audit_degrades_gracefully():
    audit = audit_report("Totally unstructured reply.", k_researcher=3, k_literature=4)
    assert audit["researcher"]["llm_registers"] == 0
    assert audit["researcher"]["divergence_rate"] is None
    assert audit["literature"]["divergence_rate"] is None
    assert len(audit["notes"]) == 2


def test_audit_negation_in_mapping_field_is_one_to_one():
    """ "no splitting or merging" inside a mapping field is 1:1, not merge+split."""
    report = (
        "### Cluster A: x\n"
        "## 3. Researcher vs. Literature Divergence\n"
        "### Literature Register L-I\n"
        "- **k-means/LLM mapping:** does not merge or split; it maps 1:1 onto cluster 0.\n"
    )
    audit = audit_report(report, k_researcher=1, k_literature=1)
    assert audit["literature"]["merge"] == 0
    assert audit["literature"]["split"] == 0
    assert audit["literature"]["one_to_one"] == 1
    assert audit["literature"]["divergence_rate"] == 0.0


def test_audit_structured_field_excludes_surrounding_prose():
    """Only bolded '**...mapping:**' bullets count; prose mentions are ignored."""
    report = (
        "### Cluster A: x\n"
        "## 3. Researcher vs. Literature Divergence\n"
        "- **k-means/LLM mapping:** Cluster 0 is coherent; no splitting or merging required.\n"
        "- **k-means/LLM mapping:** Cluster 1 is coherent; no splitting or merging required.\n"
        "In prose, these could be merged into one super-register for discussion.\n"
    )
    audit = audit_report(report, k_researcher=2, k_literature=2)
    assert audit["literature"]["one_to_one"] == 2
    assert audit["literature"]["merge"] == 0  # the prose "merged" is excluded
    assert audit["literature"]["divergence_rate"] == 0.0


def test_audit_table_mapping_is_recomputed_independently():
    """Consensus-style register table → rate recomputed from the table itself,
    not the LLM's stated figure; a disagreement with that figure is flagged."""
    report = (
        "### Cluster A: x\n"
        "## 3. Researcher vs. Literature Divergence\n"
        "| Literature Register | K-means clusters absorbed | Definition |\n"
        "|---|---|---|\n"
        "| **L-R1** | Cluster 0 (partial), Cluster 3 | ... |\n"
        "| **L-R2** | Cluster 1, Cluster 0 (partial) | ... |\n"
        "| **L-R3** | Cluster 2 | ... |\n"
        "| **L-R4** | Cluster 4, Cluster 5 | ... |\n"
        "\n"
        "- Literature corpus: Divergence rate: 1/6 ≈ 17%.\n"
    )
    audit = audit_report(report, k_researcher=1, k_literature=6)
    lit = audit["literature"]
    # 3 registers absorb >1 cluster (merge), L-R3 is 1:1 → (3+0)/4 = 0.75 —
    # NOT the LLM's stated 1/6, which ignored its own merges.
    assert (lit["merge"], lit["one_to_one"], lit["split"]) == (3, 1, 0)
    assert lit["divergence_rate"] == 0.75
    assert lit["rate_source"] == "recomputed"
    assert any("disagrees" in note and "0.167" in note for note in audit["notes"])
    assert "0.75" in render_audit_markdown(audit)


def test_audit_prose_only_mapping_is_na_even_with_a_stated_rate():
    """Prose mapping with only an LLM-stated rate (no bullets, no table) → n/a.

    The audit no longer echoes the LLM's figure: with nothing structured to
    recompute from, it abstains rather than adopt the LLM's arithmetic.
    """
    report = (
        "### Cluster A: x\n"
        "## 3. Researcher vs. Literature Divergence\n"
        "*(Literature Cluster 5 is split across L-R1 and L-R4.)*\n"
        "- Literature corpus: 1 boundary revised. Divergence rate: 1/6 ≈ 17%.\n"
    )
    audit = audit_report(report, k_researcher=1, k_literature=6)
    assert audit["literature"]["divergence_rate"] is None
    assert audit["literature"]["rate_source"] is None
    assert "not parsed" in render_audit_markdown(audit)


def test_audit_recomputed_rate_disagreeing_with_stated_is_flagged():
    """Bullets recompute a rate; if it contradicts the stated one, flag it."""
    report = (
        "### Cluster A: x\n"
        "## 3. Researcher vs. Literature Divergence\n"
        "### Literature Register L-I\n"
        "- **k-means/LLM mapping:** merges clusters 0 and 1.\n"
        "### Literature Register L-II\n"
        "- **k-means/LLM mapping:** merges clusters 2 and 3.\n"
        "- Literature corpus: no boundaries revised. Divergence rate: 0%.\n"
    )
    audit = audit_report(report, k_researcher=1, k_literature=4)
    # Two affirmative merges → recomputed 2/2 = 1.0, but the LLM stated 0%.
    assert audit["literature"]["divergence_rate"] == 1.0
    assert audit["literature"]["rate_source"] == "recomputed"
    assert any("disagrees" in note for note in audit["notes"])


def test_audit_markdown_render():
    audit = audit_report(SAMPLE_REPORT, k_researcher=3, k_literature=4)
    text = render_audit_markdown(audit)
    assert "| Researcher | 3 " in text
    assert "1×1:1, 1×merge, 1×split" in text
    assert "documented heuristic" in text
