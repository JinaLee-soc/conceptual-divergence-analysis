"""Parsing rules of the cluster–register audit: negation scope, stated rates, cluster lists."""

import pytest

from cda.registers.audit import _classify_mapping_line, audit_report, stated_rates

_FIELD = "- **k-means/LLM mapping:** "


# ---- negation scope -----------------------------------------------------


@pytest.mark.parametrize(
    "mapping",
    [
        "1:1 with embedding cluster 0 (no k-means merge)",
        "1:1 with embedding cluster 2 (no cross-cluster merging)",
        "1:1 with cluster 2 (not a re-split of cluster 1)",
        "1:1 with cluster 2; not merged, split, or reassigned",
        "1:1 with embedding cluster 3; merges: none",
        "1:1 with embedding cluster 3; splits: none",
        "1:1 with embedding cluster 3; merges: 0",
        "1:1 with embedding cluster 3; no merges",
        "no merges; 1:1 with cluster 2",
        "1:1 with embedding cluster 3 (does not split or merge)",
        "1:1 with embedding cluster 3 (neither merges nor splits)",
        "1:1 with embedding cluster 2 - not merged with any other",
        "1:1 with embedding cluster 2, which is not split",
    ],
)
def test_hyphenated_words_and_negated_lists_keep_negation(mapping):
    assert _classify_mapping_line(_FIELD + mapping) == "one_to_one"


@pytest.mark.parametrize(
    "mapping",
    [
        "no merge, but cluster 4 splits into L-R2 and L-R3",
        "no merge; however, cluster 4 splits into L-R2 and L-R3",
        "no merge, whereas cluster 4 splits into L-R2 and L-R3",
        "no merge, while cluster 4 splits into L-R2 and L-R3",
        "no merge; Cluster 4 splits into L-R2 and L-R3",
        "not 1:1 – Cluster 4 is split across L-R2/L-R3",
        "not 1:1 — Cluster 4 is split across L-R2/L-R3",
        "(not merged) cluster 4 splits into L-R2 and L-R3",
    ],
)
def test_contrastive_connectors_and_clause_ends_stop_negation(mapping):
    assert _classify_mapping_line(_FIELD + mapping) == "split"


def test_affirmative_merge_with_a_hyphenated_word_is_still_a_merge():
    assert _classify_mapping_line(_FIELD + "merges k-means clusters 1 and 2") == "merge"
    assert _classify_mapping_line(_FIELD + "merges: 2 clusters (1 and 2)") == "merge"


# ---- stated-rate cross-check --------------------------------------------

_SECTION = "## 2. Researcher vs. Literature Divergence\n"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "The divergence rate is 1/4 for the researcher corpus and 2/5 for the literature corpus.",
            {"researcher": 0.25, "literature": 0.4},
        ),
        (
            "Divergence rate: 0/3 (researcher) and 1/4 (literature).",
            {"researcher": 0.0, "literature": 0.25},
        ),
        (
            "k-means/LLM divergence rate: researcher 0/3 (0%); literature 1/4 (25%)",
            {"researcher": 0.0, "literature": 0.25},
        ),
        (
            "divergence rate: researcher 0/3 (0%), literature 2/4 (50%)",
            {"researcher": 0.0, "literature": 0.5},
        ),
        (
            "divergence rate — researcher: 0.0; literature: 0.25",
            {"researcher": 0.0, "literature": 0.25},
        ),
        (
            "Researcher corpus (3 registers, k=3) divergence rate 1/3; literature (k=4) 0/4",
            {"researcher": pytest.approx(1 / 3), "literature": 0.0},
        ),
    ],
)
def test_stated_rate_belongs_to_the_corpus_in_its_own_clause(line, expected):
    assert stated_rates(_SECTION + line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "Researcher and literature divergence rates are 0/2 and 1/4.",
        "Divergence rate for researcher and literature: 1/3, 2/4.",
        "Divergence rate: researcher literature 1/3 2/4.",
    ],
)
def test_ambiguous_stated_rate_is_none_and_noted(line):
    report = "### Cluster A: x\n" + _SECTION + line + "\n"
    assert stated_rates(report) == {"researcher": None, "literature": None}
    notes = audit_report(report, 1, 1)["notes"]
    assert any("could not be paired unambiguously" in n and "researcher" in n for n in notes)
    assert any("could not be paired unambiguously" in n and "literature" in n for n in notes)


# ---- cluster lists in the register table --------------------------------


def _table(*cells: str) -> str:
    rows = "\n".join(f"| L-R{i} | def | {cell} |" for i, cell in enumerate(cells, 1))
    return (
        "### Cluster A: x\n"
        + _SECTION
        + "| Register | Definition | k-means clusters absorbed |\n|---|---|---|\n"
        + rows
        + "\n"
    )


@pytest.mark.parametrize(
    "cell",
    [
        "Clusters 0–2",
        "Clusters 0-2",
        "Clusters 0 to 2",
        "Clusters 0 through 2",
        "Cluster 0, 1, and 2",
        "clusters 0, 1 and 2",
        "Cluster 0, 1, 2",
        "C0, C1, C2",
        "Cluster 0 & 1 + 2",
    ],
)
def test_cluster_ranges_and_lists_name_every_cluster(cell):
    # One register absorbing clusters 0, 1, 2 and another taking cluster 2:
    # a merge plus a split of the shared cluster 2 only if 2 was parsed.
    lit = audit_report(_table(cell, "Cluster 2"), 1, 3)["literature"]
    assert (lit["merge"], lit["split"], lit["one_to_one"]) == (1, 1, 0)


def test_reversed_cluster_range_is_noted_not_counted():
    audit = audit_report(_table("Clusters 3–1", "Cluster 4"), 1, 5)
    assert audit["literature"]["one_to_one"] == 1
    assert audit["literature"]["merge"] == 0
    assert any("'L-R1'" in n and "backwards" in n for n in audit["notes"])


# ---- end to end -------------------------------------------------------------

_ONE_TO_ONE_REPORT = """## Summary
x
## 1. Polysemy Map
### Cluster A: maintenance
- **k-means/LLM mapping:** 1:1 with embedding cluster 0 (no k-means merge)
### Cluster B: repair
- **k-means/LLM mapping:** 1:1 with embedding cluster 1 (not a re-split of cluster 0)
## 2. Researcher vs. Literature Divergence
### Literature Register I: x
- **k-means/LLM mapping:** 1:1 with embedding cluster 0 (no cross-cluster merging)
### Literature Register II: y
- **k-means/LLM mapping:** 1:1 with embedding cluster 1; merges: none
### Divergence
The k-means/LLM divergence rate is 0/2 for the researcher corpus and 0/2 for the literature corpus.
## 3. Actionable Recommendations
"""


def test_true_one_to_one_report_with_asides_audits_to_zero_without_notes():
    audit = audit_report(_ONE_TO_ONE_REPORT, 2, 2)
    assert audit["researcher"]["divergence_rate"] == 0.0
    assert audit["literature"]["divergence_rate"] == 0.0
    assert audit["notes"] == []


_RANGE_TABLE_REPORT = """### Cluster A: x
## 2. Researcher vs. Literature Divergence
| Register | Definition | k-means clusters absorbed | n |
|---|---|---|---|
| L-R1 | a | Clusters 0–2 | 40 |
| L-R2 | b | Cluster 3 | 10 |
| L-R3 | c | Cluster 4, 5, and 6 | 12 |
| L-R4 | d | Cluster 6 | 3 |
Divergence rate: literature {stated}
"""


def test_range_and_oxford_comma_table_recomputes_the_careful_rate():
    agree = audit_report(_RANGE_TABLE_REPORT.format(stated="3/4 (75%)"), 1, 7)
    lit = agree["literature"]
    assert (lit["merge"], lit["split"], lit["one_to_one"]) == (2, 1, 1)
    assert lit["divergence_rate"] == 0.75
    assert not any("disagrees" in n for n in agree["notes"])
    disagree = audit_report(_RANGE_TABLE_REPORT.format(stated="1/4 (25%)"), 1, 7)
    assert any("literature rate 0.75 disagrees" in n for n in disagree["notes"])


def test_unspaced_em_dash_ends_the_negated_clause():
    from cda.registers.audit import _has_affirmative

    assert _has_affirmative("no merge—cluster 4 splits into two registers", "split")
    assert not _has_affirmative("no merge—cluster 4 splits into two registers", "merge")
