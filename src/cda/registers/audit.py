"""Deterministic cluster–register audit.

The paper reports a match/split/merge divergence rate between k-means
clusters and LLM registers for both corpora (§3.3). The LLM states its
mapping (and often a rate) in the report; this module recomputes the rate
independently from the LLM's *declared* mapping so the number never rests on
the LLM's own arithmetic.

Two structured forms of the literature mapping are read:
per-register ``**...mapping:**`` bullets, and a "k-means clusters absorbed"
register table. Either yields per-register 1:1 / merge / split counts.

Denominator: the rate is (merge + split) / declared registers, where each
per-register declaration (one bullet or one table row) counts once. The
researcher rate is reported only when every researcher register heading has a
declaration. The LLM's own stated rates (fractions or percentages) are used
only as a cross-check — a disagreement is noted, never silently adopted. When
no structured mapping is present, the audit reports n/a rather than count prose.
"""

from __future__ import annotations

import re
from collections import Counter

# One heading rule splits the report: the literature-divergence section starts
# here, and everything before it is the researcher part.
_LITERATURE_HEADING = re.compile(
    r"^##\s+(?:\d+\.\s*)?(?:Researcher vs\.? Literature )?Divergence\b",
    re.MULTILINE,
)
_NEXT_SECTION = re.compile(r"^##\s", re.MULTILINE)

# A negation cue applies only within its own clause, so "no merge; Cluster 4
# splits" still declares a split. Clauses end at ; . : and a closing
# parenthesis, at a dash with spaces on both sides or at any en or em dash
# (never a hyphen inside a word such as "k-means" or "re-split"), and at a
# contrastive connector. A colon in "1:1" and a decimal point are not clause
# ends.
_NEGATION = r"(?:\bno\b|\bnot\b|\bwithout\b|\bneither\b|n't)"
_CLAUSE_BOUNDARY = re.compile(
    r"[;)–—]|\.(?!\d)|(?<!\d):(?!\d)|\s--?\s"
    r"|\b(?:but|however|whereas|while|although|though)\b"
)
# Commas inside a negated list keep the negation ("not merged, split, or
# reassigned"); a comma followed by anything other than a bare list item starts
# a new clause ("without merging, cluster 2 is split").
_LIST_CONTINUATION = re.compile(r"\s*(?:(?:or|nor|and)\s+)?")
_LIST_ITEM = re.compile(r"\s*(?:(?:or|nor|and)\s+)?[\w-]+\s*")
# "merges: none", "splits: 0" state the count as zero.
_ZERO_COUNT = re.compile(r"\w*\s*:\s*(?:none|no|n/a|0(?![\d./]))\b")

# A per-register mapping declaration is a markdown bullet whose *bolded label*
# names the mapping, e.g. "- **k-means/LLM mapping:** merges clusters 1 and 2".
# Only this structured field is counted, never any line that merely mentions
# "merge"/"split", so silhouette asides, narrative recaps, and the rate summary
# line cannot inflate the rate.
_MAPPING_FIELD = re.compile(r"^\s*[-*]\s*\*\*[^*]*\bmapping\b[^*]*:\*\*", re.IGNORECASE)

# Cluster references in a table cell: "Cluster 0", "Clusters 0 and 3",
# "clusters 0, 2 & 5", "Cluster 0, 2, and 3", "Cluster 0, Cluster 3", "C2",
# and inclusive ranges "Clusters 0–2", "0 to 2", "1 through 3".
_CLUSTER_PREFIX = r"(?:clusters?\s*|c(?=\d))"
_RANGE_DASH = r"(?:-|–|—|\bto\b|\bthrough\b|\bthru\b)"
_CLUSTER_ITEM = re.compile(
    rf"(?P<start>\d+)(?:\s*{_RANGE_DASH}\s*{_CLUSTER_PREFIX}?(?P<end>\d+))?", re.IGNORECASE
)
_ITEM = rf"\d+(?:\s*{_RANGE_DASH}\s*{_CLUSTER_PREFIX}?\d+)?"
_LIST_SEPARATOR = r"\s*(?:,\s*(?:(?:and\b|&)\s*)?|\band\b|&|\+|/)\s*"
_CLUSTER_LIST = re.compile(
    rf"\b{_CLUSTER_PREFIX}({_ITEM}(?:{_LIST_SEPARATOR}{_CLUSTER_PREFIX}?{_ITEM})*)",
    re.IGNORECASE,
)

# A stated rate is read only from lines that say "divergence rate". The line is
# cut into clauses at ";", "," and "and" (never inside parentheses), and a rate
# belongs to the corpus word in its own clause, before or after it:
# "researcher 0/3 (0%); literature 1/4", "1/4 for the researcher corpus and
# 2/5 for the literature corpus", "0/3 (researcher) and 1/4 (literature)".
# Restatements of one value ("1/6 ≈ 17%", "0/3 (0%)") count once.
_CORPUS = re.compile(r"\b(researcher|literature)\b", re.IGNORECASE)
_RATE_VALUE = re.compile(
    r"(?P<num>\d+)\s*/\s*(?P<den>\d+)"
    r"|(?P<pct>\d+(?:\.\d+)?)\s*%"
    r"|(?<![\d.])(?P<dec>[01]?\.\d+)(?![\d./%])"
)
_RATE_CLAUSE_SPLIT = re.compile(r",|;|\band\b", re.IGNORECASE)
_PARENTHETICAL = re.compile(r"\([^()]*\)")


def split_report_sections(report_text: str) -> tuple[str, str]:
    """Return (researcher part, literature-divergence section) of the report."""
    heading = _LITERATURE_HEADING.search(report_text)
    if not heading:
        return report_text, ""
    researcher = report_text[: heading.start()]
    rest = report_text[heading.start() :]
    following = _NEXT_SECTION.search(rest, heading.end() - heading.start())
    literature = rest[: following.start()] if following else rest
    return researcher, literature


def _count_researcher_registers(researcher_section: str) -> int:
    """Registers the LLM formed for the researcher corpus.

    The output format mandates one '### Cluster X: label' heading per
    researcher register in the Polysemy Map.
    """
    return len(re.findall(r"^###\s+Cluster\s+\S+", researcher_section, flags=re.MULTILINE))


def _classify_mapping_line(line: str) -> str | None:
    """Classify a single mapping declaration as one_to_one / merge / split.

    Returns None when the line makes no mapping claim. An affirmative merge or
    split wins; otherwise (an explicit 1:1, or merge/split all negated) it is a
    1:1 mapping.
    """
    low = line.lower()
    has_merge = bool(re.search(r"\bmerg\w*", low))
    has_split = bool(re.search(r"\bsplit\w*", low))
    has_one_to_one = bool(re.search(r"\b(?:1:1|one[- ]to[- ]one)\b", low))
    if not (has_merge or has_split or has_one_to_one):
        return None
    if has_merge and _has_affirmative(low, "merg"):
        return "merge"
    if has_split and _has_affirmative(low, "split"):
        return "split"
    return "one_to_one"


def _has_affirmative(low: str, stem: str) -> bool:
    """True when some occurrence of ``stem`` is not negated within its clause."""
    boundary_ends = [m.end() for m in _CLAUSE_BOUNDARY.finditer(low)]
    for occurrence in re.finditer(rf"\b{stem}", low):
        if _ZERO_COUNT.match(low, occurrence.end()):
            continue
        clause_start = max((end for end in boundary_ends if end <= occurrence.start()), default=0)
        if not _is_negated(low[clause_start : occurrence.start()]):
            return True
    return False


def _is_negated(clause_prefix: str) -> bool:
    """Whether the clause text before a merge/split word negates it.

    The negation must sit in the word's own comma segment, or in an earlier
    segment linked to it only by bare list items ("not merged, split, or ...").
    """
    *earlier, current = clause_prefix.split(",")
    if re.search(_NEGATION, current):
        return True
    if not _LIST_CONTINUATION.fullmatch(current):
        return False
    for segment in reversed(earlier):
        if re.search(_NEGATION, segment):
            return True
        if not _LIST_ITEM.fullmatch(segment):
            return False
    return False


def _bullet_counts(section: str) -> dict[str, int]:
    counts = {"one_to_one": 0, "merge": 0, "split": 0}
    for line in section.splitlines():
        if _MAPPING_FIELD.match(line):
            relation = _classify_mapping_line(line)
            if relation:
                counts[relation] += 1
    return counts


def _parse_clusters(cell: str) -> tuple[list[int], list[str]]:
    """Cluster ids named in a table cell, with ranges expanded inclusively.

    Returns (ids, reversed ranges); a reversed range such as "3–1" is reported
    rather than guessed at.
    """
    clusters: set[int] = set()
    reversed_ranges: list[str] = []
    for group in _CLUSTER_LIST.findall(cell):
        for item in _CLUSTER_ITEM.finditer(group):
            start = int(item.group("start"))
            end = int(item.group("end")) if item.group("end") else start
            if start > end:
                reversed_ranges.append(item.group(0))
            else:
                clusters.update(range(start, end + 1))
    return sorted(clusters), reversed_ranges


def _find_absorbed_column(lines: list[str]) -> tuple[int, int] | None:
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip().lower() for c in line.strip().strip("|").split("|")]
        for j, cell in enumerate(cells):
            if "cluster" in cell and ("absorb" in cell or "k-means" in cell):
                return i, j
    return None


def _literature_table_counts(section: str) -> tuple[dict[str, int] | None, list[str]]:
    """Per-register counts from a "k-means clusters absorbed" register table.

    Each data row is one register: absorbing >1 cluster is a merge; a sole
    cluster shared with another register is a split; one cluster to one
    register is 1:1. Rows that cannot be used are reported in the notes.
    Returns (None, notes) when no usable table is present.
    """
    lines = section.splitlines()
    located = _find_absorbed_column(lines)
    if located is None:
        return None, []
    header_idx, absorbed_col = located

    notes: list[str] = []
    register_clusters: dict[str, list[int]] = {}
    for line in lines[header_idx + 1 :]:
        if not line.lstrip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if re.fullmatch(r"[-:\s]*", cells[0]):
            continue  # separator row
        register = re.sub(r"[*`]", "", cells[0]).strip()
        cell = cells[absorbed_col] if len(cells) > absorbed_col else ""
        clusters, reversed_ranges = _parse_clusters(cell)
        if reversed_ranges:
            notes.append(
                f"Register table row '{register}' gives a cluster range that runs backwards "
                f"({', '.join(reversed_ranges)}); it is not counted in the literature rate."
            )
        elif not clusters:
            notes.append(
                f"Register table row '{register}' names no parseable k-means cluster; "
                "it is not counted in the literature rate."
            )
        elif register in register_clusters:
            notes.append(
                f"Register table lists '{register}' more than once; only its first row "
                "is counted in the literature rate."
            )
        else:
            register_clusters[register] = clusters
    if not register_clusters:
        return None, notes

    shared = Counter(c for clusters in register_clusters.values() for c in clusters)
    counts = {"one_to_one": 0, "merge": 0, "split": 0}
    for clusters in register_clusters.values():
        if len(clusters) > 1:
            counts["merge"] += 1
        elif shared[clusters[0]] > 1:
            counts["split"] += 1
        else:
            counts["one_to_one"] += 1
    return counts, notes


def _literature_mapping(report_text: str) -> tuple[dict[str, int], list[str]]:
    """Literature counts plus notes on table rows that could not be counted."""
    _, section = split_report_sections(report_text)
    counts = _bullet_counts(section)
    if sum(counts.values()) > 0:
        return counts, []
    table_counts, notes = _literature_table_counts(section)
    return table_counts or counts, notes


def literature_mapping_counts(report_text: str) -> dict[str, int]:
    """Independently count per-register mappings in the literature-divergence section.

    Reads whichever structured form the LLM used: bolded ``**...mapping:**``
    bullets (classified with clause-scoped negation), or a "clusters absorbed"
    register table. Bullets win when both are present. Prose that merely
    mentions merging or splitting is never counted.

    Public because it is the single source of truth for merge/split/1:1 counts:
    the stability benchmark reuses it.
    """
    return _literature_mapping(report_text)[0]


def _rate_value(match: re.Match) -> float | None:
    if match.group("num") is not None:
        den = int(match.group("den"))
        return int(match.group("num")) / den if den else None
    if match.group("pct") is not None:
        return float(match.group("pct")) / 100.0
    return float(match.group("dec"))


def _rate_clauses(line: str) -> list[tuple[set[str], list[float | None]]]:
    """(corpus words, rate values) per clause of a rate line, empty clauses dropped."""
    unbroken = _PARENTHETICAL.sub(lambda m: _RATE_CLAUSE_SPLIT.sub(" ", m.group()), line)
    clauses = []
    for text in _RATE_CLAUSE_SPLIT.split(unbroken):
        corpora = {m.group(1).lower() for m in _CORPUS.finditer(text)}
        values = [_rate_value(m) for m in _RATE_VALUE.finditer(text)]
        if corpora or values:
            clauses.append((corpora, values))
    return clauses


def _single_value(values: list[float | None]) -> float | None:
    """The one rate a clause states; None when it states none or several."""
    if not values or None in values:
        return None
    first = values[0]
    return first if all(abs(v - first) <= 0.01 for v in values) else None


def _line_rates(line: str) -> tuple[dict[str, float], set[str]]:
    """Rates a line pairs unambiguously with a corpus, and corpora it leaves unclear.

    A line is read only when every clause pairs one corpus with one value;
    a corpus word or a number left without a partner means the pairing is a
    guess, so every corpus on that line is reported unclear instead.
    """
    clauses = _rate_clauses(line)
    mentioned = set().union(*(corpora for corpora, _ in clauses)) if clauses else set()
    rates: dict[str, float] = {}
    for corpora, values in clauses:
        value = _single_value(values)
        if len(corpora) != 1 or value is None:
            return {}, mentioned
        (corpus,) = corpora
        if corpus in rates and abs(rates[corpus] - value) > 0.01:
            return {}, mentioned
        rates.setdefault(corpus, value)
    return rates, set()


def _read_stated_rates(report_text: str) -> tuple[dict[str, float | None], list[str]]:
    """Stated rates per corpus plus notes on rate statements too ambiguous to read."""
    researcher, literature = split_report_sections(report_text)
    rates: dict[str, float | None] = {"researcher": None, "literature": None}
    unclear: dict[str, str] = {}
    for line in (researcher + "\n" + literature).splitlines():
        if "divergence rate" not in line.lower():
            continue
        line_rates, line_unclear = _line_rates(line)
        for corpus, value in line_rates.items():
            if rates[corpus] is None:
                rates[corpus] = value
        for corpus in line_unclear:
            unclear.setdefault(corpus, line.strip())
    notes = [
        f"The LLM's stated {corpus} divergence rate could not be paired unambiguously "
        f"with its corpus in '{line}'; it is not cross-checked."
        for corpus, line in sorted(unclear.items())
        if rates[corpus] is None
    ]
    return rates, notes


def stated_rates(report_text: str) -> dict[str, float | None]:
    """Rates the LLM itself states for each corpus (0–1), or None when absent or ambiguous."""
    return _read_stated_rates(report_text)[0]


def _rate(counts: dict[str, int]) -> float:
    return round((counts["merge"] + counts["split"]) / sum(counts.values()), 3)


def _disagreement_note(corpus: str, recomputed: float, stated: float | None) -> list[str]:
    if stated is None or abs(stated - recomputed) <= 0.01:
        return []
    return [
        f"Independently computed {corpus} rate {recomputed} disagrees with the "
        f"LLM's stated {round(stated, 3)}: the LLM's declared mapping implies a "
        "different number than the figure it wrote; review the report's mapping declarations."
    ]


def audit_report(report_text: str, k_researcher: int, k_literature: int) -> dict:
    """Compute the divergence audit; a rate is None when it cannot be recomputed."""
    researcher_section, _ = split_report_sections(report_text)
    stated, stated_notes = _read_stated_rates(report_text)
    result: dict = {
        "researcher": {"k_means_clusters": k_researcher},
        "literature": {"k_means_clusters": k_literature},
        "notes": [],
    }

    n_researcher = _count_researcher_registers(researcher_section)
    researcher_counts = _bullet_counts(researcher_section)
    declared_researcher = sum(researcher_counts.values())
    result["researcher"]["llm_registers"] = n_researcher
    result["researcher"].update(researcher_counts)
    if n_researcher and declared_researcher == n_researcher:
        rate = _rate(researcher_counts)
        result["researcher"]["divergence_rate"] = rate
        result["notes"] += _disagreement_note("researcher", rate, stated["researcher"])
    else:
        result["researcher"]["divergence_rate"] = None
        result["notes"].append(
            f"Researcher mapping declarations missing or incomplete ({declared_researcher} "
            f"declarations for {n_researcher} register headings); rate unavailable."
        )

    counts, table_notes = _literature_mapping(report_text)
    result["literature"].update(counts)
    result["notes"] += table_notes
    if sum(counts.values()) > 0:
        rate = _rate(counts)
        result["literature"]["divergence_rate"] = rate
        result["literature"]["rate_source"] = "recomputed"
        result["notes"] += _disagreement_note("literature", rate, stated["literature"])
    else:
        result["literature"]["divergence_rate"] = None
        result["literature"]["rate_source"] = None
        result["notes"].append(
            "No structured mapping (per-register '**...mapping:**' bullets or a "
            "'clusters absorbed' register table) found in the literature-divergence section; literature "
            "rate unavailable."
        )
    result["notes"] += stated_notes
    return result


def render_audit_markdown(audit: dict) -> str:
    lines = [
        "| Corpus | k-means clusters | LLM registers / declarations | Divergence rate |",
        "|---|---:|---|---:|",
    ]
    researcher = audit["researcher"]
    lines.append(
        f"| Researcher | {researcher['k_means_clusters']} "
        f"| {researcher['llm_registers']} registers "
        f"| {researcher['divergence_rate'] if researcher['divergence_rate'] is not None else 'n/a'} |"
    )
    literature = audit["literature"]
    if literature.get("rate_source") == "recomputed":
        declarations = (
            f"{literature.get('one_to_one', 0)}×1:1, "
            f"{literature.get('merge', 0)}×merge, {literature.get('split', 0)}×split"
        )
        rate_cell = str(literature["divergence_rate"])
    else:
        declarations = "not parsed"
        rate_cell = "n/a"
    lines.append(
        f"| Literature | {literature['k_means_clusters']} | {declarations} | {rate_cell} |"
    )
    if audit.get("notes"):
        lines.append("")
        for note in audit["notes"]:
            lines.append(f"*{note}*")
    lines.append("")
    lines.append(
        "*Computed from the report text by a documented heuristic "
        "(see `cda.registers.audit`); the LLM's own prose statement of the "
        "mapping appears in the literature-divergence section above.*"
    )
    return "\n".join(lines)
