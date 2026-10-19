"""EXPERIMENTAL multi-model verification (`cda verify`).

Re-runs register assignment on the *identical* inputs of a previous analysis
— its saved user prompt, and its saved system prompt (``…_system.md``, written
beside the user prompt) — with a second model, and emits an
agreement/discrepancy table plus the full second report. Because the inputs
are byte-identical, any disagreement is attributable to the models, not the
pipeline. Runs from before system-prompt persistence have only the user
prompt on disk; for those the currently configured system prompt is used
instead, with a printed note, and the byte-identical premise holds only if
the prompt file has not changed since the run.

Enable it by setting a second model in config or environment:

    llm:
      verification_model: claude-haiku-4-5-20251001   # any Anthropic model
      # or an OpenAI-compatible endpoint: "openai:<model>" (+ OPENAI_API_KEY,
      # optionally OPENAI_BASE_URL) — no code changes needed.

Interpretation guidance: register *counts* and cluster mappings are the
structural signal; label wording always varies between models. Distance
ratings are ordinal judgments — deltas of 1 are common, deltas ≥ 2 deserve a
human look. This hook is shipped as experimental: it surfaces disagreement,
it does not adjudicate it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from cda.config import Config
from cda.divergence.report import END_OF_LLM_BODY
from cda.extraction.windows import safe_term_slug
from cda.llm.base import FixtureMissError, LLMError
from cda.llm.mock import MockProvider, require_recorded_fixtures
from cda.llm.providers import fixture_model, llm_temperature, resolve_model_spec, spec_model
from cda.outputs import unique_run_basename, write_text_atomic
from cda.registers.prompt_loader import load_system_prompt

_REGISTER_HEADING = re.compile(r"^###\s+Cluster\s+\S+?:?\s*(.+)$", re.MULTILINE)
# A rating is a 1-5 that follows a label and a colon or equals sign ("B: 2",
# "Cluster 2: 4", "**Distance:** 3") or the word "rated"; the label's own digits
# ("Cluster 2") are never ratings. Lines defining the scale ("1 = ...") are skipped.
_DISTANCE_LINE = re.compile(r"distance", re.IGNORECASE)
_SCALE_DEFINITION = re.compile(r"\b[1-5]\s*=\s*[^\W\d_]")
_RATING = re.compile(
    r"(?:(?<=[\w)*])\s*\**\s*[:=]|\brated(?:\s+(?:as|at))?)\s*\**\s*(?P<rating>[1-5])(?![\d%]|\.\d)",
    re.IGNORECASE,
)
_HEADER_MODEL = re.compile(r"^<!-- generated: [^|]*\| model: (?P<model>[^|]+?) \|", re.MULTILINE)
_LEGACY_COMPUTED_HEADING = "\n## Clustering Stability (computed)\n"

# Same shape as validation's _SESSION_DIR: exact slug, parsed timestamp,
# optional same-second collision suffix. Prompt/diagnostics/verification
# files can't match (their names continue past the timestamp).
_RUN_REPORT = re.compile(r"^drift_(?P<slug>.+)_(?P<ts>\d{8}_\d{6})(?:-(?P<n>\d+))?\.md$")


def find_latest_run(output_dir: Path, term: str) -> tuple[Path, Path]:
    """Newest report for exactly this term's slug, with the SAME run's prompt.

    A run writes ``drift_<term>_<ts>.md`` and ``drift_<term>_<ts>_prompt.md``
    with one shared timestamp, so the prompt is derived from the report
    rather than picked independently — otherwise a later dry-run prompt could
    be paired with an earlier live report (mismatched inputs).

    The slug is compared exactly and ordering uses the parsed timestamp —
    mirroring ``find_latest_session``: with a prefix glob and lexicographic
    sort, a run for "novelty claims" would always shadow every run for
    "novelty" ('c' sorts above the timestamp digits), silently verifying the
    wrong term's analysis.
    """
    slug = safe_term_slug(term)
    candidates = []
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            match = _RUN_REPORT.match(path.name)
            if match is None or match.group("slug") != slug:
                continue
            prompt = path.with_name(f"{path.stem}_prompt.md")
            if prompt.exists():
                candidates.append((match.group("ts"), int(match.group("n") or 0), prompt, path))
    if not candidates:
        raise FileNotFoundError(
            f"No complete previous run for '{term}' found in {output_dir} (a report "
            "with its matching saved prompt). Run `cda analyze --term "
            f"{term}` (or the demo) first."
        )
    _, _, prompt, report = max(candidates, key=lambda entry: (entry[0], entry[1]))
    return prompt, report


def _read_run_file(path: Path) -> str:
    """Read a previous run's saved prompt or report as UTF-8, or say why not.

    A file damaged by a full disk can end mid-character, and the codec error
    that raises names a byte offset and nothing else. The remedy is the same
    as for any unreadable run file, so it is worth saying out loud.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise LLMError(
            f"{path} is not readable as UTF-8 text ({exc.reason} at byte "
            f"{exc.start}), so the previous run cannot be verified. The file "
            "is probably damaged (a full disk, an interrupted copy): delete "
            "it and rerun `cda analyze` for this term to produce a fresh "
            "report and prompt pair."
        ) from exc


def _comparable_body(report_text: str) -> str:
    """The LLM's own text, without the pipeline's appended computed sections.

    The primary side is parsed from the *rendered* report while the secondary
    side is the raw model reply; without stripping, the distance-scale legend
    and other computed boilerplate would feed spurious "distance N" mentions
    into the primary side only, skewing the ratings comparison. Reports carry
    a structural end-of-body marker; reports rendered before it existed are
    cut at their first computed section heading.
    """
    index = report_text.find(END_OF_LLM_BODY)
    if index == -1:
        index = report_text.find(_LEGACY_COMPUTED_HEADING)
    return report_text[:index] if index != -1 else report_text


def primary_model_label(report_text: str) -> str:
    """The model named in the primary report's own metadata header."""
    match = _HEADER_MODEL.search(report_text)
    if match is None:
        raise LLMError(
            "The primary report has no '<!-- generated: ... | model: ... -->' header, "
            "so the model that wrote it cannot be named. Rerun `cda analyze` for this term."
        )
    return match.group("model")


def _distance_ratings(line: str) -> list[int]:
    """Every rating stated on a line that mentions distance, in order."""
    if not _DISTANCE_LINE.search(line) or _SCALE_DEFINITION.search(line):
        return []
    return [int(m.group("rating")) for m in _RATING.finditer(line)]


def extract_register_structure(report_text: str) -> dict:
    """The comparable skeleton of a report: labels + distance mentions."""
    body = _comparable_body(report_text)
    labels = [label.strip().rstrip("*").strip() for label in _REGISTER_HEADING.findall(body)]
    ratings = [rating for line in body.splitlines() for rating in _distance_ratings(line)]
    return {"researcher_register_labels": labels, "distance_ratings": ratings}


def verification_verdict(primary: dict, secondary: dict) -> tuple[str, str]:
    """One unmissable verdict line for the top of the verification report.

    Register *count* is the structural signal (label wording always varies
    between models), so the flag keys on it: matching counts → agreement;
    differing counts → a discrepancy that deserves a human read of both
    reports. Returns (verdict_key, banner_markdown).
    """
    n_primary = len(primary["researcher_register_labels"])
    n_secondary = len(secondary["researcher_register_labels"])
    if n_primary == n_secondary and n_primary > 0:
        return (
            "agreement",
            f"**VERIFICATION: STRUCTURAL AGREEMENT** — both models formed "
            f"{n_primary} researcher registers.",
        )
    return (
        "discrepancy",
        f"⚠️ **VERIFICATION: DISCREPANCY** — researcher register counts differ "
        f"({n_primary} vs {n_secondary}); read both reports before relying on "
        "either register structure.",
    )


def render_agreement(
    primary: dict, secondary: dict, primary_model: str, secondary_model: str
) -> str:
    p_labels = primary["researcher_register_labels"]
    s_labels = secondary["researcher_register_labels"]
    count_match = len(p_labels) == len(s_labels)
    lines = [
        "## Agreement summary (computed)",
        "",
        f"| | `{primary_model}` (primary) | `{secondary_model}` (verification) |",
        "|---|---|---|",
        f"| Researcher registers | {len(p_labels)} | {len(s_labels)} |",
        f"| Structural match (register count) | {'✅ yes' if count_match else '⚠️ NO'} | |",
        "",
        "**Register labels, side by side** (wording is expected to vary; the "
        "structure is the signal):",
        "",
        "| # | primary | verification |",
        "|---|---|---|",
    ]
    for i in range(max(len(p_labels), len(s_labels))):
        left = p_labels[i] if i < len(p_labels) else "—"
        right = s_labels[i] if i < len(s_labels) else "—"
        lines.append(f"| {i + 1} | {left} | {right} |")

    p_ratings, s_ratings = primary["distance_ratings"], secondary["distance_ratings"]
    lines += [
        "",
        f"**Distance ratings mentioned**: primary {p_ratings or '—'} · "
        f"verification {s_ratings or '—'}.",
    ]
    if p_ratings and s_ratings:
        delta = abs(max(p_ratings) - max(s_ratings))
        lines.append(
            f"Max-rating delta: **{delta}** "
            f"({'within ordinary ordinal variation' if delta <= 1 else 'worth a human look'})."
        )
    lines += [
        "",
        "*Computed by a documented heuristic parse of both reports "
        "(`cda.registers.verify`); read both full reports before concluding "
        "anything from disagreement.*",
    ]
    return "\n".join(lines)


def run_verify(
    term: str,
    config: Config,
    *,
    mock: bool = False,
    fixtures_dir: str | Path | None = None,
    record_fixtures: bool = False,
    progress: Callable[[str], None] = print,
) -> dict:
    """Re-run register assignment on the latest saved prompt with a second model.

    Writes the verification report and returns its path, the primary report
    path, and the computed verdict ("agreement" | "discrepancy").

    ``record_fixtures`` wraps the live provider so the verification model's
    reply lands on disk as a replayable fixture — the only shipped path that
    records one, so it is what makes ``cda verify --mock-llm`` reachable and
    its own fixture-miss remedy followable.
    """
    spec = str(config.get("llm.verification_model") or "").strip()
    if not spec:
        raise LLMError(
            "No verification model configured. Set llm.verification_model in "
            "your config (or the CDA_VERIFICATION_MODEL environment variable), "
            "e.g. claude-haiku-4-5-20251001 or openai:<model>. This hook is "
            "experimental — see docs/interpretation_guide.md."
        )
    max_tokens = config.get_int("llm.max_tokens", minimum=1)
    temperature = llm_temperature(config, spec)

    if mock:
        if fixtures_dir is None:
            raise LLMError(
                "--mock-llm verification needs a fixtures directory: pass "
                "--fixtures-dir PATH (or --demo, which uses demo/fixtures)."
            )
        # Same refusal `cda validate --mock-llm` makes, and for the same
        # reason: an empty fixtures directory is checkable before any file
        # reading or model work, not after the first prompt is assembled.
        require_recorded_fixtures(fixtures_dir, what="cda verify")

    output_dir = config.path("paths.output_dir")
    prompt_path, report_path = find_latest_run(output_dir, term)
    progress(f"[verify] reusing prompt {prompt_path.name} and report {report_path.name}")

    user_prompt = _read_run_file(prompt_path)
    # The run's own system prompt, when the run saved one (written as
    # ``<base>_system.md`` beside ``<base>_prompt.md``): the module's premise
    # is byte-identical inputs, and the configured prompt file can have
    # changed between the analysis and this verification — silently pairing
    # the old user prompt with a new system prompt would attribute the prompt
    # diff to the models. Pre-persistence runs have no saved copy, so the
    # configured prompt is the only candidate; say so out loud.
    system_path = report_path.with_name(f"{report_path.stem}_system.md")
    if system_path.is_file():
        system_prompt = _read_run_file(system_path)
    else:
        system_prompt = load_system_prompt(config)
        progress(
            f"[verify] note: this run predates system-prompt persistence (no "
            f"{system_path.name}), so verification assumes the currently "
            "configured system prompt is the one the run used."
        )
    primary_report = _read_run_file(report_path)

    if mock:
        provider = MockProvider(
            fixtures_dir=Path(fixtures_dir),
            model=spec_model(spec),
            fingerprint_model=fixture_model(spec),
        )
    else:
        provider = resolve_model_spec(spec)
        if record_fixtures:
            from cda.llm.mock import RecordingProvider

            if fixtures_dir is None:
                raise LLMError("--record-fixtures needs --fixtures-dir.")
            provider = RecordingProvider(
                provider, Path(fixtures_dir), fingerprint_model=fixture_model(spec)
            )

    progress(f"[verify] EXPERIMENTAL: re-running register assignment via {spec}")
    try:
        secondary_report = provider.complete(
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except FixtureMissError as exc:
        # The generic miss advice ("run cda demo --mock-llm") is wrong here:
        # the demo runs the *primary* model only, so it never records anything
        # for a verification model — following it would loop. No verification
        # fixture ships with the repository either, because recording one
        # requires a live call to whichever second model the user configured.
        raise FixtureMissError(
            f"{exc}\n"
            f"Note for `cda verify`: no recorded reply exists for the "
            f"verification model '{spec}', and none ships with CDA — the demo "
            "records the primary model only. The one way to create it is a "
            "single live run of this same command:\n"
            f"  cda verify --term {term} --record-fixtures "
            f"--fixtures-dir {fixtures_dir}\n"
            "with ANTHROPIC_API_KEY (or OPENAI_API_KEY) set and --mock-llm "
            "dropped. After that, --mock-llm replays it for free."
        ) from exc

    # The raw reply saved before post-processing is the exact counterpart of
    # the secondary reply; older runs only have the rendered report.
    response_path = report_path.with_name(f"{report_path.stem}_llm_response.txt")
    primary_body = _read_run_file(response_path) if response_path.is_file() else primary_report
    primary_structure = extract_register_structure(primary_body)
    secondary_structure = extract_register_structure(secondary_report)
    verdict, banner = verification_verdict(primary_structure, secondary_structure)
    agreement = render_agreement(
        primary_structure,
        secondary_structure,
        primary_model_label(primary_report),
        spec,
    )

    now = datetime.now()
    model_slug = safe_term_slug(spec)
    base = unique_run_basename(
        output_dir,
        f"drift_{safe_term_slug(term)}_{now.strftime('%Y%m%d_%H%M%S')}_verification_{model_slug}",
        (".md",),
    )
    out_path = output_dir / f"{base}.md"
    write_text_atomic(
        out_path,
        f"<!-- generated: {now.isoformat()} | cda verify (EXPERIMENTAL) -->\n\n"
        f'# Multi-model verification: "{term}"\n\n'
        f"{banner}\n\n"
        f"> **Experimental.** Same cluster summaries, second model (`{spec}`).\n"
        f"> Primary report: `{report_path.name}`.\n\n"
        f"{agreement}\n\n---\n\n"
        f"## Full verification-model report\n\n{secondary_report}\n",
    )
    progress(f"[verify] {banner}")
    progress(f"[verify] report: {out_path}")
    return {
        "verification_file": str(out_path),
        "primary_report": str(report_path),
        "verdict": verdict,
    }
