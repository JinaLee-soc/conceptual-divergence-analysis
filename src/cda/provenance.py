"""Run receipts for the derived-corpus analysis boundary, without copying source text."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from cda.config import ConfigError
from cda.outputs import write_text_immutable

_JSON_SCALARS = (str, int, float, bool, type(None))


def _non_json_location(value, location: str = "") -> tuple[str, object] | None:
    """The dotted key and value of the first entry JSON cannot encode, if any."""
    if isinstance(value, dict):
        for key, item in value.items():
            dotted = f"{location}.{key}" if location else str(key)
            if not isinstance(key, _JSON_SCALARS):
                return dotted, key
            found = _non_json_location(item, dotted)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _non_json_location(item, f"{location}[{index}]")
            if found:
                return found
        return None
    return None if isinstance(value, _JSON_SCALARS) else (location or "<value>", value)


def digest(value) -> str:
    """Hash canonical JSON; nonfinite numbers must never enter a receipt.

    A value JSON cannot encode is refused by name rather than coerced with
    ``str()``: coercion would make two different objects hash alike.
    """
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except TypeError as exc:
        found = _non_json_location(value)
        if found is None:
            raise
        key, bad = found
        raise ConfigError(
            f"Cannot record '{key}' in the run manifest: {type(bad).__name__} value {bad!r} "
            "is not plain text, a number, true/false, a list, or a mapping. If it is a "
            'date in your config, quote it (for example "2026-05-01").'
        ) from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def analysis_metadata(metadata: dict) -> dict:
    """Physical PDF-root location does not change an observation's identity."""
    return {key: value for key, value in metadata.items() if key != "pdf_root"}


def input_snapshot(hits: dict, drafts: dict[str, str]) -> dict:
    """Identify all retrieved records and supplied drafts before filtering.

    This is not a raw-library census: records that the literal query did not
    retrieve are outside this boundary. Keep the derived store for replay.
    """
    records = [
        {
            "id": key,
            "sha256": digest(dict(value, metadata=analysis_metadata(value.get("metadata") or {}))),
        }
        for key, value in sorted(hits.items())
    ]
    draft_records = [{"id": key, "sha256": digest(value)} for key, value in sorted(drafts.items())]
    return {
        "boundary": "retrieved derived corpus records and supplied draft texts, before filtering",
        "source_hashes": sorted(
            {
                (str((record.get("metadata") or {}).get("pdf_relpath", key)), kind, value)
                for key, record in hits.items()
                for kind, value in (record.get("metadata") or {}).items()
                if kind in {"source_file_sha256", "source_text_sha256", "zotero_csv_sha256"}
            }
        ),
        "retrieved_records": records,
        "drafts": draft_records,
        "sha256": digest({"records": records, "drafts": draft_records}),
    }


def sample_snapshot(researcher, literature) -> dict:
    """Record ordered observation IDs, content hashes and available vectors."""
    result = {}
    for side, passages in (("researcher", researcher), ("literature", literature)):
        ids = [p.chunk_id for p in passages]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate passage identifiers in {side}; cannot certify the sample.")
        rows = [
            {
                "id": p.chunk_id,
                "source": p.source,
                "sentence_index": p.metadata.get("sentence_index"),
                "text_sha256": digest(p.text),
                "metadata_sha256": digest(analysis_metadata(p.metadata)),
                "embedding_sha256": digest(p.embedding) if p.embedding is not None else None,
                "cluster": p.cluster_id,
            }
            for p in passages
        ]
        result[side] = {"n": len(rows), "sha256": digest(rows), "observations": rows}
    result["sha256"] = digest(result)
    return result


def code_snapshot() -> dict:
    """Hash installed CDA sources, even for wheels or dirty checkouts."""
    package = Path(__file__).resolve().parent
    files = {
        str(path.relative_to(package)): file_digest(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml", ".md"}
    }
    result = {
        "package_files": files,
        "sha256": digest(files),
        "git_commit": None,
        "git_dirty": None,
    }
    root = package.parent.parent
    if (root / ".git").exists():
        try:
            result["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5
            ).strip()
            result["git_dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, text=True, timeout=5
                ).strip()
            )
            result["pyproject_sha256"] = file_digest(root / "pyproject.toml")
        except (OSError, subprocess.SubprocessError):
            result["git_status"] = "unavailable; installed source hashes remain authoritative"
    return result


def normalize_package_name(name: str) -> str:
    """PEP 503 normalized form: "PyYAML" and "pyyaml" are one project."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_record(dist: importlib.metadata.Distribution) -> dict:
    """Name and version of one installed distribution, never raising.

    A broken install can leave a dist-info whose METADATA has no Name; it is
    still on the path, so it is recorded under a placeholder naming its
    location instead of aborting the run at the manifest.
    """
    # .get(): a missing field indexes to None today, with KeyError announced.
    name = dist.metadata.get("Name")
    if not name:
        location = getattr(dist, "_path", None) or dist.locate_file("")
        name = f"<unnamed:{location}>"
    return {"name": name, "version": dist.metadata.get("Version")}


def _package_sort_key(item: dict) -> tuple[str, str, str]:
    return (normalize_package_name(item["name"]), item["name"], item["version"] or "")


def environment_snapshot() -> dict:
    """Capture the full installed distribution inventory, not just direct deps."""
    packages = sorted(
        (_distribution_record(d) for d in importlib.metadata.distributions()),
        key=_package_sort_key,
    )
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "packages_sha256": digest(packages),
    }


def redact_config(value):
    """Never persist credentials accidentally supplied in a custom config."""
    sensitive = {"api_key", "access_token", "auth_token", "password", "secret", "authorization"}
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if str(key).lower() in sensitive else redact_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    return value


def embedding_snapshot(embedder, config) -> dict:
    """The embedding model as configured and, when loaded, as actually resolved."""
    embedding = {
        "name": embedder.name,
        "implementation": f"{type(embedder).__module__}.{type(embedder).__qualname__}",
        "configured_revision": config.get("embedding.revision", None),
    }
    if hasattr(embedder, "provenance"):
        embedding.update(embedder.provenance())
    else:
        embedding["status"] = "custom embedder; model revision unavailable"
    return embedding


def build_manifest(
    *,
    term,
    config,
    embedder,
    inputs,
    extracted,
    researcher,
    literature,
    stats,
    diagnostics,
    mock,
    dry_run,
    now,
) -> dict:
    """Freeze the deterministic stage before any model call can fail.

    ``config_sha256`` covers every setting except ``paths.*``: where inputs and
    outputs live is recorded separately and identifies a location, not an analysis.
    """
    settings = redact_config(config.data)
    analysis_settings = {key: value for key, value in settings.items() if key != "paths"}
    return {
        "schema_version": 2,
        "created_at": now.astimezone().isoformat(),
        "phase": "prepared; only a matching completion receipt certifies finished outputs",
        "term": term,
        "execution": "dry-run" if dry_run else ("mock" if mock else "live"),
        "research_adoption": "not assessed",
        "code": code_snapshot(),
        "environment": environment_snapshot(),
        "config": settings,
        "resolved_paths": {
            key: str(Path(value).expanduser().resolve())
            for key, value in settings.get("paths", {}).items()
            if isinstance(value, str) and value
        },
        "config_sha256": digest(analysis_settings),
        "embedding": embedding_snapshot(embedder, config),
        "inputs": inputs,
        "extracted_sample": extracted,
        "analyzed_sample": sample_snapshot(researcher, literature),
        "sample_flow": {key: value for key, value in stats.items() if key != "input_snapshot"},
        "diagnostics_sha256": digest(diagnostics),
    }


def seal_run(written: dict, *, status: str) -> str:
    """Write a separate immutable completion receipt binding every output."""
    artifacts = {
        key: {"file": Path(value).name, "sha256": file_digest(Path(value))}
        for key, value in written.items()
        if key.endswith("_file")
    }
    receipt = {
        "schema_version": 2,
        "run_id": written["base"],
        "status": status,
        "artifacts": artifacts,
    }
    path = Path(written["manifest_file"]).with_name(f"{written['base']}_receipt.json")
    write_text_immutable(path, json.dumps(receipt, indent=2))
    return str(path)


def verify_receipt(path: Path) -> dict:
    """Check a completed run's bytes; this does not assert scientific validity."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {"manifest_file", "prompt_file", "system_file", "diagnostics_file"}
    if receipt.get("status") == "complete":
        required.add("report_file")
        if receipt.get("schema_version", 1) >= 2:
            required.add("llm_response_file")
    elif receipt.get("status") != "dry-run":
        raise ValueError("Receipt does not mark a completed analysis or dry-run.")
    if not required <= receipt.get("artifacts", {}).keys():
        raise ValueError("Incomplete receipt: required artifacts are missing.")
    for item in receipt["artifacts"].values():
        if Path(item["file"]).name != item["file"]:
            raise ValueError("Receipt artifact must be in the run directory.")
        artifact = path.parent / item["file"]
        if not artifact.is_file() or file_digest(artifact) != item["sha256"]:
            raise ValueError(f"Missing or changed run artifact: {artifact}")
    return {"status": "verified", "run_id": receipt["run_id"], "scope": "artifact integrity only"}


# Fields that describe the machine rather than the analysis. They are reported
# by compare_runs and must match only under strict_environment.
_MACHINE_EMBEDDING_FIELDS = ("device",)


def _environment_differences(manifests: list[dict]) -> dict:
    """Machine-specific differences between two manifests, by field."""
    current, reference = (m["environment"] for m in manifests)
    differences = {
        field: {"current": current.get(field), "reference": reference.get(field)}
        for field in sorted(set(current) | set(reference))
        if field not in ("packages", "packages_sha256")
        and current.get(field) != reference.get(field)
    }
    differences.update(_package_differences(current["packages"], reference["packages"]))
    for field in _MACHINE_EMBEDDING_FIELDS:
        values = [m["embedding"].get(field) for m in manifests]
        if values[0] != values[1]:
            differences[f"embedding.{field}"] = {"current": values[0], "reference": values[1]}
    return differences


def _versions_by_name(packages: list[dict]) -> dict[str, list]:
    """All installed versions per PEP 503 name, sorted; duplicates kept."""
    versions: dict[str, list] = {}
    for package in packages:
        versions.setdefault(normalize_package_name(package["name"]), []).append(package["version"])
    return {name: sorted(found, key=lambda v: v or "") for name, found in versions.items()}


def _package_differences(current: list[dict], reference: list[dict]) -> dict:
    """Package inventory differences, comparing every installed distribution.

    Two distributions of one project (say numpy 1.26 and 2.0 both on the path)
    are a different environment from a single one, so they are compared as
    lists and reported under ``duplicate_packages`` rather than collapsed.
    """
    sides = [_versions_by_name(current), _versions_by_name(reference)]
    single: dict = {}
    duplicated: dict = {}
    for name in sorted(set(sides[0]) | set(sides[1])):
        found = [side.get(name, []) for side in sides]
        if found[0] == found[1]:
            continue
        if max(len(versions) for versions in found) > 1:
            duplicated[name] = {"current": found[0], "reference": found[1]}
        else:
            single[name] = {
                "current": found[0][0] if found[0] else None,
                "reference": found[1][0] if found[1] else None,
            }
    differences = {}
    if single:
        differences["packages"] = single
    if duplicated:
        differences["duplicate_packages"] = duplicated
    return differences


def _deterministic_embedding(manifest: dict) -> dict:
    return {
        key: value
        for key, value in manifest["embedding"].items()
        if key not in _MACHINE_EMBEDDING_FIELDS
    }


def compare_runs(current: Path, reference: Path, *, strict_environment: bool = False) -> dict:
    """Compare deterministic stages exactly after verifying both run receipts.

    Timestamps, output basenames, output/input locations (``paths.*``) and
    live model prose are outside this comparison. Machine-specific fields
    (platform, Python build, package inventory, embedding device) are reported
    under ``environment_differences``; with ``strict_environment`` they must
    match too. No numerical tolerance or researcher adoption is implied.
    """
    for path in (current, reference):
        verify_receipt(path)
    receipts = [json.loads(p.read_text(encoding="utf-8")) for p in (current, reference)]
    manifests = [
        json.loads((p.parent / r["artifacts"]["manifest_file"]["file"]).read_text(encoding="utf-8"))
        for p, r in zip((current, reference), receipts, strict=True)
    ]
    if manifests[0].get("schema_version") != manifests[1].get("schema_version"):
        raise ValueError(
            "Reference comparison failed: manifests use different schema versions "
            f"({manifests[0].get('schema_version')} vs {manifests[1].get('schema_version')})."
        )
    mismatches = [
        field
        for field in (
            "term",
            "execution",
            "config_sha256",
            "inputs",
            "extracted_sample",
            "analyzed_sample",
            "diagnostics_sha256",
        )
        if manifests[0][field] != manifests[1][field]
    ]
    if _deterministic_embedding(manifests[0]) != _deterministic_embedding(manifests[1]):
        mismatches.append("embedding")
    if manifests[0]["code"]["sha256"] != manifests[1]["code"]["sha256"]:
        mismatches.append("installed_code")
    for field in ("prompt_file", "system_file", "diagnostics_file"):
        if receipts[0]["artifacts"][field]["sha256"] != receipts[1]["artifacts"][field]["sha256"]:
            mismatches.append(field)
    environment = _environment_differences(manifests)
    if strict_environment and environment:
        mismatches += [f"environment.{field}" for field in environment]
    if mismatches:
        raise ValueError("Reference comparison failed (exact equality): " + ", ".join(mismatches))
    return {
        "status": "exact agreement",
        "scope": "deterministic analysis stages only",
        "reference_run": receipts[1]["run_id"],
        "current_run": receipts[0]["run_id"],
        "environment_differences": environment,
        "strict_environment": strict_environment,
        "excluded": ["timestamps", "output names", "paths.*", "LLM prose", "figure bytes"],
        "research_adoption": "not assessed",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify the integrity of a CDA run receipt.")
    parser.add_argument("receipt", type=Path)
    parser.add_argument(
        "--compare", type=Path, help="Reference receipt for exact deterministic-stage comparison"
    )
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help="With --compare, also require identical platform, Python build, "
        "package inventory, and embedding device",
    )
    args = parser.parse_args()
    if args.strict_environment and not args.compare:
        parser.error("--strict-environment requires --compare")
    result = (
        compare_runs(args.receipt, args.compare, strict_environment=args.strict_environment)
        if args.compare
        else verify_receipt(args.receipt)
    )
    print(json.dumps(result, indent=2))
