"""Build the public demo from an explicit allowlist of committed toy reports.

No CDA imports, model downloads, corpus reads, or analysis execution are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    term: ROOT / "demo" / "expected_reports" / f"drift_{term}_v5_20261018.md"
    for term in ("novelty", "consensus")
}
STATIC_FILES = ("index.html", "style.css", "app.js")


def inline(text: str) -> str:
    """Escape first; support only the inline markup present in the toy reports."""
    value = html.escape(text)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    return re.sub(r"\*([^*]+)\*", r"<em>\1</em>", value)


def blocks(text: str) -> list[dict]:
    """A small, deliberately bounded parser for saved Markdown reports."""
    # Only complete standalone metadata comments are markup. An unclosed comment
    # quoted in the novelty table is literal source text and must stay visible.
    text = re.sub(r"^<!--[^\n]*-->\s*$", "", text, flags=re.M)
    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            result.append({"kind": "table", "rows": rows})
            continue
        heading = re.match(r"^(#{1,3}) (.*)", line)
        if heading:
            result.append({"kind": "heading", "level": len(heading[1]), "text": heading[2]})
        elif re.match(r"^(?:- |\d+\. )", line):
            result.append({"kind": "item", "text": re.sub(r"^(?:- |\d+\. )", "", line)})
        else:
            result.append({"kind": "paragraph", "text": line})
        i += 1
    return result


def render_html(text: str) -> str:
    output = []
    in_list = False
    for block in blocks(text):
        kind = block["kind"]
        if kind != "item" and in_list:
            output.append("</ul>")
            in_list = False
        if kind == "item":
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{inline(block['text'])}</li>")
        elif kind == "table":
            output.append(
                '<div class="table-scroll" tabindex="0" role="region" aria-label="Report table"><table>'
            )
            for index, row in enumerate(block["rows"]):
                tag = "th" if index == 0 else "td"
                output.append(
                    "<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in row) + "</tr>"
                )
            output.append("</table></div>")
        elif kind == "heading":
            level = min(block["level"] + 1, 4)
            output.append(f"<h{level}>{inline(block['text'])}</h{level}>")
        else:
            output.append(f"<p>{inline(block['text'])}</p>")
    if in_list:
        output.append("</ul>")
    return "\n".join(output)


def section(text: str, title: str) -> str:
    match = re.search(r"^## " + re.escape(title) + r"\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not match:
        raise ValueError(f"Missing report section: {title}")
    return match[1].strip()


def case_data(term: str, path: Path) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    counts = re.search(r"researcher_passages: (\d+), literature_passages: (\d+)", text)
    partition = re.search(r"joint_k=(\d+), joint_silhouette=([\d.]+)", text)
    stability = re.search(r"matched (\d+)% of (\d+) shuffled orderings \(k ranged ([\d–]+)", text)
    grounding = re.search(r"\*\*(\d+)/(\d+)\*\* quoted passages verified", text)
    if not all((counts, partition, stability, grounding)):
        raise ValueError(f"Unexpected report metadata: {path.name}")
    flag_section = section(text, "Computed Divergence Flags")
    rows = next(b["rows"][1:] for b in blocks(flag_section) if b["kind"] == "table")
    clusters = [
        dict(id=int(r[0].split()[-1]), literature=int(r[1]), researcher=int(r[3]), flag=r[5])
        for r in rows
    ]
    researcher, literature = map(int, counts.groups())
    if (
        sum(c["researcher"] for c in clusters) != researcher
        or sum(c["literature"] for c in clusters) != literature
    ):
        raise ValueError("Cluster counts do not match report totals")
    if len(clusters) != int(partition[1]):
        raise ValueError("Cluster count does not match k")
    evidence = re.findall(r'"([^"\n]+)" - \*([^*\n]+)\*', text)
    if not evidence:
        raise ValueError("No attributed example passages found")
    return {
        "term": term,
        "source": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "date": "2026-10-18",
        "researcher": researcher,
        "literature": literature,
        "k": int(partition[1]),
        "silhouette": float(partition[2]),
        "stability": int(stability[1]),
        "trials": int(stability[2]),
        "kRange": stability[3],
        "grounded": int(grounding[1]),
        "quotations": int(grounding[2]),
        "clusters": clusters,
        "summary": section(text, "Summary"),
        "evidence": [
            {
                "text": q,
                "source": s,
                "corpus": "Fictional draft" if s.startswith("draft_") else "Toy literature",
            }
            for q, s in evidence
        ],
        "interpretationHtml": render_html(section(text, "2. Researcher vs. Literature Divergence")),
        "recommendationsHtml": render_html(section(text, "3. Actionable Recommendations")),
        "reportHtml": render_html(text),
        "markdown": f"downloads/{term}.md",
    }


def check_public_text(text: str, name: str) -> None:
    """Fail closed on common accidental private paths, emails, or credentials."""
    patterns = [
        r"/(?:Users|home)/[^\s/]+",
        r"[A-Z]:\\Users\\",
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,})\b",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    ]
    if any(re.search(p, text) for p in patterns):
        raise ValueError(f"Potential private information in {name}; inspect before publishing")


def build(output: Path) -> None:
    # Refuse an existing directory, so stale or private files cannot ride along.
    output.mkdir(parents=True, exist_ok=False)
    (output / "downloads").mkdir()
    data = [case_data(term, path) for term, path in REPORTS.items()]
    for name in STATIC_FILES:
        source = ROOT / "web" / name
        check_public_text(source.read_text(), name)
        shutil.copyfile(source, output / name)
    for case in data:
        source = REPORTS[case["term"]]
        check_public_text(source.read_text(), source.name)
        shutil.copyfile(source, output / case["markdown"])
    payload = json.dumps({"schemaVersion": 1, "cases": data}, ensure_ascii=False, indent=2)
    check_public_text(payload, "data.json")
    (output / "data.json").write_text(payload + "\n", encoding="utf-8")
    (output / ".nojekyll").write_text("")
    manifest = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {len(manifest) + 1} allowlisted public files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new, empty output path")
    build(parser.parse_args().output)
