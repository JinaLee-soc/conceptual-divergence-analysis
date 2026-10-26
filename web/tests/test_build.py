"""Public artifact boundaries and source fidelity, without CDA dependencies."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_pages import REPORTS, build, case_data, check_public_text, render_html


class PagesBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "site"
        build(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_only_allowlisted_files_are_published(self):
        expected = {"index.html", "style.css", "app.js", "data.json", "manifest.json", ".nojekyll"}
        expected.update(f"downloads/{term}.md" for term in REPORTS)
        actual = {
            p.relative_to(self.output).as_posix() for p in self.output.rglob("*") if p.is_file()
        }
        self.assertEqual(actual, expected)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(set(manifest), expected - {"manifest.json"})
        for name, digest in manifest.items():
            self.assertEqual(digest, hashlib.sha256((self.output / name).read_bytes()).hexdigest())

    def test_existing_directory_is_refused(self):
        with self.assertRaises(FileExistsError):
            build(self.output)

    def test_original_reports_and_counts_are_preserved(self):
        expectations = {"novelty": (38, 56, 2, 33), "consensus": (33, 103, 3, 37)}
        for term, source in REPORTS.items():
            data = case_data(term, source)
            self.assertEqual(
                tuple(data[k] for k in ("researcher", "literature", "k", "stability")),
                expectations[term],
            )
            self.assertEqual(
                source.read_bytes(), (self.output / f"downloads/{term}.md").read_bytes()
            )
        consensus = case_data("consensus", REPORTS["consensus"])
        self.assertEqual(
            [c["id"] for c in consensus["clusters"] if c["flag"] == "underrepresented"], [1]
        )

    def test_html_is_escaped_and_literal_comment_is_preserved(self):
        rendered = render_html(
            '# Example\n\n<script>alert(1)</script>\n\n| a | b |\n|---|---|\n| x | "<!-- SYNTHETIC DOCUMENT…" |'
        )
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;!-- SYNTHETIC", rendered)

    def test_private_material_is_rejected(self):
        for text in (
            "/home/private/example.txt",
            "contact@example.invalid",
            "-----BEGIN PRIVATE KEY-----",  # pragma: allowlist secret (marker-only test)
        ):
            with self.assertRaises(ValueError):
                check_public_text(text, "fixture")

    def test_byte_reproducible_exports(self):
        second = Path(self.temp.name) / "second"
        build(second)
        self.assertEqual(
            (self.output / "manifest.json").read_bytes(), (second / "manifest.json").read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
