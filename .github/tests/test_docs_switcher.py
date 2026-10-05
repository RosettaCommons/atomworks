"""Regression checks for repairing archived documentation navigation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "repair_docs_switcher", Path(__file__).parents[1] / "repair_docs_switcher.py"
)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class SwitcherRepairTests(unittest.TestCase):
    def test_repairs_nested_archives_without_changing_content_or_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pages = Path(directory)
            manifest = pages / "latest/_static/switcher.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps([{"version": "2.2.1"}, {"version": "3.0.0"}]))
            original = (
                "<h1>Archived content</h1><script>"
                "DOCUMENTATION_OPTIONS.theme_switcher_json_url = "
                "'https://baker-laboratory.github.io/atomworks-dev/latest/_static/switcher.json';"
                "DOCUMENTATION_OPTIONS.theme_switcher_version_match = '2.2.1';</script>"
            )
            expected = original.replace(
                "'https://baker-laboratory.github.io/atomworks-dev/latest/_static/switcher.json'",
                '"https://rosettacommons.github.io/atomworks/latest/_static/switcher.json"',
            )
            for name in ("v2.2.1/index.html", "v2.2.1/io/parser.html"):
                page = pages / name
                page.parent.mkdir(parents=True, exist_ok=True)
                page.write_text(original)
            untouched = pages / "index.html"
            untouched.write_text('<meta http-equiv="refresh" content="0;url=latest/">')
            self.assertEqual(repair.repair_switcher_urls(pages, "https://rosettacommons.github.io/atomworks/"), 2)
            for page in (pages / "v2.2.1").rglob("*.html"):
                self.assertEqual(page.read_text(), expected)
            self.assertEqual(repair.repair_switcher_urls(pages, "https://rosettacommons.github.io/atomworks"), 0)
            self.assertEqual(untouched.read_text(), '<meta http-equiv="refresh" content="0;url=latest/">')

    def test_missing_manifest_prevents_archive_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "index.html"
            page.write_text("Archived content")
            with self.assertRaises(FileNotFoundError):
                repair.repair_switcher_urls(Path(directory), "https://example.com/docs")
            self.assertEqual(page.read_text(), "Archived content")


if __name__ == "__main__":
    unittest.main()
