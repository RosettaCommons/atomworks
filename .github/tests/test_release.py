"""Check package contents, tag consistency, and docs promotion without publishing."""

import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release import prepare_docs, verify_distributions, verify_tag


class ReleaseTests(unittest.TestCase):
    def setUp(self) -> None:  # noqa: N802
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "pyproject.toml"
        self.html = self.root / "html"
        self.html.mkdir()
        (self.html / "index.html").write_text("new documentation")
        self.pages = self.root / "pages"
        self.pages.mkdir()

    def test_tag_must_match_project_version(self) -> None:
        for value in ("2.3.0", "2.3.0rc1"):
            with self.subTest(value=value):
                self.project.write_text(f'[project]\nversion = "{value}"\n')
                self.assertEqual(str(verify_tag(f"v{value}", self.project)), value)
                with self.assertRaises(ValueError):
                    verify_tag("v9.0.0", self.project)

    def test_distribution_contents(self) -> None:
        source = self.root / "src"
        package = source / "atomworks"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('"""AtomWorks."""\n')
        (package / "parameters.json").write_text('{"charge": -1}\n')
        dist = self.root / "dist"
        dist.mkdir()
        for case in ("complete", "empty-wheel", "missing-data", "changed-data", "relocated-source"):
            with self.subTest(case=case):
                with zipfile.ZipFile(dist / "atomworks-2.3.0-py3-none-any.whl", "w") as wheel:
                    for path in package.iterdir():
                        if case == "empty-wheel" or (case == "missing-data" and path.suffix == ".json"):
                            continue
                        data = b"{}" if case == "changed-data" and path.suffix == ".json" else path.read_bytes()
                        wheel.writestr(path.relative_to(source).as_posix(), data)
                with tarfile.open(dist / "atomworks-2.3.0.tar.gz", "w:gz") as archive:
                    prefix = "" if case == "relocated-source" else "src/"
                    archive.add(package, arcname=f"atomworks-2.3.0/{prefix}atomworks")
                if case == "complete":
                    verify_distributions(dist, source)
                else:
                    with self.assertRaises((KeyError, ValueError)):
                        verify_distributions(dist, source)

    def test_rejects_nonrelease_tags(self) -> None:
        self.project.write_text('[project]\nversion = "2.3.0"\n')
        for tag in ("2.3.0", "production", "v2.3.0.dev1", "v2.3.0+local", "v2.3.0/extra"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                verify_tag(tag, self.project)

    def test_old_release_and_prerelease_do_not_replace_latest(self) -> None:
        for version in ("v2.2.10", "v2.2.9", "v2.3.0rc1"):
            (self.html / "index.html").write_text(version)
            prepare_docs(version, self.html, self.pages)
        self.assertEqual((self.pages / "latest/index.html").read_text(), "v2.2.10")
        switcher = json.loads((self.pages / "_static/switcher.json").read_text())
        self.assertEqual([entry["name"] for entry in switcher], ["latest", "v2.3.0rc1", "v2.2.10", "v2.2.9"])
        self.assertEqual([entry["name"] for entry in switcher if entry.get("preferred")], ["v2.2.10"])
        self.assertEqual(switcher, json.loads((self.pages / "v2.2.9/_static/switcher.json").read_text()))

    def test_new_stable_release_updates_latest_without_stale_files(self) -> None:
        prepare_docs("v2.0.0", self.html, self.pages)
        (self.pages / "latest/removed.html").write_text("old page")
        (self.html / "index.html").write_text("new stable")
        prepare_docs("v2.3.0", self.html, self.pages)
        self.assertEqual((self.pages / "latest/index.html").read_text(), "new stable")
        self.assertFalse((self.pages / "latest/removed.html").exists())

    def test_retry_preserves_published_version_pages(self) -> None:
        prepare_docs("v2.3.0", self.html, self.pages)
        (self.html / "index.html").write_text("different build")
        prepare_docs("v2.3.0", self.html, self.pages)
        self.assertEqual((self.pages / "v2.3.0/index.html").read_text(), "new documentation")

    def test_first_prerelease_has_switcher_without_stable_alias(self) -> None:
        prepare_docs("v2.3.0rc1", self.html, self.pages)
        self.assertFalse((self.pages / "latest").exists())
        switcher = json.loads((self.pages / "_static/switcher.json").read_text())
        self.assertEqual([entry["name"] for entry in switcher], ["v2.3.0rc1"])


if __name__ == "__main__":
    unittest.main()
