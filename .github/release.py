"""Validate AtomWorks release tags and prepare versioned documentation."""

import json
import os
import shutil
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

from packaging.version import Version

DOCS_URL = "https://rosettacommons.github.io/atomworks"


def verify_distributions(dist: Path, source: Path) -> None:
    """Require every package source and data file in both release archives, byte for byte."""
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")
    files = [path for path in source.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    if not files:
        raise ValueError(f"No package files found in {source}")
    with zipfile.ZipFile(wheel) as binary, tarfile.open(sdist) as archive:
        for path in files:
            relative = path.relative_to(source).as_posix()
            member = f"{sdist.name.removesuffix('.tar.gz')}/src/{relative}"
            archived = archive.extractfile(member)
            expected = path.read_bytes()
            if archived is None or archived.read() != expected or binary.read(relative) != expected:
                raise ValueError(f"Release archive differs from source: {relative}")


def release_version(tag: str) -> Version:
    """Accept canonical version tags suitable for a public package release."""
    version = Version(tag.removeprefix("v"))
    if tag != f"v{version}" or version.is_devrelease or version.local:
        raise ValueError(f"Expected a canonical release tag, got {tag!r}")
    return version


def verify_tag(tag: str, project_file: Path) -> Version:
    """Require the release tag to match the package's declared version exactly."""
    version = release_version(tag)
    project_version = tomllib.loads(project_file.read_text())["project"]["version"]
    if tag != f"v{project_version}":
        raise ValueError(f"Tag {tag!r} does not match project.version {project_version!r}")
    return version


def prepare_docs(tag: str, html: Path, pages: Path) -> None:
    """Keep published version pages intact and point latest at the newest stable release."""
    release_version(tag)
    if not (html / "index.html").is_file():
        raise FileNotFoundError(f"Documentation index missing from {html}")
    destination = pages / tag
    if not destination.exists():
        shutil.copytree(html, destination)
    versions = []
    for directory in pages.iterdir():
        if not directory.is_dir() or not (directory / "index.html").is_file():
            continue
        try:
            version = release_version(directory.name)
        except ValueError:
            continue
        versions.append((version, directory.name))
    versions.sort(reverse=True)
    stable = next((name for version, name in versions if not version.is_prerelease), None)
    latest = pages / "latest"
    if stable:
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(pages / stable, latest)
    switcher = [{"version": "latest", "url": f"{DOCS_URL}/latest/index.html", "name": "latest"}] if stable else []
    switcher.extend(
        {"version": str(version), "url": f"{DOCS_URL}/{name}/index.html", "name": name, "preferred": name == stable}
        for version, name in versions
    )
    targets = [pages, *(pages / name for _, name in versions)]
    if stable:
        targets.append(latest)
    for directory in targets:
        switcher_path = directory / "_static" / "switcher.json"
        switcher_path.parent.mkdir(parents=True, exist_ok=True)
        switcher_path.write_text(json.dumps(switcher, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["artifacts"]:
        verify_distributions(Path("dist"), Path("src"))
        raise SystemExit(0)
    tag = os.environ["GITHUB_REF_NAME"]
    version = verify_tag(tag, Path("pyproject.toml"))
    if sys.argv[1:] == ["verify"]:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"prerelease={str(version.is_prerelease).lower()}\n")
    elif sys.argv[1:] == ["docs"]:
        prepare_docs(tag, Path("docs/_build/html"), Path("gh-pages"))
    else:
        raise SystemExit("Usage: python .github/release.py {verify|docs|artifacts}")
