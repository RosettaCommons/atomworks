"""Setup utilities for AtomWorks."""
# ruff: noqa: B008

from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import typer
from tqdm import tqdm

from .pdb import PDB_PORT, PDB_REMOTE, _collect_pdb_ids, _pdb_id_to_relpath, _rsync_fetch_specific, _run_rsync_list

IPD_DOWNLOAD_URL = "https://files.ipd.uw.edu/pub/atomworks"

TEST_PACK_URL = f"{IPD_DOWNLOAD_URL}/test_pack_latest.tar.gz"
"""The URL for the latest AtomWorks test pack. Should be untared in `tests/data/shared`."""

METADATA_URL = f"{IPD_DOWNLOAD_URL}/pdb_metadata_latest.tar.gz"
"""The URL for the latest AtomWorks PDB metadata. Should be untared at the specified location."""

app = typer.Typer(help="Setup utilities for AtomWorks.")


def _download_file(url: str, dest_path: Path) -> None:
    """Download a file from URL to dest_path, showing a progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
        total_str = response.headers.get("Content-Length") or response.headers.get("content-length")
        total = int(total_str) if total_str and total_str.isdigit() else None

        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading test pack",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            chunk_size = 1024 * 64  # 64 KiB
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))


def _extract_tar_gz(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir, filter="data")


def _find_missing_mmCIFs(pdb_ids: Iterable[str], mirror_root: Path) -> list[str]:  # noqa: N802
    """Return pdb_ids whose corresponding mmCIF files are missing under mirror_root."""
    missing: list[str] = []
    for pid in pdb_ids:
        rel = _pdb_id_to_relpath(pid)
        cif_path = mirror_root / rel
        if not cif_path.is_file():
            missing.append(pid)
    return missing


@app.command("tests")
def setup_tests(
    tests_data_dir: Path = typer.Option(Path("tests/data"), "--tests-data-dir", help="Where to extract the test pack."),
    keep_archive: bool = typer.Option(False, "--keep-archive", help="Keep downloaded test pack archive."),
) -> None:
    """Download the test pack and ensure required PDB mmCIFs are present in the mirror.

    NOTE: It's expected that you run this command from the root of the repository.

    Steps:
    1) Download and extract the AtomWorks test pack into `tests/data/shared` (by default).
    2) Read `test_pdb_ids.txt` from the test pack.
    3) Download any missing PDB mmCIFs listed there into the given PDB mirror path using rsync.

    Example:
        atomworks setup tests --pdb-mirror-path /data/rcsb/mmcif
    """
    typer.echo("Setting up AtomWorks test environment...")

    # Resolve PDB mirror path
    pdb_mirror_path = Path(os.getenv("PDB_MIRROR_PATH") or tests_data_dir / "pdb")

    # Download and extract test pack
    tests_data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "test_pack_latest.tar.gz"
        typer.echo(f"Downloading test pack from {TEST_PACK_URL} ...")
        _download_file(TEST_PACK_URL, archive_path)
        typer.secho("Download complete", fg=typer.colors.GREEN)

        typer.echo(f"Extracting test pack into {tests_data_dir} ...")
        _extract_tar_gz(archive_path, tests_data_dir)
        typer.secho("Extraction complete", fg=typer.colors.GREEN)

        if keep_archive:
            keep_path = tests_data_dir / archive_path.name
            keep_path.write_bytes(archive_path.read_bytes())

    # Read PDB IDs from the test pack
    ids_file = tests_data_dir / "shared" / "test_pdb_ids.txt"
    if not ids_file.is_file():
        typer.secho(f"Missing PDB id list: {ids_file}", fg=typer.colors.RED)
        raise typer.Exit(code=3)
    typer.echo(f"Reading PDB ids from {ids_file}")
    ids = _collect_pdb_ids(None, ids_file)
    typer.echo(f"Found {len(ids)} unique PDB ids in the test pack list")

    # Determine missing mmCIFs
    typer.echo(f"Checking for missing PDB mmCIFs at {pdb_mirror_path} ...")
    missing = _find_missing_mmCIFs(ids, pdb_mirror_path)
    if missing:
        typer.echo(f"{len(missing)} PDB ids are missing from the mirror; testing rsync connectivity...")
        ok, output = _run_rsync_list(PDB_REMOTE, PDB_PORT)
        if not ok:
            typer.secho("Connection test failed", fg=typer.colors.RED)
            typer.echo("Error output:")
            typer.echo(output)
            raise typer.Exit(code=4)
        typer.secho("Connection test successful", fg=typer.colors.GREEN)

        typer.echo("Fetching missing PDB mmCIFs via rsync...")
        _rsync_fetch_specific(PDB_REMOTE, pdb_mirror_path, missing, PDB_PORT)
        typer.secho("PDB sync complete", fg=typer.colors.GREEN)
    else:
        typer.secho("All required PDB mmCIFs are already present", fg=typer.colors.GREEN)

    typer.secho("Test setup completed successfully!", fg=typer.colors.GREEN)
    typer.secho("To run tests use: PDB_MIRROR_PATH=tests/data/pdb pytest -n auto tests")


@app.command("metadata")
def setup_metadata(
    output_dir: Path = typer.Argument(
        ...,
        help="Directory where the PDB metadata archive should be extracted.",
    ),
    keep_archive: bool = typer.Option(False, "--keep-archive", help="Keep downloaded metadata archive."),
) -> None:
    """Download the latest PDB metadata archive and extract it to the given directory.

    NOTE: It's expected that you run this command from the root of the repository.

    The metadata archive is structured to extract under `shared/` inside the provided directory by default.

    Example:
        atomworks setup metadata --output-dir tests/data
    """
    typer.echo("Setting up AtomWorks PDB metadata...")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "pdb_metadata_latest.tar.gz"
        typer.echo(f"Downloading PDB metadata from {METADATA_URL} ...")
        _download_file(METADATA_URL, archive_path)
        typer.secho("Download complete", fg=typer.colors.GREEN)

        typer.echo(f"Extracting PDB metadata into {output_dir} ...")
        _extract_tar_gz(archive_path, output_dir)
        typer.secho("Extraction complete", fg=typer.colors.GREEN)

        if keep_archive:
            keep_path = output_dir / archive_path.name
            keep_path.write_bytes(archive_path.read_bytes())

    typer.secho("PDB metadata setup completed successfully!", fg=typer.colors.GREEN)
