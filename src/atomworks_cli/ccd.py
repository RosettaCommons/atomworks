"""Command line interface for managing a local CCD mirror."""

# ruff: noqa: B008

from __future__ import annotations

import functools
import operator
import os
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import typer

CCD_REMOTE = "rsync://ftp.ebi.ac.uk/pub/databases/msd/pdbechem_v2/ccd/"


def _normalize_ccd_code(ccd_code: str) -> str:
    """Return a normalized, uppercase CCD code or raise ValueError.

    Args:
        ccd_code: The CCD code to normalize (1-3 characters).

    Returns:
        Normalized uppercase CCD code.

    Raises:
        ValueError: If the CCD code format is invalid.
    """
    ccd_code = ccd_code.strip().upper()
    return ccd_code


def _ccd_code_to_relpath(ccd_code: str) -> Path:
    """Map a CCD code to its relative CIF path under the divided layout.

    Example: 'ALA' -> 'A/ALA/ALA.cif'

    Args:
        ccd_code: The CCD code.

    Returns:
        Relative path to the CCD entry.
    """
    code = _normalize_ccd_code(ccd_code)
    subdir = code[0]
    return Path(subdir) / code / f"{code}.cif"


def _run_rsync_list(remote_path: str) -> tuple[bool, str]:
    """Try to list a remote rsync path and return success and output/error.

    Args:
        remote_path: The remote rsync path to list.

    Returns:
        A tuple of (success, output_or_error).
    """
    try:
        completed = subprocess.run(
            ["rsync", "--list-only", remote_path],
            check=False,
            capture_output=True,
            text=True,
        )
        success = completed.returncode == 0
        output = completed.stdout if success else completed.stderr
        return success, output
    except FileNotFoundError:
        return False, "rsync executable not found. Please install rsync."


def _rsync_sync(remote_path: str, dest_path: Path) -> None:
    """Synchronize CCD CIF files from rsync remote to local destination.

    Mirrors the shell behavior: recursive, times preserved where possible, include only directories and .cif files.
    """
    cmd = [
        "rsync",
        "-rltvz",
        "--stats",
        "--no-perms",
        "--chmod=ug=rwX,o=rX",
        "--delete",
        "--omit-dir-times",
        "--include=*/",
        "--include",
        "*.cif",
        "--exclude",
        "*",
        remote_path,
        str(dest_path),
    ]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"rsync failed with exit code {completed.returncode}")


def _rsync_fetch_specific(remote_base: str, dest_path: Path, ccd_codes: Iterable[str]) -> None:
    """Fetch only specific CCD codes by rsync-ing their individual files in a single command.

    Uses rsync's --files-from option with --relative to preserve directory structure
    and fetch all files in one efficient operation.

    Args:
        remote_base: The base remote rsync path.
        dest_path: Local destination directory.
        ccd_codes: Iterable of CCD codes to fetch.
    """
    ccd_list = list(ccd_codes)  # Convert to list in case it's a generator
    if not ccd_list:
        return

    # Create a temporary file with the list of relative paths to sync
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
        for ccd_code in ccd_list:
            rel = _ccd_code_to_relpath(ccd_code)
            tmp_file.write(f"{rel.as_posix()}\n")
        tmp_file_path = tmp_file.name

    try:
        # Ensure destination directory exists
        dest_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "rsync",
            "-rltvz",
            "--stats",
            "--no-perms",
            "--chmod=ug=rwX,o=rX",
            "--files-from",
            tmp_file_path,
            "--relative",
            remote_base,
            str(dest_path),
        ]

        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"rsync failed with exit code {completed.returncode}")
    finally:
        # Clean up the temporary file
        Path(tmp_file_path).unlink(missing_ok=True)


def _collect_ccd_codes(ccd_codes: list[str] | None, ccd_codes_file: Path | None) -> list[str]:
    """Combine codes from CLI list and an optional file; return normalized unique codes.

    Args:
        ccd_codes: List of CCD codes from CLI arguments.
        ccd_codes_file: Path to file containing CCD codes.

    Returns:
        List of normalized unique CCD codes.
    """
    collected: list[str] = []
    if ccd_codes:
        collected.extend(ccd_codes)
    if ccd_codes_file:
        with open(ccd_codes_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                collected.append(line)

    normalized: list[str] = []
    seen: set[str] = set()
    for code in collected:
        try:
            norm = _normalize_ccd_code(code)
        except ValueError:
            continue
        if norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


app = typer.Typer(help="PDBeChem CCD utilities")


@app.command("sync")
def sync_ccd(
    destination_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
        help="Destination directory to mirror CCD into.",
    ),
    remote: str = typer.Option(
        CCD_REMOTE,
        "--remote",
        help="Rsync remote path for CCD.",
    ),
    ccd_code: list[str] = typer.Option(
        None,
        "--ccd-code",
        help="CCD code to sync (may be used multiple times). If omitted and no file is provided, full sync is done.",
    ),
    ccd_codes_file: Path | None = typer.Option(
        None,
        "--ccd-codes-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a file containing one CCD code per line.",
    ),
) -> None:
    """Mirror the PDBeChem CCD to a local directory fully or for specific CCD codes.

    If no codes are provided via --ccd-code or --ccd-codes-file, a full mirror is performed.
    This replicates the behavior of scripts/setup_ccd_mirror.sh using Python-only dependencies.
    """
    typer.echo("Setting up CCD mirror...")
    destination_path.mkdir(parents=True, exist_ok=True)

    codes = _collect_ccd_codes(ccd_code, ccd_codes_file)

    typer.echo("Testing rsync connection to EBI server...")
    ok, output = _run_rsync_list(remote)
    if not ok:
        typer.secho("Connection test failed", fg=typer.colors.RED)
        typer.echo("Error output:")
        typer.echo(output)
        raise typer.Exit(code=1)
    typer.secho("Connection test successful", fg=typer.colors.GREEN)

    if codes:
        typer.echo(f"Syncing {len(codes)} selected CCD codes...")
        _rsync_fetch_specific(remote, destination_path, codes)
    else:
        typer.echo("Syncing full PDBeChem CCD (this may take a few minutes)...")
        _rsync_sync(remote, destination_path)
    typer.secho("Sync completed successfully!", fg=typer.colors.GREEN)

    # Write README
    readme_path = destination_path / "README"
    mode = "a" if readme_path.exists() else "w"
    with open(readme_path, mode, encoding="utf-8") as fh:
        fh.write("# CCD Mirror Information\n\n")
        fh.write(f"Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        cmd_repr = ["atomworks", "sync_ccd", str(destination_path)]
        if codes:
            cmd_repr.extend(functools.reduce(operator.iadd, (["--ccd-code", c] for c in codes), []))
        fh.write(f"Sync command: {' '.join(cmd_repr)}\n")
        fh.write(f"Sync user: {os.getenv('USER') or os.getenv('USERNAME') or 'unknown'}\n")

    typer.echo("")
    typer.echo(f"Set 'CCD_MIRROR_PATH={destination_path}' in your .env file")
    typer.echo(f"  ... or run 'export CCD_MIRROR_PATH={destination_path}' in your shell")
    typer.echo(f"  ... or add 'export CCD_MIRROR_PATH={destination_path}' to your shell profile.")
    typer.echo("")
