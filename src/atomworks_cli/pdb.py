"""Command line interface for managing a local PDB mmCIF mirror."""

# ruff: noqa: B008
from __future__ import annotations

import functools
import operator
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import typer

PDB_ID_REGEX = re.compile(r"^[0-9a-zA-Z]{4}$")
PDB_REMOTE = "rsync.wwpdb.org::ftp/data/structures/divided/mmCIF/"
PDB_PORT = 33444


def _normalize_pdb_id(pdb_id: str) -> str:
    """Return a normalized, lower-case 4-char PDB id or raise ValueError.

    Args:
        pdb_id: The PDB ID to normalize.

    Returns:
        Normalized lowercase PDB ID.

    Raises:
        ValueError: If the PDB ID is invalid.
    """
    pdb_id = pdb_id.strip().lower()
    if not PDB_ID_REGEX.match(pdb_id):
        raise ValueError(f"Invalid PDB id: {pdb_id}")
    return pdb_id


def _pdb_id_to_relpath(pdb_id: str) -> Path:
    """Map a PDB id to its relative mmCIF path under the divided layout.

    Example: '1a0i' -> 'a0/1a0i.cif.gz'

    Args:
        pdb_id: The PDB ID to map.

    Returns:
        The relative path to the mmCIF file.
    """
    pid = _normalize_pdb_id(pdb_id)
    subdir = pid[1:3]
    return Path(subdir) / f"{pid}.cif.gz"


def _run_rsync_list(remote_path: str, port: int | None) -> tuple[bool, str]:
    """Try to list a remote rsync path and return success and output/error.

    Args:
        remote_path: The remote rsync path to list.
        port: The port to use for rsync connection.

    Returns:
        Tuple of (success, output) where success is a boolean and output is the stdout/stderr.
    """
    cmd = ["rsync", "--list-only"]
    if port is not None:
        cmd.extend(["--port", str(port)])
    cmd.append(remote_path)
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        success = completed.returncode == 0
        output = completed.stdout if success else completed.stderr
        return success, output
    except FileNotFoundError:
        return False, "rsync executable not found. Please install rsync."


def _rsync_sync_full(remote_path: str, dest_path: Path, port: int | None) -> None:
    """Perform a full mirror of the mmCIF divided tree into dest_path."""
    cmd = [
        "rsync",
        "-rltvz",
        "--stats",
        "--no-perms",
        "--chmod=ug=rwX,o=rX",
        "--delete",
        "--omit-dir-times",
    ]
    if port is not None:
        cmd.extend(["--port", str(port)])
    cmd.extend([remote_path, str(dest_path)])
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"rsync full sync failed with exit code {completed.returncode}")


def _rsync_fetch_specific(remote_base: str, dest_path: Path, pdb_ids: Iterable[str], port: int | None) -> None:
    """Fetch only specific PDB ids by rsync-ing their individual files in a single command.

    Uses rsync's --files-from option with --relative to preserve directory structure
    and fetch all files in one efficient operation.
    """
    pdb_list = list(pdb_ids)  # Convert to list in case it's a generator
    if not pdb_list:
        return

    # Create a temporary file with the list of relative paths to sync
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
        for pdb_id in pdb_list:
            rel = _pdb_id_to_relpath(pdb_id)
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
        ]
        if port is not None:
            cmd.extend(["--port", str(port)])
        cmd.extend([remote_base, str(dest_path)])

        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"rsync failed with exit code {completed.returncode}")
    finally:
        # Clean up the temporary file
        Path(tmp_file_path).unlink(missing_ok=True)


def _collect_pdb_ids(pdb_ids: list[str] | None, pdb_ids_file: Path | None) -> list[str]:
    """Combine ids from CLI list and an optional file; return normalized unique ids."""
    collected: list[str] = []
    if pdb_ids:
        collected.extend(pdb_ids)
    if pdb_ids_file:
        with open(pdb_ids_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                collected.append(line)
    normalized: list[str] = []
    seen: set[str] = set()
    for pid in collected:
        try:
            norm = _normalize_pdb_id(pid)
        except ValueError:
            continue
        if norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


app = typer.Typer(
    help="RCSB PDB mmCIF mirror utilities. Allows you to synchronize a local copy of the RCSB PDB mmCIF archive."
)


@app.command("sync")
def sync_pdb(
    destination_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
        help="Destination directory to mirror PDB mmCIFs into.",
    ),
    remote: str = typer.Option(
        PDB_REMOTE,
        "--remote",
        help="Rsync remote base path for divided mmCIF tree.",
    ),
    port: int = typer.Option(PDB_PORT, "--port", help="Rsync server port for RCSB."),
    pdb_id: list[str] = typer.Option(
        None,
        "--pdb-id",
        help="PDB id to sync (may be used multiple times). If omitted and no file is provided, full sync is done.",
    ),
    pdb_ids_file: Path | None = typer.Option(
        None,
        "--pdb-ids-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a file containing one PDB id per line.",
    ),
) -> None:
    """Mirror the RCSB PDB mmCIF archive fully or for specific PDB ids.

    If no ids are provided via --pdb-id or --pdb-ids-file, a full mirror is performed.
    As of August 2025, a full RCSB PDB mirror requires about 100 GB of disk space.
    """
    typer.echo("Setting up RCSB PDB mirror...")
    destination_path.mkdir(parents=True, exist_ok=True)

    ids = _collect_pdb_ids(pdb_id, pdb_ids_file)

    # Test rsync connection
    typer.echo("Testing rsync connection to RCSB server...")
    ok, output = _run_rsync_list(remote, port)
    if not ok:
        typer.secho("Connection test failed", fg=typer.colors.RED)
        typer.echo("Error output:")
        typer.echo(output)
        raise typer.Exit(code=1)
    typer.secho("Connection test successful", fg=typer.colors.GREEN)

    if ids:
        typer.echo(f"Syncing {len(ids)} selected PDB ids...")
        _rsync_fetch_specific(remote, destination_path, ids, port)
    else:
        typer.echo(
            "Syncing full RCSB PDB mmCIF tree (this may take a while and requires about 100 GB of disk space)..."
        )
        _rsync_sync_full(remote, destination_path, port)
    typer.secho("Sync completed successfully!", fg=typer.colors.GREEN)

    # Write README
    readme_path = destination_path / "README"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("# RCSB PDB Mirror Information\n\n")
        from datetime import datetime

        fh.write(f"Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        cmd_repr = ["atomworks", "sync_pdb", str(destination_path)]
        if ids:
            cmd_repr.extend(functools.reduce(operator.iadd, (["--pdb-id", i] for i in ids), []))
        fh.write(f"Sync command: {' '.join(cmd_repr)}\n")
        fh.write(f"Sync user: {os.getenv('USER') or os.getenv('USERNAME') or 'unknown'}\n")

    typer.echo("")
    typer.echo(f"Set 'PDB_MIRROR_PATH={destination_path}' in your .env file")
    typer.echo(f"  ... or run 'export PDB_MIRROR_PATH={destination_path}' in your shell")
    typer.echo(f"  ... or add 'export PDB_MIRROR_PATH={destination_path}' to your shell profile.")
    typer.echo("")
