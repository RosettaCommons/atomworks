"""MSA organization command for standardized AtomWorks-compatible MSA directory structures."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from atomworks.enums import MSAFileExtension
from atomworks.ml.preprocessing.msa.organizing import MSAOrganizationConfig, organize_msas

from .common import enable_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def organize(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Source directory containing MSA files to organize.",
    ),
    output_dir: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Destination directory for organized MSA files.",
    ),
    input_extension: str = typer.Option(
        MSAFileExtension.A3M_GZ.value,
        "--input-extension",
        "-i",
        help="File extension for input MSA files (e.g., .a3m, .a3m.gz, .a3m.zst, .afa, .afa.gz, .afa.zst).",
    ),
    output_extension: str = typer.Option(
        MSAFileExtension.A3M_GZ.value,
        "--output-extension",
        "-o",
        help="File extension for output MSA files (e.g., .a3m, .a3m.gz, .a3m.zst, .afa, .afa.gz, .afa.zst).",
    ),
    sharding_pattern: str | None = typer.Option(
        "/0:2/",
        "--sharding-pattern",
        "-s",
        help="Sharding pattern for organizing files (e.g., '/0:2/'). Set to empty string to disable sharding.",
    ),
    copy_files: bool = typer.Option(
        True,
        "--copy/--move",
        help="Whether to copy (--copy) or move (--move) input files.",
    ),
    num_workers: int | None = typer.Option(
        None,
        "--num-workers",
        "-j",
        help="Number of parallel workers (defaults to min(CPU_COUNT, 16)).",
    ),
    check_existing: bool = typer.Option(
        False,
        "--check-existing",
        help="Check for existing MSAs before organization.",
    ),
    existing_msa_dirs: str | None = typer.Option(
        None,
        "--existing-msa-dirs",
        help="Comma-separated MSA directories to check (uses LOCAL_MSA_DIRS env var if not specified).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Organize MSA files into a standardized directory structure and naming convention.

    MSA files are renamed using the SHA-256 hash of their first sequence and organized
    into a directory structure that enables efficient lookup. Automatic compression/decompression
    is applied based on the input and output file extensions.

    The implementation uses smart parallelism: thread-based for I/O-bound tasks (same compression
    format) and process-based for CPU-bound tasks (compression/decompression). This provides
    excellent performance (~700+ files/second) for typical workloads.

    Examples:
        # Organize MSA files with default extensions (both .a3m.gz)
        atomworks msa organize ./msas ./msas_organized

        # Convert uncompressed MSA files to compressed ones
        atomworks msa organize ./msas ./msas_organized --input-extension .a3m --output-extension .a3m.gz

        # Convert between compression formats (gzip to zstd)
        atomworks msa organize ./msas ./msas_organized --input-extension .a3m.gz --output-extension .a3m.zst

        # Organize with custom sharding pattern
        atomworks msa organize ./msas ./msas_organized --sharding-pattern "/0:2/2:4/"

        # Check existing MSAs before organizing
        atomworks msa organize ./msas ./msas_organized --check-existing --existing-msa-dirs ./dir1,./dir2
    """
    if not sharding_pattern:
        sharding_pattern = None

    # Parse MSA directories if provided (same pattern as generate command)
    msa_dirs = None
    if existing_msa_dirs:
        msa_dirs = [Path(d.strip()) for d in existing_msa_dirs.split(",")]

    config = MSAOrganizationConfig(
        input_extension=input_extension,
        output_extension=output_extension,
        sharding_pattern=sharding_pattern,
        copy_files=copy_files,
        num_workers=num_workers,
        check_existing=check_existing,
        existing_msa_dirs=msa_dirs,
    )

    # Display configuration in a table
    config_table = [
        ["Input Directory", str(input_dir)],
        ["Output Directory", str(output_dir)],
        ["Input File Extension", config.input_extension],
        ["Output File Extension", config.output_extension],
        ["Sharding Pattern", str(config.sharding_pattern)],
        ["Copy Files", str(config.copy_files)],
        ["Number of Workers", str(config.num_workers)],
        ["Check Existing", str(config.check_existing)],
    ]
    typer.secho("MSA Organization Configuration", fg=typer.colors.CYAN, bold=True)
    typer.secho("=" * 40, fg=typer.colors.CYAN)
    for key, value in config_table:
        typer.secho(f"{key:<18}: ", fg=typer.colors.BLUE, nl=False)
        typer.echo(f"{value}")
    typer.secho("=" * 40, fg=typer.colors.CYAN)
    if config.check_existing:
        dirs_display = config.existing_msa_dirs if config.existing_msa_dirs else "LOCAL_MSA_DIRS env var"
        typer.secho(f"MSA Directories: {dirs_display}", fg=typer.colors.BLUE)

    enable_logging(verbose)

    try:
        typer.secho("\n🚀 Starting MSA organization...", fg=typer.colors.CYAN, bold=True)
        organize_msas(input_dir=input_dir, output_dir=output_dir, config=config)
        typer.secho("✅ MSA organization completed successfully!", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        typer.secho(f"Error during MSA organization: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
