"""MSA generation command using MMseqs2 or HHblits."""

import logging
from pathlib import Path

import torch
import typer

from atomworks.enums import MSAFileExtension
from atomworks.ml.preprocessing.msa.generating import (
    HHblitsSearchConfig,
    MMseqs2SearchConfig,
    MSAGenerationConfig,
    make_msas_from_csv,
)

from .common import enable_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def generate(
    csv_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to and file name of the CSV file containing protein sequences.",
    ),
    output_dir: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Output directory for generated MSA files.",
    ),
    sequence_column: str | None = typer.Option(
        None,
        "--sequence-column",
        "-c",
        help="Name of column containing sequences (required if CSV has multiple columns).",
    ),
    # MSAGenerationConfig parameters
    sharding_pattern: str = typer.Option(
        "/0:2/",
        "--sharding-pattern",
        "-s",
        help="Directory sharding pattern (e.g., '/0:2/').",
    ),
    output_extension: str = typer.Option(
        MSAFileExtension.A3M_GZ.value,
        "--output-extension",
        "-o",
        help="Output file extension (.a3m, .a3m.gz, .a3m.zst, .afa, .afa.gz, .afa.zst).",
    ),
    gpu: bool | None = typer.Option(
        None,
        "--gpu/--no-gpu",
        help="(MMseqs2) Use GPU acceleration (auto-detects if not specified)",
    ),
    num_iterations: int = typer.Option(
        3,
        "--num-iterations",
        "-n",
        help="(MMseqs2) Number of search iterations",
    ),
    max_final_sequences: int = typer.Option(
        10_000,
        "--max-final-sequences",
        help="Maximum number of sequences in final MSAs.",
    ),
    use_env: bool = typer.Option(
        True,
        "--use-env/--no-env",
        help="(MMseqs2) Include environmental (metagenomic) database",
    ),
    threads: int = typer.Option(
        4,
        "--threads",
        "-j",
        help="Number of CPU threads for search operations (used by both MMseqs2 and HHblits)",
    ),
    sensitivity: float | None = typer.Option(
        8.0,
        "--sensitivity",
        help="(MMseqs2) Sensitivity (lower = faster, sparser MSAs)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
    check_existing: bool = typer.Option(
        False,
        "--check-existing/--no-check-existing",
        help="Check for existing MSAs before generation.",
    ),
    existing_msa_dirs: str | None = typer.Option(
        None,
        "--existing-msa-dirs",
        help="Comma-separated MSA directories to check (uses PROTEIN_MSA_DIRS env var if not specified)",
    ),
    backend: str = typer.Option(
        "mmseqs2",
        "--backend",
        "-b",
        help="MSA generation backend: 'mmseqs2' (default) or 'hhblits' (CPU-only)",
    ),
    hhblits_mem: int = typer.Option(
        64,
        "--hhblits-mem",
        help="(HHblits) Memory limit in GB",
    ),
) -> None:
    """Generate MSAs from sequences in a CSV file using MMseqs2 or HHblits.

    Examples:
        # Single-column CSV
        atomworks msa generate sequences.csv output_msas/

        # Multi-column CSV
        atomworks msa generate data.csv output_msas/ --sequence-column seq

        # With custom parameters
        atomworks msa generate sequences.csv output_msas/ --gpu --max-final-sequences 5000 --threads 16
    """
    enable_logging(verbose)

    # Auto-detect GPU if not specified
    if gpu is None:
        gpu = torch.cuda.is_available()

    # Parse MSA directories if provided
    msa_dirs = None
    if existing_msa_dirs:
        msa_dirs = [Path(d.strip()) for d in existing_msa_dirs.split(",")]

    # Build backend-specific config
    hhblits_search_config = None
    if backend == "hhblits":
        hhblits_search_config = HHblitsSearchConfig(mem=hhblits_mem)
        if gpu:
            logger.info("Note: HHblits is CPU-only; --gpu flag will be ignored for the HHblits backend")

    # Create search config with sensitivity and iterations control
    search_config = MMseqs2SearchConfig(
        s=sensitivity,
        num_iterations=num_iterations,
    )

    # Create generation config
    config = MSAGenerationConfig(
        sharding_pattern=sharding_pattern,
        output_extension=output_extension,
        gpu=gpu,
        use_env=use_env,
        threads=threads,
        max_final_sequences=max_final_sequences,
        check_existing=check_existing,
        existing_msa_dirs=msa_dirs,
        search_config=search_config,
        backend=backend,
        hhblits_search_config=hhblits_search_config,
    )

    # Display configuration
    typer.secho("MSA Generation Configuration:", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  CSV File: {csv_file}")
    typer.echo(f"  Sequence Column: {sequence_column or 'auto-detect'}")
    typer.echo(f"  Output Directory: {output_dir}")
    typer.echo(f"  Backend: {config.backend}")
    typer.echo(f"  Max Final Sequences: {config.max_final_sequences}")
    typer.echo(f"  Output Extension: {config.output_extension}")
    typer.echo(f"  Sharding Pattern: {config.sharding_pattern}")
    typer.echo(f"  Check Existing: {config.check_existing}")
    if config.check_existing:
        dirs_display = config.existing_msa_dirs if config.existing_msa_dirs else "PROTEIN_MSA_DIRS env var"
        typer.echo(f"  MSA Directories: {dirs_display}")
    if config.backend == "hhblits":
        typer.echo(f"  Threads: {config.threads}")
        typer.echo(f"  HHblits Memory: {hhblits_search_config.mem} GB")
    else:
        typer.echo(f"  GPU Enabled: {config.gpu}")
        typer.echo(f"  Iterations: {config.search_config.num_iterations}")
        typer.echo(f"  Threads: {config.threads}")
        typer.echo(f"  Use Environmental DB: {config.use_env}")
        typer.echo(f"  Sensitivity: {config.search_config.s}")

    try:
        typer.secho("\n🚀 Starting MSA generation...", fg=typer.colors.CYAN, bold=True)
        make_msas_from_csv(csv_file=csv_file, output_dir=output_dir, sequence_column=sequence_column, config=config)
        typer.secho("✅ MSA generation completed successfully!", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        typer.secho(f"Error during MSA generation: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
