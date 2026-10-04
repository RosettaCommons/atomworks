"""MSA generation command using MMseqs2 or HHblits."""

import logging
from pathlib import Path

import typer

from atomworks.constants import DEFAULT_MSA_SERVER_URL
from atomworks.enums import MSAFileExtension

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
        help="CSV file containing protein sequences",
    ),
    output_dir: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Output directory for generated MSA files",
    ),
    sequence_column: str | None = typer.Option(
        None,
        "--sequence-column",
        "-c",
        help="Name of column containing sequences (required if CSV has multiple columns)",
    ),
    # MSAGenerationConfig parameters
    sharding_pattern: str = typer.Option(
        "/0:2/",
        "--sharding-pattern",
        "-s",
        help="Directory sharding pattern (e.g., '/0:2/')",
    ),
    output_extension: str = typer.Option(
        MSAFileExtension.A3M_GZ.value,
        "--output-extension",
        "-o",
        help="Output file extension (.a3m, .a3m.gz, .a3m.zst, .afa, .afa.gz, .afa.zst)",
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
    max_final_sequences: int | None = typer.Option(
        None,
        "--max-final-sequences",
        help="Local HHfilter limit (default: 10000 for local backends; disabled for the server)",
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
        help="Enable verbose logging",
    ),
    check_existing: bool = typer.Option(
        False,
        "--check-existing/--no-check-existing",
        help="Check for existing MSAs before generation",
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
        help="MSA backend: 'mmseqs2' (default), 'hhblits', or 'mmseqs2_server'",
    ),
    server_url: str = typer.Option(DEFAULT_MSA_SERVER_URL, "--server-url", help="Remote MMseqs2 server URL"),
    server_job_timeout: float = typer.Option(1800.0, "--server-job-timeout", help="Remote batch deadline in seconds"),
    server_api_key_header: str | None = typer.Option(
        None, "--server-api-key-header", help="Header for MSA_SERVER_API_KEY"
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
    import torch

    from atomworks.ml.preprocessing.msa.generating import (
        HHblitsSearchConfig,
        MMseqs2SearchConfig,
        MSAGenerationConfig,
        make_msas_from_csv,
    )
    from atomworks.ml.preprocessing.msa.server import MSAServerConfig

    enable_logging(verbose)

    # Auto-detect GPU if not specified
    if gpu is None:
        gpu = backend == "mmseqs2" and torch.cuda.is_available()

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
        max_final_sequences=max_final_sequences if max_final_sequences is not None else 10000,
        server_max_final_sequences=max_final_sequences,
        check_existing=check_existing,
        existing_msa_dirs=msa_dirs,
        search_config=search_config,
        backend=backend,
        server_config=MSAServerConfig(
            host_url=server_url,
            job_timeout=server_job_timeout,
            use_env=use_env,
            api_key_header=server_api_key_header,
        )
        if backend == "mmseqs2_server"
        else None,
        hhblits_search_config=hhblits_search_config,
    )

    # Display configuration
    typer.secho("MSA Generation Configuration:", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  CSV File: {csv_file}")
    typer.echo(f"  Sequence Column: {sequence_column or 'auto-detect'}")
    typer.echo(f"  Output Directory: {output_dir}")
    typer.echo(f"  Backend: {config.backend}")
    typer.echo(
        f"  Max Final Sequences: {config.server_max_final_sequences if backend == 'mmseqs2_server' else config.max_final_sequences}"
    )
    typer.echo(f"  Output Extension: {config.output_extension}")
    typer.echo(f"  Sharding Pattern: {config.sharding_pattern}")
    typer.echo(f"  Check Existing: {config.check_existing}")
    if config.check_existing:
        dirs_display = config.existing_msa_dirs if config.existing_msa_dirs else "PROTEIN_MSA_DIRS env var"
        typer.echo(f"  MSA Directories: {dirs_display}")
    if config.backend == "mmseqs2_server":
        typer.echo(f"  Server: {config.server_config.host_url}")
        typer.echo(f"  Batch Deadline: {config.server_config.job_timeout} s")
    elif config.backend == "hhblits":
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
