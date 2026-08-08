"""MSA generation command using MMseqs2, either locally or against a remote MSA server."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import torch
import typer

from atomworks.enums import MSAFileExtension
from atomworks.ml.preprocessing.msa.generating import (
    MMseqs2SearchConfig,
    MSAGenerationConfig,
    make_msas_from_csv,
)
from atomworks.ml.preprocessing.msa.server import DEFAULT_MSA_SERVER_URL, MSAServerConfig

from .common import enable_logging

app = typer.Typer()
logger = logging.getLogger(__name__)

# Options that only make sense for the local MMseqs2 pipeline
_LOCAL_ONLY_OPTIONS = {
    "gpu": "--gpu/--no-gpu",
    "num_workers": "--num-workers/-j",
    "sensitivity": "--sensitivity",
    "num_iterations": "--num-iterations/-n",
}


class MSABackend(str, Enum):
    """Where the MMseqs2 search runs."""

    LOCAL = "local"
    SERVER = "server"


def _was_passed_on_the_command_line(ctx: typer.Context, parameter_name: str) -> bool:
    """Whether the user explicitly passed a parameter (as opposed to it taking its default value).

    Note:
        We compare the parameter source by name rather than importing `click.core.ParameterSource`,
        since recent Typer releases vendor Click rather than depending on it.
    """
    source = ctx.get_parameter_source(parameter_name)
    return source is not None and source.name == "COMMANDLINE"


def _reject_local_only_options(ctx: typer.Context) -> None:
    """Error out if the user passed local-backend options together with `--backend server`."""
    passed = [flag for name, flag in _LOCAL_ONLY_OPTIONS.items() if _was_passed_on_the_command_line(ctx, name)]
    if passed:
        raise typer.BadParameter(
            f"{', '.join(passed)} {'is' if len(passed) == 1 else 'are'} only supported with '--backend local'; "
            "the remote MSA server controls its own search parameters."
        )


@app.command()
def generate(
    ctx: typer.Context,
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
    backend: MSABackend = typer.Option(
        MSABackend.LOCAL.value,
        "--backend",
        help="Run MMseqs2 against local ColabFold databases, or submit to a remote MSA server",
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
        help="Use GPU acceleration (auto-detects if not specified). Local backend only",
    ),
    num_iterations: int = typer.Option(
        3,
        "--num-iterations",
        "-n",
        help="Number of MMseqs2 search iterations. Local backend only",
    ),
    max_final_sequences: int | None = typer.Option(
        None,
        "--max-final-sequences",
        help=(
            "Maximum number of sequences in final MSAs "
            "(default: 10000 for the local backend, no HHfilter pass for the server backend)"
        ),
    ),
    use_env: bool = typer.Option(
        True,
        "--use-env/--no-env",
        help="Include environmental (metagenomic) database",
    ),
    num_workers: int = typer.Option(
        32,
        "--num-workers",
        "-j",
        help="Number of CPU threads. Local backend only",
    ),
    sensitivity: float | None = typer.Option(
        8.0,
        "--sensitivity",
        help="MMseqs2 sensitivity (lower = faster, sparser MSAs). Local backend only",
    ),
    # MSAServerConfig parameters
    server_url: str = typer.Option(
        DEFAULT_MSA_SERVER_URL,
        "--server-url",
        help="Base URL of the MMseqs2 server. Server backend only",
    ),
    server_username: str | None = typer.Option(
        None,
        "--server-username",
        help="Username for HTTP basic auth (or set MSA_SERVER_USERNAME). Server backend only",
    ),
    server_password: str | None = typer.Option(
        None,
        "--server-password",
        help="Password for HTTP basic auth (or set MSA_SERVER_PASSWORD). Server backend only",
    ),
    api_key_header: str | None = typer.Option(
        None,
        "--api-key-header",
        help="Header name to carry an API key, e.g. 'X-API-Key'. Server backend only",
    ),
    api_key_value: str | None = typer.Option(
        None,
        "--api-key-value",
        help="API key value (or set MSA_SERVER_API_KEY). Server backend only",
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
        help="Comma-separated MSA directories to check (uses LOCAL_MSA_DIRS env var if not specified)",
    ),
) -> None:
    """Generate MSAs from sequences in a CSV file using MMseqs2.

    Examples:
        # Single-column CSV
        atomworks msa generate sequences.csv output_msas/

        # Multi-column CSV
        atomworks msa generate data.csv output_msas/ --sequence-column seq

        # With custom parameters
        atomworks msa generate sequences.csv output_msas/ --gpu --max-final-sequences 5000 --num-workers 16

        # Without local databases, using the public ColabFold MSA server
        atomworks msa generate sequences.csv output_msas/ --backend server
    """
    enable_logging(verbose)

    is_server = backend is MSABackend.SERVER
    if is_server:
        _reject_local_only_options(ctx)

    # Auto-detect GPU if not specified
    if gpu is None:
        gpu = False if is_server else torch.cuda.is_available()

    # HHfilter is a local binary; don't require it by default when the search itself ran remotely
    if max_final_sequences is None and not is_server:
        max_final_sequences = 10_000

    # Parse MSA directories if provided
    msa_dirs = None
    if existing_msa_dirs:
        msa_dirs = [Path(d.strip()) for d in existing_msa_dirs.split(",")]

    # Create search config with only sensitivity control
    search_config = MMseqs2SearchConfig(
        s=sensitivity,
    )

    try:
        server_config = MSAServerConfig(
            host_url=server_url,
            use_env=use_env,
            username=server_username,
            password=server_password,
            api_key_header=api_key_header,
            api_key_value=api_key_value,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    # Create generation config
    config = MSAGenerationConfig(
        backend=backend.value,
        sharding_pattern=sharding_pattern,
        output_extension=output_extension,
        gpu=gpu,
        num_iterations=num_iterations,
        use_env=use_env,
        threads=num_workers,
        max_final_sequences=max_final_sequences,
        check_existing=check_existing,
        existing_msa_dirs=msa_dirs,
        search_config=search_config,
        server=server_config,
    )

    # Display configuration
    typer.secho("MSA Generation Configuration:", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  CSV File: {csv_file}")
    typer.echo(f"  Sequence Column: {sequence_column or 'auto-detect'}")
    typer.echo(f"  Output Directory: {output_dir}")
    typer.echo(f"  Backend: {config.backend}")
    if is_server:
        typer.echo(f"  Server URL: {config.server.host_url}")
        typer.echo(f"  Server Auth: {_describe_server_auth(config.server)}")
    else:
        typer.echo(f"  GPU Enabled: {config.gpu}")
        typer.echo(f"  Iterations: {config.num_iterations}")
        typer.echo(f"  Threads: {config.threads}")
        typer.echo(f"  Sensitivity: {config.search_config.s}")
    typer.echo(f"  Max Final Sequences: {config.max_final_sequences if config.max_final_sequences else 'no filtering'}")
    typer.echo(f"  Use Environmental DB: {config.use_env}")
    typer.echo(f"  Output Extension: {config.output_extension}")
    typer.echo(f"  Sharding Pattern: {config.sharding_pattern}")
    typer.echo(f"  Check Existing: {config.check_existing}")
    if config.check_existing:
        dirs_display = config.existing_msa_dirs if config.existing_msa_dirs else "LOCAL_MSA_DIRS env var"
        typer.echo(f"  MSA Directories: {dirs_display}")

    try:
        typer.secho("\n🚀 Starting MSA generation...", fg=typer.colors.CYAN, bold=True)
        make_msas_from_csv(csv_file=csv_file, output_dir=output_dir, sequence_column=sequence_column, config=config)
        typer.secho("✅ MSA generation completed successfully!", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        typer.secho(f"Error during MSA generation: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


def _describe_server_auth(config: MSAServerConfig) -> str:
    """Summarize how requests to the MSA server will be authenticated (without echoing secrets)."""
    if config.username is not None or config.password is not None:
        return f"basic auth (user: {config.username})"
    if config.api_key_value is not None:
        return f"API key in header '{config.api_key_header}'"
    return "none"
