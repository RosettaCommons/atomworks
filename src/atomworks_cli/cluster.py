"""CLI commands for sequence clustering."""

from pathlib import Path

import typer

from .common import enable_logging

app = typer.Typer(help="Sequence clustering utilities")


@app.command(name="add")
def add(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        resolve_path=True,
        help="Path to parquet file with sequences",
    ),
    sequence_column: str = typer.Option(
        ...,
        "-s",
        "--sequence-column",
        help="Name of column containing sequences",
    ),
    output_column: str = typer.Option(
        "cluster",
        "-o",
        "--output-column",
        help="Name of output cluster column",
    ),
    identity: float = typer.Option(
        0.4,
        "-i",
        "--identity",
        help="Sequence identity threshold (0.0-1.0)",
    ),
    coverage: float = typer.Option(
        0.8,
        "-c",
        "--coverage",
        help="Coverage threshold (0.0-1.0)",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        help="Output file path. If not specified, modifies input in place.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Cluster sequences and add cluster column using MMseqs2.

    Examples:
        atomworks cluster add data.parquet -s sequence -i 0.4 -o cluster_40
        atomworks cluster add proteins.parquet -s seq -i 0.9 -o cluster_90
    """
    enable_logging(verbose)

    from atomworks.ml.preprocessing.clustering.cluster import add_cluster_column_to_file
    from atomworks.ml.preprocessing.clustering.config import MMseqs2Config

    typer.echo(f"Clustering {input_file}")
    typer.echo(f"  Sequence column: {sequence_column}")
    typer.echo(f"  Identity: {identity}")
    typer.echo(f"  Output column: {output_column}")

    config = MMseqs2Config(cluster_identity=identity, coverage=coverage)
    result = add_cluster_column_to_file(
        input_file,
        sequence_column,
        output_column,
        config,
        output_file,
    )
    typer.secho(f"Done! Processed {len(result):,} rows", fg=typer.colors.GREEN)
