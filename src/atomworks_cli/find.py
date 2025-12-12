"""Finding pre-computed MSAs on disk, and reporting missing sequences."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from atomworks.ml.preprocessing.msa.finding import find_msas

from .common import enable_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def find(
    csv_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="CSV file containing protein sequences",
    ),
    sequence_column: str | None = typer.Option(
        None,
        "--sequence-column",
        "-c",
        help="Name of column containing sequences (required if CSV has multiple columns)",
    ),
    existing_msa_dirs: str | None = typer.Option(
        None,
        "--existing-msa-dirs",
        help="Comma-separated MSA directories to find (uses LOCAL_MSA_DIRS env var if not specified)",
    ),
    missing_output: Path | None = typer.Option(
        None,
        "--missing-output",
        help="Optional path to save CSV with missing sequences",
    ),
    found_output: Path | None = typer.Option(
        None,
        "--found-output",
        help="Optional path to save CSV with found sequences and their MSA paths",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Find MSA files for sequences in a CSV file.

    Analyzes a CSV file to find existing MSA files for sequences and
    optionally saves missing and found sequences to separate CSV files.

    Examples:
        # Find MSAs for single-column CSV
        atomworks msa find sequences.csv

        # Find MSAs for multi-column CSV with specific sequence column
        atomworks msa find data.csv --sequence-column seq

        # Find MSAs and save missing sequences
        atomworks msa find sequences.csv --missing-output missing.csv

        # Find MSAs and save both missing and found sequences
        atomworks msa find sequences.csv --missing-output missing.csv --found-output found.csv

        # Find MSAs with custom MSA directories
        atomworks msa find sequences.csv --existing-msa-dirs /path/msa1,/path/msa2
    """
    enable_logging(verbose)

    # Parse MSA directories if provided
    msa_dirs = None
    if existing_msa_dirs:
        msa_dirs = [Path(d.strip()) for d in existing_msa_dirs.split(",")]

    # Display configuration
    typer.secho("MSA Finding Configuration:", fg=typer.colors.CYAN, bold=True)
    typer.secho("=" * 35, fg=typer.colors.CYAN)
    typer.echo(f"  CSV File: {csv_file}")
    typer.echo(f"  Sequence Column: {sequence_column or 'auto-detect'}")
    if missing_output:
        typer.echo(f"  Missing Output: {missing_output}")
    if found_output:
        typer.echo(f"  Found Output: {found_output}")
    if msa_dirs:
        typer.echo(f"  MSA Directories: {msa_dirs}")
    else:
        typer.echo("  MSA Directories: LOCAL_MSA_DIRS env var")
    typer.secho("=" * 35, fg=typer.colors.CYAN)

    try:
        # Load CSV and extract sequences
        typer.secho("\n📂 Loading sequences from CSV...", fg=typer.colors.CYAN, bold=True)

        df = pd.read_csv(csv_file)

        # Determine sequence column
        if sequence_column is None:
            if len(df.columns) != 1:
                raise ValueError(
                    f"CSV has {len(df.columns)} columns. Either provide exactly 1 column or specify --sequence-column"
                )
            sequence_column = df.columns[0]

        if sequence_column not in df.columns:
            raise ValueError(f"Column '{sequence_column}' not found in CSV. Available columns: {list(df.columns)}")

        # Extract unique sequences
        sequences = df[sequence_column].dropna().unique().tolist()
        total_sequences = len(sequences)
        typer.echo(f"Loaded {total_sequences} unique sequences from column '{sequence_column}'")

        # Find MSAs
        typer.secho("\n🔍 Finding existing MSAs...", fg=typer.colors.CYAN, bold=True)
        missing_sequences, sequence_to_msa_path = find_msas(
            sequences=sequences,
            msa_dirs=msa_dirs,
        )

        # Calculate statistics
        found_count = len(sequence_to_msa_path)
        missing_count = len(missing_sequences)
        coverage_percent = (found_count / total_sequences * 100) if total_sequences > 0 else 0

        # Display results with colors
        typer.secho("\n📊 Results:", fg=typer.colors.CYAN, bold=True)
        typer.secho(f"  Total sequences: {total_sequences:,}", fg=typer.colors.BLUE)

        found_color = (
            typer.colors.GREEN
            if coverage_percent > 80
            else typer.colors.YELLOW
            if coverage_percent > 50
            else typer.colors.RED
        )
        typer.secho(f"  Found MSAs: {found_count:,} ({coverage_percent:.1f}%)", fg=found_color)
        typer.secho(
            f"  Missing MSAs: {missing_count:,} ({100 - coverage_percent:.1f}%)",
            fg=typer.colors.RED if missing_count > 0 else typer.colors.GREEN,
        )

        # Save results to files if requested
        if missing_output and missing_sequences:
            missing_path = Path(missing_output)
            missing_path.parent.mkdir(parents=True, exist_ok=True)
            missing_df = pd.DataFrame({"sequence": missing_sequences})
            missing_df.to_csv(missing_path, index=False)
            typer.secho(f"💾 Missing sequences saved to: {missing_path}", fg=typer.colors.BLUE)
        elif missing_output and not missing_sequences:
            typer.echo("No missing sequences to save")

        if found_output and sequence_to_msa_path:
            found_path = Path(found_output)
            found_path.parent.mkdir(parents=True, exist_ok=True)
            found_df = pd.DataFrame(
                [{"sequence": seq, "msa_path": str(path)} for seq, path in sequence_to_msa_path.items()]
            )
            found_df.to_csv(found_path, index=False)
            typer.secho(f"💾 Found sequences saved to: {found_path}", fg=typer.colors.BLUE)
        elif found_output and not sequence_to_msa_path:
            typer.echo("No found sequences to save")

        if missing_count == 0:
            typer.secho("✅ All sequences have MSAs!", fg=typer.colors.GREEN, bold=True)

    except Exception as e:
        typer.secho(f"Error during MSA finding: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
