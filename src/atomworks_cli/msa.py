"""Command line interface for MSA (Multiple Sequence Alignment) operations."""

import typer

app = typer.Typer(help="MSA (Multiple Sequence Alignment) utilities")

try:
    from .filter import filter
    from .find import find
    from .generate import generate
    from .organize import organize
except ModuleNotFoundError as error:
    if error.name not in {"torch", "einops", "beartype", "numba", "filelock"}:
        raise
    app.info.help = r"Install 'atomworks\[ml]' to use MSA commands."

    @app.callback(invoke_without_command=True)
    def require_ml() -> None:
        """Install 'atomworks[ml]' to use MSA commands."""
        typer.echo("MSA commands require the ML extra: pip install 'atomworks[ml]'", err=True)
        raise typer.Exit(1)

else:
    app.command(name="find", help="Find MSA files for sequences in a CSV file")(find)
    app.command(name="filter", help="Filter MSA files using HHfilter")(filter)
    app.command(name="generate", help="Generate MSAs from sequences using MMseqs2 or HHblits")(generate)
    app.command(name="organize", help="Organize MSA files into standardized structure")(organize)
