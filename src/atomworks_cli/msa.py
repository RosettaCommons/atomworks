"""Command line interface for MSA (Multiple Sequence Alignment) operations."""

import typer

from .filter import filter
from .find import find
from .generate import generate
from .organize import organize

app = typer.Typer(help="MSA (Multiple Sequence Alignment) utilities")

# Add subcommands
app.command(name="find", help="Find MSA files for sequences in a CSV file")(find)
app.command(name="filter", help="Filter MSA files using HHfilter")(filter)
app.command(name="generate", help="Generate MSAs from sequences using MMseqs2 or HHblits")(generate)
app.command(name="organize", help="Organize MSA files into standardized structure")(organize)
