"""Entry point for the AtomWorks command-line interface."""

import typer

app = typer.Typer(help="AtomWorks command-line interface")

# Import commands to register them with the root app and expose sub-apps
from . import ccd as _ccd  # noqa: E402
from . import pdb as _pdb  # noqa: E402, T100
from . import setup as _setup  # noqa: E402

# Expose namespaced groups: `atomworks ccd ...`, ...
app.add_typer(_ccd.app, name="ccd")
app.add_typer(_pdb.app, name="pdb")
app.add_typer(_setup.app, name="setup")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
