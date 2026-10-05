"""Protonation of a structure: :func:`assign_hydrogens` decides each heavy atom's charge and hydrogen count at a
pH, :func:`place_hydrogens` names, positions and inserts those hydrogens (TMol's internal coordinates where they
apply), and :func:`add_hydrogens` does both.

Example::

    from atomworks.experimental.protonation import add_hydrogens
    from atomworks.io import parse
    from atomworks.io.config import ParseConfig

    config = ParseConfig(add_bond_types_from_struct_conn=("covale", "metalc", "disulf"))
    atoms = parse("1abc.cif", config=config)["asym_unit"][0]
    protonated = add_hydrogens(atoms, ph=7.4)
"""

from atomworks.experimental.protonation._assign import assign_hydrogens
from atomworks.experimental.protonation._placement import add_hydrogens, place_hydrogens

__all__ = ["add_hydrogens", "assign_hydrogens", "place_hydrogens"]
