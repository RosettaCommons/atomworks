"""
Parse, protonate and round-trip a molecule
==========================================

This offline example uses ethanol from Biotite's bundled CCD. The same code runs
against the installed core wheel in CI, without PyTorch or a PDB mirror.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from atomworks.experimental.protonation import assign_hydrogens, place_hydrogens
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.utils.ccd import get_ccd_component_from_biotite
from atomworks.io.utils.io_utils import to_cif_file

# %%
# Prepare a heavy-atom input and parse it with explicit configuration.
component = get_ccd_component_from_biotite("EOH")
component.chain_id[:] = "A"
component.res_id[:] = 1
component.hetero[:] = True
heavy = component[component.element != "H"]
config = ParseConfig.from_preset("minimal", hydrogen_policy="keep", add_id_and_entity_annotations=True)

with TemporaryDirectory() as directory:
    path = Path(directory) / "ethanol.cif"
    to_cif_file(heavy, path, ccd_entries={"EOH": component})
    atoms = parse(path, config=config)["asym_unit"][0]

    # Assignment chooses the state; placement supplies hydrogen coordinates.
    state = assign_hydrogens(atoms, ph=7.4)
    protonated = place_hydrogens(state)
    np.testing.assert_array_equal(protonated.coord[protonated.element != "H"], atoms.coord)
    assert np.count_nonzero(protonated.element == "H") == 6
    assert np.isfinite(protonated.coord).all()

    # Write and reparse the complete molecule without changing its chemistry.
    to_cif_file(protonated, path, ccd_entries={"EOH": component})
    restored = parse(path, config=config)["asym_unit"][0]
    np.testing.assert_array_equal(restored.atom_name, protonated.atom_name)
    np.testing.assert_allclose(restored.coord, protonated.coord, atol=0.001)
    np.testing.assert_array_equal(restored.bonds.as_array(), protonated.bonds.as_array())

print("Ethanol: 3 heavy atoms and 6 hydrogens; coordinates and bonds round-trip.")
