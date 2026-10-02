"""Inspect a real derived OMol P graph; the P=O edit is a diagnostic, not a repair policy."""
import json
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray, BondList, BondType
from atomworks.experimental.protonation import assign_hydrogens, place_hydrogens

fixture = json.loads(Path(__file__).with_name("tetrahedral-phosphorus-10552741.json").read_text())
atoms = AtomArray(len(fixture["atom_name"]))
for field in ("atom_name", "element", "coord"):
    setattr(atoms, field, np.array(fixture[field]))
atoms.chain_id[:] = "A"
atoms.res_id[:] = 1
atoms.res_name[:] = "A:1"
atoms.set_annotation("charge", np.array(fixture["charge"], dtype=np.int8))
atoms.set_annotation("pn_unit_iid", np.zeros(len(atoms), dtype=int))
atoms.bonds = BondList(len(atoms), np.array(fixture["bonds"], dtype=np.uint32))
parent = int(np.flatnonzero(atoms.atom_name == "P1")[0])
oxygen = int(np.flatnonzero(atoms.atom_name == "O3")[0])
neighbors, orders = atoms.bonds.get_bonds(parent)
assert atoms.element[neighbors].tolist() == ["N", "O", "N", "N"]
assert np.all(orders == BondType.SINGLE)
assert atoms.charge[parent] == 0
state = assign_hydrogens(atoms)
assert state.nhyd[state.atom_name == "P1"].item() == 1
print("Derived all-single, neutral P graph requests one P-H despite no supplied P-H.")
try:
    place_hydrogens(state)
except ValueError as error:
    assert "more hydrogens than free directions" in str(error)
    print(f"Current safeguard: {error}")
else:
    raise AssertionError("Expected the current unsupported-geometry rejection")

# Diagnostic counterfactual: change only this P-O order, not coordinates or formal charges.
double_bond = atoms.copy()
bonds = double_bond.bonds.as_array()
selected = ((bonds[:, 0] == parent) & (bonds[:, 1] == oxygen)) | (
    (bonds[:, 1] == parent) & (bonds[:, 0] == oxygen)
)
assert selected.sum() == 1
bonds[selected, 2] = BondType.DOUBLE
double_bond.bonds = BondList(len(double_bond), bonds)
state = assign_hydrogens(double_bond)
assert state.nhyd[state.atom_name == "P1"].item() == 0
print("P=O diagnostic control requests zero P-H.")

# Stated-count control: zero declared H is retained on the original graph.
declared = np.full(len(atoms), -1, dtype=int)
declared[parent] = 0
state = assign_hydrogens(atoms, hydrogens=declared)
assert state.nhyd[state.atom_name == "P1"].item() == 0
print("Explicit zero-P-H control retains zero P-H without editing the source graph.")
