"""Reproduce a declared P-H placement failure using an OMol phosphorane.

Install AtomWorks from public commit e0f2e4da0aef2cf6c81a2c506eece3bc4204b75d
and run: python reproduce.py
"""
import json
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray, BondList
from atomworks.experimental.protonation import assign_hydrogens, place_hydrogens

fixture = json.loads(Path(__file__).with_name("phosphorane-0594f821.json").read_text())
atoms = AtomArray(len(fixture["atom_name"]))
for field in ("atom_name", "element", "coord"):
    setattr(atoms, field, np.array(fixture[field]))
atoms.chain_id[:] = "A"
atoms.res_id[:] = 1
atoms.res_name[:] = "A:1"
atoms.set_annotation("charge", np.array(fixture["charge"], dtype=np.int8))
atoms.set_annotation("pn_unit_iid", np.zeros(len(atoms), dtype=int))
atoms.set_annotation("atom_id", np.arange(len(atoms)))
atoms.bonds = BondList(len(atoms), np.array(fixture["bonds"], dtype=np.uint32))

# Control: the complete supplied molecule needs no P-H construction.
kept = place_hydrogens(assign_hydrogens(atoms))
assert len(kept) == len(atoms) == 38
for atom_id, coord in zip(atoms.atom_id, atoms.coord, strict=True):
    assert np.array_equal(kept.coord[kept.atom_id == atom_id][0], coord)
print("PASS: all 38 supplied atoms and coordinates survive when P-H is present.")

# Remove only P-H7. Its required count comes from the original explicit bond.
parent = int(np.flatnonzero(atoms.atom_name == "P1")[0])
neighbors, orders = atoms.bonds.get_bonds(parent)
assert len(neighbors) == 5 and np.all(orders == 1)
hydrogen = neighbors[atoms.element[neighbors] == "H"]
assert len(hydrogen) == 1 and atoms.atom_name[hydrogen[0]] == "H7"
missing_h = atoms[np.arange(len(atoms)) != hydrogen[0]]
declared = np.full(len(missing_h), -1, dtype=int)
declared[missing_h.atom_name == "P1"] = 1
state = assign_hydrogens(missing_h, hydrogens=declared)
assert state.nhyd[state.atom_name == "P1"].item() == 1
assert state.charge[state.atom_name == "P1"].item() == 0
print("PASS: assigning the declared neutral P-H state requests exactly one H.")
# Current result: ValueError: Atom 4 (P) has more hydrogens than free directions.
place_hydrogens(state)
