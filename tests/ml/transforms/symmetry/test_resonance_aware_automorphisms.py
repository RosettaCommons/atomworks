"""Regression tests for resonance-aware, bond-order-aware automorphism search."""

import json
from pathlib import Path

import numpy as np
import pytest
from biotite import structure as struc
from biotite.structure import AtomArray

from atomworks.ml.transforms.covalent_modifications import AnnotateCovalentModifications
from atomworks.ml.transforms.symmetry import build_automorphism_units, find_automorphisms_with_networkx
from atomworks.ml.utils.testing import cached_parse

REFERENCE = json.loads((Path(__file__).resolve().parents[3] / "data" / "automorphism_reference.json").read_text())


def as_ligand(ccd_code: str) -> AtomArray:
    """Return a heavy-atom CCD entry annotated as a single non-polymer unit."""
    atom_array = struc.info.residue(ccd_code)
    atom_array = atom_array[atom_array.element != "H"]
    atom_array.res_name[:] = "LIG"
    atom_array.hetero[:] = True
    atom_array.chain_id[:] = "A"
    atom_array.res_id[:] = 1
    atom_array.set_annotation("pn_unit_iid", np.full(atom_array.array_length(), "L0"))
    atom_array.set_annotation("is_polymer", np.zeros(atom_array.array_length(), dtype=bool))
    return atom_array


def swapped_atom_names(atom_array: AtomArray) -> set[str]:
    """Atom names that move under some automorphism of ``atom_array``."""
    return {
        str(atom_array.atom_name[source])
        for group in find_automorphisms_with_networkx(atom_array)
        for row in group[1:]
        for source, target in zip(group[0], row, strict=True)
        if source != target
    }


@pytest.mark.parametrize("ccd_code", sorted(REFERENCE), ids=sorted(REFERENCE))
def test_automorphisms_match_reference(ccd_code: str) -> None:
    """Each hard molecule keeps the automorphism group recorded on disk."""
    expected = REFERENCE[ccd_code]
    groups = find_automorphisms_with_networkx(as_ligand(ccd_code))
    n_automorphisms = sum(len(group) for group in groups if len(group) > 1) or 1
    assert n_automorphisms == expected["n_automorphisms"]
    assert sorted(swapped_atom_names(as_ligand(ccd_code))) == expected["swapped_atoms"]


def test_covalent_modifications_on_1ivo() -> None:
    """The search respects covalent modifications on a real N-glycosylated protein (1ivo)."""
    # ``raw`` is a pristine snapshot (the transform reassigns pn_units in place, so reassign a
    # separate copy and keep ``raw`` for the original pn_unit layout).
    raw = cached_parse("1ivo", hydrogen_policy="remove")["atom_array"].copy()
    atom_array = AnnotateCovalentModifications()(
        {"atom_array": cached_parse("1ivo", hydrogen_policy="remove")["atom_array"]}
    )["atom_array"]
    units = build_automorphism_units(atom_array)

    # Units partition every atom and none spans a pn_unit.
    assert np.array_equal(np.sort(np.concatenate(units)), np.arange(atom_array.array_length()))
    for unit in units:
        assert len(set(atom_array.pn_unit_iid[unit])) == 1, "a unit spans several pn_units"

    # Every covalent modification (a bond across pn_units in the raw parse) shares a unit.
    unit_of = np.empty(atom_array.array_length(), dtype=int)
    for index, unit in enumerate(units):
        unit_of[unit] = index
    covalent_bonds = [(int(i), int(j)) for i, j, _ in raw.bonds.as_array() if raw.pn_unit_iid[i] != raw.pn_unit_iid[j]]
    assert covalent_bonds, "expected 1ivo to carry covalent modifications"
    for i, j in covalent_bonds:
        assert unit_of[i] == unit_of[j], "a glycosylated residue and its glycan should share a unit"

    # No backbone atom is swappable (a terminal carboxylate genuinely swaps O/OXT and is exempt).
    for group in find_automorphisms_with_networkx(atom_array):
        if "OXT" in atom_array.atom_name[group[0]]:
            continue
        moved = {
            str(atom_array.atom_name[source])
            for row in group[1:]
            for source, target in zip(group[0], row, strict=True)
            if source != target
        }
        assert not moved & {"N", "CA", "C", "O"}, f"backbone atoms {sorted(moved)} permuted"
