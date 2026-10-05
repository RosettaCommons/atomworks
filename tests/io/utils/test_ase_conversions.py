"""Tests for ASE-Biotite conversion utilities."""

from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from biotite.structure import AtomArray, BondList, BondType  # noqa: E402

from atomworks.io.parser import parse  # noqa: E402
from atomworks.io.utils.ase_conversions import ase_to_atom_array, atom_array_to_ase  # noqa: E402
from atomworks.io.utils.testing import get_pdb_path  # noqa: E402

DATA = Path(__file__).parents[2] / "data" / "io"

# structures chosen for what they carry: a ligand whose bonds the file states, a non-canonical
# residue with charged atoms, 1QFE's LYS170 linked to its ligand, and 101M's haem protein
WITH_BONDS = [
    "test_unl_ligand_with_bonds.cif",
    "example_ncaa.cif",
    "1qfe",
    "101m_arginine_nh1nh2_swapped.cif",
]


def _structure(name):
    """The asymmetric unit of a test file, or of a PDB ID's mmCIF, as a single model."""
    path = DATA / name if name.endswith(".cif") else get_pdb_path(name)
    return parse(path, config="minimal")["asym_unit"][0]


def _bond_names(array):
    """Bonds as the atoms they join, so a reordering is visible as a change."""
    if array.bonds is None:
        return set()
    return {
        (
            *sorted((str(array.atom_name[i]) + str(array.res_id[i]), str(array.atom_name[j]) + str(array.res_id[j]))),
            int(order),
        )
        for i, j, order in array.bonds.as_array()
    }


@pytest.mark.parametrize("filename", WITH_BONDS)
def test_a_structure_survives_repeated_round_trips(filename):
    """Out to ASE and back, twice, changes nothing about the structure (101M lost its 1307 bonds)."""
    original = _structure(filename)
    assert original.bonds is not None and original.bonds.get_bond_count() > 0

    current = original
    for _ in range(2):
        current = ase_to_atom_array(atom_array_to_ase(current), formal_charges=True)

    np.testing.assert_array_equal(current.element, original.element)
    np.testing.assert_array_equal(current.atom_name, original.atom_name)
    np.testing.assert_array_equal(current.charge, original.charge)
    np.testing.assert_allclose(current.coord, original.coord, atol=1e-5)
    assert _bond_names(current) == _bond_names(original)


def test_roundtrip_conversion_with_metadata():
    """ASE -> Biotite -> ASE keeps symbols, positions, ``info`` and extra arrays."""
    original_atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    original_atoms.info["energy"] = -10.5
    original_atoms.info["method"] = "DFT"
    original_atoms.arrays["forces"] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    final_atoms = atom_array_to_ase(ase_to_atom_array(original_atoms))

    assert original_atoms.get_chemical_symbols() == final_atoms.get_chemical_symbols()
    np.testing.assert_array_almost_equal(original_atoms.get_positions(), final_atoms.get_positions(), decimal=5)
    assert final_atoms.info["energy"] == -10.5
    assert final_atoms.info["method"] == "DFT"
    np.testing.assert_array_almost_equal(original_atoms.arrays["forces"], final_atoms.arrays["forces"])


@pytest.mark.parametrize("filename", WITH_BONDS)
def test_bonds_follow_their_atoms_when_ase_reorders_them(filename):
    """ASE keeps a per-atom array in step with the atoms, and the bonds ride on it."""
    original = _structure(filename)

    atoms = atom_array_to_ase(original)
    order = np.arange(len(atoms))[::-1]
    reversed_back = ase_to_atom_array(atoms[order])

    assert _bond_names(reversed_back) == _bond_names(original)


@pytest.mark.parametrize("bond_type", BondType)
def test_bond_types_survive_ase_reordering_and_deletion(bond_type):
    original = AtomArray(3)
    original.element[:] = "C"
    original.set_annotation("atom_id", np.array([30, 10, 20]))
    original.bonds = BondList(3, np.array([[0, 1, bond_type], [1, 2, bond_type]]))

    reordered = atom_array_to_ase(original)[[1, 0]]
    restored = ase_to_atom_array(reordered)

    np.testing.assert_array_equal(restored.atom_id, [10, 30])
    np.testing.assert_array_equal(restored.bonds.as_array(), [[0, 1, bond_type]])


@pytest.mark.parametrize("filename", WITH_BONDS)
def test_a_repeated_atom_id_moves_no_bond(filename, caplog):
    """ASE repeats ids when structures are joined; no bond may land on the wrong copy."""
    atoms = atom_array_to_ase(_structure(filename))
    joined = ase_to_atom_array(atoms + atoms.copy())
    n = len(atoms)

    bonds = joined.bonds.as_array()
    assert not ((bonds[:, 0] < n) != (bonds[:, 1] < n)).any()
    assert len(bonds[bonds[:, 0] < n]) == len(bonds[bonds[:, 0] >= n])
    assert f"Dropped {len(atoms.info['atomworks_bonds'])} bonds whose atom_ids are duplicated" in caplog.text


def test_a_bond_whose_atom_is_gone_is_dropped_and_the_rest_are_kept():
    """Half the ligand deleted in ASE leaves the bonds among what is left."""
    original = _structure("test_unl_ligand_with_bonds.cif")

    keep = np.arange(0, original.array_length(), 2)
    kept = ase_to_atom_array(atom_array_to_ase(original)[keep])

    assert _bond_names(kept) == _bond_names(original[keep])


def test_a_calculators_partial_charges_are_not_taken_for_formal_charges():
    """``initial_charges`` holds a formal charge or a partial one; both survive."""
    original = _structure("test_unl_ligand_with_bonds.cif")

    atoms = atom_array_to_ase(original)
    atoms.set_initial_charges(np.linspace(-0.4, 0.4, len(atoms)))
    back = ase_to_atom_array(atoms)

    np.testing.assert_allclose(back.initial_charges, atoms.get_initial_charges())
    assert "charge" not in back.get_annotation_categories()
    np.testing.assert_allclose(atom_array_to_ase(back).get_initial_charges(), atoms.get_initial_charges())


def test_a_bond_table_that_cannot_be_read_is_ignored():
    """``info`` is whatever the caller put there; it is not a promise."""
    original = _structure("test_unl_ligand_with_bonds.cif")

    atoms = atom_array_to_ase(original)
    atoms.info["atomworks_bonds"] = "not a bond table"

    assert ase_to_atom_array(atoms).bonds is None
