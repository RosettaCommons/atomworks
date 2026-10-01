"""Tests for the ASE -> RDKit -> AtomArray pipeline against CCD reference."""

import pytest

pytest.importorskip("ase")

from atomworks.io.tools.rdkit import atom_array_from_rdkit, atom_array_to_rdkit
from atomworks.io.utils.ase_conversions import ase_to_atom_array, atom_array_to_ase
from atomworks.io.utils.ccd import atom_array_from_ccd_code
from atomworks.io.utils.testing import assert_same_atom_array_or_stack


@pytest.mark.parametrize("ccd_code", ["H5C", "ALA", "GLY", "NAG", "ATP"])
def test_ase_pipeline_matches_ccd(ccd_code):
    """ASE -> RDKit -> AtomArray pipeline recovers the CCD reference structure."""
    ccd_arr = atom_array_from_ccd_code(ccd_code)

    # ASE -> RDKit -> AtomArray pipeline (simulates what happens with OMOL/ASE data)
    ase_atoms = atom_array_to_ase(ccd_arr)
    bare = ase_to_atom_array(ase_atoms)
    mol = atom_array_to_rdkit(bare, infer_bonds=True, hydrogen_policy="keep", system_charge=0)
    result = atom_array_from_rdkit(mol, remove_hydrogens=False)

    assert_same_atom_array_or_stack(
        ccd_arr,
        result,
        compare_bonds=True,
        compare_bond_order=False,
        compare_coords=True,
        enforce_order=False,
        annotations_to_compare=["element", "charge"],
    )
