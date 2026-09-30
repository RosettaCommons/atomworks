"""Tests for ASE-Biotite conversion utilities."""

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402

from atomworks.io.utils.ase_conversions import ase_to_atom_array, atom_array_to_ase  # noqa: E402


def test_roundtrip_conversion_with_metadata():
    """Test round-trip conversion: ASE -> Biotite -> ASE with metadata."""
    # Create atoms with metadata but no PBC
    original_atoms = Atoms(
        "H2O",
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    )
    original_atoms.info["energy"] = -10.5
    original_atoms.info["method"] = "DFT"
    original_atoms.arrays["forces"] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    # ASE -> Biotite -> ASE
    array = ase_to_atom_array(original_atoms)
    final_atoms = atom_array_to_ase(array)

    # Verify preservation of key properties
    assert original_atoms.get_chemical_symbols() == final_atoms.get_chemical_symbols()
    np.testing.assert_array_almost_equal(original_atoms.get_positions(), final_atoms.get_positions(), decimal=5)
    assert final_atoms.info["energy"] == -10.5
    assert final_atoms.info["method"] == "DFT"
    np.testing.assert_array_almost_equal(original_atoms.arrays["forces"], final_atoms.arrays["forces"])
