"""Tests for bond distance-based filtering functionality."""

import numpy as np
import pytest

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.bonds import filter_bonds_by_distance
from atomworks.io.utils.testing import get_pdb_path_or_buffer

# (pdb_id, atom_selection, displacement) - displacement must exceed the element's threshold
BOND_FILTER_TEST_CASES = [
    # Metal bonds: threshold 3.6A, displace 10A
    pytest.param("1a6l", "element == 'FE'", 10.0, id="metal_FeS_cluster_1a6l"),
    pytest.param("1aco", "element == 'FE'", 10.0, id="metal_FeS_cluster_1aco"),
    # Organic bonds: threshold 1.8A, displace 5A
    pytest.param("1a6l", "(res_name == 'ALA') & (atom_name == 'CA')", 5.0, id="organic_alanine_CA"),
    pytest.param("1aco", "(res_name == 'GLY') & (atom_name == 'CA')", 5.0, id="organic_glycine_CA"),
]


class TestBondDistanceFiltering:
    """Test filtering of bonds based on distance thresholds."""

    @pytest.mark.parametrize("pdb_id, atom_selection, displacement", BOND_FILTER_TEST_CASES)
    def test_bond_filtered_when_moved_beyond_threshold(self, pdb_id: str, atom_selection: str, displacement: float):
        """Moving an atom beyond its threshold should cause its bonds to be filtered."""
        path = get_pdb_path_or_buffer(pdb_id)
        result = parse(path, config=ParseConfig(build_assembly="all", long_bond_policy="keep"))
        atom_array = result["assemblies"]["1"][0]

        # Find the atom to move
        mask = eval(
            atom_selection,
            {"__builtins__": {}},
            {
                "element": atom_array.element,
                "res_name": atom_array.res_name,
                "atom_name": atom_array.atom_name,
            },
        )
        atom_idx = np.where(mask)[0][0]

        # Count bonds involving this atom before filtering
        bonds_before = atom_array.bonds.as_array()
        bonds_with_atom = np.sum((bonds_before[:, 0] == atom_idx) | (bonds_before[:, 1] == atom_idx))
        assert bonds_with_atom > 0, f"Atom {atom_idx} should have bonds"

        # Move the atom beyond threshold
        atom_array.coord[atom_idx] += np.array([displacement, 0.0, 0.0])

        # Filter and verify bonds were removed (policy="filter" is the default for the function)
        filtered = filter_bonds_by_distance(atom_array, policy="filter")
        bonds_after = filtered.bonds.as_array()
        bonds_with_atom_after = np.sum((bonds_after[:, 0] == atom_idx) | (bonds_after[:, 1] == atom_idx))

        assert (
            bonds_with_atom_after < bonds_with_atom
        ), f"Bonds to moved atom should be filtered (before: {bonds_with_atom}, after: {bonds_with_atom_after})"

    def test_warn_policy_keeps_all_bonds(self):
        """With long_bond_policy='warn' (default), bond count should remain unchanged."""
        path = get_pdb_path_or_buffer("1a6l")

        result_default = parse(path, config=ParseConfig(build_assembly="all"))
        result_keep = parse(path, config=ParseConfig(build_assembly="all", long_bond_policy="keep"))

        assert len(result_default["assemblies"]["1"][0].bonds.as_array()) == len(
            result_keep["assemblies"]["1"][0].bonds.as_array()
        )

    def test_filter_policy_removes_long_bonds(self):
        """With long_bond_policy='filter', long bonds should be removed."""
        path = get_pdb_path_or_buffer("1a6l")

        result_keep = parse(path, config=ParseConfig(build_assembly="all", long_bond_policy="keep"))
        result_filter = parse(path, config=ParseConfig(build_assembly="all", long_bond_policy="filter"))

        # Filter policy should not increase bond count (may decrease or stay same)
        assert len(result_filter["assemblies"]["1"][0].bonds.as_array()) <= len(
            result_keep["assemblies"]["1"][0].bonds.as_array()
        )

    def test_raise_policy_raises_on_long_bonds(self):
        """With long_bond_policy='raise', should raise ValueError if long bonds detected."""
        path = get_pdb_path_or_buffer("1a6l")
        result = parse(path, config=ParseConfig(build_assembly="all", long_bond_policy="keep"))
        atom_array = result["assemblies"]["1"][0]

        # Move an atom beyond threshold to create a long bond
        fe_mask = atom_array.element == "FE"
        fe_idx = np.where(fe_mask)[0][0]
        atom_array.coord[fe_idx] += np.array([10.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="bonds exceeding distance thresholds"):
            filter_bonds_by_distance(atom_array, policy="raise")

    def test_filter_nonstandard_only_preserves_standard_residues(self):
        """filter_nonstandard_only preserves bonds in standard residues."""
        path = get_pdb_path_or_buffer("1a6l")
        result = parse(path, config=ParseConfig(build_assembly="all", long_bond_policy="keep"))
        atom_array = result["assemblies"]["1"][0]

        # Move a standard AA atom beyond threshold
        ala_mask = (atom_array.res_name == "ALA") & (atom_array.atom_name == "CA")
        ala_idx = np.where(ala_mask)[0][0]
        atom_array.coord[ala_idx] += np.array([5.0, 0.0, 0.0])

        # Count bonds before and after
        bonds_before = len(atom_array.bonds.as_array())
        filtered = filter_bonds_by_distance(atom_array, policy="filter_nonstandard_only")
        bonds_after = len(filtered.bonds.as_array())

        assert bonds_after == bonds_before, "Standard residue bonds should not be filtered"


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
