import numpy as np
import pytest

from atomworks.io.utils.ccd import (
    _get_base_ccd_template,
    atom_array_from_ccd_code,
    get_polymerization_atoms,
    register_custom_ccd_entry,
    unregister_custom_ccd_entry,
)


@pytest.mark.parametrize(
    "ccd_code,expected_coord_type,coord_preferences",
    [
        ("H5C", "model", ("ideal_pdbx", "model")),  # H5C should fall back to model
        ("HEM", "ideal_pdbx", ("ideal_pdbx", "model")),  # HEM should use ideal_pdbx
    ],
)
def test_coordinate_fallback_behavior(ccd_code: str, expected_coord_type: str, coord_preferences: tuple[str, ...]):
    """Test coordinate fallback behavior for different CCD codes."""
    # Test with fallback preferences
    fallback_result = atom_array_from_ccd_code(ccd_code, coords=coord_preferences)
    expected_result = atom_array_from_ccd_code(ccd_code, coords=(expected_coord_type,))

    # Should match the expected coordinate type
    np.testing.assert_array_equal(
        fallback_result.coord,
        expected_result.coord,
        err_msg=f"{ccd_code} with preferences {coord_preferences} should match {expected_coord_type} coordinates",
    )

    # Test with different preference order
    if len(coord_preferences) > 1:
        reversed_prefs = tuple(reversed(coord_preferences))
        if reversed_prefs[0] != expected_coord_type:
            # If the first preference in reversed order is different from expected, result should be different
            reversed_result = atom_array_from_ccd_code(ccd_code, coords=reversed_prefs)
            first_pref_result = atom_array_from_ccd_code(ccd_code, coords=(reversed_prefs[0],))

            np.testing.assert_array_equal(
                reversed_result.coord,
                first_pref_result.coord,
                err_msg=f"{ccd_code} with reversed preferences should match first preference {reversed_prefs[0]}",
            )


def test_coordinate_fallback_with_mirror():
    """Test coordinate fallback using atom_array_from_ccd_code (works with Biotite fallback)."""
    # Test H5C - should fall back to model
    h5c_fallback = atom_array_from_ccd_code("H5C", coords=("ideal_pdbx", "model"))
    h5c_model = atom_array_from_ccd_code("H5C", coords=("model",))

    np.testing.assert_array_equal(
        h5c_fallback.coord,
        h5c_model.coord,
        err_msg="H5C fallback should match model coordinates",
    )

    # Test HEM - should use ideal_pdbx
    hem_ideal = atom_array_from_ccd_code("HEM", coords=("ideal_pdbx", "model"))
    hem_model = atom_array_from_ccd_code("HEM", coords=("model",))

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            hem_ideal.coord,
            hem_model.coord,
            err_msg="HEM ideal and model coordinates should be different",
        )


def test_coordinate_types_validation():
    """Test that invalid coordinate types raise appropriate errors."""
    with pytest.raises(ValueError, match="Invalid coordinate type"):
        atom_array_from_ccd_code("ALA", coords=("invalid_type",))

    with pytest.raises(ValueError, match="Invalid coordinate type"):
        atom_array_from_ccd_code("ALA", coords=("ideal_pdbx", "invalid_type"))


def test_single_coordinate_type_compatibility():
    """Test that single coordinate type still works (backward compatibility)."""
    try:
        # Test that single string still works
        ala_single = atom_array_from_ccd_code("ALA", coords="ideal_pdbx")
        ala_tuple = atom_array_from_ccd_code("ALA", coords=("ideal_pdbx",))

        np.testing.assert_array_equal(
            ala_single.coord, ala_tuple.coord, err_msg="Single string and single-item tuple should give same results"
        )

    except ValueError as e:
        if "not found" in str(e):
            pytest.skip(f"ALA not available for testing: {e}")
        else:
            raise


def test_no_coordinates_fallback():
    """Test behavior when no coordinate preferences can be satisfied."""
    result = atom_array_from_ccd_code("H5C", coords=("ideal_pdbx",))  # H5C doesn't have ideal PDBx coordinates
    # Should be all NaNs
    assert np.all(np.isnan(result.coord)), "Should have NaN coordinates"


def test_custom_ccd_codes_follow_reregistration():
    """Non-standard CCD lookups must reflect the latest registered definition."""
    custom_code = "ZZTEST:0"
    mol_a = atom_array_from_ccd_code("ALA", coords=None)
    mol_b = atom_array_from_ccd_code("GLY", coords=None)

    try:
        register_custom_ccd_entry(custom_code, mol_a)
        assert len(_get_base_ccd_template(custom_code, "", "keep")) == len(mol_a)

        register_custom_ccd_entry(custom_code, mol_b)
        assert len(_get_base_ccd_template(custom_code, "", "keep")) == len(mol_b)
    finally:
        unregister_custom_ccd_entry(custom_code)


@pytest.mark.parametrize(
    "ccd_code,expected",
    [
        ("ALA", ("C", "N")),  # canonical peptide atom names
        ("DA", ("O3'", "P")),  # canonical nucleotide atom names
        ("CRO", ("C3", "N1")),  # GFP chromophore: derived from the CCD's leaving-atom flags
        ("3DA", ("O2'", "P")),  # cordycepin: no O3', so it links 2'->5' via an override
        ("4DG", (None, "P")),  # acyclic guanine phosphonate: a 3' terminus, nothing exits
    ],
)
def test_get_polymerization_atoms(ccd_code: str, expected: tuple[str | None, str | None]):
    """Polymerization atoms resolve via canonical names, the CCD's own annotations, or an override."""
    assert get_polymerization_atoms(ccd_code) == expected


def test_get_polymerization_atoms_prefers_canonical_names_over_overrides(cleanup_registry):
    """A custom component registered under an overridden code resolves by its own atom names."""
    custom_ncaa = atom_array_from_ccd_code("TRP", coords=None)
    custom_ncaa.res_name[:] = "3DA"
    register_custom_ccd_entry("3DA", custom_ncaa)

    # Without the canonical-names check first, this would return the nucleotide override ("O2'", "P").
    assert get_polymerization_atoms("3DA") == ("C", "N")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
