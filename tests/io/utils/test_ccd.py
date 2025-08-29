import os

import numpy as np
import pytest

from atomworks.constants import CCD_MIRROR_PATH
from atomworks.io.utils.ccd import (
    atom_array_from_ccd_code,
    get_ccd_component_from_mirror,
)


@pytest.mark.parametrize("ccd_mirror_path", [CCD_MIRROR_PATH, None])
@pytest.mark.parametrize(
    "ccd_code,expected_coord_type,coord_preferences",
    [
        ("H5C", "model", ("ideal_pdbx", "model")),  # H5C should fall back to model
        ("HEM", "ideal_pdbx", ("ideal_pdbx", "model")),  # HEM should use ideal_pdbx
    ],
)
def test_coordinate_fallback_behavior(
    ccd_mirror_path: os.PathLike | None, ccd_code: str, expected_coord_type: str, coord_preferences: tuple[str, ...]
):
    """Test coordinate fallback behavior for different CCD codes."""
    # Test with fallback preferences
    fallback_result = atom_array_from_ccd_code(ccd_code, ccd_mirror_path=ccd_mirror_path, coords=coord_preferences)
    expected_result = atom_array_from_ccd_code(ccd_code, ccd_mirror_path=ccd_mirror_path, coords=(expected_coord_type,))

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
            reversed_result = atom_array_from_ccd_code(ccd_code, ccd_mirror_path=ccd_mirror_path, coords=reversed_prefs)
            first_pref_result = atom_array_from_ccd_code(
                ccd_code, ccd_mirror_path=ccd_mirror_path, coords=(reversed_prefs[0],)
            )

            np.testing.assert_array_equal(
                reversed_result.coord,
                first_pref_result.coord,
                err_msg=f"{ccd_code} with reversed preferences should match first preference {reversed_prefs[0]}",
            )


def test_coordinate_fallback_with_mirror():
    """Test coordinate fallback with get_ccd_component_from_mirror."""
    # Test H5C with mirror - should fall back to model
    h5c_mirror_fallback = get_ccd_component_from_mirror("H5C", coords=("ideal_pdbx", "model"))
    h5c_mirror_model = get_ccd_component_from_mirror("H5C", coords=("model",))

    np.testing.assert_array_equal(
        h5c_mirror_fallback.coord,
        h5c_mirror_model.coord,
        err_msg="H5C mirror fallback should match model coordinates",
    )

    # Test HEM with mirror - should use ideal_pdbx
    hem_mirror_ideal = get_ccd_component_from_mirror("HEM", coords=("ideal_pdbx", "model"))
    hem_mirror_model = get_ccd_component_from_mirror("HEM", coords=("model",))

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            hem_mirror_ideal.coord,
            hem_mirror_model.coord,
            err_msg="HEM mirror ideal and model coordinates should be different",
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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
