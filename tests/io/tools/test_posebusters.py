"""Tests for PoseBusters validation using CCD components with ideal coordinates."""

import numpy as np
import pytest

posebusters = pytest.importorskip("posebusters")

from atomworks.io.tools.posebusters import validate_molecule
from atomworks.io.utils.ccd import atom_array_from_ccd_code


@pytest.fixture(scope="module")
def pb():
    """Shared PoseBusters instance for the test module."""
    from posebusters import PoseBusters

    return PoseBusters(config="mol")


@pytest.mark.parametrize(
    "ccd_code",
    [
        "ALA",  # amino acid
        "ATP",  # nucleotide
        "NAG",  # sugar (N-acetylglucosamine)
        "BEN",  # benzene
        "HEM",  # iron porphyrin (Fe disconnects, organic porphyrin validated)
        "BCL",  # bacteriochlorophyll (Mg disconnects, organic fragment validated)
        "CLA",  # chlorophyll a (Mg disconnects, organic fragment validated)
    ],
)
def test_validate_molecule_from_ccd(pb, ccd_code):
    """CCD components with ideal coordinates should pass, perturbed should fail."""
    # ... first, check that vanilla CCD molecule passes
    atom_array = atom_array_from_ccd_code(ccd_code, coords="ideal_pdbx")
    failed, checks = validate_molecule(atom_array, pb=pb)
    assert not failed, f"{ccd_code} failed PB checks: {[k for k, v in checks.items() if not v]}"

    # ... next, check that a 2A perturbation of a carbon atom causes failure

    # Perturb a carbon atom by 2A — should now fail
    perturbed = atom_array.copy()
    carbon_indices = np.where(perturbed.element == "C")[0]
    assert len(carbon_indices) > 0, f"No carbon atoms in {ccd_code}"
    perturbed.coord[carbon_indices[0], 2] += 2.0

    failed, checks = validate_molecule(perturbed, pb=pb)
    assert failed, f"{ccd_code} should fail PB checks after 2A perturbation of carbon atom {carbon_indices[0]}"
