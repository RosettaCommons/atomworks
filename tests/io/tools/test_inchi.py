"""Tests for InChI support."""

import os
import random

import pytest
from biotite.structure import AtomArray
from rdkit.Chem.inchi import MolToInchi

from atomworks.constants import CCD_MIRROR_PATH
from atomworks.io.tools.rdkit import atom_array_to_rdkit, inchi_to_rdkit
from atomworks.io.utils.ccd import (
    atom_array_from_ccd_code,
    get_available_ccd_codes_in_mirror,
    get_ccd_descriptors,
)
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from tests.conftest import skip_if_no_ccd_mirror


def _inchi_from_ccd_array(arr: AtomArray) -> str | None:
    """Convert a CCD AtomArray (with 3D coords) to a canonical InChI string."""
    heavy = arr[arr.element != "H"]
    mol = atom_array_to_rdkit(heavy, hydrogen_policy="remove", sanitize=True)
    return MolToInchi(mol)


def _inchi_from_inchi_string(inchi: str) -> str | None:
    """Round-trip an InChI string through RDKit to get a canonical form."""
    mol = inchi_to_rdkit(inchi, generate_conformers=False)
    return MolToInchi(mol)


@skip_if_no_ccd_mirror
class TestMolecularEquivalence:
    """Tests verifying molecular equivalence between InChI and CCD atom array paths."""

    @pytest.mark.parametrize(
        "ccd_code",
        [
            "ATP",  # Adenosine triphosphate
            "NAG",  # N-acetylglucosamine
            "ALA",  # Alanine
            "GLY",  # Glycine
            "GDP",  # Guanosine diphosphate
        ],
    )
    def test_inchi_vs_ccd_atom_array_equivalence(self, ccd_code: str):
        """Test that InChI -> AtomArray produces same heavy atoms as CCD -> AtomArray."""
        descriptors = get_ccd_descriptors(ccd_code)
        inchi = descriptors.get("InChI")
        assert inchi is not None, f"No InChI descriptor for {ccd_code}"

        ccd_arr = atom_array_from_ccd_code(ccd_code)
        inchi_from_ccd = _inchi_from_ccd_array(ccd_arr)
        inchi_canonical = _inchi_from_inchi_string(inchi)

        assert inchi_from_ccd == inchi_canonical, (
            f"Canonical InChI mismatch for {ccd_code}:\n"
            f"  from CCD:   {inchi_from_ccd}\n"
            f"  from InChI: {inchi_canonical}"
        )

    @pytest.fixture(scope="class")
    def random_ccd_codes_with_inchi(self):
        """Get random CCD codes that have InChI descriptors."""
        if not CCD_MIRROR_PATH or not os.path.isdir(CCD_MIRROR_PATH):
            pytest.skip("CCD_MIRROR_PATH not set or not a valid directory")

        available_codes = list(get_available_ccd_codes_in_mirror(CCD_MIRROR_PATH))
        if len(available_codes) == 0:
            pytest.skip("No CCD codes available in mirror")

        with rng_state(create_rng_state_from_seeds(py_seed=42)):
            sample_size = min(500, len(available_codes))
            sampled_codes = random.sample(available_codes, sample_size)

        valid_codes = []
        for code in sampled_codes:
            if len(valid_codes) >= 100:
                break
            descriptors = get_ccd_descriptors(code)
            if descriptors.get("InChI"):
                valid_codes.append((code, descriptors["InChI"]))

        if len(valid_codes) < 10:
            pytest.skip("Not enough CCD codes with InChI descriptors")

        return valid_codes

    def test_random_ccd_inchi_equivalence(self, random_ccd_codes_with_inchi):
        """Test molecular equivalence for random CCD codes.

        Allows up to 10% failure rate for edge cases (metal complexes, exotic coordination, etc.).
        """
        codes = random_ccd_codes_with_inchi
        failures = []

        for ccd_code, inchi in codes:
            try:
                ccd_arr = atom_array_from_ccd_code(ccd_code)
                inchi_from_ccd = _inchi_from_ccd_array(ccd_arr)
                inchi_canonical = _inchi_from_inchi_string(inchi)
                assert inchi_from_ccd == inchi_canonical, (
                    f"Canonical InChI mismatch:\n"
                    f"  from CCD:   {inchi_from_ccd}\n"
                    f"  from InChI: {inchi_canonical}"
                )
            except Exception as e:
                failures.append(f"{ccd_code}: {e}")

        max_failures = max(1, int(len(codes) * 0.1))  # Allow up to 10% failures
        assert len(failures) <= max_failures, f"Too many failures ({len(failures)}/{len(codes)}): {failures[:10]}..."


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
