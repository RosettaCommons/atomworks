"""Tests for DSSP secondary structure annotation."""

import os

import numpy as np
import pytest

from atomworks.constants import STANDARD_AA
from atomworks.enums import ChainType
from atomworks.ml.executables import _EXECUTABLES
from atomworks.ml.executables.dssp import DSSPExecutable
from atomworks.ml.transforms.atom_array import AddGlobalAtomIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.crop import CropSpatialLikeAF3
from atomworks.ml.transforms.dssp import AnnotateSecondaryStructure, SSEnum
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import get_token_starts

DSSP_PATH = os.environ.get("DSSP_PATH", "/projects/ml/dssp/install/bin/mkdssp")

DSSP_TEST_CASES = [
    "1a1e",  # Protein structure
    "1fu2",  # Multi-chain protein
]


def _has_dssp() -> bool:
    """Check if DSSP is installed."""
    if os.path.isfile(DSSP_PATH) and os.access(DSSP_PATH, os.X_OK):
        try:
            DSSPExecutable.get_or_initialize(DSSP_PATH)
            return True
        except Exception:
            return False
    return False


has_dssp = _has_dssp()


skip_if_no_dssp = pytest.mark.skipif(
    not has_dssp,
    reason="DSSP is not installed",
)


@pytest.fixture
def cleanup_dssp():
    """Reinitialize DSSPExecutable for test isolation.

    Ensures the DSSP executable singleton is reset between tests.
    """
    if has_dssp:
        DSSPExecutable.reinitialize(DSSP_PATH)
    yield
    _EXECUTABLES.pop(DSSPExecutable.name, None)
    DSSPExecutable._is_initialized = False


def check_dssp_annotations(atom_array):
    """Verify DSSP annotations are present and valid."""
    # Check annotation exists
    assert "dssp_sse" in atom_array.get_annotation_categories()
    assert "dssp_sse_is_valid" in atom_array.get_annotation_categories()

    # Check annotation length and type
    sse = atom_array.get_annotation("dssp_sse")
    assert len(sse) == atom_array.array_length()
    assert sse.dtype in [np.int8, np.int16, np.int32, np.int64]

    # Check dssp_sse_is_valid length and type
    sse_is_valid = atom_array.get_annotation("dssp_sse_is_valid")
    assert len(sse_is_valid) == atom_array.array_length()
    assert sse_is_valid.dtype == bool or sse_is_valid.dtype == np.bool_

    # Token-level checks
    token_starts = get_token_starts(atom_array)
    token_chain_types = atom_array.chain_type[token_starts]
    protein_token_mask = np.isin(token_chain_types, ChainType.get_proteins())

    # Get the DSSP group for each token (first atom of each token)
    token_dssp_groups = sse[token_starts]
    token_sse_is_valid = sse_is_valid[token_starts]

    # Protein tokens with valid DSSP should have non-NON_PROTEIN groups
    protein_valid_mask = protein_token_mask & token_sse_is_valid
    if np.any(protein_valid_mask):
        assert np.all(
            token_dssp_groups[protein_valid_mask] != SSEnum.NON_PROTEIN
        ), "Protein tokens with valid DSSP must have non-NON_PROTEIN groups"

    # All non-protein tokens should have NON_PROTEIN group
    assert np.all(
        token_dssp_groups[~protein_token_mask] == SSEnum.NON_PROTEIN
    ), "All non-protein tokens must have NON_PROTEIN DSSP group"

    # All non-protein tokens should have is_valid False
    assert np.all(~token_sse_is_valid[~protein_token_mask]), "All non-protein tokens must have dssp_sse_is_valid False"

    # Check all values are valid SSEnum indices
    valid_values = {0, 1, 2, 3, 4}  # 0 for NONE, 1-4 for HELIX, SHEET, LOOP, NON_PROTEIN
    assert set(np.unique(sse)).issubset(valid_values), f"Invalid DSSP values: {np.unique(sse)}"


@pytest.mark.requires_dssp
@skip_if_no_dssp
@pytest.mark.parametrize("pdb_id", DSSP_TEST_CASES)
def test_annotate_secondary_structure(pdb_id, cleanup_dssp):
    """Test DSSP annotation on various structures."""
    data = cached_parse(pdb_id)
    pipe = Compose(
        [
            AtomizeByCCDName(
                atomize_by_default=True,
                res_names_to_ignore=STANDARD_AA,
            ),
            AnnotateSecondaryStructure(),  # Uses environment variable or default
        ]
    )
    data = pipe(data)
    atom_array = data["atom_array"]
    check_dssp_annotations(atom_array)


@pytest.mark.requires_dssp
@skip_if_no_dssp
@pytest.mark.parametrize("pdb_id", ["1fu2"])
def test_annotate_secondary_structure_after_spatial_crop(pdb_id, cleanup_dssp):
    """Test DSSP annotation works correctly after spatial cropping."""

    data = cached_parse(pdb_id)
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AtomizeByCCDName(
                atomize_by_default=True,
                res_names_to_ignore=STANDARD_AA,
            ),
            CropSpatialLikeAF3(crop_size=100, seed=42),
            AnnotateSecondaryStructure(),  # Uses environment variable or default
        ]
    )

    data = pipe(data)
    atom_array = data["atom_array"]
    check_dssp_annotations(atom_array)


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=INFO", __file__])
