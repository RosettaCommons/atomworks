import logging
from typing import Any

import numpy as np
import pytest

from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.covalent_modifications import AnnotateCovalentModifications
from atomworks.ml.transforms.crop import (
    CropSpatialLikeAF3,
)
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveTerminalOxygen
from atomworks.ml.transforms.masks import AddSpatialKNNMask, compute_spatial_knn_mask
from atomworks.ml.utils.testing import cached_parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_CASES = [
    {
        "pdb_id": "3p42",
    },
]


def test_compute_spatial_knn_mask():
    # fmt: off
    coords = np.array([
        [np.nan, np.nan, np.nan], # 0
        [0.0, 0.0, 0.0], # 1
        [1.0, 1.0, 1.0], # 2
        [2.0, 2.0, 2.0], # 3
        [3.0, 3.1, 3.0], # 4
        [4.0, 4.0, 4.0], # 5
        [np.inf, 5.0, 5.0], # 6
        [6.0, 6.0, 6.0], # 7
        [7.0, 7.0, np.nan], # 8
    ])
    # fmt: on
    # Test warning is raised when atoms have no coordinates
    with pytest.warns(UserWarning, match="Some atoms have no coordinates"):
        mask = compute_spatial_knn_mask(coords, 2)
    assert mask.shape == (9, 9)
    # check that the rows with no coordinates have all False values
    assert mask[0].sum() == 0, "Row 0 should have no neighbors"
    assert mask[6].sum() == 0, "Row 6 should have no neighbors"
    assert mask[8].sum() == 0, "Row 8 should have no neighbors"
    # check that all the other rows have exctly 2 neighbors
    assert np.all(mask[1:6].sum(axis=1) == 2), "Rows 1-5 should have 2 neighbors"
    assert np.all(mask[7:8].sum(axis=1) == 2), "Rows 7-8 should have 2 neighbors"
    # spot-check that the neighbors are correct
    assert all(np.where(mask[1])[0] == np.array([2, 3])), "Row 1 should have neighbors 2 and 3"
    # spot-check that the neighbors are correct
    assert all(np.where(mask[7])[0] == np.array([4, 5])), "Row 7 should have neighbors 4 and 5"
    assert all(np.where(mask[5])[0] == np.array([3, 4])), "Row 5 should have neighbors 3 and 4"


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_spatial_knn_mask(test_case: dict[str, Any]):
    """
    test the AddSpatialKNNMask class to ensure we bave a "spatial_knn_masks" key in the output data dictionary of correct shape
    """
    pdb_id = test_case["pdb_id"]

    # Ensure PDB ID is lowercase in the dataframe
    data = cached_parse(pdb_id)

    # Apply the transform
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            RemoveHydrogens(),
            RemoveTerminalOxygen(),
            AnnotateCovalentModifications(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
        ]
    )

    output = pipe(data)
    output = AddSpatialKNNMask(num_neighbors=4)(output)

    mask = output["spatial_knn_masks"]

    # Check the key is in the output dictionary
    assert "spatial_knn_masks" in output
    # check have correct neighbors
    assert (mask[100].nonzero() == np.array([96, 97, 101, 102])).all()
    # makse[0] nonzero all smaller than 20
    assert (mask[0].nonzero()[0] < 20).all()

    # Check if Croping is compatible with the AddSpatialKNNMask
    pipe_with_crop = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            RemoveHydrogens(),
            RemoveTerminalOxygen(),
            AnnotateCovalentModifications(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
            CropSpatialLikeAF3(crop_size=128),
        ]
    )

    # Ensure PDB ID is lowercase in the dataframe
    data = cached_parse(pdb_id)
    output = pipe_with_crop(data)
    output = AddSpatialKNNMask(num_neighbors=4)(output)

    # Check the shape of the mask
    mask = output["spatial_knn_masks"]
    assert mask.shape == (output["atom_array"].array_length(), output["atom_array"].array_length())

    # Check each row has 4 neighbors
    is_finite = np.isfinite(output["atom_array"].coord).all(axis=1)
    if not np.all(mask[is_finite].sum(axis=1) == 4):
        # problematic rows:
        problematic_rows = np.where(mask[is_finite].sum(axis=1) != 4)[0]
        n_neighbors_problematic_rows = mask[is_finite].sum(axis=1)[problematic_rows]
        raise AssertionError(
            "Not all rows have 4 neighbors. Showing up to 5 problematic rows:"
            f"Problematic rows: {problematic_rows[:5]} with {n_neighbors_problematic_rows[:5]} neighbors"
        )
