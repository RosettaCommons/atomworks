from typing import Any

import numpy as np
import pytest

from atomworks.io.parser import parse
from atomworks.io.utils.ccd import add_annotations_from_ccd
from tests.io.conftest import get_pdb_path

TEST_CASES = [{"pdb_id": "1jj8", "charge_sum": 7}, {"pdb_id": "2r5z", "charge_sum": 32}]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_add_charge_from_ccd(test_case: dict[str, Any]):
    path = get_pdb_path(test_case["pdb_id"])

    result = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="remove",
    )

    atom_array = result["assemblies"]["1"][0]  # First bioassembly, first model
    has_resolved_coordinates = ~np.isnan(atom_array.coord).any(axis=-1)
    non_nan_array = atom_array[has_resolved_coordinates]

    # parse() should already have applied CCD charges via its annotation pipeline.
    assert np.sum(non_nan_array.charge) == test_case["charge_sum"]

    # add_annotations_from_ccd should reproduce the same sum from scratch.
    non_nan_array.del_annotation("charge")

    # check that we can add charge back in and get the same sum
    non_nan_array = add_annotations_from_ccd(non_nan_array, annotations=["charge"])
    assert np.sum(non_nan_array.charge) == test_case["charge_sum"]
