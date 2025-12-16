"""
PyTest function to test filtering criteria.
Tested criteria includes:
- The detection and resolutions of clashes within a structure
- The exclusion of non-polymers bonded to a polymer via a non-biological bond
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from atomworks.ml.utils.testing import get_pdb_mirror_path
from tests.ml.preprocessing.conftest import DATA_PREPROCESSOR

FILTERING_CRITERIA_TEST_CASES = [
    # Clashing test cases
    {"pdb_id": "6qhp", "pn_units_to_keep": ["A_1"], "pn_units_to_remove": ["C_1"]},
    {
        "pdb_id": "3voz",  # NOTE: In this example, the pn_units are bonded via a sulfer bridge, but still clash since we superimpose across the axis of symmetry
        "pn_units_to_keep": ["B_1", "B_3"],
        "pn_units_to_remove": ["B_2", "B_4"],
    },
    {
        "pdb_id": "2voy",
        "pn_units_to_keep": ["D_1", "J_1", "F_1", "C_1", "I_1"],
        "pn_units_to_remove": ["H_1", "B_1", "A_1"],
    },
    # Non-biological bond test cases
    {"pdb_id": "1pak", "pn_units_to_keep": ["A_1"], "pn_units_to_remove": ["B_1"]},
    # 2pnf - non-biological bonds
]


@pytest.mark.parametrize("test_case", FILTERING_CRITERIA_TEST_CASES)
def test_filtering_criteria(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]
    path = get_pdb_mirror_path(pdb_id)

    rows = DATA_PREPROCESSOR.get_rows(path)
    df = pd.DataFrame(rows)
    pn_unit_iids = eval(df.iloc[0]["all_pn_unit_iids_after_processing"])

    assert set(pn_unit_iids) >= set(df["q_pn_unit_iid"].unique().tolist())

    # ...assert that all of the rows have the same PN units
    assert (
        df["all_pn_unit_iids_after_processing"] == df["all_pn_unit_iids_after_processing"].iloc[0]
    ).all(), "Not all rows have the same pn_units"

    pn_units_to_keep = set(test_case["pn_units_to_keep"])
    pn_units_to_remove = set(test_case["pn_units_to_remove"])

    # Assert that we are keeping the correct PN units
    assert pn_units_to_keep.issubset(pn_unit_iids), f"Missing PN unit to keep in {pdb_id}."

    # Assert we are removing the correct PN units
    assert not pn_units_to_remove.intersection(pn_unit_iids), f"Removing PN unit that should be kept in {pdb_id}."


if __name__ == "__main__":
    pytest.main(["-v", __file__])
