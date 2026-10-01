"""Pytest function to test the detection and assignment of contacting PN units."""

from typing import Any

import pytest

from atomworks.ml.preprocessing.preprocess import preprocess
from atomworks.ml.utils.testing import get_pdb_mirror_path
from tests.ml.preprocessing.conftest import TEST_CONFIG

FIND_CONTACTS_TEST_CASES = [
    # Defined with contact_distance = 5
    {
        # Simple protein complex
        "pdb_id": "1fu2",
        "contact_information": [
            {
                "assembly_id": "1",
                "pn_unit_iid": "A_1",
                "num_contacting_pn_units": 2,
                "num_contacts": 461,  # 458 for (A,B) and 3 for (A, D)
            }
        ],
    },
    {
        # RNA complex
        "pdb_id": "4gxy",
        "contact_information": [
            {
                "assembly_id": "1",
                "pn_unit_iid": "C_1",
                "num_contacting_pn_units": 1,
                "num_contacts": 94,  # 94 for (C, A)
            }
        ],
    },
]


@pytest.mark.parametrize("test_case", FIND_CONTACTS_TEST_CASES)
def test_find_contacts(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]
    path = get_pdb_mirror_path(pdb_id)

    _, pn_units, _ = preprocess(path, TEST_CONFIG)

    for example in test_case["contact_information"]:
        assembly_id = example["assembly_id"]
        target_iid = example["pn_unit_iid"]

        # Find the PN unit of interest
        pn_unit = next(
            (p for p in pn_units if p.assembly_id == assembly_id and p.pn_unit_iid == target_iid),
            None,
        )
        assert pn_unit is not None, f"PN unit {target_iid} not found"

        # Check contacting PN units count
        assert len(pn_unit.contacting_pn_unit_iids) == example["num_contacting_pn_units"]

        # Check total contact count
        total_contacts = sum(c["num_contacts"] for c in pn_unit.contacting_pn_unit_iids)
        assert (
            total_contacts == example["num_contacts"]
        ), f"Expected {example['num_contacts']} contacts, got {total_contacts}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
