"""Pytest for LOI - SOI (subject of investigation) extraction"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from atomworks.ml.utils.testing import get_pdb_mirror_path
from tests.conftest import skip_if_no_internet
from tests.ml.preprocessing.conftest import DATA_PREPROCESSOR

LOI_EXTRACTION_TEST_CASES = [
    {
        # TEST CASE 0
        "pdb_id": "7lad",
        "loi": {"HEM", "XRD"},
    },
    {
        # TEST CASE 1
        "pdb_id": "5ocm",
        "loi": set(),
    },
    {
        # TEST CASE 2
        "pdb_id": "6wjc",
        "loi": {"Y01", "OIN"},
    },
    {
        # TEST CASE 3
        "pdb_id": "7mub",
        "loi": {"K"},
    },
    {
        # TEST CASE 4
        "pdb_id": "1qk0",
        "loi": {"IOB", "GLC", "XYS"},
        # NOTE: GLC & XYS are only specified in the cif file, not on
        #  the PDB summary page
        "has_covalently_bonded_loi": True,
    },
]


@skip_if_no_internet
@pytest.mark.parametrize("test_case", LOI_EXTRACTION_TEST_CASES)
def test_loi_extraction(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]
    path = get_pdb_mirror_path(pdb_id)

    # Check that the LOI is extracted correctly from the CIF file
    parsed = DATA_PREPROCESSOR._load_structure_with_atomworks(path)
    loi_set = set(parsed["ligand_info"]["ligand_of_interest"])
    assert loi_set == test_case["loi"]

    # Check that the LOI examples give the correct molecule
    rows = DATA_PREPROCESSOR.get_rows(path)
    df = pd.DataFrame(rows)

    loi_seen = {k: 0 for k in loi_set}
    for _, row in df.iterrows():
        if row.q_pn_unit_is_loi:
            assembly_id = row.assembly_id
            chain_ids = row.q_pn_unit_id.split(",")
            structure = parsed["assemblies"][assembly_id][0]
            res_names = np.unique(
                structure[(np.isin(structure.chain_id, chain_ids)) & (structure.occupancy > 0)].res_name
            )
            if test_case.get("has_covalently_bonded_loi", False):
                assert any(
                    res in loi_set for res in res_names
                ), f"No LOI molecule found for {row.q_pn_unit_iid} in {res_names}. LOIs: {loi_set}"
                for res in res_names:
                    if res in loi_set:
                        loi_seen[res] += 1
            else:
                assert len(res_names) == 1, f"Multiple LOI molecules found for {row.q_pn_unit_iid}: {res_names}"
                assert res_names[0] in loi_set, f"LOI molecule {res_names[0]} not found in {loi_set}"
                loi_seen[res_names[0]] += 1

    # Check that all LOI molecules have been extracted
    assert all(count > 0 for count in loi_seen.values()), f"Some LOI molecules have not been extracted: {loi_seen}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
