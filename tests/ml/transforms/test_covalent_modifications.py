from typing import Any

import numpy as np
import pytest

from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.covalent_modifications import AnnotateCovalentModifications
from atomworks.ml.utils.testing import cached_parse

COVALENT_MODIFICATION_TEST_CASES = [
    {
        # 4js1: A_1 61 (protein residue) is covalently bound to B_1 (multi-chain sugar)
        "pdb_id": "4js1",
        "residues_to_be_atomized": [
            {
                "polymer_pn_unit_iid": "A_1",
                "polymer_res_id": 61,
                "non_polymer_pn_unit_iid": "B_1",
                "non_polymer_pn_unit_id": "B",
            }
        ],
    },
]


@pytest.mark.parametrize("test_case", COVALENT_MODIFICATION_TEST_CASES)
def test_covalent_modifications(test_case: dict[str, Any]):
    data = cached_parse(test_case["pdb_id"])

    # Apply transforms
    unprocessed_atom_array = data["atom_array"]
    covalent_modification_pipeline = Compose(
        [
            AnnotateCovalentModifications(),
        ]
    )
    result_with_covalent_modifications = covalent_modification_pipeline(data)

    processed_atom_array = result_with_covalent_modifications["atom_array"]
    # For all residues bound to non-polymers...
    for residue_dict in test_case["residues_to_be_atomized"]:
        residue_atom_mask = (processed_atom_array.pn_unit_iid == residue_dict["non_polymer_pn_unit_iid"]) & (
            processed_atom_array.res_id == residue_dict["polymer_res_id"]
        )
        residue_atom_array = processed_atom_array[residue_atom_mask]

        assert len(residue_atom_array) > 0

        # ... ensure that we set atomize = True for all atoms in the residue
        assert np.all(residue_atom_array.atomize)

        # ... ensure that we set atomize = True for all atoms in the non-polymer residue
        non_polymer_residue_atom_mask = processed_atom_array.pn_unit_iid == residue_dict["non_polymer_pn_unit_iid"]
        non_polymer_residue_atom_array = processed_atom_array[non_polymer_residue_atom_mask]
        assert np.all(non_polymer_residue_atom_array.atomize)

        # ... ensure that we set pn_unit_id and pn_unit_iid to that of the non-polymer PN unit
        assert np.all(
            residue_atom_array.pn_unit_iid
            == np.array([residue_dict["non_polymer_pn_unit_iid"]] * len(residue_atom_array))
        )
        assert np.all(
            residue_atom_array.pn_unit_id
            == np.array([residue_dict["non_polymer_pn_unit_id"]] * len(residue_atom_array))
        )

        # ... check that for all other residues, atomize was unchanged
        combined_residue_mask = residue_atom_mask | non_polymer_residue_atom_mask
        other_residues_processed = processed_atom_array[~combined_residue_mask]
        other_residues_unprocessed = unprocessed_atom_array[~combined_residue_mask]
        assert np.all(other_residues_processed.atomize == other_residues_unprocessed.atomize)


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
