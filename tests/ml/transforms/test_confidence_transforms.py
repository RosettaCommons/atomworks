from typing import Any

import pytest
import torch

from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING, AF3SequenceEncoding
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
)
from atomworks.ml.transforms.atom_frames import (
    AddIsRealAtom,
    AddPolymerFrameIndices,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import (
    Compose,
    ConvertToTorch,
)
from atomworks.ml.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from atomworks.ml.transforms.encoding import EncodeAF3TokenLevelFeatures, EncodeAtomArray
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveNucleicAcidTerminalOxygen, RemoveTerminalOxygen
from atomworks.ml.utils.testing import cached_parse

CONFIDENCE_MODIFICATION_TEST_CASES = [
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

EXPECTED_OUTPUT_SHAPES: dict[str, Any] = {
    "4js1": {
        "is_real_atom": (453, 36),
        "pae_frame_idx_token_lvl_from_atom_lvl": (453, 3),
        "alignment_mask_atm_lvl": (2723,),
    }
}

TERMINAL_OXYGEN_TEST_CASES = [
    {
        "pdb_id": "4z3c",
        "expected_terminal_oxygen_idx": torch.tensor([0, 247]),
    }
]


@pytest.mark.parametrize("test_case", CONFIDENCE_MODIFICATION_TEST_CASES)
def test_add_is_real_atom(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]

    data = cached_parse(test_case["pdb_id"])

    # Apply base transforms
    af3_sequence_encoding = AF3SequenceEncoding()
    base_pipeline = Compose(
        [
            # Base pipeline
            RemoveHydrogens(),
            FlagAndReassignCovalentModifications(),
            AddGlobalAtomIdAnnotation(),
            AtomizeByCCDName(
                atomize_by_default=True,
                res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
                move_atomized_part_to_end=False,
                validate_atomize=False,
            ),
            RemoveTerminalOxygen(),
            RemoveNucleicAcidTerminalOxygen(),
            AddWithinChainInstanceResIdx(),
            AddWithinPolyResIdxAnnotation(),
            AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
            EncodeAF3TokenLevelFeatures(sequence_encoding=af3_sequence_encoding),
            ComputeAtomToTokenMap(),
            ConvertToTorch(
                keys=[
                    "feats",
                ]
            ),
            # Additions required for confidence calculation
            EncodeAtomArray(RF2AA_ATOM36_ENCODING),
        ]
    )

    prepared_data = base_pipeline(data)

    add_is_real_atom_pipeline = Compose(
        [
            AddIsRealAtom(RF2AA_ATOM36_ENCODING),
        ]
    )

    confidence_data = add_is_real_atom_pipeline(prepared_data)

    assert confidence_data["is_real_atom"].sum() == len(
        confidence_data["atom_array"]
    ), "is_real_atom must account for all atoms in the atom array"
    assert (
        confidence_data["is_real_atom"].shape == EXPECTED_OUTPUT_SHAPES[pdb_id]["is_real_atom"]
    ), "is_real_atom shape should be [n_residues, 36]"


@pytest.mark.parametrize("test_case", CONFIDENCE_MODIFICATION_TEST_CASES)
def test_add_frame_indices(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]

    data = cached_parse(test_case["pdb_id"])

    # Apply base transforms
    af3_sequence_encoding = AF3SequenceEncoding()
    base_pipeline = Compose(
        [
            # Base pipeline
            RemoveHydrogens(),
            FlagAndReassignCovalentModifications(),
            AddGlobalAtomIdAnnotation(),
            AtomizeByCCDName(
                atomize_by_default=True,
                res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
                move_atomized_part_to_end=False,
                validate_atomize=False,
            ),
            AddWithinChainInstanceResIdx(),
            AddWithinPolyResIdxAnnotation(),
            AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
            EncodeAF3TokenLevelFeatures(sequence_encoding=af3_sequence_encoding),
            ComputeAtomToTokenMap(),
            ConvertToTorch(
                keys=[
                    "feats",
                ]
            ),
            # Additions required for confidence calculation
            EncodeAtomArray(RF2AA_ATOM36_ENCODING),
        ]
    )

    prepared_data = base_pipeline(data)

    add_frame_indices_pipeline = Compose(
        [
            AddPolymerFrameIndices(),
        ]
    )

    confidence_data = add_frame_indices_pipeline(prepared_data)

    assert (
        confidence_data["pae_frame_idx_token_lvl_from_atom_lvl"].shape
        == EXPECTED_OUTPUT_SHAPES[pdb_id]["pae_frame_idx_token_lvl_from_atom_lvl"]
    ), "frame_atom_idxs shape should be [n_residues, 3]"


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
