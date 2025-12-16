import biotite.structure as struc
import numpy as np
import pytest

from atomworks.ml.encoding_definitions import (
    AF2_ATOM14_ENCODING,
    AF2_ATOM37_ENCODING,
    RF2_ATOM14_ENCODING,
    RF2_ATOM23_ENCODING,
    RF2_ATOM36_ENCODING,
    RF2AA_ATOM36_ENCODING,
    TokenEncoding,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose, Identity
from atomworks.ml.transforms.encoding import AddTokenAnnotation, EncodeAtomArray, get_token_count
from atomworks.ml.transforms.filters import (
    FilterToProteins,
    RemoveHydrogens,
    RemoveTerminalOxygen,
)
from atomworks.ml.utils.testing import cached_parse


@pytest.mark.parametrize("pdb_id", ["5ocm", "5ocn"])
def test_encoding_af2_atom37_encoding(pdb_id: str):
    data = cached_parse(pdb_id)

    encoding = AF2_ATOM37_ENCODING
    pipe = Compose(
        [
            RemoveHydrogens(),
            FilterToProteins(min_size=3),  # AF2 can only handle `protein-like` amino acids
            AddTokenAnnotation(encoding),
            EncodeAtomArray(encoding),
        ]
    )

    data = pipe(data)
    n_token_seq = len(data["encoded"]["seq"])
    n_token_struc = len(data["encoded"]["xyz"])
    n_token_mask = len(data["encoded"]["mask"])
    n_token_array = get_token_count(data["atom_array"])
    n_res = struc.get_residue_count(data["atom_array"])

    assert n_token_seq == n_token_struc, f"n_token_seq={n_token_seq} != n_token_struc={n_token_struc}"
    assert n_token_seq == n_token_mask, f"n_token_seq={n_token_seq} != n_token_mask={n_token_mask}"
    assert n_token_seq == n_token_array, f"n_token_seq={n_token_seq} != n_token_array={n_token_array}"
    assert (
        n_res == n_token_array
    ), f"n_res={n_res} != n_token_array={n_token_array} -- this should be case when not atomizing"


@pytest.mark.parametrize("pdb_id", ["5ocm", "5ocn"])
@pytest.mark.parametrize(
    "encoding",
    [AF2_ATOM14_ENCODING, RF2_ATOM14_ENCODING, RF2_ATOM23_ENCODING, RF2_ATOM36_ENCODING, RF2AA_ATOM36_ENCODING],
)
def test_encoding_atom14_proteins_only(pdb_id: str, encoding: TokenEncoding):
    data = cached_parse(
        pdb_id,
        convert_mse_to_met=True,
        remove_waters=True,
        build_assembly="first",
    )

    pipe = Compose(
        [
            RemoveHydrogens(),
            RemoveTerminalOxygen(),  # Atom14 does not encode terminal oxygen
            FilterToProteins(min_size=3),  # AF2/RF2 can only handle `protein-like` amino acids
            AddTokenAnnotation(encoding),
            EncodeAtomArray(encoding),
        ]
    )

    data = pipe(data)
    n_token_seq = len(data["encoded"]["seq"])
    n_token_struc = len(data["encoded"]["xyz"])
    n_token_mask = len(data["encoded"]["mask"])
    n_token_array = get_token_count(data["atom_array"])
    n_res = struc.get_residue_count(data["atom_array"])

    assert n_token_seq == n_token_struc, f"n_token_seq={n_token_seq} != n_token_struc={n_token_struc}"
    assert n_token_seq == n_token_mask, f"n_token_seq={n_token_seq} != n_token_mask={n_token_mask}"
    assert n_token_seq == n_token_array, f"n_token_seq={n_token_seq} != n_token_array={n_token_array}"
    assert (
        n_res == n_token_array
    ), f"n_res={n_res} != n_token_array={n_token_array} -- this should be case when not atomizing"


@pytest.mark.parametrize("pdb_id", ["5ocm"])
@pytest.mark.parametrize("encode_hydrogens", [False, True])
def test_all_atom_encoding(
    pdb_id: str,
    encode_hydrogens: bool,
    encoding: TokenEncoding = RF2AA_ATOM36_ENCODING,
):
    data = cached_parse(
        pdb_id,
        convert_mse_to_met=True,
        remove_waters=True,
        build_assembly="first",
    )

    pipe = Compose(
        [
            Identity() if encode_hydrogens else RemoveHydrogens(),
            RemoveTerminalOxygen(),  # RF2AA does not encode terminal oxygen for AA residues.
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=encoding.tokens),
            AddTokenAnnotation(encoding),
            EncodeAtomArray(encoding),
        ]
    )

    data = pipe(data)
    n_token_seq = len(data["encoded"]["seq"])
    n_token_struc = len(data["encoded"]["xyz"])
    n_token_mask = len(data["encoded"]["mask"])
    n_token_array = get_token_count(data["atom_array"])
    n_res = struc.get_residue_count(data["atom_array"])

    assert n_token_seq == n_token_struc, f"n_token_seq={n_token_seq} != n_token_struc={n_token_struc}"
    assert n_token_seq == n_token_mask, f"n_token_seq={n_token_seq} != n_token_mask={n_token_mask}"
    assert n_token_seq == n_token_array, f"n_token_seq={n_token_seq} != n_token_array={n_token_array}"
    assert (
        n_res < n_token_array
    ), f"n_res={n_res} > n_token_array={n_token_array} -- there should be more tokens than residues when atomizing"


MOLECULE_TEST_CASES = [
    {
        "pdb_id": "1ivo",
        "num_molecules": 4,
        "chain_iid_combinations": [
            ["A_1", "E_1", "F_1", "G_1", "H_1", "I_1", "J_1"],
            ["B_1", "K_1", "L_1", "M_1"],
            ["C_1"],
            ["D_1"],
        ],
    },
    {
        "pdb_id": "4js1",
        "num_molecules": 2,
        "chain_iid_combinations": [
            ["A_1", "B_1"],
            ["C_1"],
        ],
    },
]


@pytest.mark.parametrize("test_case", MOLECULE_TEST_CASES)
def test_extra_annotations(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)

    encoding = RF2AA_ATOM36_ENCODING
    pipe = Compose(
        [
            RemoveHydrogens(),
            RemoveTerminalOxygen(),  # RF2AA does not encode terminal oxygen for AA residues.
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=encoding.tokens),
            AddTokenAnnotation(encoding),
            EncodeAtomArray(encoding),
        ]
    )

    data = pipe(data)
    atom_array = data["atom_array"]

    n_token = len(data["encoded"]["seq"])

    # Check `chain_id` annotations
    assert "chain_id" in data["encoded"], "chain_id not in encoded"
    assert (
        len(data["encoded"]["chain_id"]) == n_token
    ), f"chain_id length={len(data['encoded']['chain_id'])} != n_token={n_token}"

    # Check `molecule_iid` annotations
    assert "molecule_iid" in data["encoded"], "molecule_iid not in encoded"
    assert (
        len(data["encoded"]["molecule_iid"]) == n_token
    ), f"molecule_iid length={len(data['encoded']['molecule_iid'])} != n_token={n_token}"
    assert (
        len(data["encoded"]["molecule_iid_to_int"]) == test_case["num_molecules"]
    ), f"molecule_iid_to_int length={len(data['encoded']['molecule_iid_to_int'])} != num_molecules={test_case['num_molecules']}"
    assert np.all(
        sorted(np.unique(data["encoded"]["molecule_iid"])) == np.arange(test_case["num_molecules"])
    ), f"molecule_iid unique values={np.unique(data['encoded']['molecule_iid'])} != num_molecules={test_case['num_molecules']}"

    for molecule_iid, molecule_iidx in data["encoded"]["molecule_iid_to_int"].items():
        raw_coords = atom_array[(atom_array.occupancy > 0) & (atom_array.molecule_iid == molecule_iid)].coord
        encoded_coords = data["encoded"]["xyz"][
            (data["encoded"]["molecule_iid"] == molecule_iidx).reshape(-1, 1) & (data["encoded"]["mask"])
        ]
        assert (
            raw_coords.shape == encoded_coords.shape
        ), f"raw_coords shape={raw_coords.shape} != encoded_coords shape={encoded_coords.shape}"


# TODO:
# - Test encoding-decoding roundtrip (known tokens only)
# - Test encoding-decoding roundtrip (known tokens + atomized unknowns)
# - Test encoding-decoding roundtrip (known tokens + atomized unknowns + unknowns)

if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
