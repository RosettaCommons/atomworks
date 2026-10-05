import biotite.structure as struc
import numpy as np
import pytest

from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from atomworks.io.utils.io_utils import load_any
from atomworks.ml.encoding_definitions import (
    AF2_ATOM14_ENCODING,
    AF2_ATOM37_ENCODING,
    RF2_ATOM14_ENCODING,
    RF2_ATOM23_ENCODING,
    RF2_ATOM36_ENCODING,
    RF2AA_ATOM36_ENCODING,
    UNIFIED_ATOM37_ENCODING,
    TokenEncoding,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose, Identity
from atomworks.ml.transforms.encoding import (
    AddTokenAnnotation,
    EncodeAtomArray,
    atom_array_to_encoded_resnames,
    atom_array_to_encoding,
    get_token_count,
)
from atomworks.ml.transforms.filters import (
    FilterToProteins,
    RemoveHydrogens,
    RemoveTerminalOxygen,
)
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import token_iter
from tests.conftest import TEST_DATA_DIR


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


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("atom_mode", ["element", "atomic_number", "literal"])
def test_resname_encoding_at_token_starts(override: bool, atom_mode: str):
    atoms = struc.AtomArray(6)
    atoms.res_name = np.array(["ALA", "ALA", "DA", "U", "UNL", "UNL"])
    atoms.element = np.array(["C", "N", "C", "C", "c", "n"])
    atoms.set_annotation("token_id", np.array([0, 0, 1, 2, 3, 4]))
    atoms.set_annotation("atomize", np.array([False] * 4 + [True] * 2))
    if atom_mode == "atomic_number":
        atoms.set_annotation("atomic_number", np.array([6, 7, 6, 6, 8, 26]))
    encoding = TokenEncoding(
        token_atoms={name: [str(name)] for name in ("ALA", "DA", "U", "<M>", "UNK", "<A>", 6, 7, 8, 26)},
        chemcomp_type_to_unknown={"L-PEPTIDE LINKING": "UNK"},
    )
    names = np.array(["MSE", "ALA", "<M>", "DA", "<M>", "<M>"]) if override else None
    atom_tokens = {"element": [6, 7], "atomic_number": [8, 26], "literal": ["<A>", "<A>"]}[atom_mode]
    expected_names = (["UNK", "UNK", "<M>", "DA"] if override else ["ALA", "ALA", "DA", "U"]) + atom_tokens
    expected = np.array([encoding.token_to_idx[name] for name in expected_names], dtype=int)
    actual = atom_array_to_encoded_resnames(
        atoms, encoding, atomize_token="<A>" if atom_mode == "literal" else None, res_names=names
    )
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_atomize_token_encodes_atomized_tokens():
    atom_array = load_any(str(TEST_DATA_DIR / "io" / "1qfe.pdb"), model=1)

    assert hasattr(atom_array, "hetero") and atom_array.hetero is not None
    assert np.any(atom_array.hetero), "Expected ligand atoms in 1qfe.pdb"
    atom_array.set_annotation("atomize", atom_array.hetero.copy())

    encoding = UNIFIED_ATOM37_ENCODING
    encoded = atom_array_to_encoding(
        atom_array,
        encoding,
        atomize_token="<A>",
        atomize_atom_name="X",
    )

    atomize_idx = encoding.token_to_idx["<A>"]
    saw_atomized = False
    saw_non_atomized = False
    for i, token in enumerate(token_iter(atom_array)):
        if token.atomize[0]:
            saw_atomized = True
            assert encoded["seq"][i] == atomize_idx
        else:
            saw_non_atomized = True
            assert encoded["seq"][i] != atomize_idx

    assert saw_atomized and saw_non_atomized


def test_atomize_token_none_uses_atomic_number_tokens():
    atom_array = load_any(str(TEST_DATA_DIR / "io" / "test_unl_ligand_with_bonds.cif"), model=1)
    atom_array.set_annotation("atomize", np.ones(atom_array.array_length(), dtype=bool))

    atomic_numbers = np.unique(
        atom_array.atomic_number
        if "atomic_number" in atom_array.get_annotation_categories()
        else np.array([ELEMENT_NAME_TO_ATOMIC_NUMBER[e.upper()] for e in atom_array.element])
    )
    token_atoms = {int(n): [str(int(n))] for n in atomic_numbers}
    token_atoms.setdefault(0, ["0"])
    encoding = TokenEncoding(token_atoms=token_atoms)

    encoded = atom_array_to_encoding(atom_array, encoding, atomize_token=None)

    for i, token in enumerate(token_iter(atom_array)):
        token_name = (
            token.atomic_number[0]
            if "atomic_number" in token.get_annotation_categories()
            else ELEMENT_NAME_TO_ATOMIC_NUMBER[token.element[0].upper()]
        )
        assert encoded["seq"][i] == encoding.token_to_idx[token_name]


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
