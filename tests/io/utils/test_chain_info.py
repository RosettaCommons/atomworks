"""Tests for chain_info utilities."""

import biotite.structure as struc
import numpy as np
import pytest

from atomworks.enums import ChainType
from atomworks.io.parser import parse
from atomworks.io.utils.chain_info import build_chain_info
from tests.io.conftest import CHAIN_TYPE_TEST_CASES, get_pdb_path


def _single_atom_residues(res_names: list[str], *, hetero: bool) -> struc.AtomArray:
    array = struc.AtomArray(len(res_names))
    array.chain_id[:] = "A"
    array.res_id = np.arange(1, len(res_names) + 1)
    array.res_name = res_names
    array.atom_name[:] = "CA"
    array.element[:] = "C"
    array.hetero[:] = hetero
    array.set_annotation("entity_id", np.full(len(array), "1"))
    return array


@pytest.mark.parametrize("hetero", [False, True])
def test_generic_entity_polymer_preserves_inferred_polymer_chemistry(hetero):
    atom_array = _single_atom_residues(["ALA", "GLY"], hetero=hetero)

    info = build_chain_info(atom_array, entity={"id": np.array(["1"]), "type": np.array(["polymer"])})["A"]

    assert info["chain_type"] == ChainType.POLYPEPTIDE_L
    assert info["is_polymer"] is True
    assert info["processed_entity_canonical_sequence"] == "AG"


def test_generic_entity_polymer_promotes_unknown_chemistry_to_other_polymer():
    atom_array = _single_atom_residues(["UNL"], hetero=True)

    info = build_chain_info(atom_array, entity={"id": np.array(["1"]), "type": np.array([" POLYMER "])})["A"]

    assert info["chain_type"] == ChainType.OTHER_POLYMER
    assert info["is_polymer"] is True


def test_entity_metadata_matches_string_identifiers():
    atom_array = _single_atom_residues(["ALA", "GLY"], hetero=False)
    atom_array.set_annotation("entity_id", np.full(len(atom_array), "001"))

    info = build_chain_info(
        atom_array,
        entity={"id": np.array(["001"]), "type": np.array(["polymer"])},
        entity_poly_seq={
            "entity_id": np.array(["001", "001", "001"]),
            "num": np.array([1, 2, 3]),
            "mon_id": np.array(["ALA", "GLY", "SER"]),
        },
    )["A"]

    assert info["label_entity"] == "001"
    assert info["processed_entity_canonical_sequence"] == "AGS"


def test_unresolved_poly_seq_residues_get_an_empty_insertion_code():
    atom_array = _single_atom_residues(["ALA", "GLY"], hetero=False)
    atom_array.res_id = np.array([2, 3])
    atom_array.ins_code = np.array(["A", "B"])
    poly_seq = {
        "entity_id": np.array(["1"] * 4),
        "num": np.array([1, 2, 3, 4]),
        "mon_id": np.array(["MET", "ALA", "GLY", "SER"]),
    }

    info = build_chain_info(atom_array, entity_poly_seq=poly_seq)["A"]

    assert info["res_id"] == [1, 2, 3, 4]
    assert info["ins_code"] == ["", "A", "B", ""]
    assert info["res_name"] == ["MET", "ALA", "GLY", "SER"]


@pytest.mark.parametrize("test_case", CHAIN_TYPE_TEST_CASES)
def test_infer_chain_info_from_atom_array(test_case: dict):
    cif_path = get_pdb_path(test_case["pdb_id"])
    atom_array = parse(
        filename=cif_path,
        add_missing_atoms=False,
        remove_waters=True,
    )["asym_unit"][0]

    chain_info = build_chain_info(atom_array)

    for chain_id, info_dict in chain_info.items():
        got = info_dict["chain_type"]
        expected = ChainType.as_enum(test_case["chain_types"][chain_id])

        if got.is_non_polymer():
            # We allow all non-polymers to be interchanged
            assert expected.is_non_polymer()
        else:
            # Enforce strict equality for polymers
            assert got == expected


if __name__ == "__main__":
    pytest.main([__file__])
