"""
Tests for chain type assignment and annotation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from atomworks.enums import ChainType
from atomworks.io.parser import parse
from tests.io.conftest import CHAIN_TYPE_TEST_CASES, get_pdb_path

# General Enum tests


def test_chain_type_to_int():
    assert ChainType.DNA.value == 3


def test_chain_type_from_int():
    assert ChainType(3) == ChainType.DNA


def test_chain_type_from_string():
    assert ChainType.as_enum("polydeoxyribonucleotide") == ChainType.DNA
    assert ChainType.as_enum("POLYDEOXYRIBONUCLEOTIDE") == ChainType.DNA
    assert ChainType.as_enum("PolyDeoxyRibonucleotide") == ChainType.DNA
    with pytest.raises(ValueError):
        ChainType.as_enum("invalid_chain_type")


def test_chain_type_get_chain_type_strings():
    assert "POLYDEOXYRIBONUCLEOTIDE" in ChainType.get_chain_type_strings()


def test_chain_type_equality():
    assert ChainType(3) == ChainType.DNA
    assert ChainType.DNA == 3
    assert ChainType.DNA == "POLYDEOXYRIBONUCLEOTIDE"


@pytest.mark.parametrize("test_case", CHAIN_TYPE_TEST_CASES)
def test_chain_types(test_case: dict[str, Any]):
    path = get_pdb_path(test_case["pdb_id"])
    result = parse(
        filename=path,
        build_assembly="all",
    )

    atom_array = result["assemblies"]["1"][0]  # Choose first model
    for pn_unit_id in np.unique(atom_array.pn_unit_id):
        pn_unit_atom_array = atom_array[atom_array.pn_unit_id == pn_unit_id]
        # ...check if all chains in a PN unit have the same type
        assert np.unique(pn_unit_atom_array.chain_type).size == 1

        # ...check that the type matches the expected type for chains that we care about
        if pn_unit_id.astype(str) in test_case["chain_types"]:
            # Check ChainType
            got_chain_type = ChainType.as_enum(pn_unit_atom_array.chain_type[0])
            expected_chain_type = test_case["chain_types"][pn_unit_id]
            assert (
                got_chain_type == expected_chain_type
            ), f"Mismatch for {pn_unit_id=}: {got_chain_type=}, {expected_chain_type=}"

            # Check is_polymer
            got_is_polymer = pn_unit_atom_array.is_polymer[0]
            expected_is_polymer = expected_chain_type.is_polymer()
            assert (
                got_is_polymer == expected_is_polymer
            ), f"Mismatch for {pn_unit_id=}: {got_is_polymer=}, {expected_is_polymer=}"
