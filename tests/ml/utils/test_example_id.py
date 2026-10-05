"""Tests for example_id parsing."""

import pytest

from atomworks.ml.example_id import parse_example_id


@pytest.mark.parametrize(
    "example_id,keys,permissive,expected",
    [
        # Short format
        ("{mgnify}{123}", ("datasets", "pdb_id"), False, {"datasets": "mgnify", "pdb_id": "123"}),
        # Full PDB format with lists
        (
            "{['pdb']}{6vyb}{1}{['A_1']}",
            ("datasets", "pdb_id", "assembly_id", "query_pn_unit_iids"),
            False,
            {"datasets": ["pdb"], "pdb_id": "6vyb", "assembly_id": "1", "query_pn_unit_iids": ["A_1"]},
        ),
        # Permissive with missing keys
        (
            "{mgnify}{123}",
            ("datasets", "pdb_id", "assembly_id"),
            True,
            {"datasets": "mgnify", "pdb_id": "123", "assembly_id": None},
        ),
    ],
)
def test_parse_example_id(example_id, keys, permissive, expected):
    """Test parsing various example ID formats."""
    assert parse_example_id(example_id, keys=keys, permissive=permissive) == expected


def test_parse_example_id_strict_raises():
    """Test that strict mode raises on mismatch."""
    with pytest.raises(ValueError, match="Expected 4 values"):
        parse_example_id("{mgnify}{123}")
