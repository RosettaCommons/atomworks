import pytest

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

TEST_CASES = [
    "6xa4",
    "1qh9",
    "6qhw",
    "5vo3",
    "5t4j",
    "3t44",
    "6t4v",
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_atom_order(pdb_id: str):
    path = get_pdb_path(pdb_id)
    result = parse(path, config=ParseConfig(add_missing_atoms=True, build_assembly=None))
    assert result is not None
