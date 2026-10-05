import pytest

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

TEST_CASES = [
    # With the wrong version of biotite, these will lead to cif deserialization errors as the assembly category is represented slightly differently in these files
    "5xa9",
    "5xag",
    "5xaf",
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_deserialize_assembly(pdb_id: str):
    digs_path = get_pdb_path(pdb_id)
    result = parse(digs_path, config=ParseConfig(build_assembly="first"))
    assert result is not None
