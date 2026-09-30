import numpy as np
import pytest

from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

TEST_CASES = ["1iau"]
# has multiple NAG with same res number loaded as -1.
# if hydrogen addtion is done too early they are not able to be resolved.


@pytest.mark.parametrize("pdbid", TEST_CASES)
def test_resnum_duplication_resolve(pdbid: str):
    # Not excluding crystallization aids
    out1 = parse(get_pdb_path(pdbid))
    out1 = out1["assemblies"]["1"][0]
    ids = out1[out1.res_name == "NAG"].res_id
    print(ids)
    assert np.all(ids != -1)
    assert len(np.unique(ids)) == 4  # [1,2,4,301]
