import pytest

from atomworks.constants import CCD_MIRROR_PATH
from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

TEST_CASES = [
    {"pdb_id": "6lyz"},  # small
    {"pdb_id": "5ocm"},  # medium
    {"pdb_id": "6tqn"},  # large
]


@pytest.mark.benchmark(
    group="parse-speed",
    cprofile=[
        "ncalls_recursion",
        "ncalls",
        "tottime",
        "tottime_per",
        "cumtime",
        "cumtime_per",
        "function_name",
    ],
)
@pytest.mark.parametrize("case", TEST_CASES)
def test_parse_no_ccd_mirror(case, benchmark):
    # NOTE: Requires pytest-benchmark to be installed
    path = get_pdb_path(case["pdb_id"])
    benchmark(parse, filename=path)


@pytest.mark.benchmark(
    group="parse-speed",
    cprofile=["ncalls_recursion", "ncalls", "tottime", "tottime_per", "cumtime", "cumtime_per", "function_name"],
)
@pytest.mark.parametrize("case", TEST_CASES)
def test_parse_with_ccd_mirror(case, benchmark):
    # NOTE: Requires pytest-benchmark to be installed
    path = get_pdb_path(case["pdb_id"])
    benchmark(parse, filename=path, ccd_mirror_path=CCD_MIRROR_PATH)
