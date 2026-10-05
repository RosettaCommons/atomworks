from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from tests.io.conftest import TEST_DATA_IO, get_pdb_path


def test_arginine_renaming():
    correct_file = get_pdb_path("101m")
    renamed_file = TEST_DATA_IO / "101m_arginine_nh1nh2_swapped.cif"  # Manually renamed the arginine atoms.

    result1 = parse(correct_file, config=ParseConfig(build_assembly=None))
    result2 = parse(renamed_file, config=ParseConfig(build_assembly=None, fix_arginines=True))

    assert_same_atom_array_or_stack(
        result1["asym_unit"][0],
        result2["asym_unit"][0],
        annotations_to_compare=["chain_id", "res_name", "res_id", "atom_name", "element"],
    )
