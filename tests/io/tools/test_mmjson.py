
from atomworks.io.parser import parse
from atomworks.io.utils.io_utils import infer_pdb_file_type
from atomworks.io.utils.testing import assert_same_atom_array

from tests.conftest import TEST_DATA_DIR


def test_mmjson_inference_and_parsing():
    json_path = TEST_DATA_DIR / "io" / "2hhb.json.gz"
    cif_path = TEST_DATA_DIR / "io" / "2hhb.cif.gz"

    assert json_path.exists(), f"mmJSON file not found at {json_path}"
    assert cif_path.exists(), f"CIF file not found at {cif_path}"

    # 1. Test File Type Inference
    inferred_type = infer_pdb_file_type(json_path)
    assert inferred_type == "mmjson", f"Failed to infer 'mmjson'. Got: {inferred_type}"

    # 2. Parse mmJSON
    result_json = parse(json_path, file_type="mmjson")
    atoms_json = result_json["asym_unit"]

    # 3. Parse CIF for Comparison
    result_cif = parse(cif_path, file_type="cif")
    atoms_cif = result_cif["asym_unit"]

    # 4. Compare Results
    # This utility checks atom count, coordinates, and other annotations
    assert_same_atom_array(atoms_json, atoms_cif)
