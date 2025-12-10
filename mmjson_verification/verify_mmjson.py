
import sys
from pathlib import Path
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_mmjson")

try:
    from atomworks.io.parser import parse
    from atomworks.io.utils.io_utils import infer_pdb_file_type
except ImportError:
    # Add src to path if running from root without package installed in editable mode
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from atomworks.io.parser import parse
    from atomworks.io.utils.io_utils import infer_pdb_file_type

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLE_DIR = Path(__file__).parent

def verify_mmjson_parsing():
    print("Starting mmJSON verification...")
    
    json_path = EXAMPLE_DIR / "2hhb.json.gz"
    cif_path = EXAMPLE_DIR / "2hhb.cif.gz"
    
    if not json_path.exists():
        print(f"Error: mmJSON file not found at {json_path}")
        return
    if not cif_path.exists():
        print(f"Error: CIF file not found at {cif_path}")
        return

    # 1. Test File Type Inference
    print("\n1. Testing File Type Inference...")
    inferred_type = infer_pdb_file_type(json_path)
    if inferred_type == "mmjson":
        print(f"✅ Correctly inferred 'mmjson' from {json_path.name}")
    else:
        print(f"❌ Failed to infer 'mmjson'. Got: {inferred_type}")

    # 2. Parse mmJSON
    print("\n2. Parsing mmJSON file...")
    try:
        result_json = parse(json_path, file_type="mmjson")
        atoms_json = result_json["asym_unit"]
        print(f"✅ Successfully parsed mmJSON. Loaded {atoms_json.array_length()} atoms.")
    except Exception as e:
        print(f"❌ Failed to parse mmJSON: {e}")
        return

    # 3. Parse CIF for Comparison
    print("\n3. Parsing CIF file for comparison...")
    try:
        result_cif = parse(cif_path, file_type="cif")
        atoms_cif = result_cif["asym_unit"]
        print(f"✅ Successfully parsed CIF. Loaded {atoms_cif.array_length()} atoms.")
    except Exception as e:
        print(f"❌ Failed to parse CIF: {e}")
        return

    # 4. Compare Results
    print("\n4. Comparing mmJSON vs CIF results...")
    
    # Compare Atom Count
    if atoms_json.array_length() == atoms_cif.array_length():
        print("✅ Atom count matches.")
    else:
        print(f"❌ Atom count mismatch: JSON {atoms_json.array_length()} != CIF {atoms_cif.array_length()}")
        
    # Compare Coordinates
    try:
        np.testing.assert_allclose(atoms_json.coord, atoms_cif.coord, rtol=1e-3, atol=1e-3)
        print("✅ Coordinates match (within tolerance).")
    except AssertionError as e:
        print(f"❌ Coordinates mismatch: {e}")

    # Compare Elements
    if np.array_equal(atoms_json.element, atoms_cif.element):
        print("✅ Elements match.")
    else:
        print("❌ Elements mismatch.")

    # Compare Residue Names
    if np.array_equal(atoms_json.res_name, atoms_cif.res_name):
        print("✅ Residue names match.")
    else:
        print("❌ Residue names mismatch.")
        
    # Compare Chain IDs
    if np.array_equal(atoms_json.chain_id, atoms_cif.chain_id):
        print("✅ Chain IDs match.")
    else:
        print("❌ Chain IDs mismatch.")
    
    print("\nVerification finished.")

if __name__ == "__main__":
    verify_mmjson_parsing()
