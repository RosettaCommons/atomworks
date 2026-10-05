import pytest

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path
from tests.io.utils.charge_invariants import check_charge_invariants

TEST_CASES = [
    "5e5j",  # Comes from more than 1 experimental method (X-ray & neutron scattering)
    "1j8z",  # Contains misordered atoms in a residue
    "2fs3",  # Contains an unusual operation expression for assembly building
    "1fp7",  # Contains bonds between crystallization aids in struct_conn
    "6lzb",  # Multiple occupancy; some prior issues with `struct_conn` (since fixed in latest PDB updates)
    "5t39",  # Contains misordered atoms in a residue (`SAH`)
    "1nci",  # Issues with arginine resolving, seems to have differing number of NH1/NH2
    "1aym",  # Issues with patching symmetric ligands (invalid literal)
    "8bc3",  # Covalent bond involving chains that were removed during cleaning
    "1twr",  # Residue name not in biotite's CCD
    "6q9t",  # Contains residue `QUK` which uses a mix of `std` and `alt` atom ids
    "1xvk",  # Final boss of PDB parsing: non-canonical macrocycle with disulfide bond and multiple occupancy; cannot infer standard inter-residue bonds
    "1qfe",  # A170 NZ incorrectly assigned charge +2 unless we remove a hydrogen as a leaving atom (there should be no protonated amides in the final structure; Lysine NZ becomes part of an amide bond to small molecule DHS)
    "1tqh",  # Serine hydrolase intermediate requires converting C=O to C-O(-) to satisfy valence requirements
    "1dpn",  # A single P-O link replaces a double-bond leaving group and restores an equivalent terminal P=O
    "6h9v",  # Includes isopeptide bond with IAS NCAA where naively building N-CA-C backbone would create too many bonds
    "1rcq",  # Enzyme active site where PLP exists in two partially occupied states, one with a covalent bond to a lysine and one without
    # Additional charge edge cases
    "6lyz",  # Pure protein
    "1d9d",  # DNA/RNA hybrid + protein
    "1ivo",  # Oligosaccharide
    "6qhp",  # Simple protein
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
@pytest.mark.parametrize("hydrogen_policy", ["keep", "remove"], ids=["with_h", "no_h"])
def test_parse_invariants(pdb_id: str, hydrogen_policy: str):
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
        hydrogen_policy=hydrogen_policy,
    )
    assert result is not None  # Check if processing runs through

    atom_array = result["asym_unit"][0]

    check_charge_invariants(pdb_id, atom_array)


# PDB IDs that are malformed or contain unresolvable data, and should raise
# specific errors during parsing.
ERROR_TEST_CASES = [
    # 8cuy contains UNL (unknown ligand) residues — free-floating lipids with no CCD
    # template. Atoms have inter-residue bonds but zero intra-residue bonds.
    ("8cuy", ValueError, "missing CCD template"),
    # 4v4s authors both O3'-P and O3'-OP2 links from A36 to YYG37; reject the three-bond oxygen.
    ("4v4s", ValueError, r"Unresolved link valence at C/36/A/O3'.*bond-order sum=3"),
]


@pytest.mark.parametrize("pdb_id,expected_error,match", ERROR_TEST_CASES)
@pytest.mark.parametrize("hydrogen_policy", ["keep", "remove"], ids=["with_h", "no_h"])
def test_parse_expected_errors(pdb_id: str, expected_error: type, match: str, hydrogen_policy: str):
    """PDB structures that should raise specific errors during parsing."""
    path = get_pdb_path(pdb_id)
    with pytest.raises(expected_error, match=match):
        parse(path, config=ParseConfig(hydrogen_policy=hydrogen_policy))


if __name__ == "__main__":
    pytest.main([__file__])
