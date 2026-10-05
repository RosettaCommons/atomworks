"""Regression tests for complex cases to ensure consistent behavior."""

import pickle
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.constants import ATOMWORKS_COMMON_ANNOTATIONS, CRYSTALLIZATION_AIDS
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.transforms import atom_array as ta
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from tests.io.conftest import TEST_DATA_IO, get_pdb_path

TEST_CASES = [
    "6mub",  # Symmetry center clash
    "1j8z",  # Contains misordered atoms in a residue
    "1fp7",  # Contains bonds between crystallization aids in struct_conn
    "1twr",  # Residue VER provided with custom bonds (aromatic in crystal structure but not in CCD)
    "6q9t",  # Contains residue `QUK` which uses a mix of `std` and `alt` atom ids; also contains various unusual ligands and NCAA's
    "1d9d",  # DNA_RNA_HYBRID + protein
    "1ivo",  # protein + oligosaccharides (branched)
    "4js1",  # Protein-glycosylation with covalent bonds
    "4ndz",  # Large assembly with ligands covalently bound
    "3ne7",  # Small assembly with two ligands covalently bound
    "1rcq",  # PLP,DLY ligands with 2 transformations
    "1qh9",  # 2 transformations + LAC (lactose)
    "1cbn",  # Sequence heterogeneity in residue 22 of chain A
    "6h9v",  # Partial hydrogens cause fast-path miscount in add_missing_atoms
    "1d3z",  # NMR protein (ubiquitin), 10 models
    "2koc",  # NMR RNA hairpin, 20 models
    "3ne2",  # Two simple proteins
    "6dmg",  # Multiconformer ligand
    "104d",  # DNA with MG ions
    "1a3g",  # Covalent modification, protein-ligand
    "3n95",  # Terminal VAL-NH2 amide inferred from polymer sequence without an authored link.
    "4aah",  # Vicinal Cys103-Cys104 disulfide must not suppress the C103-N104 backbone bond.
]

ANNOTATIONS_TO_EXCLUDE = {
    "ins_code",  # Legacy PDB artifact
    "label_alt_id",  # Altloc selection may not round-trip
}

# Additional annotations to exclude only for CIF roundtrip tests
ROUNDTRIP_ANNOTATIONS_TO_EXCLUDE = {
    "chain_type",  # CIF roundtrip collapses non-polymer types (BRANCHED→NON_POLYMER, etc.)
}

# Common parse config used in regression tests
REGRESSION_PARSE_CONFIG = ParseConfig(
    add_missing_atoms=True,
    remove_waters=True,
    remove_ccds=CRYSTALLIZATION_AIDS,
    fix_ligands_at_symmetry_centers=True,
    build_assembly="all",
    fix_arginines=True,
    convert_mse_to_met=True,
    hydrogen_policy="keep",
    model=None,
    return_atom_array_plus=True,
)


@pytest.fixture
def regression_config(pdb_id: str) -> ParseConfig:
    # Include disulfides to exercise 4AAH's adjacent-cysteine backbone attachment.
    if pdb_id == "4aah":
        return replace(REGRESSION_PARSE_CONFIG, add_bond_types_from_struct_conn=("covale", "disulf"))
    return REGRESSION_PARSE_CONFIG


def get_annotations_to_compare(
    atom_array1: AtomArray | AtomArrayStack,
    atom_array2: AtomArray | AtomArrayStack,
    extra_exclude: set[str] | None = None,
):
    """Get the set of annotations to compare between atom arrays."""
    annots1 = set(ATOMWORKS_COMMON_ANNOTATIONS) & set(atom_array1.get_annotation_categories())
    annots2 = set(ATOMWORKS_COMMON_ANNOTATIONS) & set(atom_array2.get_annotation_categories())
    annotations = annots1 & annots2

    exclude = ANNOTATIONS_TO_EXCLUDE
    if extra_exclude:
        exclude = exclude | extra_exclude

    return annotations - exclude


def _compare_parse_results(
    result: dict,
    expected_result: dict,
    remove_hydrogens: bool = True,
    extra_annotations_to_exclude: set[str] | None = None,
    metadata_keys_to_compare: set[str] | None = None,
    chain_info_fields_to_skip: set[str] | None = None,
) -> None:
    """Compare two parse results for structural and metadata equality.

    Compares:
    - Asymmetric unit atoms and annotations
    - All assemblies atoms and annotations
    - Metadata sections specified by metadata_keys_to_compare
    """
    # +----- Asymmetric unit -----+
    # ... remove hydrogens, if requested
    if remove_hydrogens:
        expected_result["asym_unit"] = ta.remove_hydrogens(expected_result["asym_unit"])
        result["asym_unit"] = ta.remove_hydrogens(result["asym_unit"])

    assert_same_atom_array_or_stack(
        result["asym_unit"],
        expected_result["asym_unit"],
        annotations_to_compare=list(
            get_annotations_to_compare(
                result["asym_unit"],
                expected_result["asym_unit"],
                extra_exclude=extra_annotations_to_exclude,
            )
        ),
        compare_coords=True,
        compare_bonds=True,
        enforce_order=False,
        cast_to_common_dtype=True,
    )

    # +----- Assemblies -----+
    for assembly_id in result["assemblies"]:
        # ... remove hydrogens, if requested
        if remove_hydrogens:
            result["assemblies"][assembly_id] = ta.remove_hydrogens(result["assemblies"][assembly_id])
            expected_result["assemblies"][assembly_id] = ta.remove_hydrogens(expected_result["assemblies"][assembly_id])

        result_asm = result["assemblies"][assembly_id]
        expected_asm = expected_result["assemblies"][assembly_id]

        assert_same_atom_array_or_stack(
            result_asm,
            expected_asm,
            annotations_to_compare=list(
                get_annotations_to_compare(
                    result_asm,
                    expected_asm,
                    extra_exclude=extra_annotations_to_exclude,
                )
            ),
            compare_coords=True,
            compare_bonds=True,
            enforce_order=False,
            cast_to_common_dtype=True,
        )

    # +----- Metadata -----+
    # Determine which metadata keys to compare (default = all)
    if metadata_keys_to_compare is None:
        metadata_keys_to_compare = {"ligand_info", "chain_info", "extra_info", "metadata"}

    # ... the ligand of interest information
    if "ligand_info" in metadata_keys_to_compare:
        assert result["ligand_info"] == expected_result["ligand_info"]

    # ... the chain information
    if "chain_info" in metadata_keys_to_compare:
        assert set(result["chain_info"].keys()) == set(expected_result["chain_info"].keys())

        # Initialize skip set
        if chain_info_fields_to_skip is None:
            chain_info_fields_to_skip = set()

        for chain in result["chain_info"]:
            # Only compare fields that aren't in the skip set
            if "chain_type" not in chain_info_fields_to_skip:
                got = result["chain_info"][chain]["chain_type"]
                expected = expected_result["chain_info"][chain]["chain_type"]
                assert got == expected, f"Chain info for {chain=} does not match: {got} != {expected}"

            if "res_name" not in chain_info_fields_to_skip:
                got = result["chain_info"][chain]["res_name"]
                expected = expected_result["chain_info"][chain]["res_name"]
                assert np.array_equal(got, expected), f"Chain info for {chain=} does not match: {got} != {expected}"

            if "res_id" not in chain_info_fields_to_skip:
                got = result["chain_info"][chain]["res_id"]
                expected = expected_result["chain_info"][chain]["res_id"]
                assert np.array_equal(got, expected), f"Chain info for {chain=} does not match: {got} != {expected}"

            if "is_polymer" not in chain_info_fields_to_skip:
                got = result["chain_info"][chain]["is_polymer"]
                expected = expected_result["chain_info"][chain]["is_polymer"]
                assert got == expected, f"Chain info for {chain=} does not match: {got} != {expected}"

    # ... the extra information
    if "extra_info" in metadata_keys_to_compare:
        assert result["extra_info"] == expected_result["extra_info"]

    # ... any additional metadata keys
    if "metadata" in metadata_keys_to_compare:
        for key in expected_result.get("metadata", {}):
            assert key in result["metadata"], f"Missing metadata key: {key}"
            assert result["metadata"][key] == expected_result["metadata"][key]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
@pytest.mark.parametrize("remove_hydrogens", [True, False], ids=["no_h", "with_h"])
def test_regression_against_stored_result(pdb_id: str, remove_hydrogens: bool, regression_config: ParseConfig):
    """Test that parse output matches stored regression baseline."""
    regression_dir = TEST_DATA_IO / "regression_tests"
    pickle_path = regression_dir / f"{pdb_id}.pkl"

    with pickle_path.open("rb") as f:
        expected_result = pickle.load(f)

    result = parse(get_pdb_path(pdb_id), config=regression_config)
    assert result is not None

    _compare_parse_results(result, expected_result, remove_hydrogens=remove_hydrogens)


@pytest.mark.parametrize("pdb_id", TEST_CASES)
@pytest.mark.parametrize("remove_hydrogens", [True, False], ids=["no_h", "with_h"])
@pytest.mark.parametrize("file_type", ["cif", "bcif.zst"])
def test_regression_cif_roundtrip(pdb_id: str, remove_hydrogens: bool, file_type: str, regression_config: ParseConfig):
    """Test CIF/BCIF roundtrip.

    Gold-standard test that ensures:
    (a) Saving to CIF preserves all atoms
    (b) struct_oper categories are preserved
    (c) Assemblies can be rebuilt from saved struct_oper
    (d) Roundtrip produces same result as original
    """
    regression_dir = TEST_DATA_IO / "regression_tests"
    pickle_path = regression_dir / f"{pdb_id}.pkl"

    with pickle_path.open("rb") as f:
        expected_result = pickle.load(f)

    result = parse(get_pdb_path(pdb_id), config=regression_config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cif_path = Path(tmp_dir) / f"{pdb_id}_roundtrip.{file_type}"

        extra_cats = {
            "pdbx_struct_oper_list": category_to_dict(result["extra_info"]["struct_oper_category"]),
            "pdbx_struct_assembly_gen": category_to_dict(result["extra_info"]["assembly_gen_category"]),
        }

        to_cif_file(
            result["asym_unit"],
            cif_path,
            include_entity_categories=True,
            extra_categories=extra_cats,
        )

        reloaded_result = parse(
            cif_path,
            config=ParseConfig(
                add_bond_types_from_struct_conn=regression_config.add_bond_types_from_struct_conn,
                add_missing_atoms=False,
                remove_waters=False,
                remove_ccds=None,
                fix_ligands_at_symmetry_centers=False,
                build_assembly="all",
                convert_mse_to_met=False,
                hydrogen_policy="keep",
            ),
        )

    _compare_parse_results(
        reloaded_result,
        expected_result,
        remove_hydrogens=remove_hydrogens,
        extra_annotations_to_exclude=ROUNDTRIP_ANNOTATIONS_TO_EXCLUDE,
        metadata_keys_to_compare={"chain_info"},
        chain_info_fields_to_skip={"chain_type"},
    )


if __name__ == "__main__":
    pytest.main([__file__])
