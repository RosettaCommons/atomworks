import copy
import gzip
import io
import pickle
import tempfile
from contextlib import nullcontext
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
from biotite.database import rcsb
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.constants import ATOMWORKS_COMMON_ANNOTATIONS, CCD_MIRROR_PATH
from atomworks.enums import ChainType
from atomworks.io.config import ParseConfig, PrepareConfig
from atomworks.io.parser import parse, parse_atom_array, prepare_atom_array
from atomworks.io.tools.inference import (
    ChemicalComponent,
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
)
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, as_atom_array_plus, concatenate_atom_array_plus
from atomworks.io.utils.ccd import (
    _get_ccd_block,
    _get_standard_ccd_codes_cached,
    atom_array_from_ccd_code,
    get_atom_names_for_residue,
    register_custom_ccd_entry,
)
from atomworks.io.utils.io_utils import (
    CIF_LIKE_EXTENSIONS,
    CIFWriteConfig,
    _build_chem_comp_atom,
    get_structure,
    infer_pdb_file_type,
    load_any,
    read_any,
    to_cif_buffer,
    to_cif_file,
    to_cif_string,
    to_pdb_string,
)
from atomworks.io.utils.testing import (
    assert_same_annotation_cardinality,
    assert_same_atom_array_or_stack,
    get_pdb_path,
)
from atomworks.ml.utils.testing import assert_equal
from tests.conftest import skip_if_no_internet
from tests.io.conftest import TEST_DATA_IO


@pytest.mark.requires_internet
@skip_if_no_internet
@pytest.mark.parametrize(
    "pdb_id, file_type, directory",
    [
        ("6lyz", "pdb", True),
        ("6lyz", "pdb", False),
        ("6lyz", "cif", True),
        ("6lyz", "cif", False),
        pytest.param(
            "6lyz",
            "bcif",
            True,
            marks=pytest.mark.skip(reason="bcif loading began failing for reasons that have not been chased down yet"),
        ),
        pytest.param(
            "6lyz",
            "bcif",
            False,
            marks=pytest.mark.skip(reason="bcif loading began failing for reasons that have not been chased down yet"),
        ),
    ],
)
def test_load_any(pdb_id, file_type, directory):
    with tempfile.TemporaryDirectory() if directory else nullcontext() as tmp_dir:
        # Test loading from a buffer or file
        loaded_structure = load_any(rcsb.fetch(pdb_id, file_type, tmp_dir), file_type=file_type)
        assert isinstance(loaded_structure, AtomArray | AtomArrayStack)
        assert loaded_structure.array_length() > 0


def test_infer_filetype():
    assert infer_pdb_file_type("6lyz.pdb") == "pdb"
    assert infer_pdb_file_type("6lyz.pdb.gz") == "pdb"
    assert infer_pdb_file_type("6lyz.pdb.gzip") == "pdb"
    assert infer_pdb_file_type("6lyz.mmcif") == "cif"
    assert infer_pdb_file_type("6lyz.mmcif.gz") == "cif"
    assert infer_pdb_file_type("6lyz.mmcif.gzip") == "cif"
    assert infer_pdb_file_type("6lyz.pdbx") == "cif"
    assert infer_pdb_file_type("6lyz.pdbx.gz") == "cif"
    assert infer_pdb_file_type("6lyz.pdbx.gzip") == "cif"
    assert infer_pdb_file_type("6lyz.bcif") == "bcif"
    assert infer_pdb_file_type("6lyz.bcif.gz") == "bcif"
    assert infer_pdb_file_type("6lyz.bcif.gzip") == "bcif"

    for compression in ("", ".gz", ".gzip", ".zst"):
        assert infer_pdb_file_type(f"2hhb.mmjson{compression}") == "mmjson"
        assert f".mmjson{compression}" in CIF_LIKE_EXTENSIONS
        assert f".json{compression}" not in CIF_LIKE_EXTENSIONS
        with pytest.raises(ValueError, match="Unsupported file type: json"):
            infer_pdb_file_type(f"2hhb.json{compression}")

    with open(TEST_DATA_IO / "6lyz.bcif", "rb") as f:
        buffer = io.BytesIO(f.read())
        assert infer_pdb_file_type(buffer) == "bcif"

    with open(TEST_DATA_IO / "1a8o_modified.cif") as f:
        buffer = io.StringIO(f.read())
        assert infer_pdb_file_type(buffer) == "cif"

    with open(TEST_DATA_IO / "UniRef50_A0A0S8JQ92_AF2_predicted.pdb") as f:
        buffer = io.StringIO(f.read())
        assert infer_pdb_file_type(buffer) == "pdb"


@pytest.mark.parametrize("source_kind", ["path", "buffer"])
def test_parse_mmjson_matches_cif(source_kind: str) -> None:
    json_path = TEST_DATA_IO / "2hhb.mmjson.gz"
    cif_path = TEST_DATA_IO / "2hhb.cif.gz"

    assert infer_pdb_file_type(json_path) == "mmjson"

    if source_kind == "path":
        json_source = json_path
    else:
        with gzip.open(json_path, "rt") as stream:
            json_source = io.StringIO(stream.read())

    file_type = None if source_kind == "path" else "mmjson"
    atoms_json = parse(json_source, config=ParseConfig(file_type=file_type))["asym_unit"]
    atoms_cif = parse(cif_path, config=ParseConfig(file_type="cif"))["asym_unit"]

    assert_same_atom_array_or_stack(atoms_json, atoms_cif)


@pytest.mark.parametrize(
    "extra_fields, include_bonds, model",
    [
        ([], True, None),
        (["b_factor", "occupancy"], True, None),
        ([], False, None),
        ([], True, 1),
    ],
)
def test_get_structure_configurations(extra_fields, include_bonds, model):
    # Read 6lyz CIF into a CIFFile object
    cif_file = read_any(get_pdb_path("6lyz"), file_type="cif")

    # Get the structure with different configurations
    structure = get_structure(
        cif_file,
        extra_fields=extra_fields,
        include_bonds=include_bonds,
        model=model,
    )

    assert isinstance(structure, AtomArray | AtomArrayStack)
    assert structure.array_length() > 0

    # Check if extra fields are present
    for field in extra_fields:
        assert field in structure.get_annotation_categories()

    # Check if bonds are included
    if include_bonds:
        assert structure.bonds is not None
    else:
        assert structure.bonds is None

    # Check if the correct model is returned when specified
    if model is not None:
        assert isinstance(structure, AtomArray)
    else:
        assert isinstance(structure, AtomArrayStack)


def test_parse_atom_array_with_multiple_transformations():
    data_dict = parse(get_pdb_path("1out"), config=ParseConfig(add_missing_atoms=True))
    parsed_from_file = data_dict["assemblies"]["1"]
    assert any(chain_iid.endswith("_2") for chain_iid in np.unique(parsed_from_file.chain_iid))

    new_data_dict = parse_atom_array(parsed_from_file, config=PrepareConfig(add_missing_atoms=True))
    parsed_from_atom_array = new_data_dict["asym_unit"][0]
    assert "chain_iid" in parsed_from_atom_array.get_annotation_categories()

    chain_iid = parsed_from_atom_array.chain_iid
    res_id = parsed_from_atom_array.res_id
    res_name = parsed_from_atom_array.res_name
    atom_name = parsed_from_atom_array.atom_name

    # Get indices for the test case atoms
    tfm_1_atom_1_idx = np.where((chain_iid == "A_1") & (res_name == "ACE") & (res_id == 1) & (atom_name == "C"))[0]
    tfm_1_atom_2_idx = np.where((chain_iid == "A_1") & (res_name == "SER") & (res_id == 2) & (atom_name == "N"))[0]
    tfm_2_atom_1_idx = np.where((chain_iid == "A_2") & (res_name == "ACE") & (res_id == 1) & (atom_name == "C"))[0]
    tfm_2_atom_2_idx = np.where((chain_iid == "A_2") & (res_name == "SER") & (res_id == 2) & (atom_name == "N"))[0]
    for arr in [tfm_1_atom_1_idx, tfm_1_atom_2_idx, tfm_2_atom_1_idx, tfm_2_atom_2_idx]:
        assert len(arr) == 1
    tfm_1_atom_1_idx = tfm_1_atom_1_idx[0]
    tfm_1_atom_2_idx = tfm_1_atom_2_idx[0]
    tfm_2_atom_1_idx = tfm_2_atom_1_idx[0]
    tfm_2_atom_2_idx = tfm_2_atom_2_idx[0]

    tfm_1_atom_1_bonds = parsed_from_atom_array.bonds.get_bonds(tfm_1_atom_1_idx)[0]
    tfm_2_atom_1_bonds = parsed_from_atom_array.bonds.get_bonds(tfm_2_atom_1_idx)[0]

    # Assert the intra-transform bonds are present
    assert tfm_1_atom_2_idx in tfm_1_atom_1_bonds
    assert tfm_2_atom_2_idx in tfm_2_atom_1_bonds

    # Assert that no inter-transform bonds are present
    assert tfm_2_atom_2_idx not in tfm_1_atom_1_bonds
    assert tfm_2_atom_1_idx not in tfm_1_atom_1_bonds
    assert tfm_1_atom_2_idx not in tfm_2_atom_1_bonds
    assert tfm_1_atom_1_idx not in tfm_2_atom_1_bonds


def test_to_cif_string():
    result = parse(get_pdb_path("6lyz"), config="minimal")
    cif_structure = result["asym_unit"]

    # Make identifiers unique
    cif_structure.res_id = struc.spread_residue_wise(cif_structure, np.arange(struc.get_residue_count(cif_structure)))
    # ... drop HOH
    cif_structure = cif_structure[0, cif_structure.res_name != "HOH"]
    cif_string = to_cif_string(cif_structure)

    assert isinstance(cif_string, str)
    assert len(cif_string) > 0

    result2 = parse(io.StringIO(cif_string), config="minimal")
    cif_structure2 = result2["asym_unit"][0]
    assert np.allclose(cif_structure.coord, cif_structure2.coord)
    assert np.all(cif_structure.atom_name == cif_structure2.atom_name)
    assert np.all(cif_structure.element == cif_structure2.element)
    assert np.all(cif_structure.charge == cif_structure2.charge)
    assert np.all(cif_structure.chain_id == cif_structure2.chain_id)
    assert np.all(cif_structure.res_name == cif_structure2.res_name)
    assert np.all(cif_structure.res_id == cif_structure2.res_id)
    assert np.all(cif_structure.b_factor == cif_structure2.b_factor)
    assert np.all(cif_structure.occupancy == cif_structure2.occupancy)

    # Test if we can write custom metadata
    metadata = {
        "test_category": {"test_col1": "data", "test_col2": "data2"},
        "test_category2": {"test_col1": np.arange(10), "test_col2": np.arange(10)},
        "test_category3": {"test_col1": [1, 3, 4], "test_col2": [2, 5, "a"]},
    }

    cif_string2 = to_cif_string(
        cif_structure,
        config=CIFWriteConfig(id="test_id", extra_categories=metadata),
    )

    metadata_serialized = (
        "#\n"
        "_test_category.test_col1   data\n"
        "_test_category.test_col2   data2\n"
        "#\n"
        "loop_\n"
        "_test_category2.test_col1 \n"
        "_test_category2.test_col2 \n"
        "0 0\n"
        "1 1\n"
        "2 2\n"
        "3 3\n"
        "4 4\n"
        "5 5\n"
        "6 6\n"
        "7 7\n"
        "8 8\n"
        "9 9\n"
        "#\n"
        "loop_\n"
        "_test_category3.test_col1 \n"
        "_test_category3.test_col2 \n"
        "1 2\n"
        "3 5\n"
        "4 a\n"
        "#\n"
    )
    assert metadata_serialized in cif_string2, "Metadata not found in serialized CIF string."


def test_to_pdb_string():
    result = parse(get_pdb_path("6lyz"), config="minimal")
    pdb_structure = result["asym_unit"][0]
    n_atoms = pdb_structure.array_length()
    pdb_string = to_pdb_string(pdb_structure)
    assert isinstance(pdb_string, str)
    assert len(pdb_string) > 0

    # Test that we can load the pdb string back into an AtomArray
    result2 = parse(io.StringIO(pdb_string), config="minimal")
    pdb_structure2 = result2["asym_unit"][0]
    assert pdb_structure2.array_length() == n_atoms
    assert np.allclose(pdb_structure.coord, pdb_structure2.coord)
    assert np.all(pdb_structure.atom_name == pdb_structure2.atom_name)
    assert np.all(pdb_structure.element == pdb_structure2.element)
    assert np.all(pdb_structure.charge == pdb_structure2.charge)
    assert np.all(pdb_structure.chain_id == pdb_structure2.chain_id)
    assert np.all(pdb_structure.res_name == pdb_structure2.res_name)
    assert np.all(pdb_structure.res_id == pdb_structure2.res_id)
    assert np.all(pdb_structure.b_factor == pdb_structure2.b_factor)
    assert np.all(pdb_structure.occupancy == pdb_structure2.occupancy)


@pytest.mark.parametrize(
    "pdb_id,assembly_id,mode",
    [
        # General edge cases from the PDB
        ("1d9d", "1", "transformation_id"),  # DNA_RNA_HYBRID + protein
        ("6qhp", "1", "transformation_id"),  # protein + non-polymers
        ("1fix", "1", "transformation_id"),  # DNA + RNA only
        ("1ivo", "1", "transformation_id"),  # protein + oligosaccharides (branched)
        # Assembly tests with transformations - test both modes on 1rxz
        ("1rxz", "1", "transformation_id"),  # Complex assembly with 3 transformations
        ("1rxz", "1", "chain_iid"),  # Same structure, chain_iid mode
        # Struct_conn preservation tests
        ("4js1", "1", "transformation_id"),  # Protein-glycosylation with covalent bonds
        ("4ndz", "1", "transformation_id"),  # Large assembly with ligands covalently bound
        ("3ne7", "1", "transformation_id"),  # Small assembly with two ligands covalently bound
        # Multi-residue ligand tests
        ("7zcy", "1", "transformation_id"),  # Selenium ligands with 3 transformations
        ("1rcq", "1", "transformation_id"),  # PLP,DLY ligands with 2 transformations
        # Local assembly tests with glycosylations
        ("4i7z", "1", "transformation_id"),  # 2 transformations + UMQ (ubiquinone)
        ("1qh9", "1", "transformation_id"),  # 2 transformations + LAC (lactose)
    ],
)
def test_cif_assembly_roundtrip(pdb_id: str, assembly_id: str, mode: str):
    """CIF roundtrip test for assemblies with various configurations."""
    # Parse original structure
    source = get_pdb_path(pdb_id)

    data_dict = parse(
        source,
        config=ParseConfig(
            hydrogen_policy="remove",
            build_assembly=[assembly_id],
            add_bond_types_from_struct_conn=("covale",),
            add_missing_atoms=True,
        ),
    )
    # Original atom array (for comparison later)
    original = data_dict["assemblies"][assembly_id][0]

    # Write to CIF (with specified mode)
    with tempfile.TemporaryDirectory() as tmp_dir:
        cif_path = Path(tmp_dir) / f"{pdb_id}_assembly_{assembly_id}.cif"
        to_cif_file(
            original,
            cif_path,
            include_entity_categories=True,
            chain_disambiguation=mode,
        )

        # Read back
        reloaded_dict = parse(
            cif_path,
            config=ParseConfig(
                hydrogen_policy="remove",
                add_bond_types_from_struct_conn=("covale",),
                add_missing_atoms=False,
            ),
        )
        reloaded = reloaded_dict["assemblies"][assembly_id][0]

        common_annotations = (
            set(original.get_annotation_categories())
            & set(reloaded.get_annotation_categories())
            & set(ATOMWORKS_COMMON_ANNOTATIONS)
        )
        # Annotations that are order-dependent and may not roundtrip with the same values (but should have same cardinality)
        exclude_annotations = {
            "chain_type",
            "pn_unit_entity",
            "molecule_id",
            "molecule_iid",
            "molecule_entity",
            "label_entity_id",
            "atom_id",
            "auth_seq_id",
            "label_alt_id",
        }

        if mode == "chain_iid":
            # Annotations derive fro chain_id also won't roundtrip
            # (Since we convert chain_id -> chain_iid)
            exclude_annotations = exclude_annotations | {
                "chain_id",
                "transformation_id",
                "chain_iid",
                "pn_unit_iid",
                "pn_unit_id",
            }
        annotations_to_compare = common_annotations - exclude_annotations

        # Compare structures
        assert_same_atom_array_or_stack(
            original,
            reloaded,
            annotations_to_compare=annotations_to_compare,
            compare_coords=True,
            compare_bonds=True,
            compare_bond_order=True,
            enforce_order=False,
            cast_to_common_dtype=True,  # CIF roundtrips may change dtypes (e.g., int8→int64)
        )

        # Validate cardinality of excluded annotations
        # (For simplicity, only when disambiguating via transformation_id)
        if mode == "transformation_id":
            cardinality_annotations = common_annotations - {"chain_type", "label_alt_id"}
            assert_same_annotation_cardinality(original, reloaded, cardinality_annotations)


@pytest.fixture
def forget_ccd_lookups():
    """Drops the CCD lookups a test cached, keyed by mirror path alone, while it had turned the CCD off."""
    yield
    for lookup in (_get_standard_ccd_codes_cached, get_atom_names_for_residue):
        lookup.cache_clear()


@pytest.mark.parametrize(
    "pdb_id",
    [
        # General edge cases
        "1d9d",  # DNA_RNA_HYBRID + protein; bridging oxygen +1 charge from unremoved leaving hydrogen
        "6qhp",  # protein + non-polymers
        "1fix",  # DNA + RNA only
        "1ivo",  # protein + oligosaccharides (branched); covalent modification had +1 charge on amide nitrogen
        # Covalent modification charge bugs
        "4js1",  # covalent modification had +1 charge on amide nitrogen (same as 1IVO)
        "1qfe",  # erroneous +2 on nitrogen from unremoved hydrogen leaving group in covalent modification
        # Multi-residue ligand / bond inference bugs
        "4ndz",  # multi-residue ligand with spurious bond causing positive charge on oxygen
        "1a3g",  # positive charge on carbon from missing double bond inference in Schiff base linkage
        # Altloc / occupancy handling
        "1rcq",  # lysine NZ +2 from improper altloc handling in struct conn; multiple occupancy
        # Peptide bond inference
        "6h9v",  # C-gamma linking L-beta peptide bond not correctly inferred across chains
        # Hydrogen leaving atom bugs
        "1j8z",  # ACE carbonyl carbon incorrectly charged due to unremoved hydrogen leaving atom
        "1a1e",  # positive charge on carbon from unremoved hydrogen leaving atom (not labeled in CCD)
        # Custom CCD edge cases
        "1twr",  # Residue VER provided with custom bonds (aromatic in crystal structure but not in CCD)
        "6q9t",  # Contains residue `QUK` which uses a mix of `std` and `alt` atom ids; also contains various unusual ligands and NCAA's
    ],
)
@pytest.mark.parametrize("biotite_ccd", ["on", "off"])
@pytest.mark.parametrize(
    "reload_config",
    [
        pytest.param(
            ParseConfig(hydrogen_policy="remove", add_missing_atoms=False),
            id="no_rebuild",
        ),
        pytest.param(ParseConfig(hydrogen_policy="remove", add_missing_atoms=True), id="with_rebuild"),
    ],
)
def test_to_cif_file_roundtrip(
    pdb_id: str, reload_config: ParseConfig, biotite_ccd: str, monkeypatch, forget_ccd_lookups
):
    """CIF roundtrip on diverse asymmetric units, with and without biotite CCD access.

    ``biotite_ccd="off"`` disables ``ALLOW_BIOTITE_CCD`` and clears
    ``CCD_MIRROR_PATH`` so the reloader must reconstruct residue chemistry
    from the CIF's own ``chem_comp*`` categories alone.
    """
    source = get_pdb_path(pdb_id)
    data_dict = parse(
        source, config=ParseConfig(hydrogen_policy="remove", build_assembly=None, return_atom_array_plus=True)
    )
    original = data_dict["asym_unit"][0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        cif_path = Path(tmp_dir) / "test.cif"
        to_cif_file(original, cif_path, include_entity_categories=True)

        if biotite_ccd == "off":
            # Force reload to reconstruct residue chemistry from the CIF's own
            # chem_comp* categories
            monkeypatch.setattr("atomworks.io.utils.ccd.ALLOW_BIOTITE_CCD", False)
            monkeypatch.setattr("atomworks.io.utils.ccd.CCD_MIRROR_PATH", None)

        reloaded = parse(cif_path, config=reload_config)["asym_unit"][0]

        # Compare all standard annotations present in both arrays, plus the
        # chem_comp-specific annotations the chem_comp* categories preserve.
        annotations_to_compare = (
            set(ATOMWORKS_COMMON_ANNOTATIONS)
            & set(original.get_annotation_categories())
            & set(reloaded.get_annotation_categories())
        ) - {"label_alt_id"}

        # Misc. extra annotations to compare if present (but that may not always be present)
        for extra in ("charge", "is_aromatic", "is_leaving_atom", "chem_comp_type"):
            if extra in original.get_annotation_categories() and extra in reloaded.get_annotation_categories():
                annotations_to_compare.add(extra)

        assert_same_atom_array_or_stack(
            original, reloaded, annotations_to_compare=annotations_to_compare, enforce_order=False
        )


def test_entity_id_consistent_when_atom_order_disagrees_with_chain_id_order():
    """`_entity` rows and `atom_site.label_entity_id` must reference matching IDs."""
    polymer = as_atom_array_plus(atom_array_from_ccd_code("ALA"))
    polymer.chain_id[:] = "A"
    polymer.set_annotation("chain_type", np.full(polymer.array_length(), int(ChainType.POLYPEPTIDE_L), dtype=np.int8))

    ligand = as_atom_array_plus(atom_array_from_ccd_code("HEM"))
    ligand.chain_id[:] = "L"
    ligand.set_annotation("chain_type", np.full(ligand.array_length(), int(ChainType.NON_POLYMER), dtype=np.int8))

    # Non-polymer first in physical atom order, but chain_id "L" sorts after "A".
    combined = concatenate_atom_array_plus([ligand, polymer])

    with tempfile.TemporaryDirectory() as tmp_dir:
        cif_path = Path(tmp_dir) / "entity_id_swap.cif"
        to_cif_file(combined, cif_path)
        reloaded = parse(
            cif_path,
            config=ParseConfig.from_preset("minimal", load_standard_annotations=True, return_atom_array_plus=True),
        )["asym_unit"][0]

    chain_types = {
        ch: set(reloaded.chain_type[reloaded.chain_id == ch].tolist()) for ch in set(reloaded.chain_id.tolist())
    }
    assert chain_types == {
        "A": {int(ChainType.POLYPEPTIDE_L)},
        "L": {int(ChainType.NON_POLYMER)},
    }, f"chain_type mismatch on round-trip: {chain_types}"


@pytest.mark.parametrize(
    "pdb_id",
    [
        "1rxz",  # 3 transformations, 256 residues, protein
        "1qh9",  # 2 transformations, 221 residues, protein + lactose
        "6lyz",  # 1 transformation, 129 residues, monomer (edge case)
    ],
)
def test_struct_oper_roundtrip(pdb_id: str):
    """Test struct_oper categories enable assembly building from saved CIF.

    Verifies that struct_oper categories can be preserved when saving CIF files,
    allowing biological assemblies to be rebuilt from the saved file.
    """
    source = get_pdb_path(pdb_id)
    assembly_id = "1"

    # Load original file and build assembly directly (for later comparison)
    result_original = parse(
        source, config=ParseConfig(hydrogen_policy="remove", build_assembly=[assembly_id], add_missing_atoms=True)
    )
    assembly_original = result_original["assemblies"][assembly_id][0]

    # Load asym unit only (no assembly building)
    result_asym = parse(
        source, config=ParseConfig(hydrogen_policy="remove", build_assembly=None, add_missing_atoms=True)
    )
    assert (
        "transformation_id" not in result_asym["asym_unit"][0].get_annotation_categories()
    ), "transformation_id should not be present without assembly building"

    with tempfile.TemporaryDirectory() as tmp_dir:
        cif_path = Path(tmp_dir) / "roundtrip.cif"

        # Save asym unit with struct_oper categories preserved
        extra_cats = {
            "pdbx_struct_oper_list": category_to_dict(result_asym["extra_info"]["struct_oper_category"]),
            "pdbx_struct_assembly_gen": category_to_dict(result_asym["extra_info"]["assembly_gen_category"]),
        }

        # Include pdbx_struct_assembly if available
        if (
            "cif_block" in result_asym
            and result_asym["cif_block"]
            and "pdbx_struct_assembly" in result_asym["cif_block"]
        ):
            extra_cats["pdbx_struct_assembly"] = category_to_dict(result_asym["cif_block"], "pdbx_struct_assembly")

        to_cif_file(result_asym["asym_unit"], cif_path, include_entity_categories=True, extra_categories=extra_cats)

        # Load saved file WITH and WITHOUT missing atoms and build assembly from it
        # ... without missing atoms
        result_rebuilt_without_missing = parse(
            cif_path,
            config=ParseConfig(hydrogen_policy="remove", build_assembly=[assembly_id], add_missing_atoms=False),
        )
        assembly_rebuilt_without_missing = result_rebuilt_without_missing["assemblies"][assembly_id][0]
        # ... with missing atoms (should be identical)
        result_rebuilt_with_missing = parse(
            cif_path,
            config=ParseConfig(hydrogen_policy="remove", build_assembly=[assembly_id], add_missing_atoms=True),
        )
        assembly_rebuilt_with_missing = result_rebuilt_with_missing["assemblies"][assembly_id][0]

        # Verify rebuilt assemblies matches original assembly
        annotations_to_compare = [
            annot
            for annot in set(ATOMWORKS_COMMON_ANNOTATIONS)
            if annot in assembly_original.get_annotation_categories()
            and annot in assembly_rebuilt_without_missing.get_annotation_categories()
        ]

        assert_same_atom_array_or_stack(
            assembly_rebuilt_without_missing,
            assembly_original,
            annotations_to_compare=annotations_to_compare,
            compare_bond_order=True,
            enforce_order=False,
        )
        assert_same_atom_array_or_stack(
            assembly_rebuilt_with_missing,
            assembly_original,
            annotations_to_compare=annotations_to_compare,
            compare_bond_order=True,
            enforce_order=False,
        )


def test_chain_disambiguation_errors():
    """Test error cases for chain disambiguation modes.

    Verifies that appropriate exceptions are raised when:
    - No disambiguation method is provided for ambiguous structures
    - transformation_id mode is used without transformation_id annotation
    - chain_iid mode is used without chain_iid annotation
    """
    pdb_id = "1rxz"
    assembly_id = "1"
    source = get_pdb_path(pdb_id)

    # Load assembly with ambiguous chain annotations
    result = parse(
        source, config=ParseConfig(hydrogen_policy="remove", build_assembly=[assembly_id], add_missing_atoms=True)
    )
    assembly_original = result["assemblies"][assembly_id][0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Error case 1: No disambiguation for ambiguous structure
        with pytest.raises(ValueError, match="Ambiguous chain annotations detected"):
            to_cif_file(assembly_original, Path(tmp_dir) / "fail.cif", chain_disambiguation=None)

        # Error case 2: transformation_id mode without transformation_id annotation
        assembly_no_tid = assembly_original.copy()
        assembly_no_tid.del_annotation("transformation_id")
        with pytest.raises(
            ValueError, match="chain_disambiguation='transformation_id' specified.*lacks.*'transformation_id'"
        ):
            to_cif_file(assembly_no_tid, Path(tmp_dir) / "fail.cif", chain_disambiguation="transformation_id")

        # Error case 3: chain_iid mode without chain_iid annotation
        assembly_no_ciid = assembly_original.copy()
        assembly_no_ciid.del_annotation("chain_iid")
        with pytest.raises(ValueError, match="chain_disambiguation='chain_iid' specified.*lacks.*'chain_iid'"):
            to_cif_file(assembly_no_ciid, Path(tmp_dir) / "fail.cif", chain_disambiguation="chain_iid")


def test_parse_with_no_resolved_atoms(tmpdir):
    # Spoof the input data using the inference pipeline
    smiles = "C[C@]12CC[C@@H](C[C@H]1CC[C@@H]3[C@@H]2C[C@H]([C@]4([C@@]3(CC[C@@H]4C5=CC(=O)OC5)O)C)O)O"
    inputs = [
        {
            "smiles": smiles,
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "A",
        }
    ]
    atom_array = components_to_atom_array(inputs)

    # Use the tmpdir fixture to create a temporary file path
    cif_path = Path(tmpdir) / "test.cif"
    cif_path = to_cif_file(atom_array, cif_path, include_nan_coords=True)

    # ... parse the atom array
    out = parse(Path(cif_path))

    # Smoke test
    assert out is not None


def test_inject_msa_information_into_chain_info():
    # Spoof the input data using the inference pipeline
    inputs = [
        {
            "seq": "MSSKQVQLSLPVLVSLVLVSLQVR",
            "msa_path": "sequence_1.a3m",
        },
        {
            "seq": "MKTAYIAKQRQISFVKSHFS",
            "msa_path": "sequence_2.a3m",
        },
    ]
    atom_array, components = components_to_atom_array(inputs, return_components=True)

    msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(components)

    cif_buffer_with_metadata = to_cif_buffer(
        atom_array,
        config=CIFWriteConfig(id="test_inject_msa", extra_categories={"msa_paths_by_chain_id": msa_paths_by_chain_id}),
    )

    # ... parse
    out = parse(cif_buffer_with_metadata)

    assert out["chain_info"]["A"]["msa_path"] == Path("sequence_1.a3m")
    assert out["chain_info"]["B"]["msa_path"] == Path("sequence_2.a3m")


@pytest.mark.parametrize(
    "af3_cif_filename,pdb_id",
    [
        ("8cjg_from_af3.cif", "8cjg"),
        ("7ubd_from_af3.cif", "7ubd"),
    ],
)
def test_load_from_af3_output(af3_cif_filename, pdb_id, compressed_example):
    cif_path_af3 = compressed_example(TEST_DATA_IO / af3_cif_filename)
    cif_path_rcsb = get_pdb_path(pdb_id)

    # Parse the structure without CCD mirror path
    atom_array_from_af3 = parse(cif_path_af3, config=ParseConfig(hydrogen_policy="remove"))["assemblies"]["1"][0]
    atom_array_from_rcsb = parse(cif_path_rcsb, config=ParseConfig(hydrogen_policy="remove"))["assemblies"]["1"][0]

    assert len(atom_array_from_af3) == len(atom_array_from_rcsb), "Atom arrays are not the same length"

    # Ensure full occupancy from AF-3
    assert np.all(atom_array_from_af3.occupancy == 1)

    # Check that annotations match, where applicable (may have different chain ID's)
    assert len(np.unique(atom_array_from_af3.chain_id)) == len(np.unique(atom_array_from_rcsb.chain_id))
    assert np.array_equal(
        np.sort(np.unique(atom_array_from_af3.res_name)), np.sort(np.unique(atom_array_from_rcsb.res_name))
    )
    assert np.array_equal(
        np.sort(np.unique(atom_array_from_af3.atom_name)), np.sort(np.unique(atom_array_from_rcsb.atom_name))
    )


def _drop_empty_keys(d: dict) -> dict:
    """Recursively drop keys whose value is ``None`` or an empty dict."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            v = _drop_empty_keys(v)
            if not v:
                continue
        out[k] = v
    return out


def assert_data_dicts_equal(
    obtained_data_dict,
    expected_data_dict,
    compare_assemblies=True,
    asym_unit_annotations_to_compare=None,
    assembly_annotations_to_compare=None,
):
    # Copy both dicts to avoid modifying the originals
    # Deep copy is needed because we modify nested dicts (extra_info, metadata)
    obtained_data_dict = copy.deepcopy(obtained_data_dict)
    expected_data_dict = copy.deepcopy(expected_data_dict)

    obtained_asym_unit, expected_asym_unit = obtained_data_dict.pop("asym_unit"), expected_data_dict.pop("asym_unit")
    obtained_assemblies, expected_assemblies = (
        obtained_data_dict.pop("assemblies"),
        expected_data_dict.pop("assemblies"),
    )

    ensure_atom_array_stack(obtained_asym_unit)
    ensure_atom_array_stack(expected_asym_unit)
    assert len(obtained_asym_unit) == len(expected_asym_unit), "Asym unit stack depths do not match"
    for i in range(len(obtained_asym_unit)):
        assert_same_atom_array_or_stack(
            obtained_asym_unit[i],
            expected_asym_unit[i],
            annotations_to_compare=asym_unit_annotations_to_compare,
            enforce_order=False,  # Allow bond reordering
        )

    if compare_assemblies:
        for assembly_id, obtained_assembly in obtained_assemblies.items():
            expected_assembly = expected_assemblies[assembly_id]
            ensure_atom_array_stack(obtained_assembly)
            ensure_atom_array_stack(expected_assembly)
            assert len(obtained_assembly) == len(expected_assembly), "Asym unit stack depths do not match"
            for i in range(len(obtained_assembly)):
                assert_same_atom_array_or_stack(
                    obtained_assembly[i],
                    expected_assembly[i],
                    annotations_to_compare=assembly_annotations_to_compare,
                    enforce_order=False,  # Allow bond reordering
                )
    else:
        obtained_data_dict["extra_info"].pop("struct_oper_category", None)
        expected_data_dict["extra_info"].pop("struct_oper_category", None)
        obtained_data_dict["extra_info"].pop("assembly_gen_category", None)
        expected_data_dict["extra_info"].pop("assembly_gen_category", None)

    # Compare only common chain_info keys
    for chain_id in obtained_data_dict["chain_info"]:
        if chain_id in expected_data_dict["chain_info"]:
            common_keys = set(obtained_data_dict["chain_info"][chain_id].keys()) & set(
                expected_data_dict["chain_info"][chain_id].keys()
            )
            for data_dict in [obtained_data_dict, expected_data_dict]:
                data_dict["chain_info"][chain_id] = {k: data_dict["chain_info"][chain_id][k] for k in common_keys}

    # Drop None/empty values so missing-vs-None doesn't cause spurious failures
    obtained_data_dict = _drop_empty_keys(obtained_data_dict)
    expected_data_dict = _drop_empty_keys(expected_data_dict)

    assert_equal(obtained_data_dict, expected_data_dict, allow_extra_keys=True)


CONSISTENCY_TEST_CASES_FULL_DICT = [
    (get_pdb_path("1a1e"), TEST_DATA_IO / "1a1e_cif_data_dict.pkl"),
    (TEST_DATA_IO / "1qfe.pdb", TEST_DATA_IO / "1qfe_pdb_data_dict.pkl"),
]


@pytest.mark.parametrize("filepath, expected_data_dict_path", CONSISTENCY_TEST_CASES_FULL_DICT)
def test_parse_consistency_full_dict(filepath, expected_data_dict_path):
    """Compare the parsed structure to a reference structure (computed pre-refactor)."""
    obtained_data_dict = parse(
        filepath,
        config=ParseConfig(convert_mse_to_met=True, hydrogen_policy="remove", build_assembly=["1"]),
    )

    # Uncomment to update the regression test data
    # with open(expected_data_dict_path, "wb") as f:
    #     pickle.dump(obtained_data_dict, f)

    with open(expected_data_dict_path, "rb") as f:
        expected_data_dict = pickle.load(f)

    # Get annotations from expected, excluding ones we don't care about
    expected_asym_annots = set(expected_data_dict["asym_unit"][0].get_annotation_categories())
    annots_to_compare = expected_asym_annots - {"alt_atom_id", "uses_alt_atom_id"}

    assert_data_dicts_equal(
        obtained_data_dict,
        expected_data_dict,
        asym_unit_annotations_to_compare=annots_to_compare,
        assembly_annotations_to_compare=annots_to_compare,
    )


@pytest.fixture
def dict_inputs(compressed_example):
    """Fixture providing example chemical components for testing."""
    monomer = [
        {
            "seq": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
            "chain_type": "polypeptide(l)",
            "chain_id": "A",
            "is_polymer": True,
        }
    ]

    dimer = [
        {
            "seq": "MRDTDVTVLGLGLMGQALAGAFLKDGHATTVWNRSEGKAGQLAEQGAVLASSARDAAEASPLVVVCVSDHAAVRAVLDPLGDVLAGRVLVNLTSGTSEQARATAEWAAERGITYLDGAIMAIPQVVGTADAFLLYSGPEAAYEAHEPTLRSLGAGTTYLGADHGLSSLYDVALLGIMWGTLNSFLHGAALLGTAKVEATTFAPFANRWIEAVTGFVSAYAGQVDQGAYPALDATIDTHVATVDHLIHESEAAGVNTELPRLVRTLADRALAGGQGGLGYAAMIEQFRSPSA",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
            "chain_id": "B",
        },
        {
            "seq": "MRDTDVTVLGLGLMGQALAGAFLKDGHATTVWNRSEGKAGQLAEQGAVLASSARDAAEASPLVVVCVSDHAAVRAVLDPLGDVLAGRVLVNLTSGTSEQARATAEWAAERGITYLDGAIMAIPQVVGTADAFLLYSGPEAAYEAHEPTLRSLGAGTTYLGADHGLSSLYDVALLGIMWGTLNSFLHGAALLGTAKVEATTFAPFANRWIEAVTGFVSAYAGQVDQGAYPALDATIDTHVATVDHLIHESEAAGVNTELPRLVRTLADRALAGGQGGLGYAAMIEQFRSPSA",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
            "chain_id": "C",
        },
    ]

    noncanonical = [
        {
            "seq": "KVFGRCE(SEP)AAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
        }
    ]

    custom_residues = [
        {
            "seq": "G(C:0)G(SEP)G",
            "chain_type": "polypeptide(l)",
        }
    ]

    ligand = [
        {
            "smiles": "O=C1OCC(=C1)C5C4(C(O)CC3C(CCC2CC(O)CCC23C)C4(O)CC5)C",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "E",
        }
    ]

    glycan_1 = [
        {
            "ccd_code": "NAG",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "F",
        }
    ]
    glycan_2 = [
        {
            "ccd_code": "NAG",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "G",
        }
    ]

    sdf = [
        {
            "path": str(compressed_example(TEST_DATA_IO / "HEM_ideal.sdf")),
        }
    ]

    return {
        "monomer": monomer,
        "dimer": dimer,
        "noncanonical": noncanonical,
        "custom_residues": custom_residues,
        "ligand": ligand,
        "glycan_1": glycan_1,
        "glycan_2": glycan_2,
        "sdf": sdf,
    }


@pytest.fixture
def custom_residues():
    return {
        "C:0": {
            "path": f"{TEST_DATA_IO}/example_ncaa.cif",
        }
    }


def test_write_read_vs_parse_atom_array(dict_inputs, custom_residues, cleanup_registry):
    """Compare the write-read to/from CIF vs simply parsing an AtomArray."""

    custom_res_dict = custom_residues["C:0"]
    custom_component = ChemicalComponent.from_dict(custom_res_dict)
    register_custom_ccd_entry("C:0", custom_component.atom_array)

    # Parse input components, as in inference code
    components = sum(dict_inputs.values(), start=[])
    input_atom_array = components_to_atom_array(components, return_components=False)

    # Write-read, as was formerly done in the inference code
    with tempfile.TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        to_cif_file(input_atom_array, cif_path, include_nan_coords=True)

        config_kwargs = {"convert_mse_to_met": True, "hydrogen_policy": "remove"}

        parsed_from_cif = parse(cif_path, config=ParseConfig(**config_kwargs))

        # Directly parse the input AtomArray using new code
        parsed_from_atom_array = parse_atom_array(input_atom_array, config=PrepareConfig(**config_kwargs))

        # The asym_unit does not typically have iid annotations or transformation_id, but it will if parsing from an AtomArray that already had them
        # These annotations don't survive CIF roundtrips so we exclude them from comparison
        asym_unit_annotations_to_compare = [
            annot
            for annot in parsed_from_atom_array["asym_unit"].get_annotation_categories()
            if annot not in ("chain_iid", "pn_unit_iid", "molecule_iid", "transformation_id")
        ]

        # Subset chain_info to comparable keys (CIF has extra metadata)
        keys_to_match = {
            "entity",
            "chain_type",
            "is_polymer",
            "res_id",
            "res_name",
            "processed_entity_canonical_sequence",
            "processed_entity_non_canonical_sequence",
        }
        for chain_id in parsed_from_cif.get("chain_info", {}):
            common_keys = (
                set(parsed_from_cif["chain_info"][chain_id].keys())
                & set(parsed_from_atom_array["chain_info"][chain_id].keys())
            ) & keys_to_match
            parsed_from_cif["chain_info"][chain_id] = {
                k: parsed_from_cif["chain_info"][chain_id][k] for k in common_keys
            }
            parsed_from_atom_array["chain_info"][chain_id] = {
                k: parsed_from_atom_array["chain_info"][chain_id][k] for k in common_keys
            }

        # Compare everything including the subsetted chain_info
        assert_data_dicts_equal(
            parsed_from_cif,
            parsed_from_atom_array,
            compare_assemblies=False,
            asym_unit_annotations_to_compare=asym_unit_annotations_to_compare,
        )


def test_parse_preserves_atom_array_plus():
    """Test that parse_atom_array preserves AtomArrayPlus type and 2D annotations."""
    input_atom_array = parse(get_pdb_path("1out"), config=ParseConfig(add_missing_atoms=True))["assemblies"]["1"][0]
    input_atom_array = as_atom_array_plus(input_atom_array)

    # Add a test 2D annotation to verify it's preserved
    test_pairs = [(0, 1), (1, 2), (2, 3), (5, 10)]
    test_values = [1.5, 2.3, 3.7, 4.2]
    input_atom_array.set_annotation_2d("test_distances", test_pairs, test_values)

    output_atom_array = parse_atom_array(input_atom_array, config=PrepareConfig(add_missing_atoms=False))["asym_unit"][
        0
    ]
    assert isinstance(output_atom_array, AtomArrayPlus)

    # Verify 2D annotations are preserved
    assert "test_distances" in output_atom_array.get_annotation_2d_categories()
    annotation_2d = output_atom_array.get_annotation_2d("test_distances")
    assert np.array_equal(annotation_2d.pairs, np.array(test_pairs, dtype=np.int32))
    assert np.allclose(annotation_2d.values, test_values)


def _ccd_category_arrays(ccd_code: str, category: str) -> dict[str, np.ndarray]:
    """Return {column: array} for one CCD category."""
    block = _get_ccd_block(ccd_code, CCD_MIRROR_PATH)
    cat = block[category]
    return {key: cat[key].as_array() for key in cat}


def _assert_result_matches_ccd(
    result: dict,
    ccd_code: str,
    category: str,
    match_cols: tuple[str, ...],
    *,
    sort_by: str,
) -> None:
    """Assert result category matches CCD exactly on match_cols.

    Both sides are sorted by sort_by and each stable column is compared as a
    numpy array.  Row counts must match.
    """
    assert category in result, f"Category {category!r} missing from result"

    needed = set(match_cols) | {sort_by}
    res = {col: np.asarray(result[category][col]) for col in needed}
    ccd = _ccd_category_arrays(ccd_code, category)

    ccd_order = np.argsort(ccd[sort_by])
    res_order = np.argsort(res[sort_by])

    for col in match_cols:
        np.testing.assert_array_equal(
            ccd[col][ccd_order].astype(str),
            res[col][res_order].astype(str),
            err_msg=f"[{category}] column {col!r} mismatch",
        )


def test_build_chem_comp_atom():
    """_build_chem_comp_atom output matches CCD template columns."""
    ccd_code = "NAD"
    atom_array = atom_array_from_ccd_code(ccd_code)

    match_cols = ("comp_id", "type_symbol", "charge")

    result = _build_chem_comp_atom(atom_array)
    _assert_result_matches_ccd(result, ccd_code, "chem_comp_atom", match_cols, sort_by="atom_id")


CIF_PATHS = [TEST_DATA_IO / "example_distillation_output.cif"]


@pytest.mark.parametrize("path", CIF_PATHS)
def test_load_with_all_resolved(path: str, compressed_example):
    result = parse(
        filename=compressed_example(path),
        config=ParseConfig(add_missing_atoms=True, remove_ccds=(), hydrogen_policy="remove"),
    )
    # Check if processing runs through
    assert result is not None

    # Check if the extra metadata is present (from the custom `_extra_metadata` CIFCategory)
    assert result["metadata"]["extra_metadata"] is not None


def test_bcif_example(compressed_example):
    result = parse(
        filename=compressed_example(TEST_DATA_IO / "6lyz.bcif"),
    )
    # Check if processing runs through
    assert result is not None


def test_af2_predicted_pdb_example(compressed_example):
    result = parse(
        filename=compressed_example(TEST_DATA_IO / "UniRef50_A0A0S8JQ92_AF2_predicted.pdb"),
        config=ParseConfig(remove_waters=True, remove_ccds=()),
    )
    # Check if processing runs through
    assert result is not None


@pytest.mark.parametrize(
    "pdb_id,config_name",
    [("1ivo", "annotations_only"), ("1a8o", "lightweight")],
)
def test_prepare_atom_array(pdb_id: str, config_name: str):
    """Ensure that prepare_atom_array produces expected annotations."""
    data = parse(get_pdb_path(pdb_id))
    atoms = data["asym_unit"][0]
    result = prepare_atom_array(atoms, config=config_name)

    assert isinstance(result, struc.AtomArray)
    annots = result.get_annotation_categories()
    assert "atomic_number" in annots
    assert "pn_unit_id" in annots
    assert "molecule_id" in annots
