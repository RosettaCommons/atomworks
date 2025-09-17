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

from atomworks.io.parser import parse, parse_atom_array
from atomworks.io.tools.inference import build_msa_paths_by_chain_id_from_component_list, components_to_atom_array
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.io_utils import (
    get_structure,
    infer_pdb_file_type,
    load_any,
    read_any,
    to_cif_buffer,
    to_cif_file,
    to_cif_string,
    to_pdb_string,
)
from atomworks.io.utils.testing import assert_same_atom_array, get_pdb_path
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
        ("6lyz", "bcif", True),
        ("6lyz", "bcif", False),
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

    with open(TEST_DATA_IO / "6lyz.bcif", "rb") as f:
        buffer = io.BytesIO(f.read())
        assert infer_pdb_file_type(buffer) == "bcif"

    with open(TEST_DATA_IO / "1a8o_modified.cif") as f:
        buffer = io.StringIO(f.read())
        assert infer_pdb_file_type(buffer) == "cif"

    with open(TEST_DATA_IO / "UniRef50_A0A0S8JQ92_AF2_predicted.pdb") as f:
        buffer = io.StringIO(f.read())
        assert infer_pdb_file_type(buffer) == "pdb"


@pytest.mark.requires_internet
@skip_if_no_internet
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
    # Fetch 6lyz.cif as a buffer
    cif_buffer = rcsb.fetch("6lyz", "cif")

    # Read the buffer into a CIFFile object
    cif_file = read_any(cif_buffer, file_type="cif")

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


@pytest.mark.requires_internet
@skip_if_no_internet
def test_parse_atom_array_with_multiple_transformations():
    data_dict = parse(rcsb.fetch("1out", "cif"), file_type="cif", add_missing_atoms=True)
    parsed_from_file = data_dict["assemblies"]["1"]
    assert any(chain_iid.endswith("_2") for chain_iid in np.unique(parsed_from_file.chain_iid))
    new_data_dict = parse_atom_array(parsed_from_file, add_missing_atoms=True)
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


@pytest.mark.requires_internet
@skip_if_no_internet
def test_to_cif_string():
    cif_buffer = rcsb.fetch("6lyz", "cif")
    cif_structure = load_any(cif_buffer, file_type="cif", extra_fields=["b_factor", "occupancy", "charge"])

    with pytest.raises(ValueError, match="Ambiguous bond annotations detected"):
        cif_string = to_cif_string(cif_structure)

    # Make identifiers unique
    cif_structure.res_id = struc.spread_residue_wise(cif_structure, np.arange(struc.get_residue_count(cif_structure)))
    # ... drop HOH
    cif_structure = cif_structure[0, cif_structure.res_name != "HOH"]
    cif_string = to_cif_string(cif_structure)

    assert isinstance(cif_string, str)
    assert len(cif_string) > 0

    cif_structure2 = load_any(
        io.StringIO(cif_string),
        file_type="cif",
        extra_fields=["b_factor", "occupancy", "charge"],
    )
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
        id="test_id",
        extra_categories=metadata,
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


@pytest.mark.requires_internet
@skip_if_no_internet
def test_to_pdb_string():
    pdb_buffer = rcsb.fetch("6lyz", "pdb")
    pdb_structure = load_any(pdb_buffer, file_type="pdb", extra_fields=["b_factor", "occupancy", "charge"], model=1)
    n_atoms = pdb_structure.array_length()
    pdb_string = to_pdb_string(pdb_structure)
    assert isinstance(pdb_string, str)
    assert len(pdb_string) > 0

    # Test that we can load the pdb string back into an AtomArray
    pdb_buffer2 = io.StringIO(pdb_string)
    pdb_structure2 = load_any(pdb_buffer2, file_type="pdb", extra_fields=["b_factor", "occupancy", "charge"], model=1)
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
        id="test_inject_msa",
        extra_categories={"msa_paths_by_chain_id": msa_paths_by_chain_id},
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
def test_load_from_af3_output(af3_cif_filename, pdb_id):
    cif_path_af3 = TEST_DATA_IO / af3_cif_filename
    cif_path_rcsb = get_pdb_path(pdb_id)

    # Parse the structure without CCD mirror path
    atom_array_from_af3 = parse(cif_path_af3, hydrogen_policy="remove")["assemblies"]["1"][0]
    atom_array_from_rcsb = parse(cif_path_rcsb, hydrogen_policy="remove")["assemblies"]["1"][0]

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


def assert_data_dicts_equal(
    obtained_data_dict,
    expected_data_dict,
    ignore_metadata_id=False,
    compare_assemblies=True,
    asym_unit_annotations_to_compare=None,
    assembly_annotations_to_compare=None,
    compare_box=True,
):
    obtained_asym_unit, expected_asym_unit = obtained_data_dict.pop("asym_unit"), expected_data_dict.pop("asym_unit")
    obtained_assemblies, expected_assemblies = (
        obtained_data_dict.pop("assemblies"),
        expected_data_dict.pop("assemblies"),
    )

    ensure_atom_array_stack(obtained_asym_unit)
    ensure_atom_array_stack(expected_asym_unit)
    assert len(obtained_asym_unit) == len(expected_asym_unit), "Asym unit stack depths do not match"
    for i in range(len(obtained_asym_unit)):
        assert_same_atom_array(
            obtained_asym_unit[i], expected_asym_unit[i], annotations_to_compare=asym_unit_annotations_to_compare
        )

    if compare_assemblies:
        for assembly_id, obtained_assembly in obtained_assemblies.items():
            expected_assembly = expected_assemblies[assembly_id]
            ensure_atom_array_stack(obtained_assembly)
            ensure_atom_array_stack(expected_assembly)
            assert len(obtained_assembly) == len(expected_assembly), "Asym unit stack depths do not match"
            for i in range(len(obtained_assembly)):
                assert_same_atom_array(
                    obtained_assembly[i], expected_assembly[i], annotations_to_compare=assembly_annotations_to_compare
                )
    else:
        obtained_data_dict["extra_info"].pop("struct_oper_category", None)
        expected_data_dict["extra_info"].pop("struct_oper_category", None)
        obtained_data_dict["extra_info"].pop("assembly_gen_category", None)
        expected_data_dict["extra_info"].pop("assembly_gen_category", None)

    if ignore_metadata_id:
        obtained_data_dict["metadata"].pop("id")
        expected_data_dict["metadata"].pop("id")

    assert obtained_data_dict == expected_data_dict


CONSISTENCY_TEST_CASES_FULL_DICT = [
    (get_pdb_path("1a1e"), TEST_DATA_IO / "1a1e_cif_data_dict.pkl"),
    (TEST_DATA_IO / "1qfe.pdb", TEST_DATA_IO / "1qfe_pdb_data_dict.pkl"),
]


@pytest.mark.parametrize("filepath, expected_data_dict_path", CONSISTENCY_TEST_CASES_FULL_DICT)
def test_parse_consistency_full_dict(filepath, expected_data_dict_path):
    """
    Compare the parsed structure to a reference structure (computed pre-refactor).
    """
    parse_kwargs = {
        "convert_mse_to_met": True,
        "hydrogen_policy": "remove",
        "build_assembly": ["1"],
    }

    obtained_data_dict = parse(filepath, **parse_kwargs)

    # Uncomment to update the regression test data
    # with open(expected_data_dict_path, "wb") as f:
    #     pickle.dump(obtained_data_dict, f)

    with open(expected_data_dict_path, "rb") as f:
        expected_data_dict = pickle.load(f)

    assert_data_dicts_equal(obtained_data_dict, expected_data_dict)


@pytest.fixture
def dict_inputs():
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
            "path": f"{TEST_DATA_IO}/HEM_ideal.sdf",
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


def test_write_read_vs_parse_atom_array(dict_inputs, custom_residues):
    """Compare the write-read to/from CIF vs simply parsing an AtomArray."""

    # Parse input components, as in inference code
    components = sum(dict_inputs.values(), start=[])
    input_atom_array = components_to_atom_array(components, return_components=False, custom_residues=custom_residues)

    # Write-read, as was formerly done in the inference code
    with tempfile.TemporaryDirectory() as temp_dir:
        cif_path = Path(temp_dir) / "test.cif"
        to_cif_file(input_atom_array, cif_path, include_nan_coords=True)

        parse_kwargs = {
            "convert_mse_to_met": True,
            "hydrogen_policy": "remove",
            "build_assembly": ["1"],
        }

        parsed_from_cif = parse(cif_path, **parse_kwargs)

        # Directly parse the input AtomArray using new code
        parsed_from_atom_array = parse_atom_array(input_atom_array, **parse_kwargs)

        # The asym_unit does not typically have a chain_iid, but it will if parsing from an AtomArray that already had it
        asym_unit_annotations_to_compare = [
            annot for annot in parsed_from_atom_array["asym_unit"].get_annotation_categories() if annot != "chain_iid"
        ]

        # Check for equivalence, allowing differences only in the metadata ID and the assemblies (parsing an AtomArray
        # directly does not build assemblies)
        assert_data_dicts_equal(
            parsed_from_cif,
            parsed_from_atom_array,
            ignore_metadata_id=True,
            compare_assemblies=False,
            asym_unit_annotations_to_compare=asym_unit_annotations_to_compare,
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
