"""Tests for custom CCD registry functionality."""

import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest

from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.tools.rdkit import atom_array_to_rdkit, ccd_code_to_rdkit
from atomworks.io.utils.ccd import (
    _get_base_ccd_template,
    atom_array_from_ccd_code,
    build_ccd_entries_from_cif_block,
    clear_custom_ccd_registry,
    custom_ccd_residues,
    get_atom_names_for_residue,
    get_polymerization_atoms,
    parse_ccd_cif,
    register_custom_ccd_entry,
    unregister_custom_ccd_entry,
)
from atomworks.io.utils.io_utils import read_any, to_cif_file
from atomworks.io.utils.link_chemistry import get_chem_comp_leaving_atom_groups
from tests.conftest import TEST_DATA_DIR


def test_registry_basic_operations(cleanup_registry):
    """Registration and lookup, incl. overriding a real code whose template was already cached."""
    custom_ala = atom_array_from_ccd_code("ALA").copy()
    custom_ala.charge[:] = 99  # Marker to verify custom entry

    # Populate the standard template cache before registering an override.
    assert not np.all(_get_base_ccd_template("ALA", "", hydrogen_policy="keep").charge == 99)

    register_custom_ccd_entry("ALA", custom_ala)

    # Both the direct lookup and the memoized template path (add_annotations_from_ccd,
    # get_atom_names_for_residue) now resolve the custom entry.
    assert np.all(atom_array_from_ccd_code("ALA").charge == 99)
    assert np.all(_get_base_ccd_template("ALA", "", hydrogen_policy="keep").charge == 99)


@pytest.mark.parametrize("code", ["ALA", "CUSTOM_CACHE_TEST"])
def test_registry_cached_chemistry_tracks_replacement_and_nested_contexts(cleanup_registry, code):
    """Derived templates and leaving groups follow registry changes and exception restoration."""
    ala = atom_array_from_ccd_code("ALA")
    gly = atom_array_from_ccd_code("GLY")
    gly.is_leaving_atom[:] = False
    gly.charge[:] = 2
    gly.chem_comp_type[:] = "NON-POLYMER"

    with custom_ccd_residues({code: ala}):
        expected_groups = get_chem_comp_leaving_atom_groups(code)
        assert expected_groups
        assert "CB" in get_atom_names_for_residue(code, "")[0]
        assert get_polymerization_atoms(code) == ("C", "N")
        with (
            pytest.raises(RuntimeError, match="restore registry"),
            custom_ccd_residues({code: gly}),
        ):
            assert get_chem_comp_leaving_atom_groups(code) == {}
            assert "CB" not in get_atom_names_for_residue(code, "")[0]
            assert np.all(_get_base_ccd_template(code, "", "keep").charge == 2)
            assert get_polymerization_atoms(code) == (None, None)
            raise RuntimeError("restore registry")
        assert get_chem_comp_leaving_atom_groups(code) == expected_groups
        assert "CB" in get_atom_names_for_residue(code, "")[0]
        assert get_polymerization_atoms(code) == ("C", "N")

        register_custom_ccd_entry(code, gly)
        assert get_chem_comp_leaving_atom_groups(code) == {}
        assert "CB" not in get_atom_names_for_residue(code, "")[0]
        assert unregister_custom_ccd_entry(code)
        assert ("CB" in get_atom_names_for_residue(code, "")[0]) == (code == "ALA")
        register_custom_ccd_entry(code, gly)
        assert get_chem_comp_leaving_atom_groups(code) == {}
        clear_custom_ccd_registry()
        assert ("CB" in get_atom_names_for_residue(code, "")[0]) == (code == "ALA")


def test_registry_leaving_groups_reuse_work_without_sharing_mutable_input(cleanup_registry, monkeypatch):
    """Repeated chemistry lookups reuse graph work while registered inputs remain isolated."""
    from atomworks.io.utils import link_chemistry

    link_chemistry._get_chem_comp_leaving_atom_groups_cached.cache_clear()
    entry = atom_array_from_ccd_code("ALA")
    register_custom_ccd_entry("ALA", entry)
    original = link_chemistry._find_connected_components_after_removal
    traversals = 0

    def count_traversals(graph, node):
        nonlocal traversals
        traversals += 1
        return original(graph, node)

    monkeypatch.setattr(link_chemistry, "_find_connected_components_after_removal", count_traversals)
    groups = get_chem_comp_leaving_atom_groups("ALA")
    first_traversals = traversals
    assert first_traversals > 0
    unregister_custom_ccd_entry("ALA")
    register_custom_ccd_entry("ALA", entry.copy())
    assert get_chem_comp_leaving_atom_groups("ALA") == groups
    assert traversals == first_traversals
    entry.is_leaving_atom[:] = False
    returned = atom_array_from_ccd_code("ALA")
    returned.is_leaving_atom[:] = False
    assert get_chem_comp_leaving_atom_groups("ALA") == groups
    assert traversals == first_traversals
    register_custom_ccd_entry("ALA", entry)
    assert get_chem_comp_leaving_atom_groups("ALA") == {}


def test_registry_rdkit_fallback_with_problematic_cif(cleanup_registry):
    """Test registry works with CIF that fails RDKit sanitization (H:A).

    H:A is 3-hydroxy-3-carboxy-adipic acid - RDKit sanitization fails
    with AtomValenceException, but ideal coordinates from CIF should
    still be usable via the registry.
    """
    # Parse H:A.cif from test data
    custom_cif_path = TEST_DATA_DIR / "io" / "H_A.cif"
    cif = pdbx.CIFFile.read(str(custom_cif_path))
    custom_ligand = parse_ccd_cif(cif, coords="ideal_pdbx")

    # Verify structure is valid (14 atoms: 7 carbons + 7 oxygens)
    assert len(custom_ligand) == 14
    assert custom_ligand.res_name[0] == "H:A"
    assert not np.any(np.isnan(custom_ligand.coord))

    # Verify coordinates are valid (not all zeros, reasonable values)
    assert not np.all(custom_ligand.coord == 0)
    coord_range = np.ptp(custom_ligand.coord, axis=0)  # peak-to-peak (max - min)
    assert np.all(coord_range > 1.0), "Coordinates should span reasonable distances"

    # Register in global registry
    register_custom_ccd_entry("H:A", custom_ligand)

    # Retrieve via standard lookup
    result = atom_array_from_ccd_code("H:A")
    assert result is not None
    assert len(result) == 14

    # Verify retrieved coordinates are valid (not NaN, not zeros)
    assert not np.any(np.isnan(result.coord))
    assert not np.all(result.coord == 0)

    # Verify coordinates match original (preserved through registry)
    np.testing.assert_allclose(result.coord, custom_ligand.coord, rtol=1e-5)

    # Test RDKit conversion (will fail sanitization but shouldn't crash)
    mol = atom_array_to_rdkit(result, sanitize=True, attempt_fixing_corrupted_molecules=True)

    # Verify molecule was created despite sanitization failure
    assert mol is not None
    assert mol.GetNumAtoms() > 0

    # Verify coordinates are preserved in the molecule
    conformer = mol.GetConformer()
    assert conformer.GetNumAtoms() == 14

    # Extract RDKit coordinates and verify they're valid
    rdkit_coords = np.array([[pos.x, pos.y, pos.z] for pos in [conformer.GetAtomPosition(i) for i in range(14)]])
    assert not np.any(np.isnan(rdkit_coords))
    assert not np.all(rdkit_coords == 0)
    np.testing.assert_allclose(rdkit_coords, custom_ligand.coord, rtol=1e-5)


def test_build_ccd_entries_collision_policy(cleanup_registry):
    """A custom ligand reusing a real CCD code (ATP + foreign atom ZZ9): strict raises, lenient keeps it."""
    block = pdbx.CIFBlock(
        {
            "chem_comp": pdbx.CIFCategory({"id": ["ATP"], "type": ["NON-POLYMER"]}),
            "chem_comp_atom": pdbx.CIFCategory(
                {"comp_id": ["ATP"] * 3, "atom_id": ["C1", "C2", "ZZ9"], "type_symbol": ["C", "C", "O"]}
            ),
            "chem_comp_bond": pdbx.CIFCategory(
                {
                    "comp_id": ["ATP", "ATP"],
                    "atom_id_1": ["C1", "C1"],
                    "atom_id_2": ["C2", "ZZ9"],
                    "value_order": ["SING", "SING"],
                    "pdbx_aromatic_flag": ["N", "N"],
                    "pdbx_stereo_config": ["N", "N"],
                }
            ),
        }
    )

    for options in ({}, {"on_mismatch": "error_heavy"}, {"on_mismatch": "error"}):
        with pytest.raises(ValueError):
            build_ccd_entries_from_cif_block(block, **options)

    entries = build_ccd_entries_from_cif_block(block, on_mismatch="ignore")
    assert set(map(str, entries["ATP"].atom_name)) == {"C1", "C2", "ZZ9"}


def test_unknown_peptide_without_component_metadata(tmp_path):
    """Unknown peptides without component metadata use the existing backbone-type inference."""
    atoms = atom_array_from_ccd_code("ALA")
    atoms.set_annotation("res_name", np.full(len(atoms), "UNKNOWN_PEPTIDE"))
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    path = tmp_path / "peptide.cif"
    to_cif_file(atoms, path, ccd_entries={"UNKNOWN_PEPTIDE": atoms})
    cif = read_any(path)
    del cif.block["chem_comp"]
    cif.write(path)
    parsed = parse(path, config="minimal")["asym_unit"][0]
    assert np.all(parsed.chem_comp_type == "L-PEPTIDE LINKING")
    assert set(parsed.atom_name) == set(atoms.atom_name)


def test_cif_ccd_registry_preserves_ideal_coords(cleanup_registry):
    """A CIF-derived registry entry must not shadow the bundled component's ideal coords."""
    src = TEST_DATA_DIR / "io" / "101m_arginine_nh1nh2_swapped.cif"  # myoglobin: HEM, no in-file ideal coords
    plus = parse(src, config=ParseConfig.from_preset("rcsb", return_atom_array_plus=True))["asym_unit"]

    with custom_ccd_residues(plus._custom_ccd_registry):
        assert not np.isnan(atom_array_from_ccd_code("HEM").coord).any()
        assert np.isnan(atom_array_from_ccd_code("HEM", coords=None).coord).all()


@pytest.mark.parametrize("ccd_code", ["PO4", "PHE"])
@pytest.mark.parametrize("authored", [True, False])
def test_authored_annotations_during_atom_completion(tmp_path, ccd_code, authored):
    """Rebuilding keeps authored zero charges/false flags and supplements omitted columns."""
    # Capture CCD defaults, then author explicit zero charges and false flags.
    atoms = atom_array_from_ccd_code(ccd_code)
    ccd_annotations = {name: atoms.get_annotation(name).copy() for name in ("charge", "is_aromatic", "is_leaving_atom")}
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    atoms.charge[:] = 0
    atoms.is_aromatic[:] = False
    atoms.is_leaving_atom[:] = False

    # Write the component with either authored columns or columns omitted for supplementation.
    path = tmp_path / "component.cif"
    to_cif_file(atoms, path, ccd_entries={ccd_code: atoms})
    if not authored:
        cif = read_any(path)
        for column in ("charge", "pdbx_aromatic_flag", "pdbx_leaving_atom_flag"):
            del cif.block["chem_comp_atom"][column]
        cif.write(path)

    # Rebuild and check both the component template and the retained output annotations.
    parsed = parse(path, config=ParseConfig.from_preset("minimal", add_missing_atoms=True))["asym_unit"][0]
    template = build_ccd_entries_from_cif_block(read_any(path).block)[ccd_code]
    assert len(parsed) == len(atoms)
    for name, values in ccd_annotations.items():
        expected_values = np.zeros_like(values) if authored else values
        assert np.array_equal(template.get_annotation(name), expected_values)
        if name != "is_leaving_atom":  # Preparation removes this temporary annotation.
            assert np.array_equal(parsed.get_annotation(name), expected_values)


def test_registry_coordinate_caches_preserve_definitions_and_return_copies(cleanup_registry):
    """Bundled coordinate filling and RDKit conversion isolate outputs and respect replacement."""
    expected = atom_array_from_ccd_code("ALA").coord.copy()
    entry = atom_array_from_ccd_code("ALA", coords=None)
    with custom_ccd_residues({"ALA": entry}):
        returned = atom_array_from_ccd_code("ALA")
        np.testing.assert_array_equal(returned.coord, expected)
        returned.coord[:] = 99
        mol = ccd_code_to_rdkit("ALA")
        np.testing.assert_array_equal(mol.GetConformer().GetPositions(), expected)
        mol.GetConformer().SetAtomPosition(0, (99, 99, 99))
        np.testing.assert_array_equal(ccd_code_to_rdkit("ALA").GetConformer().GetPositions(), expected)
    with custom_ccd_residues({"ALA": entry}):
        np.testing.assert_array_equal(atom_array_from_ccd_code("ALA").coord, expected)
    entry.coord = expected + 1
    with custom_ccd_residues({"ALA": entry}):
        np.testing.assert_array_equal(ccd_code_to_rdkit("ALA").GetConformer().GetPositions(), entry.coord)


def test_registry_accepts_unserializable_custom_annotations(cleanup_registry):
    entry = atom_array_from_ccd_code("ALA")
    entry.set_annotation("callback", np.full(len(entry), lambda: None, dtype=object))
    with custom_ccd_residues({"CUSTOM_CACHE_TEST": entry}):
        assert "CB" in get_atom_names_for_residue("CUSTOM_CACHE_TEST", "")[0]
        actual = atom_array_from_ccd_code("CUSTOM_CACHE_TEST")
        assert actual.callback[0] is entry.callback[0]
