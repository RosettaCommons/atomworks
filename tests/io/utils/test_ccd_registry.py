"""Tests for custom CCD registry functionality."""

import logging

import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest

from atomworks.io import atom_array_to_file, parse
from atomworks.io.config import ParseConfig
from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.io.utils.ccd import (
    _get_base_ccd_template,
    atom_array_from_ccd_code,
    build_ccd_entries_from_cif_block,
    custom_ccd_residues,
    parse_ccd_cif,
    register_custom_ccd_entry,
)
from atomworks.io.utils.io_utils import CIFWriteConfig, read_any
from atomworks.io.utils.standardize import standardize_atom_names
from tests.conftest import TEST_DATA_DIR


def test_registry_basic_operations(cleanup_registry):
    """Registration and lookup, incl. overriding a real code whose template was already cached."""
    custom_ala = atom_array_from_ccd_code("ALA").copy()
    custom_ala.charge[:] = 99  # Marker to verify custom entry

    # Poison the memoized template cache with the bundled ALA before registering; the
    # registry-aware bypass in _standard_ccd_only_cache must still resolve the override.
    assert not np.all(_get_base_ccd_template("ALA", "", hydrogen_policy="keep").charge == 99)

    register_custom_ccd_entry("ALA", custom_ala)

    # Both the direct lookup and the memoized template path (add_annotations_from_ccd,
    # get_atom_names_for_residue) now resolve the custom entry.
    assert np.all(atom_array_from_ccd_code("ALA").charge == 99)
    assert np.all(_get_base_ccd_template("ALA", "", hydrogen_policy="keep").charge == 99)


def test_registry_standard_override_difference_logs_at_debug(cleanup_registry, caplog):
    """A differing override remains diagnosable without warning during ordinary parsing."""
    custom_ala = atom_array_from_ccd_code("ALA")
    custom_ala = custom_ala[custom_ala.atom_name != "OXT"]
    message = "Custom CCD entry 'ALA' overrides standard CCD definition"

    with caplog.at_level(logging.DEBUG, logger="atomworks.io.utils.ccd"):
        register_custom_ccd_entry("ALA", custom_ala)

    records = [record for record in caplog.records if message in record.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG


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

    # Bond-only overrides do not define custom atoms; normal name validation explains what is missing.
    del block["chem_comp_atom"]
    assert build_ccd_entries_from_cif_block(block) == {}
    with pytest.raises(ValueError, match="custom component.*chem_comp_atom and chem_comp_bond"):
        standardize_atom_names(entries["ATP"])


def test_unknown_peptide_without_component_metadata(tmp_path):
    """Unknown peptides without component metadata use the existing backbone-type inference."""
    atoms = atom_array_from_ccd_code("ALA")
    atoms.set_annotation("res_name", np.full(len(atoms), "UNKNOWN_PEPTIDE"))
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    path = tmp_path / "peptide.cif"
    atom_array_to_file(
        atoms, path, config=CIFWriteConfig(include_entity_categories=True, ccd_entries={"UNKNOWN_PEPTIDE": atoms})
    )
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
    atom_array_to_file(
        atoms, path, config=CIFWriteConfig(include_entity_categories=True, ccd_entries={ccd_code: atoms})
    )
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


@pytest.mark.parametrize("authored", [False, True])
def test_mixed_ccd_names_preserve_leaving_atoms_and_geometry(tmp_path, cleanup_registry, authored):
    """Legacy phosphate names coexist with current sugar names in older LCC files."""
    atoms = atom_array_from_ccd_code("LCC")
    expected_coords = atoms.coord.copy()
    expected_leaving = atoms.is_leaving_atom.copy()
    aliases = dict(zip(atoms.atom_name, atoms.alt_atom_id, strict=True))
    atoms.atom_name = np.array([aliases[name] if name in ("OP1", "OP2", "OP3") else name for name in atoms.atom_name])
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    if authored:
        atoms.is_leaving_atom[:] = False
    path = tmp_path / "legacy.cif"
    atom_array_to_file(atoms, path, config=CIFWriteConfig(ccd_entries={"LCC": atoms}))
    block = read_any(path).block
    if not authored:
        del block["chem_comp_atom"]["pdbx_leaving_atom_flag"]
    for column in list(block["chem_comp_atom"]):
        if "Cartn" in column:
            del block["chem_comp_atom"][column]
    entry = build_ccd_entries_from_cif_block(block, on_mismatch="ignore")["LCC"]
    np.testing.assert_array_equal(entry.atom_name, atoms.atom_name)
    np.testing.assert_array_equal(
        entry.is_leaving_atom, np.zeros_like(expected_leaving) if authored else expected_leaving
    )
    with custom_ccd_residues({"LCC": entry}):
        resolved = atom_array_from_ccd_code("LCC")
    np.testing.assert_allclose(resolved.coord, expected_coords)
    assert np.isnan(entry.coord).all()
    np.testing.assert_array_equal(resolved.bonds.as_array(), entry.bonds.as_array())


def test_alias_coordinates_do_not_override_different_chemistry(cleanup_registry):
    """A same-named component with different bonds is not a geometry template."""
    atoms = atom_array_from_ccd_code("LCC")
    atoms.atom_name[atoms.atom_name == "OP3"] = "O1P"
    atoms.coord[:] = np.nan
    a, b, _ = atoms.bonds.as_array()[0]
    atoms.bonds.remove_bond(int(a), int(b))
    with custom_ccd_residues({"LCC": atoms}):
        assert np.isnan(atom_array_from_ccd_code("LCC").coord).all()


def test_ccd_alias_matching_rejects_duplicate_targets():
    from atomworks.io.utils.ccd import _ccd_atom_indices

    atoms = atom_array_from_ccd_code("LCC")
    np.testing.assert_array_equal(_ccd_atom_indices(np.array(["OP3", "O1P"]), atoms), [-1, -1])
