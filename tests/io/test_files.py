"""`atom_array_to_file` and `atom_array_from_file` write and read a structure file without the parse pipeline."""

import io

import numpy as np
import pytest
from biotite.structure import AtomArray, BondType
from biotite.structure.io import pdb, pdbx

from atomworks.io import atom_array_from_file, atom_array_to_file, parse

from .conftest import TEST_DATA_IO


@pytest.fixture(scope="module")
def atoms():
    atoms = parse(TEST_DATA_IO / "6lyz.bcif", config="minimal")["asym_unit"][0]
    atoms.set_annotation("custom_score", np.arange(len(atoms), dtype=float))
    atoms.set_annotation("custom_mask", np.arange(len(atoms)) % 2 == 0)
    atoms.set_annotation("auth_asym_id", np.full(len(atoms), "author_chain"))
    atoms.set_annotation("auth_seq_id", atoms.res_id + 100)
    atoms.coord[0, 0] = 1.234567
    atoms.bonds.add_bond(0, len(atoms) - 1, BondType.COORDINATION)
    return atoms


@pytest.mark.parametrize("file_type", ["cif", "bcif", "cif.gz", "cif.gzip", "cif.zst", "bcif.zst"])
def test_file_content_and_buffer_round_trip(atoms, tmp_path, file_type):
    """Round-trip coordinates and custom fields through paths/buffers; use Biotite's generated author IDs."""
    path = atom_array_to_file(atoms, tmp_path / f"out.{file_type}")
    content = atom_array_to_file(atoms, file_type=file_type)
    destination = io.StringIO() if file_type == "cif" else io.BytesIO()
    assert atom_array_to_file(atoms, destination, file_type=file_type) is None
    sources = [path]
    for name, payload in {"content": content, "buffer": destination.getvalue()}.items():
        content_path = tmp_path / f"{name}.{file_type}"
        content_path.write_bytes(payload.encode() if isinstance(payload, str) else payload)
        sources.append(content_path)
    if file_type in ("cif", "bcif"):
        destination.seek(0)
        sources.append(destination)
    for source in sources:
        again = atom_array_from_file(source)
        np.testing.assert_array_equal(again.coord, atoms.coord)
        assert set(map(tuple, again.bonds.as_array())) == set(map(tuple, atoms.bonds.as_array()))
        np.testing.assert_array_equal(again.atom_name, atoms.atom_name)
        np.testing.assert_array_equal(again.custom_score, atoms.custom_score)
        assert again.custom_mask.dtype == bool
        np.testing.assert_array_equal(again.custom_mask, atoms.custom_mask)
        np.testing.assert_array_equal(again.auth_asym_id, atoms.chain_id)
        np.testing.assert_array_equal(again.auth_seq_id, atoms.res_id)


@pytest.mark.parametrize("different_atoms", [False, True])
def test_a_model_can_be_selected(atoms, tmp_path, different_atoms):
    """Follow parser model handling and select individual models by positive or negative position."""
    first = atoms.copy()
    second = atoms[:-1].copy() if different_atoms else atoms.copy()
    second.charge[0] = 2
    second.custom_score += 10
    path = atom_array_to_file([first, second], tmp_path / "two.cif", model_ids=[5, 12])
    loaded = atom_array_from_file(path)
    assert isinstance(loaded, list) == different_atoms
    assert len(loaded) == 2
    # Equal-length models inherit the parser's shared annotations from the first model.
    expected_models = [first, second] if different_atoms else [first, first]
    for actual, expected in zip(loaded, expected_models, strict=True):
        np.testing.assert_array_equal(actual.custom_score, expected.custom_score)
        np.testing.assert_array_equal(actual.charge, expected.charge)
        np.testing.assert_array_equal(actual.b_factor, expected.b_factor)
        np.testing.assert_array_equal(actual.atom_id, actual.id)
        np.testing.assert_array_equal(actual.auth_asym_id, expected.chain_id)
    for model in (2, -1):
        selected = atom_array_from_file(path, model=model, squeeze=False)
        np.testing.assert_array_equal(selected.custom_score, second.custom_score)
        np.testing.assert_array_equal(selected.charge, second.charge)
        assert np.all(selected.pdbx_PDB_model_num == 12)


def test_other_file_types_are_refused(tmp_path):
    """Reject unsupported input and output formats."""
    with pytest.raises(ValueError, match="atomworks.io.parse"):
        atom_array_from_file(tmp_path / "x.sdf", "sdf")
    with pytest.raises(ValueError, match="Could not infer file type"):
        atom_array_to_file(AtomArray(0), file_type="bogus")


@pytest.mark.parametrize("file_type", ["cif", "pdb"])
def test_read_uses_minimal_parser_conventions(file_type):
    """Select the first conformer and use the minimal parser's identifier conventions."""
    buffer = io.StringIO(
        "ATOM      1  CA AALA A   1B      0.000   1.000   2.000  1.00  0.00           C\n"
        "ATOM      2  CA BALA A   1B      3.000   4.000   5.000  1.00  0.00           C\n"
        "HETATM    3  O   HOH A   2B      6.000   7.000   8.000  1.00  0.00           O\n"
    )
    if file_type == "cif":
        atoms = pdb.PDBFile.read(buffer).get_structure(model=1, altloc="all")
        file = pdbx.CIFFile()
        pdbx.set_structure(file, atoms)
        file.block["atom_site"]["label_alt_id"] = ["A", "B", "."]
        buffer = io.StringIO()
        file.write(buffer)
        buffer.seek(0)
    loaded = atom_array_from_file(buffer, file_type)
    np.testing.assert_array_equal(loaded.coord, np.arange(9).reshape(3, 3)[[0, 2]])
    np.testing.assert_array_equal(loaded.chain_id, ["A", "A"] if file_type == "cif" else ["A", "B"])
    np.testing.assert_array_equal(loaded.ins_code, [""] * 2 if file_type == "cif" else ["B"] * 2)
