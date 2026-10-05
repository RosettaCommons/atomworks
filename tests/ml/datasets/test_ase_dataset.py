"""Tests for AseDBDataset."""

import numpy as np
import pandas as pd
import pytest
from biotite.structure import AtomArray

pytest.importorskip("ase")
pytest.importorskip("ase.db")
pytest.importorskip("ase_db_backends", reason="ase_db_backends required for AseDBDataset tests")

import ase
import ase.db

from atomworks.ml.datasets.ase_dataset import AseDBDataset  # noqa: E402


def _loader(raw_data: tuple) -> dict:
    atoms_row, _, _metadata_row = raw_data
    atoms = atoms_row.toatoms()
    atom_array = AtomArray(len(atoms))
    atom_array.coord = atoms.get_positions()
    atom_array.element = np.array(atoms.get_chemical_symbols())
    return {"atom_array": atom_array}


@pytest.fixture
def ase_db(tmp_path):
    db_path = tmp_path / "test.aselmdb"
    db = ase.db.connect(str(db_path))
    molecules = [
        ase.Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        ase.Atoms("CH4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]),
    ]
    for mol in molecules:
        db.write(mol)
    parquet_path = tmp_path / "meta.parquet"
    pd.DataFrame({"example_id": ["water", "methane"], "lmdb_idx": [0, 1]}).to_parquet(parquet_path)
    return str(db_path), str(parquet_path), molecules


def test_with_metadata(ase_db):
    db_path, parquet_path, molecules = ase_db
    with AseDBDataset(lmdb_path=db_path, name="test", data=parquet_path, loader=_loader) as ds:
        assert len(ds) == 2
        assert ds[0]["example_id"] == "water"
        assert len(ds[0]["atom_array"]) == len(molecules[0])
        assert ds.id_to_idx("methane") == 1
        assert "water" in ds


def test_without_metadata(ase_db):
    db_path, _, molecules = ase_db
    with AseDBDataset(lmdb_path=db_path, name="test", loader=_loader) as ds:
        assert len(ds) == 2
        assert ds[0]["example_id"] == "0"
        assert len(ds[1]["atom_array"]) == len(molecules[1])
        assert "0" in ds
        assert ds.id_to_idx("1") == 1
