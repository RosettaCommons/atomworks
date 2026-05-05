import pickle

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("ase_db_backends")

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from atomworks.ml.datasets import ASELMDBDataset, LMDBDataset
from atomworks.ml.datasets.loaders import create_ase_atoms_loader


def _write_lmdb(path, rows):
    """Write rows to an ASE LMDB database."""
    with connect(path, type="aselmdb", append=True) as db:
        for row in rows:
            atoms = Atoms(
                row["symbols"],
                positions=row["positions"],
                cell=row.get("cell", [10.0, 10.0, 10.0]),
                pbc=row.get("pbc", [False, False, False]),
            )
            atoms.set_initial_charges(row.get("charges", np.zeros(len(atoms))))
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=row["energy"],
                forces=np.asarray(row["forces"], dtype=float),
            )
            db.write(
                atoms,
                key_value_pairs={"sid": row["sid"], "split": row["split"]},
                data={"charge": row["charge"], "spin": row["spin"]},
            )


@pytest.fixture()
def ase_lmdb_paths(tmp_path):
    """Create two ASE LMDB shards with example records and return their paths."""
    shard_a = tmp_path / "train_a.aselmdb"
    shard_b = tmp_path / "train_b.aselmdb"
    _write_lmdb(
        shard_a,
        [
            {
                "sid": "omol-0",
                "split": "train",
                "symbols": "H2O",
                "positions": [(0, 0, 0), (0, 0, 1), (1, 0, 0)],
                "forces": np.zeros((3, 3)),
                "energy": -1.0,
                "charge": 0,
                "spin": 1,
            },
            {
                "sid": "omol-1",
                "split": "train",
                "symbols": "CO",
                "positions": [(0, 0, 0), (0, 0, 1.2)],
                "forces": np.ones((2, 3)),
                "energy": -2.0,
                "charge": -1,
                "spin": 2,
            },
        ],
    )
    _write_lmdb(
        shard_b,
        [
            {
                "sid": "omol-2",
                "split": "val",
                "symbols": "NaCl",
                "positions": [(0, 0, 0), (0, 0, 2.4)],
                "forces": np.full((2, 3), 2.0),
                "energy": -3.0,
                "charge": 1,
                "spin": 1,
            }
        ],
    )
    return [shard_a, shard_b]


def test_ase_lmdb_dataset_reads_records_and_generated_ids(ase_lmdb_paths):
    """Test that ASELMDBDataset can read records from multiple shards, generate example IDs, and access records by index and ID."""
    dataset = ASELMDBDataset(paths=ase_lmdb_paths, name="omol")

    assert LMDBDataset is ASELMDBDataset
    assert len(dataset) == 3

    example_id = dataset.idx_to_id(0)
    assert example_id == "train_a:1"
    assert example_id in dataset
    assert dataset.id_to_idx(example_id) == 0

    record = dataset[0]
    assert record["example_id"] == example_id
    assert record["db_path"] == ase_lmdb_paths[0]
    assert record["db_index"] == 0
    assert record["db_id"] == 1
    assert record["atoms"].get_chemical_formula() == "H2O"
    assert record["atoms"].info["sid"] == "omol-0"
    assert record["atoms"].info["charge"] == 0
    assert record["key_value_pairs"] == {"sid": "omol-0", "split": "train"}
    assert record["data"] == {"charge": 0, "spin": 1}
    assert record["calculator_results"]["energy"] == -1.0
    np.testing.assert_array_equal(record["calculator_results"]["forces"], np.zeros((3, 3)))

    dataset.close()


def test_ase_lmdb_dataset_can_index_metadata_ids(ase_lmdb_paths):
    """Test that ASELMDBDataset can index records by metadata IDs."""
    dataset = ASELMDBDataset(
        paths=ase_lmdb_paths,
        name="omol",
        example_id_key="sid",
        build_id_index=True,
    )

    assert dataset.idx_to_id(2) == "omol-2"
    assert dataset.id_to_idx("omol-2") == 2
    assert dataset[2]["example_id"] == "omol-2"

    dataset.close()


def test_ase_lmdb_dataset_atoms_return_type_and_pickle(ase_lmdb_paths):
    """Test that ASELMDBDataset can return Atoms objects and that the dataset can be pickled."""
    dataset = ASELMDBDataset(paths=ase_lmdb_paths[0], name="omol", return_type="atoms")

    atoms = dataset[1]
    assert atoms.get_chemical_formula() == "CO"
    assert atoms.info["example_id"] == "train_a:2"
    assert atoms.info["spin"] == 2

    unpickled = pickle.loads(pickle.dumps(dataset))
    assert len(unpickled) == 2
    assert unpickled[0].get_chemical_formula() == "H2O"

    dataset.close()
    unpickled.close()


def test_ase_atoms_loader_converts_records_to_atom_array(ase_lmdb_paths):
    """Test that the ASE atoms loader converts records to AtomArray objects."""
    dataset = ASELMDBDataset(
        paths=ase_lmdb_paths[0],
        name="omol",
        loader=create_ase_atoms_loader(chain_id="M", res_name="MOL", keep_ase_atoms=False),
    )

    loaded = dataset[0]
    atom_array = loaded["atom_array"]

    assert "atoms" not in loaded
    assert loaded["example_id"] == "train_a:1"
    assert atom_array.array_length() == 3
    assert atom_array.chain_id.tolist() == ["M", "M", "M"]
    assert atom_array.res_name.tolist() == ["MOL", "MOL", "MOL"]
    assert atom_array.element.tolist() == ["H", "H", "O"]
    np.testing.assert_array_equal(atom_array.atomic_number, np.array([1, 1, 8]))
    assert loaded["chain_info"]["M"]["chain_type"].is_non_polymer()

    dataset.close()
