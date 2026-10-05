"""Periodic geometry and worker safety through the shared ASE dataset interface."""

import multiprocessing
import pickle

import numpy as np
import pandas as pd
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("ase_db_backends")
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from atomworks.ml.datasets.ase_dataset import AseDBDataset
from atomworks.ml.datasets.loaders.materials import ase_atoms_to_material_dict, create_ase_materials_loader


@pytest.fixture
def materials(tmp_path):
    """Create two periodic ASE shards with reversed metadata ordering."""
    # Use a tilted cell with only two periodic axes and nonzero partial charges.
    atoms = ase.Atoms(
        "NaCl",
        scaled_positions=[[1.2, -0.1, 1.4], [0.3, 0.6, 0.9]],
        cell=[[3.0, 0.0, 0.0], [0.7, 4.0, 0.0], [0.4, 0.5, 5.0]],
        pbc=[True, True, False],
    )
    atoms.set_initial_charges([0.4, -0.4])
    atoms.calc = SinglePointCalculator(atoms, energy=-3.0, forces=np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]))

    # Store one row per shard; metadata selects the second shard first.
    for i in range(2):
        with connect(tmp_path / f"shard_{i}.aselmdb", type="aselmdb") as db:
            db.write(atoms, sid=f"material-{i}", data={"parent_space_group": 221, "prototype_label": "AB_cP2_221_a_b"})
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame({"example_id": ["material-1", "material-0"], "lmdb_idx": [1, 0]}).to_parquet(metadata)
    dataset = AseDBDataset(
        lmdb_path=str(tmp_path),
        name="materials",
        data=str(metadata),
        loader=create_ase_materials_loader(include_atom_array=True),
    )
    yield atoms, dataset
    dataset.close()


def test_materials_preserve_periodic_geometry_metadata_and_partial_charges(materials):
    """Preserve source geometry and labels without inferring molecular chemistry."""
    atoms, dataset = materials
    item = dataset[0]

    # Metadata indices select the intended source row.
    assert item["example_id"] == "material-1" and item["global_index"] == 1
    assert item["source_row_id"] == 1 and dataset.id_to_idx("material-1") == 0

    # Preserve the cell, periodic axes, and partial charges without inventing bonds.
    np.testing.assert_allclose(item["fractional_coordinates"] @ item["lattice_vectors"], atoms.positions)
    np.testing.assert_allclose(item["atom_array"].box, atoms.cell)
    np.testing.assert_array_equal(item["pbc"], [True, True, False])
    np.testing.assert_allclose(item["atom_array"].initial_charges, [0.4, -0.4])
    assert "charge" not in item["atom_array"].get_annotation_categories()
    assert item["atom_array"].bonds is None

    # Retain source properties and symmetry labels rather than estimating them.
    np.testing.assert_allclose(item["forces"], atoms.get_forces())
    assert item["energy"] == -3.0 and item["cell_volume"] == pytest.approx(atoms.get_volume())
    assert item["parent_space_group"] == 221 and item["space_group"] is None
    assert item["extra_info"]["prototype_label"] == "AB_cP2_221_a_b"
    assert dataset[1]["example_id"] == "material-0"

    # Wrap only periodic fractional axes, leaving source coordinates unchanged.
    wrapped = ase_atoms_to_material_dict(atoms, wrap_fractional_coordinates=True, metadata={"space_group": "1"})
    np.testing.assert_allclose(wrapped["fractional_coordinates"][0], [0.2, 0.9, 1.4])
    np.testing.assert_allclose(wrapped["cartesian_coordinates"], atoms.positions)
    assert wrapped["space_group"] == 1
    np.testing.assert_allclose(atoms.get_scaled_positions(wrap=False)[0], [1.2, -0.1, 1.4])

    # Fractional coordinates require a full-rank cell.
    atoms.cell = np.zeros((3, 3))
    with pytest.raises(ValueError, match="full-rank cell"):
        ase_atoms_to_material_dict(atoms)


def _read_material_in_worker(dataset, queue):
    """Return the worker's example order and close its database handles."""
    try:
        queue.put([dataset[i]["example_id"] for i in range(len(dataset))])
    finally:
        dataset.close()


@pytest.mark.parametrize("start_method", ["spawn", "fork"])
def test_open_materials_dataset_survives_worker_transfer(materials, start_method):
    """Keep parent and worker reads usable after pickling or inheriting an open dataset."""
    _, dataset = materials

    # Open a shard before serialization; pickling must not close the parent reader.
    assert dataset[0]["example_id"] == "material-1"
    pickle.dumps(dataset)
    assert dataset[0]["example_id"] == "material-1"

    # Read both shards in a spawned or forked worker, then check the parent again.
    context = multiprocessing.get_context(start_method)
    queue = context.Queue()
    process = context.Process(target=_read_material_in_worker, args=(dataset, queue))
    process.start()
    try:
        assert queue.get(timeout=30) == ["material-1", "material-0"]
        process.join(timeout=30)
        assert process.exitcode == 0
        assert dataset[0]["example_id"] == "material-1"
    finally:
        # Release worker resources even if an assertion fails.
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()
