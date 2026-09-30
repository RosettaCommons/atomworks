"""Tests for LmdbDataset using a real OMOL fixture (20 entries from shard_00)."""

from pathlib import Path

import biotite.structure as struc
import pandas as pd
import pytest

pytest.importorskip("lmdb")

from atomworks.ml.datasets.lmdb_dataset import LmdbDataset

OMOL_TEST_DIR = Path(__file__).resolve().parents[2] / "data" / "ml" / "omol_test"
OMOL_LMDB = str(OMOL_TEST_DIR / "shard_00.ciflmdb")
OMOL_PARQUET = str(OMOL_TEST_DIR / "metadata.parquet")


def _bytes_loader(raw_data: tuple) -> dict:
    """Simple loader that returns raw bytes as-is."""
    raw_bytes, global_idx, metadata_row = raw_data
    return {"raw_bytes": raw_bytes, "global_idx": global_idx, "metadata_row": metadata_row}


def test_load_with_metadata():
    """Load with parquet metadata, verify length and example IDs."""
    expected_ids = pd.read_parquet(OMOL_PARQUET)["example_id"].tolist()
    with LmdbDataset(lmdb_path=OMOL_LMDB, name="omol_test", data=OMOL_PARQUET) as ds:
        assert len(ds) == 20
        for i in range(3):
            example = ds[i]
            assert example["example_id"] == expected_ids[i]
            assert "assembly_id" in example
            assert "extra_info" in example


def test_load_without_metadata():
    """Load without parquet — SequentialMetadataIndex with stringified indices."""
    with LmdbDataset(lmdb_path=OMOL_LMDB, name="omol_test", loader=_bytes_loader) as ds:
        assert len(ds) == 20
        assert ds[0]["example_id"] == "0"
        assert ds[0]["metadata_row"] is None


def test_metadata_row_threaded_to_loader():
    """Metadata row is passed through to the loader as the third tuple element."""
    with LmdbDataset(lmdb_path=OMOL_LMDB, name="omol_test", data=OMOL_PARQUET, loader=_bytes_loader) as ds:
        example = ds[0]
        assert isinstance(example["metadata_row"], pd.Series)
        assert "energy" in example["metadata_row"].index
        assert "charge" in example["metadata_row"].index


def test_cif_parsing():
    """Default CIF loader produces valid AtomArrays with bonds from real OMOL entries."""
    with LmdbDataset(lmdb_path=OMOL_LMDB, name="omol_test", data=OMOL_PARQUET) as ds:
        example = ds[0]
        atom_array = example["atom_array"]
        assert isinstance(atom_array, struc.AtomArray)
        assert len(atom_array) > 0
        assert "element" in atom_array.get_annotation_categories()
        assert atom_array.bonds is not None
        assert len(atom_array.bonds.as_array()) > 0


def test_id_lookups():
    """id_to_idx, idx_to_id, and __contains__ work with real example IDs."""
    expected_ids = pd.read_parquet(OMOL_PARQUET)["example_id"].tolist()
    with LmdbDataset(lmdb_path=OMOL_LMDB, name="omol_test", data=OMOL_PARQUET) as ds:
        assert expected_ids[0] in ds
        assert ds.id_to_idx(expected_ids[0]) == 0
        assert ds.idx_to_id(0) == expected_ids[0]
        assert "nonexistent_id_xyz" not in ds
