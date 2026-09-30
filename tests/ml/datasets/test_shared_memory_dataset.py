"""Tests for memory-mapped metadata via PandasDataset(memory_map=True)."""

import os

import pandas as pd
import pytest

from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.metadata import ArrowMetadataIndex

N = 20


@pytest.fixture
def dummy_df():
    return pd.DataFrame(
        {
            "example_id": [f"example_{i}" for i in range(N)],
            "col_a": [f"path/to/file_{i}.pdb" for i in range(N)],
            "col_b": [f"chain_{i % 4}" for i in range(N)],
        }
    )


def test_feather_file_reused_on_second_init(tmp_path, monkeypatch, dummy_df):
    """A second PandasDataset(memory_map=True) with the same name must reuse the existing feather file."""
    monkeypatch.setenv("SLURM_JOB_ID", "test_job_reuse")

    kwargs = {"filters": ["col_b == 'chain_0'"], "local_drive_mount": str(tmp_path)}
    meta1 = ArrowMetadataIndex(data=dummy_df.copy(), name="reuse_dataset", **kwargs)
    ds1 = PandasDataset(data=dummy_df.copy(), name="reuse_dataset", metadata=meta1)
    mtime_after_first = os.path.getmtime(ds1.metadata.feather_path)

    meta2 = ArrowMetadataIndex(data=dummy_df.copy(), name="reuse_dataset", **kwargs)
    ds2 = PandasDataset(data=dummy_df.copy(), name="reuse_dataset", metadata=meta2)
    mtime_after_second = os.path.getmtime(ds2.metadata.feather_path)

    assert mtime_after_first == mtime_after_second
    assert len(ds1) == len(ds2) == N // 4
    pd.testing.assert_frame_equal(ds1.data.to_pandas(), dummy_df.query("col_b == 'chain_0'"))
