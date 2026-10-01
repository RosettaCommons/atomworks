from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from atomworks.ml.datasets.loaders import create_structure_loader


def _parsed_structure(assembly_id: str) -> dict:
    return {
        "assemblies": {assembly_id: ["atoms"]},
        "chain_info": {},
        "ligand_info": {},
        "metadata": {},
    }


@patch("atomworks.ml.datasets.loaders.cif.parse")
def test_filesystem_loader_builds_path_and_projects_metadata(mock_parse: MagicMock, tmp_path):
    mock_parse.return_value = _parsed_structure("2")
    row = pd.Series(
        {
            "sample_id": "interface-1",
            "record": "1abc",
            "assembly": "2",
            "left": "A_1",
            "right": "B_1",
            "resolution": 2.0,
        }
    )
    loader = create_structure_loader(
        storage="filesystem",
        example_id_colname="sample_id",
        path_colname="record",
        assembly_id_colname="assembly",
        base_path=str(tmp_path),
        extension=".cif",
        column_mapping={"query_pn_unit_iids": ["left", "right"]},
    )

    result = loader(row)

    assert result["example_id"] == "interface-1"
    assert result["path"] == tmp_path / "1abc.cif"
    assert result["query_pn_unit_iids"] == ["A_1", "B_1"]
    assert result["extra_info"] == {"resolution": 2.0}
    assert mock_parse.call_args.kwargs["config"].build_assembly == ("2",)


@pytest.mark.parametrize("seed, expected", [(np.int64(7), 7), (pd.NA, None)])
@patch("atomworks.ml.datasets.loaders.cif.parse")
def test_filesystem_loader_normalizes_altloc_seed(mock_parse: MagicMock, tmp_path, seed, expected):
    mock_parse.return_value = _parsed_structure("1")
    row = pd.Series({"example_id": "interface-1", "path": str(tmp_path / "1abc.cif"), "altloc_seed": seed})
    loader = create_structure_loader(storage="filesystem", altloc_seed_colname="altloc_seed")

    result = loader(row)

    assert result["altloc_seed"] == expected
    assert mock_parse.call_args.kwargs["config"].altloc_seed == expected


@patch("atomworks.ml.datasets.loaders.cif.parse")
def test_bytes_loader_projects_metadata(mock_parse: MagicMock):
    mock_parse.return_value = _parsed_structure("2")
    row = pd.Series(
        {"example_id": "monomer-1", "assembly_id": "2", "altloc_seed": 7, "chain": "A_1", "resolution": 1.5}
    )
    loader = create_structure_loader(
        storage="bytes",
        altloc_seed_colname="altloc_seed",
        column_mapping={"query_pn_unit_iids": ["chain"], "optional_selection": None},
    )

    result = loader((b"data_test\n#\n", 0, row))

    assert result["query_pn_unit_iids"] == ["A_1"]
    assert result["optional_selection"] is None
    assert result["extra_info"] == {"altloc_seed": 7, "resolution": 1.5}
    assert mock_parse.call_args.kwargs["config"].altloc_seed == 7


@patch("atomworks.ml.datasets.loaders.cif.parse")
def test_blob_loader_separates_record_and_example_ids(mock_parse: MagicMock, tmp_path, monkeypatch):
    mock_parse.return_value = _parsed_structure("1")
    record = b"data_test\n#\n"
    blob_dir = tmp_path / "blob"
    data_dir = blob_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "shard.bin").write_bytes(b"prefix" + record + b"suffix")
    pd.DataFrame(
        {
            "path": ["01/1abc/1abc.cif"],
            "shard": ["shard.bin"],
            "offset": [len(b"prefix")],
            "length": [len(record)],
        }
    ).to_parquet(blob_dir / "index.parquet")
    monkeypatch.setenv("LOCAL_DRIVE_MOUNT", str(tmp_path / "cache"))
    loader = create_structure_loader(
        storage="blob",
        blob_dir=str(blob_dir),
        record_id_colname="path",
        example_id_colname="sample_id",
        column_mapping={"query_pn_unit_iids": ["left", "right"]},
    )
    row = pd.Series(
        {
            "path": "01/1abc/1abc.cif",
            "sample_id": "interface-1",
            "left": "A_1",
            "right": "B_1",
        }
    )

    result = loader(row)

    assert result["example_id"] == "interface-1"
    assert result["query_pn_unit_iids"] == ["A_1", "B_1"]
    assert result["extra_info"] == {"sample_id": "interface-1"}
    assert mock_parse.call_args.args[0]().read() == record.decode()


def test_structure_loader_rejects_unknown_storage():
    with pytest.raises(ValueError, match="Unsupported structure storage"):
        create_structure_loader(storage="database")
