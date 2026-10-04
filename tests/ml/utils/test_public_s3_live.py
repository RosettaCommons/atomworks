"""Opt-in smoke test against a public, versioned structure dataset."""

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("boto3")

from atomworks.io.config import ParseConfig
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_structure_loader
from atomworks.ml.utils.io import S3ReadConfig, _s3_client


@pytest.mark.requires_internet
@pytest.mark.skipif(
    os.environ.get("ATOMWORKS_TEST_PUBLIC_S3") != "1", reason="Set ATOMWORKS_TEST_PUBLIC_S3=1 to opt in"
)
def test_anonymous_public_dataset_matches_local_cif(tmp_path, monkeypatch):
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("LOCAL_DRIVE_MOUNT", str(tmp_path / "cache"))
    config = S3ReadConfig(endpoint_url="https://cwobject.com", anonymous=True)
    prefix = "s3://rfd4-proteina-public-e04a/train_datasets/streaming-latest/synthetic/2026_09_09_tcr_af2"
    parser = ParseConfig.from_preset(
        "annotations_only", altloc="first", load_standard_annotations=True, return_atom_array_plus=True
    )
    requests, statuses = [], []
    _s3_client.cache_clear()
    client = _s3_client(config)
    client.meta.events.register("before-send.s3.GetObject", lambda request, **kw: requests.append(request))
    client.meta.events.register(
        "after-call.s3.GetObject", lambda http_response, **kw: statuses.append(http_response.status_code)
    )

    loader = create_structure_loader(
        storage="blob", blob_dir=f"{prefix}/blob", record_id_colname="path", s3_config=config, parser_args=parser
    )
    dataset = PandasDataset(
        name="public-tcr", data=f"{prefix}/examples.parquet", loader=loader, load_kwargs={"s3_config": config}
    )
    remote = dataset[0]
    row = dataset.metadata.get_row(0)
    location = loader.keywords["index"].lookup(row["path"])
    cif = loader.keywords["store"].get_bytes(*location)
    path = tmp_path / "example.cif"
    path.write_bytes(cif)
    local_loader = create_structure_loader(storage="filesystem", parser_args=parser)
    local = local_loader(pd.Series({"example_id": row["example_id"], "path": str(path)}))

    assert remote["example_id"] == local["example_id"] == row["example_id"]
    left, right = remote["atom_array"], local["atom_array"]
    assert len(left) > 0
    np.testing.assert_array_equal(left.coord, right.coord)
    np.testing.assert_array_equal(left.bonds.as_array(), right.bonds.as_array())
    assert set(left.get_annotation_categories()) == set(right.get_annotation_categories())
    for annotation in left.get_annotation_categories():
        np.testing.assert_equal(left.get_annotation(annotation), right.get_annotation(annotation))
    assert requests and all("Authorization" not in request.headers for request in requests)
    assert statuses.count(200) == 2
    assert statuses.count(206) == 2
    assert sum("Range" in request.headers for request in requests) == 2
    _s3_client.cache_clear()
