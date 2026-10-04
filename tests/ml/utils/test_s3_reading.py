"""Public and authenticated S3 reads without an external service."""

import io
import pickle
from types import MappingProxyType
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest
import zstandard as zstd

pytest.importorskip("boto3")
from botocore.awsrequest import AWSResponse

from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders import create_structure_loader
from atomworks.ml.datasets.metadata import ArrowMetadataIndex
from atomworks.ml.utils import io as dataset_io
from atomworks.ml.utils.blob_store import BlobIndex, BlobStore
from atomworks.ml.utils.io import S3ReadConfig, read_csv, read_parquet_with_metadata, read_s3_bytes


class _ResponseBody(io.BytesIO):
    def stream(self, amt=None, decode_content=False):
        yield self.read()


@pytest.fixture
def s3_transport(monkeypatch, tmp_path):
    """Capture actual boto3 requests after signing, without sending them over the network."""
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
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://wrong-endpoint.invalid")
    monkeypatch.setenv("LOCAL_DRIVE_MOUNT", str(tmp_path / "cache"))
    monkeypatch.setenv("SLURM_JOB_ID", "test-public-s3")
    monkeypatch.setattr(dataset_io.boto3, "DEFAULT_SESSION", None)
    real_client = dataset_io.boto3.client
    requests, objects = [], {}

    def serve(request, **kwargs):
        requests.append(request)
        body = objects[urlsplit(request.url).path]
        status = 200
        headers = {}
        if "Range" in request.headers:
            start, end = map(int, request.headers["Range"].decode().removeprefix("bytes=").split("-"))
            headers["Content-Range"] = f"bytes {start}-{end}/{len(body)}"
            body = body[start : end + 1]
            status = 206
        headers["Content-Length"] = str(len(body))
        return AWSResponse(request.url, status, headers, _ResponseBody(body))

    def client(*args, **kwargs):
        result = real_client(*args, **kwargs)
        result.meta.events.register("before-send.s3.GetObject", serve)
        return result

    monkeypatch.setattr(dataset_io.boto3, "client", client)
    dataset_io._s3_client.cache_clear()
    yield objects, requests
    dataset_io._s3_client.cache_clear()


@pytest.mark.parametrize("anonymous", [False, True])
@pytest.mark.parametrize("addressing_style", ["path", "virtual"])
def test_s3_range_authentication_and_addressing(s3_transport, monkeypatch, anonymous, addressing_style):
    objects, requests = s3_transport
    if not anonymous:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    path = "/public/record.bin" if addressing_style == "path" else "/record.bin"
    objects[path] = b"prefix-record-suffix"
    config = S3ReadConfig("https://objects.example", anonymous, addressing_style)

    assert read_s3_bytes("s3://public/record.bin", offset=7, length=6, s3_config=config) == b"record"
    request = requests[0]
    assert ("Authorization" in request.headers) is not anonymous
    assert request.headers["Range"] == b"bytes=7-12"
    expected_host = "objects.example" if addressing_style == "path" else "public.objects.example"
    assert urlsplit(request.url).netloc == expected_host


def test_public_metadata_blob_index_and_loader_share_config(s3_transport, tmp_path, monkeypatch):
    objects, requests = s3_transport
    config = MappingProxyType(
        {"endpoint_url": "https://objects.example", "anonymous": True, "addressing_style": "path"}
    )
    record = b"data_test\n#\n"
    compressed = zstd.ZstdCompressor().compress(record)
    shard = b"prefix" + compressed + b"suffix"
    objects["/public/blob/data/shard.bin"] = shard
    index = pd.DataFrame({"path": ["a.cif"], "shard": ["shard.bin"], "offset": [6], "length": [len(compressed)]})
    metadata = pd.DataFrame({"example_id": ["example-a"], "path": ["a.cif"], "assembly_id": ["1"]})
    metadata.attrs = {"source": "public-test"}
    objects["/public/blob/index.parquet"] = index.to_parquet()
    objects["/public/metadata.parquet"] = metadata.to_parquet()
    objects["/public/metadata.csv"] = metadata.to_csv(index=False).encode()

    def parse(source, *, config, cache_key):
        assert source().read().encode() == record
        return {"assemblies": {"1": ["atoms"]}, "chain_info": {}, "ligand_info": {}, "metadata": {}}

    monkeypatch.setattr("atomworks.ml.datasets.loaders.cif.parse", parse)
    loader = create_structure_loader(
        storage="blob", blob_dir="s3://public/blob", record_id_colname="path", s3_config=config
    )
    dataset = PandasDataset(
        name="public", data="s3://public/metadata.parquet", loader=loader, load_kwargs={"s3_config": config}
    )
    result = dataset[0]
    assert result["example_id"] == "example-a"
    assert result["atom_array"] == "atoms"
    assert read_csv("s3://public/metadata.csv", s3_config=config)["example_id"].tolist() == ["example-a"]

    restored = pickle.loads(pickle.dumps(loader))
    assert restored(dataset.data.iloc[0])["atom_array"] == "atoms"
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "shard.bin").write_bytes(shard)
    local = BlobStore(str(local_dir)).get_bytes("shard.bin", 6, len(compressed))
    remote = loader.keywords["store"].get_bytes("shard.bin", 6, len(compressed))
    assert local == remote == record
    assert all("Authorization" not in request.headers for request in requests)
    assert all(urlsplit(request.url).netloc == "objects.example" for request in requests)
    assert {urlsplit(request.url).path for request in requests} == set(objects)


def test_endpoint_is_part_of_blob_index_cache_identity(s3_transport):
    objects, _ = s3_transport
    index = pd.DataFrame({"example_id": ["a"], "shard": ["a.bin"], "offset": [0], "length": [1]})
    objects["/public/index.parquet"] = index.to_parquet()
    first = BlobIndex("s3://public/index.parquet", s3_config=S3ReadConfig("https://first.example", True, "path"))
    index["offset"] = [10]
    objects["/public/index.parquet"] = index.to_parquet()
    second = BlobIndex("s3://public/index.parquet", s3_config=S3ReadConfig("https://second.example", True, "path"))
    assert first.lookup("a") == ("a.bin", 0, 1)
    assert second.lookup("a") == ("a.bin", 10, 1)


def test_metadata_cache_resolves_environment_endpoint(s3_transport, monkeypatch, tmp_path):
    objects, _ = s3_transport
    options = {
        "data": "s3://public/metadata.parquet",
        "name": "shared-name",
        "local_drive_mount": str(tmp_path),
        "load_kwargs": {"s3_config": {"anonymous": True, "addressing_style": "path"}},
    }
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://first.example")
    objects["/public/metadata.parquet"] = pd.DataFrame({"example_id": ["first"]}).to_parquet()
    first = ArrowMetadataIndex(**options)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://second.example")
    objects["/public/metadata.parquet"] = pd.DataFrame({"example_id": ["second"]}).to_parquet()
    second = ArrowMetadataIndex(**options)
    assert first.get_example_id(0) == "first"
    assert second.get_example_id(0) == "second"


@pytest.mark.parametrize("uri", ["https://example.com/data", "hf://datasets/org/data", "gs://bucket/data"])
def test_unsupported_remote_schemes_fail_clearly(uri):
    with pytest.raises(ValueError, match="Unsupported blob URI"):
        BlobStore(uri)
    with pytest.raises(ValueError, match="Expected s3://bucket/key"):
        read_s3_bytes(uri)


def test_metadata_readers_preserve_non_s3_sources(monkeypatch):
    seen = []

    def read_csv_source(source, **kwargs):
        seen.append(source)
        return pd.DataFrame({"example_id": ["a"]})

    monkeypatch.setattr(pd, "read_csv", read_csv_source)
    source = "https://example.com/metadata.csv"
    assert read_csv(source)["example_id"].tolist() == ["a"]
    assert seen == [source]


def test_default_signed_read_uses_environment_endpoint(s3_transport, monkeypatch):
    objects, requests = s3_transport
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    objects["/value"] = b"value"
    assert read_s3_bytes("s3://public/value") == b"value"
    assert urlsplit(requests[0].url).netloc == "public.wrong-endpoint.invalid"
    assert "Authorization" in requests[0].headers


def test_local_metadata_preserves_parquet_attributes(tmp_path):
    path = tmp_path / "metadata.parquet"
    frame = pd.DataFrame({"example_id": ["a"], "value": [2]})
    frame.attrs = {"source": "local"}
    dataset_io.to_parquet_with_metadata(frame, path)
    loaded = read_parquet_with_metadata(path)
    np.testing.assert_array_equal(loaded["value"], [2])
    assert loaded.attrs["source"] == "local"
