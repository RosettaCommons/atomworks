"""Network-isolated coverage of the remote MSA protocol and release dispatch."""

import gzip
import io
import tarfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from atomworks.ml.preprocessing.msa import server


def tar_payload():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, prefix in [(server.UNIREF_A3M_FILENAME, "uni"), (server.ENV_A3M_FILENAME, "env")]:
            data = f">101\nAAA\n>{prefix}_a\nA-A\n\x00>102\nCCC\n>{prefix}_c\nC-C\n".encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    return buffer.getvalue()


def test_completed_batch_mapping_auth_retry_and_cache(tmp_path, monkeypatch):
    posts = []

    def post(url, **kwargs):
        posts.append((url, kwargs))
        if len(posts) == 1:
            raise requests.Timeout("transient")
        return SimpleNamespace(json=lambda: {"status": "PENDING", "id": "test-ticket"})

    def get(url, **kwargs):
        assert kwargs["auth"] == ("user", "test-password")
        if "/result/download/" in url:
            response = MagicMock()
            response.__enter__.return_value = response
            response.iter_content.return_value = [tar_payload()]
            return response
        return SimpleNamespace(json=lambda: {"status": "COMPLETE", "id": "test-ticket"})

    monkeypatch.setattr(server.requests, "post", post)
    monkeypatch.setattr(server.requests, "get", get)
    monkeypatch.setattr(server.time, "sleep", lambda _: None)
    config = server.MSAServerConfig(
        host_url="https://example.invalid/",
        username="user",
        password="test-password",
        poll_interval=(0, 0),
        retry_delay=0,
    )
    run_server = server.run_mmseqs2_server

    def checked_run(*args, **kwargs):
        paths = run_server(*args, **kwargs)
        assert set(paths) == {"AAA", "CCC"}
        return paths

    monkeypatch.setattr(server, "run_mmseqs2_server", checked_run)
    server.make_msas_mmseqs_server(["AAA", "CCC", "AAA"], tmp_path / "organized", config=config, existing_msa_dirs=[])
    assert len(posts) == 2
    assert posts[-1][1]["data"]["q"] == ">101\nAAA\n>102\nCCC\n"
    for sequence, hit in [("AAA", "a"), ("CCC", "c")]:
        digest = server.hash_sequence(sequence)
        matches = list(tmp_path.rglob(digest + ".a3m.gz"))
        assert len(matches) == 1
        text = gzip.decompress(matches[0].read_bytes()).decode()
        assert text.startswith(">" + digest + "\n" + sequence + "\n")
        assert ">uni_" + hit in text and ">env_" + hit in text
    monkeypatch.setattr(server.requests, "post", lambda *a, **kw: pytest.fail("cache hit submitted a request"))
    server.make_msas_mmseqs_server(["CCC", "AAA"], tmp_path / "organized", config=config, existing_msa_dirs=[])


@pytest.mark.parametrize(
    "status,error,message,attempts",
    [
        ("ERROR", RuntimeError, "bad query", 1),
        ("NETWORK_TIMEOUT", RuntimeError, "after 2 attempts", 2),
        ("PENDING", TimeoutError, "job_timeout", 3),
        ("RATELIMIT", TimeoutError, "job_timeout", 3),
    ],
)
def test_failed_jobs_stop_without_outputs(tmp_path, monkeypatch, status, error, message, attempts):
    clock = [0.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(server.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    calls = []

    def respond(*args, **kwargs):
        calls.append(kwargs)
        if status == "NETWORK_TIMEOUT":
            raise requests.Timeout("offline")
        return SimpleNamespace(json=lambda: {"status": status, "id": "stuck", "message": "bad query"})

    monkeypatch.setattr(server.requests, "post", respond)
    monkeypatch.setattr(server.requests, "get", respond)
    config = server.MSAServerConfig(max_retries=2, retry_delay=0, job_timeout=3, poll_interval=(1, 1))
    with pytest.raises(error, match=message):
        server.run_mmseqs2_server("AAA", tmp_path, config)
    pending = status in ("PENDING", "RATELIMIT")
    assert len(calls) == attempts and clock[0] == (3 if pending else 0)
    assert [call["timeout"] for call in calls] == ([3, 2, 1] if pending else [3] * attempts)
    assert not list(tmp_path.glob("*.a3m"))


def test_remote_configuration_validation_and_secret_repr():
    for field in ("job_timeout", "request_timeout"):
        for value in (0, -1, float("inf"), float("nan")):
            with pytest.raises(ValueError, match=field):
                server.MSAServerConfig(**{field: value})
    with pytest.raises(ValueError, match="same time"):
        server.MSAServerConfig(username="user", password="password", api_key_value="key", api_key_header="X-Key")
    config = server.MSAServerConfig(password="secret-password", username="user")
    assert "secret-password" not in repr(config)


@pytest.mark.parametrize("backend", ["mmseqs2", "hhblits", "mmseqs2_server"])
def test_csv_and_cli_dispatch_preserve_local_backends(tmp_path, monkeypatch, backend):
    from typer.testing import CliRunner

    from atomworks.ml.preprocessing.msa import generating
    from atomworks_cli import generate

    csv = tmp_path / "sequences.csv"
    csv.write_text("seq\nAAA\nCCC\nAAA\n")
    calls = []
    for name in ("make_msas_mmseqs", "make_msas_hhblits", "make_msas_mmseqs_server"):
        monkeypatch.setattr(generating, name, lambda _name=name, **kwargs: calls.append((_name, kwargs)))
    monkeypatch.setenv("HHBLITS_UNIREF30_DB_PATH", str(tmp_path / "dummy-db"))
    config = generating.MSAGenerationConfig(backend=backend)
    if backend == "hhblits":
        config.hhblits_search_config = generating.HHblitsSearchConfig(uniref30_db_path="dummy-db", use_bfd=False)
    generating.make_msas_from_csv(csv, tmp_path / "api", config=config)
    expected = {
        "mmseqs2": "make_msas_mmseqs",
        "hhblits": "make_msas_hhblits",
        "mmseqs2_server": "make_msas_mmseqs_server",
    }
    if backend == "mmseqs2_server":
        monkeypatch.setattr(generate.torch.cuda, "is_available", lambda: pytest.fail("Remote path queried GPU"))
    result = CliRunner().invoke(generate.app, [str(csv), str(tmp_path / "cli"), "--backend", backend])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert len(calls) == 2
    for name, kwargs in calls:
        assert name == expected[backend]
        assert kwargs["sequences"] == ["AAA", "CCC"]
        assert kwargs["max_final_sequences"] == (None if backend == "mmseqs2_server" else 10000)
