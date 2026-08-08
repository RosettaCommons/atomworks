"""Tests for the remote (ColabFold server) MSA generation backend.

All HTTP primitives (`_submit`, `_status`, `_download`) are monkeypatched; no sockets are opened.
"""

import base64
import gzip
import io
import tarfile
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from atomworks.enums import MSAFileExtension
from atomworks.ml.preprocessing.msa import server as server_module
from atomworks.ml.preprocessing.msa.finding import find_msas
from atomworks.ml.preprocessing.msa.server import (
    ENV_A3M_FILENAME,
    UNIREF_A3M_FILENAME,
    MSAServerConfig,
    _select_mode,
    _split_multi_query_a3m,
    make_msas_mmseqs_server,
    run_mmseqs2_server,
)
from atomworks.ml.utils.misc import hash_sequence

SEQ_A = "MSYIWRQLGSPTVAITLSVSTVIYVTVICPIVFIHLFGDHL"
SEQ_B = "MKKKEVEKDDLIENASRVASCISIFLIIASTTMYIFIGLKI"
SEQ_C = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFELRRQ"


@pytest.fixture(autouse=True)
def _clean_credentials_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure a developer's own server credentials don't leak into the tests."""
    for var in ("MSA_SERVER_USERNAME", "MSA_SERVER_PASSWORD", "MSA_SERVER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def config() -> MSAServerConfig:
    """A server config that never actually sleeps."""
    return MSAServerConfig(poll_interval=(0.0, 0.0), retry_delay=0.0)


def _a3m_block(query_id: int, sequence: str, hit_prefix: str) -> str:
    """Build one query's alignment block as the server would return it."""
    return (
        f">{query_id}\n{sequence}\n"
        f">UniRef100_{hit_prefix}{query_id}\t91\t0.814\t1.694E-18\t0\t40\t41\t1\t41\t41\n"
        f"{sequence[:-1]}-\n"
    )


def _multi_query_a3m(sequences: list[str], hit_prefix: str) -> str:
    """Concatenate per-query blocks the way the server does: separated by null bytes."""
    blocks = [_a3m_block(101 + i, sequence, hit_prefix) for i, sequence in enumerate(sequences)]
    return "\x00".join(blocks)


class FakeServer:
    """Stand-in for a ColabFold MMseqs2 server, recording the calls made against it."""

    def __init__(self, statuses: list[str] | None = None, submit_statuses: list[str] | None = None) -> None:
        # Status sequence returned by consecutive `_status` polls; the last one repeats
        self.statuses = statuses if statuses is not None else ["COMPLETE"]
        self.submit_statuses = submit_statuses if submit_statuses is not None else ["PENDING"]
        self.submitted_batches: list[list[str]] = []
        self.submitted_modes: list[str] = []
        self.n_status_calls = 0
        self.n_downloads = 0

    def submit(self, sequences: list[str], mode: str, config: MSAServerConfig) -> dict[str, str]:
        self.submitted_batches.append(list(sequences))
        self.submitted_modes.append(mode)
        status = self.submit_statuses[min(len(self.submitted_batches) - 1, len(self.submit_statuses) - 1)]
        return {"id": f"ticket-{len(self.submitted_batches)}", "status": status}

    def status(self, ticket_id: str, config: MSAServerConfig) -> dict[str, str]:
        status = self.statuses[min(self.n_status_calls, len(self.statuses) - 1)]
        self.n_status_calls += 1
        return {"id": ticket_id, "status": status}

    def download(self, ticket_id: str, dest: Path, config: MSAServerConfig) -> None:
        self.n_downloads += 1
        sequences = self.submitted_batches[-1]
        dest = Path(dest)
        payload_dir = dest.parent / f"payload-{self.n_downloads}"
        payload_dir.mkdir(exist_ok=True)

        members = {UNIREF_A3M_FILENAME: _multi_query_a3m(sequences, "UNI")}
        if config.use_env:
            members[ENV_A3M_FILENAME] = _multi_query_a3m(sequences, "ENV")

        with tarfile.open(dest, "w:gz") as tar:
            for name, content in members.items():
                member_path = payload_dir / name
                member_path.write_text(content)
                tar.add(member_path, arcname=name)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FakeServer":
        monkeypatch.setattr(server_module, "_submit", self.submit)
        monkeypatch.setattr(server_module, "_status", self.status)
        monkeypatch.setattr(server_module, "_download", self.download)
        return self


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    return FakeServer().install(monkeypatch)


# --- Configuration --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("use_env", "use_filter", "expected_mode"),
    [
        (True, True, "env"),
        (False, True, "all"),
        (True, False, "env-nofilter"),
        (False, False, "nofilter"),
    ],
)
def test_select_mode(use_env: bool, use_filter: bool, expected_mode: str) -> None:
    assert _select_mode(MSAServerConfig(use_env=use_env, use_filter=use_filter)) == expected_mode


def test_config_rejects_basic_auth_and_api_key_together() -> None:
    with pytest.raises(ValueError, match="basic auth"):
        MSAServerConfig(username="me", password="secret", api_key_header="X-API-Key", api_key_value="abc")


def test_config_rejects_api_key_without_header() -> None:
    with pytest.raises(ValueError, match="header name is required"):
        MSAServerConfig(api_key_value="abc")


def test_config_reads_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSA_SERVER_USERNAME", "env-user")
    monkeypatch.setenv("MSA_SERVER_PASSWORD", "env-password")
    config = MSAServerConfig()
    assert (config.username, config.password) == ("env-user", "env-password")


def test_config_strips_trailing_slash_from_host_url() -> None:
    assert MSAServerConfig(host_url="https://msa.internal/").host_url == "https://msa.internal"


# --- Parsing --------------------------------------------------------------------------


def test_split_multi_query_a3m_splits_on_numeric_headers(tmp_path: Path) -> None:
    a3m_file = tmp_path / "uniref.a3m"
    a3m_file.write_text(_multi_query_a3m([SEQ_A, SEQ_B, SEQ_C], "UNI"))

    blocks = _split_multi_query_a3m(a3m_file)

    assert sorted(blocks) == [101, 102, 103]
    assert blocks[102][0] == ">102\n"
    assert blocks[102][1] == f"{SEQ_B}\n"
    # Null bytes separating the queries are stripped, not carried into the alignment
    assert not any("\x00" in line for lines in blocks.values() for line in lines)


def test_run_mmseqs2_server_writes_one_a3m_per_sequence(tmp_path: Path, fake_server: FakeServer, config) -> None:
    paths = run_mmseqs2_server([SEQ_A, SEQ_B], tmp_path, config)

    assert set(paths) == {SEQ_A, SEQ_B}
    for sequence, path in paths.items():
        lines = path.read_text().splitlines()
        assert path.name == f"{hash_sequence(sequence)}.a3m"
        # The numeric query header is rewritten to the sequence hash, as the local backend does
        assert lines[0] == f">{hash_sequence(sequence)}"
        assert lines[1] == sequence
        # UniRef and environmental hits are concatenated, with the query kept exactly once
        assert [line for line in lines if line.startswith(">")][1:] == [
            f">UniRef100_UNI{101 + list(paths).index(sequence)}\t91\t0.814\t1.694E-18\t0\t40\t41\t1\t41\t41",
            f">UniRef100_ENV{101 + list(paths).index(sequence)}\t91\t0.814\t1.694E-18\t0\t40\t41\t1\t41\t41",
        ]


def test_use_env_false_requests_uniref_only(tmp_path: Path, fake_server: FakeServer) -> None:
    config = MSAServerConfig(use_env=False, poll_interval=(0.0, 0.0), retry_delay=0.0)

    paths = run_mmseqs2_server([SEQ_A], tmp_path, config)

    assert fake_server.submitted_modes == ["all"]
    assert paths[SEQ_A].read_text().count(">") == 2  # query + one UniRef hit


def test_duplicate_sequences_are_submitted_once(tmp_path: Path, fake_server: FakeServer, config) -> None:
    paths = run_mmseqs2_server([SEQ_A, SEQ_B, SEQ_A], tmp_path, config)

    assert fake_server.submitted_batches == [[SEQ_A, SEQ_B]]
    assert len(paths) == 2
    assert paths[SEQ_A] == tmp_path / f"{hash_sequence(SEQ_A)}.a3m"


def test_sequences_are_split_into_batches(tmp_path: Path, fake_server: FakeServer) -> None:
    config = MSAServerConfig(batch_size=2, poll_interval=(0.0, 0.0), retry_delay=0.0)

    paths = run_mmseqs2_server([SEQ_A, SEQ_B, SEQ_C], tmp_path, config)

    assert fake_server.submitted_batches == [[SEQ_A, SEQ_B], [SEQ_C]]
    assert len(paths) == 3


# --- Polling state machine ------------------------------------------------------------


def test_ratelimited_submission_is_resubmitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config) -> None:
    fake_server = FakeServer(submit_statuses=["RATELIMIT", "RATELIMIT", "PENDING"]).install(monkeypatch)

    run_mmseqs2_server([SEQ_A], tmp_path, config)

    assert len(fake_server.submitted_batches) == 3
    assert fake_server.n_downloads == 1


def test_job_is_polled_until_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config) -> None:
    fake_server = FakeServer(statuses=["PENDING", "RUNNING", "RUNNING", "COMPLETE"]).install(monkeypatch)

    run_mmseqs2_server([SEQ_A], tmp_path, config)

    assert fake_server.n_status_calls == 4
    assert fake_server.n_downloads == 1


def test_error_status_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config) -> None:
    fake_server = FakeServer(statuses=["ERROR"]).install(monkeypatch)

    with pytest.raises(RuntimeError, match="MSA server returned an error"):
        run_mmseqs2_server([SEQ_A], tmp_path, config)

    assert fake_server.n_downloads == 0


def test_maintenance_status_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config) -> None:
    FakeServer(submit_statuses=["MAINTENANCE"]).install(monkeypatch)

    with pytest.raises(RuntimeError, match="maintenance"):
        run_mmseqs2_server([SEQ_A], tmp_path, config)


def test_network_failures_are_retried_then_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config) -> None:
    n_calls = 0

    def always_fails(*args, **kwargs) -> dict[str, str]:
        nonlocal n_calls
        n_calls += 1
        raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(server_module, "_submit", always_fails)

    with pytest.raises(RuntimeError, match="after 5 attempts"):
        run_mmseqs2_server([SEQ_A], tmp_path, config)

    assert n_calls == config.max_retries


def test_transient_network_failure_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config) -> None:
    fake_server = FakeServer().install(monkeypatch)
    real_submit = fake_server.submit
    n_calls = 0

    def flaky_submit(*args, **kwargs) -> dict[str, str]:
        nonlocal n_calls
        n_calls += 1
        if n_calls == 1:
            raise TimeoutError("timed out")
        return real_submit(*args, **kwargs)

    monkeypatch.setattr(server_module, "_submit", flaky_submit)

    run_mmseqs2_server([SEQ_A], tmp_path, config)

    assert fake_server.n_downloads == 1


# --- End-to-end (mocked server) -------------------------------------------------------


def test_make_msas_mmseqs_server_writes_sharded_store(tmp_path: Path, fake_server: FakeServer, config) -> None:
    output_dir = tmp_path / "msas"

    make_msas_mmseqs_server([SEQ_A, SEQ_B], output_dir, config=config)

    for sequence in (SEQ_A, SEQ_B):
        sequence_hash = hash_sequence(sequence)
        expected = output_dir / sequence_hash[:2] / f"{sequence_hash}.a3m.gz"
        assert expected.exists(), f"missing sharded MSA for {sequence_hash}"
        with gzip.open(expected, "rt") as f:
            assert f.readline().strip() == f">{sequence_hash}"

    # ... and the store round-trips through the standard MSA lookup
    missing, found = find_msas(
        [SEQ_A, SEQ_B], msa_dirs=[output_dir], shard_depths=[1], extensions=[MSAFileExtension.A3M_GZ]
    )
    assert missing == []
    assert set(found) == {SEQ_A, SEQ_B}


def test_make_msas_mmseqs_server_honors_the_sharding_pattern(tmp_path: Path, fake_server: FakeServer, config) -> None:
    make_msas_mmseqs_server([SEQ_A], tmp_path, config=config, sharding_pattern="/0:2/2:4/")

    sequence_hash = hash_sequence(SEQ_A)
    assert (tmp_path / sequence_hash[:2] / sequence_hash[2:4] / f"{sequence_hash}.a3m.gz").exists()


def test_make_msas_mmseqs_server_skips_cached_sequences(tmp_path: Path, fake_server: FakeServer, config) -> None:
    output_dir = tmp_path / "msas"

    make_msas_mmseqs_server([SEQ_A], output_dir, config=config)
    assert len(fake_server.submitted_batches) == 1

    # A warm cache makes no further requests...
    make_msas_mmseqs_server([SEQ_A], output_dir, config=config)
    assert len(fake_server.submitted_batches) == 1

    # ... and only the uncached sequence is submitted
    make_msas_mmseqs_server([SEQ_A, SEQ_B], output_dir, config=config)
    assert fake_server.submitted_batches[-1] == [SEQ_B]


def test_make_msas_mmseqs_server_ignores_cache_when_check_existing_is_false(
    tmp_path: Path, fake_server: FakeServer, config
) -> None:
    output_dir = tmp_path / "msas"

    make_msas_mmseqs_server([SEQ_A], output_dir, config=config)
    make_msas_mmseqs_server([SEQ_A], output_dir, config=config, check_existing=False)

    assert fake_server.submitted_batches == [[SEQ_A], [SEQ_A]]


def test_make_msas_mmseqs_server_accepts_a_single_sequence(tmp_path: Path, fake_server: FakeServer, config) -> None:
    make_msas_mmseqs_server(SEQ_A, tmp_path, config=config)

    sequence_hash = hash_sequence(SEQ_A)
    assert (tmp_path / sequence_hash[:2] / f"{sequence_hash}.a3m.gz").exists()


# --- Wire protocol (real HTTP against a loopback server) --------------------------------


def _tarball_bytes(sequences: list[str]) -> bytes:
    """Build an in-memory ``out.tar.gz`` holding a UniRef alignment for `sequences`."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        payload = _multi_query_a3m(sequences, "UNI").encode()
        info = tarfile.TarInfo(UNIREF_A3M_FILENAME)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class _LoopbackHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for the ColabFold API, recording the requests it receives."""

    received: list[dict[str, object]] = []

    def log_message(self, fmt: str, *args: object) -> None:  # silence the default stderr logging
        pass

    def _record(self, body: dict[str, str] | None = None) -> None:
        self.received.append({"path": self.path, "headers": dict(self.headers), "body": body})

    def _reply_json(self, payload: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    def do_POST(self) -> None:  # noqa: N802 - name mandated by BaseHTTPRequestHandler
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        self._record({k: v[0] for k, v in urllib.parse.parse_qs(raw.decode()).items()})
        self._reply_json('{"id": "ticket-1", "status": "PENDING"}')

    def do_GET(self) -> None:  # noqa: N802 - name mandated by BaseHTTPRequestHandler
        self._record()
        if self.path.startswith("/ticket/"):
            self._reply_json('{"id": "ticket-1", "status": "COMPLETE"}')
            return
        payload = _tarball_bytes([SEQ_A])
        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def loopback_server() -> Iterator[tuple[str, list[dict[str, object]]]]:
    """Serve the ColabFold endpoints on localhost, yielding the base URL and the received requests."""
    _LoopbackHandler.received = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", _LoopbackHandler.received
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_wire_protocol_against_a_loopback_server(tmp_path: Path, loopback_server) -> None:
    host_url, received = loopback_server
    config = MSAServerConfig(
        host_url=host_url,
        use_env=False,
        username="me",
        password="secret",
        user_agent="atomworks-test",
        poll_interval=(0.0, 0.0),
        retry_delay=0.0,
    )

    paths = run_mmseqs2_server([SEQ_A], tmp_path, config)

    assert paths[SEQ_A].read_text().startswith(f">{hash_sequence(SEQ_A)}\n{SEQ_A}\n")

    submission, status_poll, download = received
    assert submission["path"] == "/ticket/msa"
    assert submission["body"] == {"q": f">101\n{SEQ_A}\n", "mode": "all"}
    assert status_poll["path"] == "/ticket/ticket-1"
    assert download["path"] == "/result/download/ticket-1"

    for request in received:
        headers = request["headers"]
        assert headers["User-Agent"] == "atomworks-test"
        assert headers["Authorization"] == f"Basic {base64.b64encode(b'me:secret').decode()}"


def test_api_key_header_is_sent(tmp_path: Path, loopback_server) -> None:
    host_url, received = loopback_server
    config = MSAServerConfig(
        host_url=host_url,
        use_env=False,
        api_key_header="X-API-Key",
        api_key_value="s3cr3t",
        poll_interval=(0.0, 0.0),
        retry_delay=0.0,
    )

    run_mmseqs2_server([SEQ_A], tmp_path, config)

    assert all(request["headers"]["X-API-Key"] == "s3cr3t" for request in received)
    assert all("Authorization" not in request["headers"] for request in received)
