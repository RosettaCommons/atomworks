"""Generate ColabFold-style multiple sequence alignments (MSAs) with a remote MMseqs2 server.

This is the remote counterpart to :py:mod:`atomworks.ml.preprocessing.msa.generating`: instead of
running MMseqs2 locally against the ~1 TB ColabFold database set, sequences are submitted to a
ColabFold-compatible MMseqs2 server (the public ``https://api.colabfold.com`` by default, or any
self-hosted deployment) and the resulting a3m files are downloaded. No local databases, no mmseqs
binary and no GPU are required.

The output is byte-compatible with the local backend: one ``<hash>.a3m.gz`` per input sequence in a
hash-sharded directory, where ``<hash>`` is :py:func:`~atomworks.ml.utils.misc.hash_sequence` of the
query. Everything downstream (finding, filtering, loading) is therefore unchanged.

Examples:
    Generate MSAs against the public ColabFold server:

    .. code-block:: python

       from atomworks.ml.preprocessing.msa.server import make_msas_mmseqs_server

       sequences = ["MSYIWRQLGSPTVAITLSVSTVIYVTVICPIVFIHLFGDHL...", "MKKKEVEKDDLIENASRVASCISIFLIIASTTMYIFIGLKI..."]
       make_msas_mmseqs_server(sequences, "output_msas/")

    Generate MSAs against a self-hosted server that requires basic authentication:

    .. code-block:: python

       from atomworks.ml.preprocessing.msa.server import MSAServerConfig, make_msas_mmseqs_server

       config = MSAServerConfig(host_url="https://msa.internal", username="me", password="secret")
       make_msas_mmseqs_server(sequences, "output_msas/", config=config)

Note:
    MSAs produced by a ColabFold server carry UniRef accessions and alignment statistics in their
    headers, but no ``TaxID=`` field. Multimer MSA pairing in AtomWorks keys on ``TaxID=``, so
    multimers built from server MSAs are effectively unpaired.

References:
    * Mirdita, M. et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*, 19, 679-682.
    * `ColabFold MMseqs2 API client`_ - Reference implementation of the wire protocol.

    .. _ColabFold MMseqs2 API client: https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py
"""

import dataclasses
import functools
import logging
import math
import os
import random
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar

import requests
from tqdm import tqdm

from atomworks.enums import MSAFileExtension
from atomworks.ml.preprocessing.msa.filtering import HHFilterConfig, MSAFilterConfig, filter_msas
from atomworks.ml.preprocessing.msa.finding import find_msas, get_msa_dirs_from_env
from atomworks.ml.preprocessing.msa.organizing import MSAOrganizationConfig, organize_msas
from atomworks.ml.utils.misc import hash_sequence

logger = logging.getLogger(__name__)

DEFAULT_MSA_SERVER_URL = "https://api.colabfold.com"
"""Public ColabFold MMseqs2 API endpoint."""

UNIREF_A3M_FILENAME = "uniref.a3m"
"""Name of the UniRef alignment inside the server's result tarball."""

ENV_A3M_FILENAME = "bfd.mgnify30.metaeuk30.smag30.a3m"
"""Name of the metagenomic (environmental) alignment inside the server's result tarball."""

_FIRST_QUERY_ID = 101
"""ColabFold servers expect numeric FASTA headers; queries are numbered from this value."""

_PENDING_STATUSES = ("UNKNOWN", "PENDING", "RUNNING")
_RESUBMIT_STATUSES = ("UNKNOWN", "RATELIMIT")

_T = TypeVar("_T")


@dataclasses.dataclass
class MSAServerConfig:
    """Configuration for a remote ColabFold-style MMseqs2 server.

    Args:
        host_url: Base URL of the MMseqs2 server.
        use_env: Whether to include the metagenomic (environmental) database.
        use_filter: Whether to let the server filter the alignment.
        username: Username for HTTP basic auth. Falls back to ``MSA_SERVER_USERNAME``.
        password: Password for HTTP basic auth. Falls back to ``MSA_SERVER_PASSWORD``.
        api_key_header: Name of the header carrying an API key (mutually exclusive with basic auth).
        api_key_value: Value of the API key header. Falls back to ``MSA_SERVER_API_KEY``.
        user_agent: ``User-Agent`` sent with every request. The public server asks clients to identify themselves.
        request_timeout: Per-request timeout, in seconds.
        poll_interval: Lower and upper bound (in seconds) of the jittered delay between status polls.
        max_retries: Maximum number of consecutive failed network calls (or job resubmissions) before giving up.
        retry_delay: Delay between retries of a failed network call, in seconds.
        batch_size: Maximum number of sequences submitted in a single ticket.

    Raises:
        ValueError: If both basic auth and an API key are configured, or if a numeric field is out of range.
    """

    host_url: str = DEFAULT_MSA_SERVER_URL
    use_env: bool = True
    use_filter: bool = True
    username: str | None = None
    password: str | None = None
    api_key_header: str | None = None
    api_key_value: str | None = None
    user_agent: str = "atomworks"
    request_timeout: float = 6.02
    poll_interval: tuple[float, float] = (5.0, 10.0)
    max_retries: int = 5
    retry_delay: float = 5.0
    batch_size: int = 50

    def __post_init__(self) -> None:
        """Resolve credentials from the environment and validate the configuration."""
        # NOTE: we read the environment directly (rather than via `atomworks.constants._load_env_var`)
        # because unset credentials are the common case and should not emit a warning.
        if self.username is None:
            self.username = os.environ.get("MSA_SERVER_USERNAME")
        if self.password is None:
            self.password = os.environ.get("MSA_SERVER_PASSWORD")

        has_basic_auth = self.username is not None or self.password is not None
        if self.api_key_value is None and not has_basic_auth:
            self.api_key_value = os.environ.get("MSA_SERVER_API_KEY")

        if has_basic_auth and self.api_key_value is not None:
            raise ValueError(
                "Cannot use HTTP basic auth (username/password) and an API key header at the same time. "
                "Provide one or the other."
            )
        if self.api_key_value is not None and not self.api_key_header:
            raise ValueError("An API key value was given without `api_key_header`; the header name is required.")

        self.host_url = self.host_url.rstrip("/")

        low, high = self.poll_interval
        if low < 0 or high < low:
            raise ValueError(f"`poll_interval` must be a non-negative (low, high) pair, got {self.poll_interval}")
        if self.batch_size < 1:
            raise ValueError(f"`batch_size` must be at least 1, got {self.batch_size}")
        if self.max_retries < 1:
            raise ValueError(f"`max_retries` must be at least 1, got {self.max_retries}")


def _select_mode(config: MSAServerConfig) -> str:
    """Map the configured databases and filtering onto a ColabFold server search mode."""
    if config.use_filter:
        return "env" if config.use_env else "all"
    return "env-nofilter" if config.use_env else "nofilter"


def _build_query_fasta(sequences: list[str]) -> str:
    """Build the FASTA payload for a batch, using the numeric headers the server expects."""
    return "".join(f">{_FIRST_QUERY_ID + i}\n{sequence}\n" for i, sequence in enumerate(sequences))


def _request_kwargs(config: MSAServerConfig) -> dict[str, Any]:
    """Build the shared `requests` keyword arguments (headers, auth, timeout) for a call."""
    headers = {"User-Agent": config.user_agent}
    if config.api_key_header and config.api_key_value:
        headers[config.api_key_header] = config.api_key_value

    kwargs: dict[str, Any] = {"headers": headers, "timeout": config.request_timeout}
    if config.username is not None or config.password is not None:
        kwargs["auth"] = (config.username or "", config.password or "")
    return kwargs


def _parse_json_response(response: requests.Response) -> dict[str, Any]:
    """Parse a ticket/status response, degrading to an ``ERROR`` status if the server didn't reply with JSON."""
    try:
        payload = response.json()
    except ValueError:
        logger.error(f"MSA server did not reply with JSON (HTTP {response.status_code}): {response.text[:500]}")
        return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:500]}"}

    if not isinstance(payload, dict):
        return {"status": "ERROR", "message": f"Unexpected JSON payload: {payload!r}"}
    return payload


def _submit(sequences: list[str], mode: str, config: MSAServerConfig) -> dict[str, Any]:
    """Submit a batch of sequences to the server.

    Args:
        sequences: Sequences to align (a single batch).
        mode: Server search mode, see :py:func:`_select_mode`.
        config: Server configuration.

    Returns:
        The decoded ticket payload, containing at least a ``status`` and (on success) an ``id``.
    """
    response = requests.post(
        f"{config.host_url}/ticket/msa",
        data={"q": _build_query_fasta(sequences), "mode": mode},
        **_request_kwargs(config),
    )
    return _parse_json_response(response)


def _status(ticket_id: str, config: MSAServerConfig) -> dict[str, Any]:
    """Query the status of a submitted ticket."""
    response = requests.get(f"{config.host_url}/ticket/{ticket_id}", **_request_kwargs(config))
    return _parse_json_response(response)


def _download(ticket_id: str, dest: PathLike, config: MSAServerConfig) -> None:
    """Download the result tarball of a completed ticket to `dest`."""
    with requests.get(
        f"{config.host_url}/result/download/{ticket_id}", stream=True, **_request_kwargs(config)
    ) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def _with_retries(call: Callable[[], _T], config: MSAServerConfig, description: str) -> _T:
    """Call `call`, retrying transient network failures up to `config.max_retries` times.

    Args:
        call: Zero-argument callable performing a single network request.
        config: Server configuration (supplies the retry count and delay).
        description: Human-readable description of the call, used in log and error messages.

    Returns:
        The return value of `call`.

    Raises:
        RuntimeError: If every attempt failed.
    """
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            return call()
        except Exception as e:  # any network failure is worth retrying
            last_error = e
            logger.warning(f"Error while {description} (attempt {attempt}/{config.max_retries}): {e}")
            if attempt < config.max_retries:
                time.sleep(config.retry_delay)

    raise RuntimeError(f"Failed while {description} after {config.max_retries} attempts") from last_error


def _sleep_between_polls(config: MSAServerConfig) -> float:
    """Sleep for a jittered interval between status polls, returning the number of seconds slept."""
    delay = random.uniform(*config.poll_interval)
    time.sleep(delay)
    return delay


def _raise_on_error_status(result: dict[str, Any]) -> None:
    """Raise if the server reported a terminal status.

    Raises:
        RuntimeError: If the ticket is in the ``ERROR`` or ``MAINTENANCE`` state.
    """
    status = result.get("status")
    if status == "ERROR":
        message = result.get("message") or "no message given"
        raise RuntimeError(
            f"MSA server returned an error: {message}. "
            "This usually means the query was malformed or too long for the server."
        )
    if status == "MAINTENANCE":
        raise RuntimeError("MSA server is undergoing maintenance; please retry later.")


def _run_batch(sequences: list[str], tar_path: Path, config: MSAServerConfig) -> None:
    """Submit one batch of sequences, poll until the job completes, and download the result tarball.

    Args:
        sequences: Sequences to align (a single batch, already deduplicated).
        tar_path: Destination for the downloaded ``out.tar.gz``.
        config: Server configuration.

    Raises:
        RuntimeError: If the server reports an error, is under maintenance, or the job never completes.
    """
    mode = _select_mode(config)
    submit = functools.partial(_submit, sequences, mode, config)

    with tqdm(
        desc=f"MMseqs2 server ({len(sequences)} sequences)", unit="s", bar_format="{l_bar}{bar}| {n:.0f}s"
    ) as bar:
        for resubmission in range(1, config.max_retries + 1):
            result = _with_retries(submit, config, "submitting sequences to the MSA server")
            while result.get("status") in _RESUBMIT_STATUSES:
                _sleep_between_polls(config)
                result = _with_retries(submit, config, "submitting sequences to the MSA server")
            _raise_on_error_status(result)

            ticket_id = result.get("id")
            if not ticket_id:
                raise RuntimeError(f"MSA server accepted the job but returned no ticket id: {result!r}")
            poll = functools.partial(_status, ticket_id, config)

            while result.get("status") in _PENDING_STATUSES:
                slept = _sleep_between_polls(config)
                bar.update(slept)
                result = _with_retries(poll, config, f"polling ticket {ticket_id}")

            _raise_on_error_status(result)
            if result.get("status") == "COMPLETE":
                _with_retries(
                    functools.partial(_download, ticket_id, tar_path, config),
                    config,
                    f"downloading results for ticket {ticket_id}",
                )
                return

            logger.warning(
                f"Unexpected status {result.get('status')!r} for ticket {ticket_id}; "
                f"resubmitting ({resubmission}/{config.max_retries})"
            )

    raise RuntimeError(f"MSA server job did not complete after {config.max_retries} submissions")


def _extract_tarball(tar_path: Path, dest_dir: Path) -> None:
    """Extract the regular files of a result tarball into `dest_dir`, flattening any directory structure."""
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, open(dest_dir / Path(member.name).name, "wb") as f:
                shutil.copyfileobj(source, f)


def _split_multi_query_a3m(a3m_path: Path) -> dict[int, list[str]]:
    """Split a multi-query a3m returned by the server into per-query blocks of lines.

    The server concatenates the alignment of every query in a batch into a single file, separated by
    null bytes and re-introduced by the numeric query header (``>101``, ``>102``, ...).

    Args:
        a3m_path: Path to the multi-query a3m file.

    Returns:
        Mapping of numeric query id to the lines of that query's alignment (headers included).

    Raises:
        ValueError: If alignment lines appear before any query header.
    """
    blocks: dict[int, list[str]] = {}
    query_id: int | None = None
    expect_header = True

    with open(a3m_path) as f:
        for raw_line in f:
            line = raw_line
            if "\x00" in line:
                # A null byte marks the boundary between two queries' alignments
                line = line.replace("\x00", "")
                expect_header = True
            if not line.strip():
                continue
            if expect_header and line.startswith(">"):
                query_id = int(line[1:].rstrip())
                expect_header = False
                blocks.setdefault(query_id, [])
            if query_id is None:
                raise ValueError(
                    f"Malformed a3m from the MSA server: alignment lines before a query header in {a3m_path}"
                )
            blocks[query_id].append(line if line.endswith("\n") else f"{line}\n")

    return blocks


def _drop_leading_query_record(block: list[str]) -> list[str]:
    """Drop the repeated query header and sequence from the head of a per-query alignment block."""
    return block[2:] if len(block) >= 2 and block[0].startswith(">") else block


def _write_a3m_files(
    blocks_per_file: list[dict[int, list[str]]], sequences: list[str], out_dir: Path
) -> dict[str, Path]:
    """Concatenate per-database alignments and write one flat ``<hash>.a3m`` per query sequence.

    Args:
        blocks_per_file: Per-query line blocks, one entry per downloaded a3m file (UniRef first).
        sequences: The batch of query sequences, in submission order.
        out_dir: Directory the flat a3m files are written to.

    Returns:
        Mapping of query sequence to the path of its a3m file.

    Raises:
        RuntimeError: If the server returned no alignment at all for one of the queries.
    """
    sequence_to_path: dict[str, Path] = {}

    for i, sequence in enumerate(sequences):
        query_id = _FIRST_QUERY_ID + i
        lines: list[str] = []
        for file_index, blocks in enumerate(blocks_per_file):
            block = blocks.get(query_id)
            if block is None:
                logger.warning(f"MSA server returned no alignment for query {query_id} in result file {file_index}")
                continue
            # The query itself is repeated at the top of every per-database alignment; keep it only once
            lines.extend(block if not lines else _drop_leading_query_record(block))

        if not lines:
            raise RuntimeError(f"MSA server returned no alignment for query {query_id}")

        # Rewrite the numeric query header to the sequence hash, matching the local backend's output
        sequence_hash = hash_sequence(sequence)
        lines[0] = f">{sequence_hash}\n"

        path = out_dir / f"{sequence_hash}{MSAFileExtension.A3M.value}"
        path.write_text("".join(lines))
        sequence_to_path[sequence] = path

    return sequence_to_path


def run_mmseqs2_server(
    sequences: str | list[str],
    output_dir: PathLike,
    config: MSAServerConfig | None = None,
) -> dict[str, Path]:
    """Align sequences with a remote MMseqs2 server and write one flat ``<hash>.a3m`` per sequence.

    This is the low-level entrypoint; it does no sharding, compression or filtering. Most callers
    want :py:func:`make_msas_mmseqs_server` instead.

    Args:
        sequences: A single protein sequence string or list of protein sequences.
        output_dir: Directory the flat a3m files are written to. Created if it doesn't exist.
        config: Server configuration. If None, uses defaults (the public ColabFold server).

    Returns:
        Mapping of each unique input sequence to the path of its a3m file.

    Raises:
        RuntimeError: If the server reports an error or the results are incomplete.
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    if config is None:
        config = MSAServerConfig()

    # Duplicate sequences would waste a server slot each; they map to the same output file anyway
    unique_sequences = list(dict.fromkeys(sequences))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sequence_to_path: dict[str, Path] = {}
    n_batches = math.ceil(len(unique_sequences) / config.batch_size)
    logger.info(
        f"Requesting MSAs for {len(unique_sequences)} unique sequences from {config.host_url} "
        f"in {n_batches} batch(es) (mode: {_select_mode(config)})"
    )

    for batch_index, start in enumerate(range(0, len(unique_sequences), config.batch_size), start=1):
        batch = unique_sequences[start : start + config.batch_size]
        logger.info(f"Submitting batch {batch_index}/{n_batches} ({len(batch)} sequences)")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tar_path = tmp_path / "out.tar.gz"

            start_time = time.time()
            _run_batch(batch, tar_path, config)
            logger.info(f"Batch {batch_index}/{n_batches} completed in {time.time() - start_time:.1f} seconds")

            _extract_tarball(tar_path, tmp_path)

            uniref_a3m = tmp_path / UNIREF_A3M_FILENAME
            if not uniref_a3m.exists():
                raise RuntimeError(f"MSA server result did not contain the expected {UNIREF_A3M_FILENAME}")

            a3m_files = [uniref_a3m]
            if config.use_env:
                env_a3m = tmp_path / ENV_A3M_FILENAME
                if env_a3m.exists():
                    a3m_files.append(env_a3m)
                else:
                    logger.warning(f"MSA server result did not contain {ENV_A3M_FILENAME}; using UniRef hits only")

            blocks_per_file = [_split_multi_query_a3m(a3m_file) for a3m_file in a3m_files]
            sequence_to_path.update(_write_a3m_files(blocks_per_file, batch, out_path))

    return sequence_to_path


def _msa_dirs_to_check(output_dir: Path, existing_msa_dirs: list[PathLike] | None) -> list[PathLike]:
    """Resolve the directories to search for already-generated MSAs.

    The output directory is always checked (it is the natural cache); explicitly requested
    directories are checked in addition, falling back to ``LOCAL_MSA_DIRS`` when none are given.
    """
    if existing_msa_dirs is None:
        existing_msa_dirs = get_msa_dirs_from_env(raise_if_not_set=False) or []
    return [output_dir, *existing_msa_dirs]


def make_msas_mmseqs_server(
    sequences: str | list[str],
    output_dir: PathLike,
    *,
    config: MSAServerConfig | None = None,
    sharding_pattern: str = "/0:2/",
    output_extension: str = MSAFileExtension.A3M_GZ.value,
    max_final_sequences: int | None = None,
    check_existing: bool = True,
    existing_msa_dirs: list[PathLike] | None = None,
) -> None:
    """Generate MSAs from protein sequences using a remote MMseqs2 server.

    Signature-compatible with :py:func:`~atomworks.ml.preprocessing.msa.generating.make_msas_mmseqs`
    (the local backend), so the two are interchangeable. Output is written to the same hash-sharded,
    compressed layout.

    Args:
        sequences: A single protein sequence string or list of protein sequences.
        output_dir: Path to the output directory where MSA files will be saved.
        config: Server configuration. If None, uses defaults (the public ColabFold server).
        sharding_pattern: Directory sharding pattern (e.g., "/0:2/").
        output_extension: Output file extension (.a3m, .a3m.gz, .a3m.zst, .afa, .afa.gz, .afa.zst).
        max_final_sequences: If set, MSAs are filtered down to this many sequences with HHfilter.
            Defaults to None: the server already filters, and HHfilter is a local binary that a
            remote-backend user may not have installed.
        check_existing: Whether to skip sequences that already have an MSA.
        existing_msa_dirs: Additional directories to check for existing MSAs. The output directory is
            always checked. If None, falls back to the LOCAL_MSA_DIRS env var (when set).

    Examples:
        .. code-block:: python

           make_msas_mmseqs_server(
               ["MSYIWRQLGSPTVAITLSVSTVIYVTVICPIVFIHLFGDHL...", "MKKKEVEKDDLIENASRVASCISIFLIIASTTMYIFIGLKI..."],
               "output_msas/",
           )
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    if config is None:
        config = MSAServerConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sequences = list(dict.fromkeys(sequences))

    if check_existing:
        logger.info(f"Finding existing MSAs among {len(sequences)} sequences...")
        sequences, _ = find_msas(
            sequences,
            msa_dirs=_msa_dirs_to_check(output_path, existing_msa_dirs),
            shard_depths=[0, 1, 2, 3, 4],
            extensions=[MSAFileExtension.A3M, MSAFileExtension.A3M_GZ, MSAFileExtension.A3M_ZST],
        )
        if not sequences:
            logger.info("All sequences already have MSAs, skipping generation")
            return
        logger.info(f"Found {len(sequences)} sequences needing MSA generation")

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_mmseqs2_server(sequences, tmp_dir, config)

        # Organize MSAs (hash-based sharding and compression) using existing organization functionality
        logger.info("Organizing MSA files...")
        organize_msas(
            tmp_dir,
            output_path,
            MSAOrganizationConfig(
                input_extension=MSAFileExtension.A3M,
                output_extension=output_extension,
                sharding_pattern=sharding_pattern,
                copy_files=False,  # Move files instead of copying
            ),
        )

    if max_final_sequences is not None:
        logger.info(f"Filtering MSA files to max {max_final_sequences} sequences (requires a local hhfilter)...")
        filter_msas(
            output_path,
            output_path,
            MSAFilterConfig(
                input_extension=output_extension,
                output_extension=output_extension,
                hhfilter=HHFilterConfig(max_sequences=max_final_sequences),
            ),
        )

    logger.info(f"MSA files saved to: {output_path.absolute()}")
