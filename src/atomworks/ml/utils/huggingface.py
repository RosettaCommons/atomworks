"""Authenticated, revision-pinned Hub metadata downloads and blob range reads."""

import os
import re
from functools import lru_cache
from typing import Any


def split_hf_url(url: str) -> tuple[str, str, str]:
    """Parse ``hf://datasets/owner/repo@<40-character commit>/path``.

    Immutable revisions keep sampling tables, blob indices and shards consistent.
    Credentials come from HF_TOKEN or the local Hugging Face login, never the URL.
    """
    match = re.fullmatch(r"hf://datasets/([\w.-]+/[\w.-]+)@([a-f0-9]{40})/(.+)", url)
    if not match or any(part in {"", ".", ".."} for part in match[3].split("/")):
        raise ValueError("Use hf://datasets/owner/repo@<full commit SHA>/path for reproducible blob reads")
    return match[1], match[2], match[3]


def download_hf_metadata(url: str) -> str:
    """Download one pinned metadata file into the Hub's shared, locked local cache."""
    from huggingface_hub import hf_hub_download

    repo_id, revision, path = split_hf_url(url)
    return hf_hub_download(repo_id, path, repo_type="dataset", revision=revision)


@lru_cache(maxsize=1)
def _filesystem(pid: int) -> Any:
    """Keep pooled clients local to each DataLoader process, outside pickled state."""
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def read_hf_bytes(url: str, *, offset: int, length: int) -> bytes:
    """Read exactly one compressed record without downloading its whole shard."""
    split_hf_url(url)
    if offset < 0 or length <= 0:
        raise ValueError("A blob range needs a nonnegative offset and positive length")
    raw = _filesystem(os.getpid()).cat_file(url, start=offset, end=offset + length, cache_type="none")
    if len(raw) != length:
        raise OSError(f"Truncated HF blob range: expected {length} bytes, received {len(raw)}")
    return raw
