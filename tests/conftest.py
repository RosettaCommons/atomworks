"""Test fixtures and utilities for atomworks tests."""

import gc
import logging
import os
import pathlib
import socket

import pytest

logger = logging.getLogger(__name__)

TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


# Conditional skip markers ----------------------------------------------------------
def _is_on_digs() -> bool:
    """Check if running on DIGS infrastructure.

    Returns:
        True if running on DIGS infrastructure, False otherwise.
    """
    return os.path.exists("/software/containers/versions/rf_diffusion_aa/ipd.txt")


skip_if_not_on_digs = pytest.mark.skipif(not _is_on_digs(), reason="Test requires DIGS infrastructure")


def _is_on_github_runner() -> bool:
    """Check if running on GitHub Actions runner.

    Returns:
        True if running on GitHub Actions runner, False otherwise.
    """
    return os.environ.get("GITHUB_ACTIONS", "false") == "true"


skip_if_on_github_runner = pytest.mark.skipif(
    _is_on_github_runner(),
    reason="Temporarily deactivated on github runners due to memory constraints on the free plan.",
)


def _has_internet_connection() -> bool:
    """Check if internet connection is available.

    Returns:
        True if internet connection is available, False otherwise.
    """
    try:
        # Try to connect to a well-known DNS server (Google's)
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


skip_if_no_internet = pytest.mark.skipif(not _has_internet_connection(), reason="Test requires an internet connection.")


def _has_gpu() -> bool:
    """Check if GPU is available.

    Returns:
        True if GPU is available, False otherwise.
    """
    import torch

    return torch.cuda.is_available()


skip_if_no_gpu = pytest.mark.skipif(not _has_gpu(), reason="Test requires a GPU")


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Force garbage collection after each test"""
    yield
    gc.collect()
