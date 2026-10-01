"""Test fixtures and utilities for atomworks tests."""

import gc
import logging
import os
import pathlib
import socket

# Standardize test environment before sub-conftest files import atomworks.
# Use Biotite's built-in CCD. For PDB_MIRROR_PATH, resolve in priority order:
#   1. An existing PDB_MIRROR_PATH env var that points at a real directory
#      (local devs with a custom mirror).
#   2. The DIGS-specific frozen PDB mirror, if present (used for the CI and available to devs with access to the DIGS cluster)
#   3. The local test data bundle (populated by `atomworks setup tests` or CI).
os.environ["ALLOW_BIOTITE_CCD"] = "True"
_ccd_mirror = os.environ.get("CCD_MIRROR_PATH", "")
if _ccd_mirror and not os.path.exists(_ccd_mirror):
    del os.environ["CCD_MIRROR_PATH"]

_LAB_PDB_MIRROR = "/projects/ml/frozen_pdb_copies/2026_01_06_pdb"
_LOCAL_PDB_MIRROR = pathlib.Path(__file__).resolve().parent / "data" / "pdb"
_pdb_mirror = os.environ.get("PDB_MIRROR_PATH", "")
if not _pdb_mirror or not os.path.exists(_pdb_mirror):
    if os.path.isdir(_LAB_PDB_MIRROR):
        os.environ["PDB_MIRROR_PATH"] = _LAB_PDB_MIRROR
    elif _LOCAL_PDB_MIRROR.is_dir():
        os.environ["PDB_MIRROR_PATH"] = str(_LOCAL_PDB_MIRROR)
    # Otherwise leave PDB_MIRROR_PATH unset so tests surface a clearer
    # "run `atomworks setup tests`" error rather than pointing at a missing dir.

import pytest

logger = logging.getLogger(__name__)

TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


# Conditional skip markers ----------------------------------------------------------
def _is_on_github_runner() -> bool:
    """Check if running on GitHub Actions runner."""
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


def _has_ccd_mirror() -> bool:
    """Check whether a usable local CCD mirror is configured."""
    path = os.environ.get("CCD_MIRROR_PATH", "")
    return bool(path) and os.path.isdir(path)


skip_if_no_ccd_mirror = pytest.mark.skipif(
    not _has_ccd_mirror(),
    reason="Test requires a local CCD mirror (set CCD_MIRROR_PATH to a valid directory).",
)


def _has_catcif_tools() -> bool:
    try:
        import catcif_tools  # noqa: F401

        return True
    except ImportError:
        return False


skip_if_no_catcif = pytest.mark.skipif(
    not _has_catcif_tools(),
    reason="Test requires catcif_tools (install atomworks[catcif]).",
)


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Force garbage collection after each test"""
    yield
    gc.collect()
