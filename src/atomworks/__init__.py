"""Unified package for biological data I/O and machine learning.

This package combines functionality from :mod:`atomworks.io` (I/O operations) and
:mod:`atomworks.ml` (ML utilities) into a unified interface for biological data
processing and machine learning.
"""

import importlib
import importlib.metadata
import logging
import os
import warnings

try:
    __version__ = importlib.metadata.version("atomworks")
except ImportError:
    __version__ = "unknown"

# Global logging configuration
logger = logging.getLogger("atomworks")
_log_level = os.environ.get("ATOMWORKS_LOG_LEVEL", "WARNING").upper()
logger.setLevel(_log_level)

# Ensure that deprecation warnings are not repeated
warnings.filterwarnings("once", category=DeprecationWarning)

# Apply monkey patching to extend AtomArray functionality
from atomworks.biotite_patch import monkey_patch_biotite  # noqa: E402

monkey_patch_biotite()


# Import version information
# Import subpackages
from . import io, ml  # noqa: E402

# Re-export key functionality from subpackages for convenience
# This maintains backward compatibility and provides a clean top-level API
# Key I/O functionality
from .io.parser import parse  # noqa: E402

__all__ = [
    "__version__",
    "io",
    "ml",
    "monkey_patch_atomarray",
    "parse",
]
