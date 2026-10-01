"""Input/Output operations for biological data structures.

This subpackage provides functionality for parsing, converting, and manipulating
biological data formats, originally from the atomworks.io package.
"""

import logging
import os
import warnings

# Set global logging level to `WARNING` if not set by user
logger = logging.getLogger("atomworks.io")
_log_level = os.environ.get("ATOMWORKS_IO_LOG_LEVEL", os.environ.get("ATOMWORKS_LOG_LEVEL", "WARNING")).upper()
logger.setLevel(_log_level)
# ... ensure that deprecation warnings are not repeated
warnings.filterwarnings("once", category=DeprecationWarning)


# Expose loading separately from preparation; importing parse also sets the version string.
from atomworks.io._loaders import load_cif, load_pdb  # noqa: E402
from atomworks.io.parser import get_config, parse, parse_atom_array, prepare_atom_array  # noqa: E402

__all__ = [
    "get_config",
    "load_cif",
    "load_pdb",
    "parse",
    "parse_atom_array",
    "prepare_atom_array",
]
