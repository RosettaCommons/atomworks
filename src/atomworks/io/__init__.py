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


# We need to import parse here to ensure that the version string is set
from atomworks.io.parser import parse  # noqa: E402
