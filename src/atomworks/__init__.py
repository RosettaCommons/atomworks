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


# Hard-enforce biotite version compatibility by importing biotite directly.
# Atomworks monkey-patches several private biotite internals that are sensitive to exact version alignment
_REQUIRED_BIOTITE_VERSION = "1.6.0"
try:
    import biotite as _biotite_check
except ImportError:
    raise ImportError(
        f"atomworks requires biotite=={_REQUIRED_BIOTITE_VERSION}, but biotite is not installed.\n"
        f"    pip install 'biotite=={_REQUIRED_BIOTITE_VERSION}'"
    ) from None
if _biotite_check.__version__ != _REQUIRED_BIOTITE_VERSION:
    raise ImportError(
        f"atomworks requires biotite=={_REQUIRED_BIOTITE_VERSION}, "
        f"but biotite {_biotite_check.__version__!r} is installed.\n"
        f"    pip install 'biotite=={_REQUIRED_BIOTITE_VERSION}'"
    )
del _biotite_check

# Global logging configuration
logger = logging.getLogger("atomworks")
_log_level = os.environ.get("ATOMWORKS_LOG_LEVEL", "WARNING").upper()
logger.setLevel(_log_level)

# Ensure that deprecation warnings are not repeated
warnings.filterwarnings("once", category=DeprecationWarning)

# Enforce strict biotite version — atomworks patches biotite internals
import biotite  # noqa: E402

_REQUIRED_BIOTITE_VERSION = "1.6.0"
if biotite.__version__ != _REQUIRED_BIOTITE_VERSION:
    raise RuntimeError(
        f"atomworks requires biotite=={_REQUIRED_BIOTITE_VERSION}, "
        f"but found biotite=={biotite.__version__}. "
        f"Install the correct version: pip install biotite=={_REQUIRED_BIOTITE_VERSION}"
    )

# Apply monkey patching to extend AtomArray functionality
from atomworks.biotite_patch import monkey_patch_biotite  # noqa: E402

monkey_patch_biotite()


# Import subpackages; also import annotators and Conditions to ensure they're registered
from atomworks.io.utils import annotator as _annotator_io  # noqa: E402
from atomworks.io.utils.standard_annotations import definitions as _standard_annotation_definitions  # noqa: E402

from . import io  # noqa: E402

# Make atomworks.ml an optional import
try:
    from atomworks.ml.conditions import definitions as _condition_definitions
    from atomworks.ml.utils import annotator as _annotator_ml

    from . import ml
except ImportError:
    pass

# Re-export key functionality from subpackages for convenience
# This maintains backward compatibility and provides a clean top-level API
# Key I/O functionality
from .io.parser import parse  # noqa: E402

__all__ = [
    "__version__",
    "io",
    "ml",
    "parse",
]
