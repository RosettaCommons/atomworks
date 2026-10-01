"""MMseqs2 executable wrapper for fast and sensitive sequence searching and clustering."""

import logging
import os
from os import PathLike

from atomworks.ml.executables import Executable, ExecutableError

logger = logging.getLogger(__name__)


# Mapping of MMseqs2 modules to their output parameter positions
# Used to detect existing outputs and skip redundant operations
MODULE_OUTPUT_POS = {
    "align": 4,
    "convertalis": 4,
    "expandaln": 5,
    "filterresult": 4,
    "lndb": 2,
    "mergedbs": 2,
    "mvdb": 2,
    "pairaln": 4,
    "result2msa": 4,
    "search": 3,
}


class MMseqs2(Executable):
    """
    Executable wrapper for the MMseqs2 program.

    MMseqs2 (Many-against-Many sequence searching) is a software suite for
    ultra-fast and sensitive protein sequence searching and clustering.

    Example:
        ```python
        mmseqs2 = MMseqs2.get_or_initialize()
        version = mmseqs2.get_version()
        bin_path = mmseqs2.get_bin_path()

        # Run mmseqs2 with parameters using the new run_command method
        result = mmseqs2.run_command("createdb", "input.fasta", "seq_db")
        ```
    """

    name = "mmseqs"
    required_verification_text = ("MMseqs2", "Version:", "Steinegger")
    version_cmd = "-h"  # MMseqs2 shows version in help output

    @classmethod
    def initialize(cls, bin_path: PathLike | None = None, *args, **kwargs) -> "MMseqs2":
        """Initialize MMseqs2 executable.

        Args:
            bin_path: Path to mmseqs executable. If None, attempts to find using MMSEQS2_PATH env variable.

        Returns:
            Initialized MMseqs2 executable.

        Raises:
            ExecutableError: If executable not found or invalid.
        """
        if bin_path is None:
            bin_path = cls._infer_bin_path_from_env_var()
        return super().initialize(bin_path, *args, **kwargs)

    @staticmethod
    def _infer_bin_path_from_env_var() -> PathLike:
        """Get the path to the mmseqs executable from environment variables."""
        mmseqs_path = os.environ.get("MMSEQS2_PATH")
        if mmseqs_path is not None and os.path.isfile(mmseqs_path) and os.access(mmseqs_path, os.X_OK):
            return mmseqs_path

        raise ExecutableError(
            "No `bin_path` provided and `MMSEQS2_PATH` environment variable not set.\n"
            "Please set the `MMSEQS2_PATH` environment variable to the path of the MMseqs2 executable "
            "or provide a `bin_path` to the `MMseqs2` constructor: "
            "`MMseqs2(bin_path='/path/to/mmseqs')`."
        )
