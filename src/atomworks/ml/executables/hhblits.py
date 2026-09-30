"""HHblits executable wrapper for MSA generation."""

import logging
import os
import subprocess
from os import PathLike

from atomworks.ml.executables import Executable, ExecutableError

logger = logging.getLogger(__name__)


class HHblits(Executable):
    """Executable wrapper for the HHblits program from HH-suite.

    HHblits is used to perform iterative sequence searches against large
    databases (UniRef30, BFD) to generate multiple sequence alignments (MSAs)
    for protein structure prediction.

    Example:
        ```python
        hhblits = HHblits.get_or_initialize()
        result = hhblits.run_command("-i", "input.fasta", "-oa3m", "output.a3m", "-d", "/path/to/db", "-e", "1e-3")
        ```
    """

    name = "hhblits"
    required_verification_text = ("HHblits", "-i", "-d")
    version_cmd = "-h"

    @classmethod
    def initialize(cls, bin_path: PathLike | None = None, *args, **kwargs) -> "HHblits":
        """Initialize HHblits executable.

        Args:
            bin_path: Path to hhblits executable. If None, attempts to find using HHBLITS_PATH env variable.

        Returns:
            Initialized HHblits executable.

        Raises:
            ExecutableError: If executable not found or invalid.
        """
        if bin_path is None:
            bin_path = cls._infer_bin_path_from_env_var()
        return super().initialize(bin_path, *args, **kwargs)

    @staticmethod
    def _infer_bin_path_from_env_var() -> PathLike:
        """Get the path to the hhblits executable from environment variables."""
        hhblits_path = os.environ.get("HHBLITS_PATH")
        if hhblits_path is not None and os.path.isfile(hhblits_path) and os.access(hhblits_path, os.X_OK):
            return hhblits_path

        raise ExecutableError(
            "No `bin_path` provided and `HHBLITS_PATH` environment variable not set.\n"
            "Please set the `HHBLITS_PATH` environment variable to the path of the hhblits executable "
            "or provide a `bin_path` to the `HHblits` constructor: "
            "`HHblits.initialize(bin_path='/path/to/hhblits')`."
        )

    @classmethod
    def run_command(cls, *args: str) -> subprocess.CompletedProcess:
        """Run hhblits with the specified arguments.

        Args:
            *args: Command line arguments to pass to hhblits.

        Returns:
            CompletedProcess instance with command output.

        Raises:
            ExecutableError: If HHblits not initialized.
            subprocess.CalledProcessError: If the command fails.
        """
        if not cls._is_initialized:
            raise ExecutableError("HHblits not initialized. Run `HHblits.initialize(...)` first.")

        cmd = [cls._bin_path, *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    @classmethod
    def _setup(cls, bin_path: PathLike, *args, **kwargs) -> None:
        """No additional setup required for HHblits."""
