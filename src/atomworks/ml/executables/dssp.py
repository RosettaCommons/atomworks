"""DSSP executable wrapper for secondary structure annotation."""

import logging
import os
from os import PathLike

from atomworks.ml.executables import Executable, ExecutableError

logger = logging.getLogger(__name__)


class DSSPExecutable(Executable):
    """Executable wrapper for the DSSP program.

    DSSP (Define Secondary Structure of Proteins) is used to annotate secondary
    structure elements in protein structures based on hydrogen bonding patterns.

    Examples:
        >>> dssp = DSSPExecutable.get_or_initialize()
        >>> version = dssp.get_version()
        >>> bin_path = dssp.get_bin_path()
    """

    name = "mkdssp"
    required_verification_text = ("DSSP", "output-format")
    version_cmd = "--version"
    verification_cmd = "--help"

    @classmethod
    def initialize(cls, bin_path: PathLike | None = None, *args, **kwargs) -> "DSSPExecutable":
        """Initialize DSSP executable.

        Args:
          bin_path: Path to DSSP executable. If ``None``, attempts to find using ``DSSP_PATH`` env variable.

        Returns:
          Initialized DSSPExecutable.

        Raises:
          ExecutableError: If executable not found or invalid.
        """
        if bin_path is None:
            bin_path = cls._infer_bin_path_from_env_var()
        return super().initialize(bin_path, *args, **kwargs)

    @staticmethod
    def _infer_bin_path_from_env_var() -> PathLike:
        """Get the path to the DSSP executable from environment variables."""
        dssp_path = os.environ.get("DSSP_PATH")
        if dssp_path is not None and os.path.isfile(dssp_path) and os.access(dssp_path, os.X_OK):
            return dssp_path

        raise ExecutableError(
            "No `bin_path` provided and `DSSP_PATH` environment variable not set.\n"
            "Please set the `DSSP_PATH` environment variable to the path of the DSSP executable "
            "or provide a `bin_path` to the `DSSPExecutable` constructor: "
            "`DSSPExecutable.initialize(bin_path='/path/to/mkdssp')`."
        )

    @classmethod
    def _setup(cls, bin_path: PathLike, *args, **kwargs) -> None:
        """Setup method for DSSP (no special setup required)."""
        pass
