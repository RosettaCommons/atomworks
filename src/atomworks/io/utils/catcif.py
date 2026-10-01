"""Utilities for reading and writing catcif archives."""

import io
import os
from pathlib import Path
from typing import Any

from biotite.structure import AtomArray


def _resolve_catcif_path(path: os.PathLike) -> str:
    """Obtain the CIF string pointed to by a catcif path."""
    import catcif_tools

    return catcif_tools.get_structure(str(path))


def resolve_catcif_source(
    source: os.PathLike | io.StringIO | io.BytesIO,
) -> tuple[io.StringIO | os.PathLike | io.BytesIO, str | None]:
    """Resolve a catcif path to a StringIO and return the inferred file type.

    If ``source`` is a catcif path (contains ``.catcif:``), reads the embedded
    CIF string and returns an ``(io.StringIO, 'cif')`` pair. Otherwise returns
    ``(source, None)`` unchanged.
    """
    if isinstance(source, str | Path) and ".catcif:" in str(source):
        return io.StringIO(_resolve_catcif_path(source)), "cif"
    return source, None


def to_catcif_file(
    structure: AtomArray | list[AtomArray],
    path: os.PathLike,
    id: str,
    *,
    scores: dict[str, Any] | None = None,
    compress: bool = False,
    **kwargs,
) -> str:
    """Write a structure into a catcif archive and return its catcif path.

    Args:
        structure: The atomic structure (or list of structures) to write.
        path: Path to the ``.catcif`` archive file. Must end with ``.catcif``.
        id: Tag identifying the entry inside the archive. Must be non-empty.
        scores: Optional score dictionary to embed alongside the structure.
        compress: Whether to write the entry in compressed form.
        **kwargs: Additional keyword arguments forwarded to
            :py:class:`~atomworks.io.utils.io_utils.CIFWriteConfig` (e.g. ``save_standard_annotations``).

    Returns:
        The ``path:id`` catcif path of the written entry.
    """
    # as an optional dependency, catcif_tools is imported as needed
    import catcif_tools

    # avoid circular import
    from atomworks.io.utils.io_utils import CIFWriteConfig, _to_cif_or_bcif

    if not id:
        raise ValueError("Structures in catcif files must have ids")

    # Turn any relative path into an absolute path
    path = str(os.path.abspath(path))
    if not path.endswith(".catcif"):
        raise ValueError(f"path passed to to_catcif_file() must end with .catcif: {path}")

    # catcif always stores CIF; reject any incompatible file_type
    file_type = kwargs.pop("file_type", None)
    if file_type is not None and file_type not in ("cif", "catcif"):
        raise ValueError(f"to_catcif_file() only supports file_type 'cif' or 'catcif', got: {file_type!r}")
    model_ids = kwargs.pop("model_ids", None)

    # Build config from parameters
    config = CIFWriteConfig(id=id, **kwargs)

    # Generate the cif text
    file_obj = _to_cif_or_bcif(structure, config, model_ids=model_ids)
    buffer = io.StringIO()
    file_obj.write(buffer)

    catcif_tools.append_to_catcif_file(path, buffer.getvalue(), id, scores=scores, compress=compress)

    return f"{path}:{id}"
