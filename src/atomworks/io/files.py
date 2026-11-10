"""Convenient atom-array IO using the parser and shared CIF writer."""

import io
import os

from biotite.structure import AtomArray, AtomArrayStack

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.io_utils import CIFWriteConfig, atom_array_to_file, infer_pdb_file_type

__all__ = ["CIFWriteConfig", "atom_array_from_file", "atom_array_to_file"]


def atom_array_from_file(
    source: str | os.PathLike | io.StringIO | io.BytesIO,
    file_type: str | None = None,
    *,
    model: int | None = None,
    squeeze: bool = True,
) -> AtomArray | AtomArrayStack | list[AtomArray]:
    """Read atoms, bonds, and StandardAnnotations using minimal parsing.

    Keeps waters and hydrogens; adds no missing atoms or assemblies. Uses the first
    alternate conformer and normalized identifiers. StandardAnnotations require CIF-family input.

    Args:
        source: Path or buffer; paths may be gzip or Zstandard compressed.
        file_type: CIF, BinaryCIF, mmJSON, PDB, or catcif; inferred when omitted.
        model: 1-based model selection (negative positions count from the end), or all models.
        squeeze: Return an array for a single model. Explicit model selection always does so.

    Returns:
        A Plus array or stack, or a list when the parser returns separate models.
    """
    # Let parse resolve catcif entries; their contents are CIF.
    if isinstance(source, str | os.PathLike) and ".catcif:" in str(source):
        file_type = "cif" if file_type in (None, "catcif") else file_type
    file_type = file_type or infer_pdb_file_type(source)
    if file_type not in ("cif", "bcif", "mmjson", "pdb"):
        raise ValueError(f"Cannot read a {file_type!r} file as an atom array; use atomworks.io.parse instead.")
    result = parse(
        source,
        config=ParseConfig.from_preset(
            "minimal",
            file_type=file_type,
            model=model,
            build_assembly=None,
            add_bond_types_from_struct_conn=("covale", "disulf", "metalc"),
            struct_conn_distance_policy="keep",
            extra_fields="all",
            load_standard_annotations=file_type != "pdb",
            return_atom_array_plus=True,
        ),
    )
    if isinstance(result, list):
        return [entry["asym_unit"][0] for entry in result]
    atoms = result["asym_unit"]
    if isinstance(atoms, AtomArrayStack) and (model is not None or (squeeze and atoms.stack_depth() == 1)):
        return atoms[0]
    return atoms
