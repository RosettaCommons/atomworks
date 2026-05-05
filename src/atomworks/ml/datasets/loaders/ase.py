"""ASE-based loader utilities."""

import functools
from collections.abc import Callable, Mapping
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from atomworks.constants import UNKNOWN_LIGAND
from atomworks.enums import ChainType
from atomworks.io.utils.non_rcsb import initialize_chain_info_from_atom_array

_ASE_IMPORT_ERROR = (
    "ASE loader support requires the optional ASE dependencies. "
    'Install them with `pip install "atomworks[ase]"` or `uv pip install "atomworks[ase]"`.'
)


def _require_ase_atoms_type() -> type:
    """Import ASE Atoms lazily so this loader module remains importable without ASE."""
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(_ASE_IMPORT_ERROR) from exc
    return Atoms


def _make_atom_names(elements: np.ndarray) -> np.ndarray:
    """Create stable PDB-like atom names from element symbols."""
    counts: dict[str, int] = {}
    atom_names: list[str] = []
    for element in elements:
        element_str = str(element).upper()
        counts[element_str] = counts.get(element_str, 0) + 1
        atom_names.append(f"{element_str}{counts[element_str]}")
    return np.asarray(atom_names, dtype="<U8")


def _get_initial_charges(atoms: Any, n_atoms: int) -> np.ndarray:
    """Return ASE initial charges when available, otherwise zeros."""
    try:
        charges = np.asarray(atoms.get_initial_charges(), dtype=float)
    except Exception:
        return np.zeros(n_atoms, dtype=float)

    if charges.shape != (n_atoms,):
        return np.zeros(n_atoms, dtype=float)
    return charges


def _extract_ase_atoms(raw_data: Any) -> Any:
    """Extract an ASE Atoms object from supported raw dataset outputs."""
    atoms_type = _require_ase_atoms_type()

    if isinstance(raw_data, atoms_type):
        return raw_data

    if isinstance(raw_data, Mapping):
        for key in ("atoms", "ase_atoms"):
            atoms = raw_data.get(key)
            if isinstance(atoms, atoms_type):
                return atoms
        row = raw_data.get("ase_row")
        if row is not None and hasattr(row, "toatoms"):
            return row.toatoms(add_additional_information=True)

    if hasattr(raw_data, "toatoms"):
        return raw_data.toatoms(add_additional_information=True)

    raise TypeError(
        "Expected an ASE Atoms object, an ASE AtomsRow, or a mapping containing "
        "an `atoms`, `ase_atoms`, or `ase_row` entry."
    )


def ase_atoms_to_atom_array(
    atoms: Any,
    *,
    chain_id: str = "A",
    res_name: str = UNKNOWN_LIGAND,
    res_id: int = 1,
    chain_type: ChainType | str | int = ChainType.NON_POLYMER,
    is_polymer: bool = False,
) -> AtomArray:
    """Convert an ASE ``Atoms`` object into a minimally annotated Biotite ``AtomArray``.

    ASE structures from molecular datasets generally do not carry PDB-style
    residue annotations. This function represents the full structure as one
    non-polymer residue by default, while preserving coordinates, elements,
    atomic numbers, initial charges, and an empty bond list.
    """
    atoms_type = _require_ase_atoms_type()
    if not isinstance(atoms, atoms_type):
        raise TypeError(f"Expected ASE Atoms, got {type(atoms)}.")

    n_atoms = len(atoms)
    atom_array = struc.AtomArray(n_atoms)

    elements = np.asarray(atoms.get_chemical_symbols(), dtype="<U3")
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)

    atom_array.coord = np.asarray(atoms.get_positions(), dtype=float)
    atom_array.element = elements
    atom_array.atom_name = _make_atom_names(elements)
    atom_array.res_name = np.full(n_atoms, res_name, dtype="<U8")
    atom_array.res_id = np.full(n_atoms, res_id, dtype=int)
    atom_array.chain_id = np.full(n_atoms, chain_id, dtype="<U4")
    atom_array.hetero = np.full(n_atoms, True)

    atom_array.set_annotation("atomic_number", atomic_numbers)
    atom_array.set_annotation("charge", _get_initial_charges(atoms, n_atoms))
    atom_array.set_annotation("occupancy", np.ones(n_atoms, dtype=float))
    atom_array.set_annotation("b_factor", np.full(n_atoms, np.nan, dtype=float))
    atom_array.set_annotation("is_polymer", np.full(n_atoms, is_polymer, dtype=bool))
    atom_array.set_annotation("chain_type", np.full(n_atoms, int(ChainType.as_enum(chain_type)), dtype=int))
    atom_array.set_annotation("stereo", np.full(n_atoms, "N", dtype="<U1"))
    atom_array.set_annotation("is_backbone_atom", np.full(n_atoms, False, dtype=bool))
    atom_array.bonds = struc.BondList(n_atoms, np.empty((0, 3), dtype=int))

    return atom_array


def _ase_atoms_loader_function(
    raw_data: Any,
    chain_id: str,
    res_name: str,
    res_id: int,
    chain_type: ChainType | str | int,
    is_polymer: bool,
    include_chain_info: bool,
    keep_ase_atoms: bool,
) -> dict[str, Any]:
    """Loader implementation for ASE atoms records."""
    atoms = _extract_ase_atoms(raw_data)
    atom_array = ase_atoms_to_atom_array(
        atoms,
        chain_id=chain_id,
        res_name=res_name,
        res_id=res_id,
        chain_type=chain_type,
        is_polymer=is_polymer,
    )

    if isinstance(raw_data, Mapping):
        result = dict(raw_data)
    else:
        result = {}

    if keep_ase_atoms:
        result.setdefault("atoms", atoms)
    else:
        result.pop("atoms", None)
        result.pop("ase_atoms", None)

    result["atom_array"] = atom_array
    if include_chain_info:
        result["chain_info"] = initialize_chain_info_from_atom_array(atom_array)
    return result


def create_ase_atoms_loader(
    *,
    chain_id: str = "A",
    res_name: str = UNKNOWN_LIGAND,
    res_id: int = 1,
    chain_type: ChainType | str | int = ChainType.NON_POLYMER,
    is_polymer: bool = False,
    include_chain_info: bool = True,
    keep_ase_atoms: bool = True,
) -> Callable[[Any], dict[str, Any]]:
    """Create a picklable loader that converts ASE ``Atoms`` into ``AtomArray`` data.

    The returned loader accepts ASE ``Atoms``, ASE ``AtomsRow`` objects, or the
    record dictionaries emitted by :class:`atomworks.ml.datasets.ASELMDBDataset`.
    """
    return functools.partial(
        _ase_atoms_loader_function,
        chain_id=chain_id,
        res_name=res_name,
        res_id=res_id,
        chain_type=chain_type,
        is_polymer=is_polymer,
        include_chain_info=include_chain_info,
        keep_ase_atoms=keep_ase_atoms,
    )
