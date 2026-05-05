"""ASE-based loader utilities."""

import functools
import re
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

_SPACE_GROUP_KEYS = (
    "space_group",
    "spacegroup",
    "space_group_number",
    "spacegroup_number",
    "spg",
    "spg_num",
    "sg",
    "sg_number",
)

_PARENT_SPACE_GROUP_KEYS = tuple(f"parent_{key}" for key in _SPACE_GROUP_KEYS)


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


def _as_plain_dict(value: Any) -> dict[str, Any]:
    """Convert mapping-like metadata to a plain dictionary."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _extract_metadata(raw_data: Any, atoms: Any) -> dict[str, Any]:
    """Collect metadata from an ASE row/record and the corresponding Atoms object."""
    metadata: dict[str, Any] = {}

    info_data = atoms.info.get("data", {}) if hasattr(atoms, "info") else {}
    metadata.update(_as_plain_dict(info_data))
    if hasattr(atoms, "info"):
        metadata.update({key: val for key, val in atoms.info.items() if key != "data"})

    if isinstance(raw_data, Mapping):
        metadata.update(_as_plain_dict(raw_data.get("key_value_pairs", {})))
        metadata.update(_as_plain_dict(raw_data.get("data", {})))
        metadata.update(_as_plain_dict(raw_data.get("extra_info", {})))
    else:
        metadata.update(_as_plain_dict(getattr(raw_data, "key_value_pairs", {})))
        if hasattr(raw_data, "get"):
            metadata.update(_as_plain_dict(raw_data.get("data", {})))

    return metadata


def _coerce_space_group_number(value: Any) -> int | None:
    """Convert a metadata value into an integer space-group number when possible."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match is not None:
            return int(match.group(0))
    return None


def _parse_space_group_from_prototype_label(prototype_label: Any) -> int | None:
    """Parse AFLOW-style prototype labels such as ``A6B_aP14_1_12a_2a:B-Na``."""
    if not isinstance(prototype_label, str):
        return None
    label = prototype_label.split(":", maxsplit=1)[0]
    label_parts = label.split("_")
    if len(label_parts) < 3:
        return None
    return _coerce_space_group_number(label_parts[2])


def _parse_space_group_from_parent_id(parent_id: Any) -> int | None:
    """Parse parent IDs containing fragments such as ``spg221``."""
    if not isinstance(parent_id, str):
        return None
    match = re.search(r"(?:^|_)spg(?P<space_group>\d+)(?:_|$)", parent_id)
    if match is None:
        return None
    return int(match.group("space_group"))


def _infer_space_group_number(metadata: Mapping[str, Any]) -> int | None:
    """Infer the current structure space-group number from metadata."""
    for key in _SPACE_GROUP_KEYS:
        if key in metadata:
            space_group = _coerce_space_group_number(metadata[key])
            if space_group is not None:
                return space_group

    space_group = _parse_space_group_from_prototype_label(metadata.get("prototype_label"))
    if space_group is not None:
        return space_group

    return _infer_parent_space_group_number(metadata)


def _infer_parent_space_group_number(metadata: Mapping[str, Any]) -> int | None:
    """Infer the parent/source structure space-group number from metadata."""
    for key in _PARENT_SPACE_GROUP_KEYS:
        if key in metadata:
            space_group = _coerce_space_group_number(metadata[key])
            if space_group is not None:
                return space_group

    space_group = _parse_space_group_from_prototype_label(metadata.get("parent_prototype_label"))
    if space_group is not None:
        return space_group

    return _parse_space_group_from_parent_id(metadata.get("parent_id"))


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


def ase_atoms_to_material_dict(
    atoms: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    wrap_fractional_coordinates: bool = False,
) -> dict[str, Any]:
    """Parse materials-science features from an ASE ``Atoms`` object.

    Args:
        atoms: ASE ``Atoms`` object with a periodic cell.
        metadata: Optional row metadata used to infer space-group labels.
        wrap_fractional_coordinates: Whether to wrap scaled positions into the
            unit cell before returning them.

    Returns:
        Dictionary containing fractional coordinates, lattice vectors, lattice
        lengths, lattice angles, combined cell parameters, volume, atomic
        numbers, chemical symbols, and inferred space-group metadata.
    """
    atoms_type = _require_ase_atoms_type()
    if not isinstance(atoms, atoms_type):
        raise TypeError(f"Expected ASE Atoms, got {type(atoms)}.")

    metadata = metadata or {}
    lattice_lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    lattice_angles = np.asarray(atoms.cell.angles(), dtype=float)

    return {
        "fractional_coordinates": np.asarray(atoms.get_scaled_positions(wrap=wrap_fractional_coordinates), dtype=float),
        "cartesian_coordinates": np.asarray(atoms.get_positions(), dtype=float),
        "lattice_vectors": np.asarray(atoms.cell, dtype=float),
        "lattice_lengths": lattice_lengths,
        "lattice_angles": lattice_angles,
        "cell_parameters": np.concatenate([lattice_lengths, lattice_angles]),
        "cell_volume": float(atoms.cell.volume),
        "pbc": np.asarray(atoms.pbc, dtype=bool),
        "atomic_numbers": np.asarray(atoms.get_atomic_numbers(), dtype=int),
        "chemical_symbols": np.asarray(atoms.get_chemical_symbols(), dtype="<U3"),
        "space_group": _infer_space_group_number(metadata),
        "parent_space_group": _infer_parent_space_group_number(metadata),
    }


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


def _ase_materials_loader_function(
    raw_data: Any,
    wrap_fractional_coordinates: bool,
    include_atom_array: bool,
    chain_id: str,
    res_name: str,
    res_id: int,
    chain_type: ChainType | str | int,
    is_polymer: bool,
    include_chain_info: bool,
    keep_ase_atoms: bool,
) -> dict[str, Any]:
    """Loader implementation for periodic ASE materials records."""
    atoms = _extract_ase_atoms(raw_data)
    metadata = _extract_metadata(raw_data, atoms)
    material_dict = ase_atoms_to_material_dict(
        atoms,
        metadata=metadata,
        wrap_fractional_coordinates=wrap_fractional_coordinates,
    )

    if isinstance(raw_data, Mapping):
        result = dict(raw_data)
    else:
        result = {}

    result.update(material_dict)

    if include_atom_array:
        atom_array = ase_atoms_to_atom_array(
            atoms,
            chain_id=chain_id,
            res_name=res_name,
            res_id=res_id,
            chain_type=chain_type,
            is_polymer=is_polymer,
        )
        result["atom_array"] = atom_array
        if include_chain_info:
            result["chain_info"] = initialize_chain_info_from_atom_array(atom_array)

    if keep_ase_atoms:
        result.setdefault("atoms", atoms)
    else:
        result.pop("atoms", None)
        result.pop("ase_atoms", None)

    return result


def create_ase_materials_loader(
    *,
    wrap_fractional_coordinates: bool = False,
    include_atom_array: bool = False,
    chain_id: str = "A",
    res_name: str = UNKNOWN_LIGAND,
    res_id: int = 1,
    chain_type: ChainType | str | int = ChainType.NON_POLYMER,
    is_polymer: bool = False,
    include_chain_info: bool = True,
    keep_ase_atoms: bool = True,
) -> Callable[[Any], dict[str, Any]]:
    """Create a loader for periodic materials records stored as ASE ``Atoms``.

    The loader adds crystal/material fields such as fractional coordinates,
    lattice lengths and angles, and parsed space-group numbers. It accepts ASE
    ``Atoms``, ASE ``AtomsRow`` objects, or records emitted by
    :class:`atomworks.ml.datasets.ASELMDBDataset`.
    """
    return functools.partial(
        _ase_materials_loader_function,
        wrap_fractional_coordinates=wrap_fractional_coordinates,
        include_atom_array=include_atom_array,
        chain_id=chain_id,
        res_name=res_name,
        res_id=res_id,
        chain_type=chain_type,
        is_polymer=is_polymer,
        include_chain_info=include_chain_info,
        keep_ase_atoms=keep_ase_atoms,
    )
