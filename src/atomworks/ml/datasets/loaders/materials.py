"""Periodic materials loaders for the existing ASE dataset interface."""

import functools
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def _space_group_label(value: Any) -> int | None:
    """Read a numeric source label without estimating symmetry from coordinates."""
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Space-group labels must be integer numbers from 1 to 230, got {value!r}") from exc
    if (isinstance(value, float | np.floating) and value != number) or not 1 <= number <= 230:
        raise ValueError(f"Space-group labels must be integer numbers from 1 to 230, got {value!r}")
    return number


def ase_atoms_to_material_dict(
    atoms: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    wrap_fractional_coordinates: bool = False,
) -> dict[str, Any]:
    """Return periodic geometry and explicit source symmetry labels from ASE Atoms.

    Coordinates and lattice lengths are in Angstroms; angles are in degrees.
    Fractional coordinates follow ASE's row-vector convention. Wrapping applies
    only to periodic axes. A full-rank cell is required, including for slabs.
    ``space_group`` and ``parent_space_group`` are separate input metadata labels,
    not symmetry estimates. Prototype labels remain in ``extra_info`` verbatim.
    """
    from ase import Atoms

    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected ASE Atoms, got {type(atoms).__name__}")
    cell = np.asarray(atoms.cell, dtype=float)
    if not np.isfinite(cell).all() or np.linalg.matrix_rank(cell) != 3:
        raise ValueError("Materials fractional coordinates require a finite, full-rank cell")
    source_metadata = dict(metadata or {})
    lengths = np.asarray(atoms.cell.lengths())
    angles = np.asarray(atoms.cell.angles())
    return {
        "fractional_coordinates": atoms.get_scaled_positions(wrap=wrap_fractional_coordinates),
        "cartesian_coordinates": atoms.get_positions(),
        "lattice_vectors": cell.copy(),
        "lattice_lengths": lengths,
        "lattice_angles": angles,
        "cell_parameters": np.concatenate([lengths, angles]),
        "cell_volume": float(atoms.cell.volume),
        "pbc": np.asarray(atoms.pbc, dtype=bool).copy(),
        "atomic_numbers": atoms.get_atomic_numbers(),
        "chemical_symbols": np.asarray(atoms.get_chemical_symbols()),
        "space_group": _space_group_label(source_metadata.get("space_group")),
        "parent_space_group": _space_group_label(source_metadata.get("parent_space_group")),
        "extra_info": source_metadata,
    }


def _load_material(
    raw_data: tuple,
    *,
    wrap_fractional_coordinates: bool,
    include_atom_array: bool,
    keep_ase_atoms: bool,
) -> dict[str, Any]:
    """Convert an ASE dataset row without molecular bond inference or preparation."""
    row, global_index, metadata_row = raw_data
    atoms = row.toatoms(add_additional_information=True)
    metadata = {**dict(row.key_value_pairs), **dict(row.data)}
    result = ase_atoms_to_material_dict(
        atoms, metadata=metadata, wrap_fractional_coordinates=wrap_fractional_coordinates
    )
    result.update(source_row_id=row.id, global_index=global_index, metadata_row=metadata_row)
    for name in ("energy", "forces", "stress"):
        value = row.get(name)
        if value is not None:
            result[name] = value
    if include_atom_array:
        from atomworks.io.utils.ase_conversions import ase_to_atom_array

        result["atom_array"] = ase_to_atom_array(atoms)
    if keep_ase_atoms:
        result["atoms"] = atoms
    return result


def create_ase_materials_loader(
    *,
    wrap_fractional_coordinates: bool = False,
    include_atom_array: bool = False,
    keep_ase_atoms: bool = True,
) -> Callable[[tuple], dict[str, Any]]:
    """Create a picklable periodic-materials loader for :class:`AseDBDataset`.

    Consumes ``(atoms_row, global_index, metadata_row)``. Dataset metadata indexing
    supplies ``example_id``. Optional AtomArrays preserve the cell, PBC and ASE
    initial charges; they do not infer bonds or assign formal charges. They are
    coordinate containers, not structures prepared for molecular ML transforms.
    """
    return functools.partial(
        _load_material,
        wrap_fractional_coordinates=wrap_fractional_coordinates,
        include_atom_array=include_atom_array,
        keep_ase_atoms=keep_ase_atoms,
    )
