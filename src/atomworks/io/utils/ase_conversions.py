"""Conversion utilities between ASE Atoms and Biotite AtomArray objects."""

import logging

import numpy as np
from ase import Atoms
from biotite.structure import AtomArray, BondList

logger = logging.getLogger(__name__)


# ASE has no bond model; ``info`` stores (atom_id, atom_id, BondType) rows while ASE reorders
# and trims the atom_id array with the atoms. ASE trajectory and database files drop the IDs.
_BONDS = "atomworks_bonds"


def _bonds_for_these_atoms(bonds: object, atom_ids: np.ndarray | None, n_atoms: int) -> BondList | None:
    """The stored ``atom_id`` bond table re-indexed to the atoms' positions; without ``atom_ids`` the
    table may describe an earlier array and is ignored."""
    try:
        bonds = np.asarray(bonds, dtype=np.int64)
    except (TypeError, ValueError):
        bonds = None
    if bonds is None or bonds.ndim != 2 or bonds.shape[1] not in (2, 3) or atom_ids is None:
        logger.warning("Ignoring info[%r]: it does not describe these %d atoms.", _BONDS, n_atoms)
        return None

    # An id seen twice (ASE fills an appended atom's with 0, repeats it on concatenation) names no atom
    ids, counts = np.unique(atom_ids, return_counts=True)
    repeated = set(ids[counts > 1].tolist())
    position_of = {int(atom_id): index for index, atom_id in enumerate(atom_ids) if atom_id not in repeated}
    kept = []
    for row in bonds:
        left, right = position_of.get(int(row[0])), position_of.get(int(row[1]))
        if left is not None and right is not None:
            kept.append((left, right, int(row[2]) if len(row) > 2 else 1))
    if len(kept) < len(bonds):
        logger.warning("Dropped %d bonds whose atom_ids are duplicated or no longer present.", len(bonds) - len(kept))
    return BondList(n_atoms, np.array(kept, dtype=np.uint32).reshape(-1, 3))


def ase_to_atom_array(atoms: Atoms, *, formal_charges: bool = False) -> AtomArray:
    """Convert ASE Atoms object to AtomArray.

    Args:
        atoms: The ASE atoms.
        formal_charges: Read whole-number ``initial_charges`` as the formal ``charge``, as
            :py:func:`atom_array_to_ase` writes it. ASE does not say which kind they are, so this is opt-in.

    See Also:
        :py:func:`atom_array_to_ase`: Convert Biotite AtomArray back to ASE format
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected ASE Atoms, got {type(atoms).__name__}")

    # Extract required data
    symbols = atoms.get_chemical_symbols()  # Returns list of str
    positions = atoms.get_positions()
    box = atoms.cell if atoms.cell is not None and np.linalg.norm(atoms.cell) > 0 else None

    # ... initialize AtomArray
    array = AtomArray(len(symbols))
    array.element = np.array(symbols)  # Biotite expects array-like
    array.coord = positions.astype(np.float32)
    array.atomic_number = atoms.get_atomic_numbers().tolist()

    # Copy box and PBC information, if available
    if box is not None:
        array.box = np.array(box)
        # Store PBC for round-trip conversion using private attribute
        if hasattr(atoms, "pbc"):
            array._pbc = tuple(atoms.pbc)

    # Transfer any additional arrays (annotations) from ASE to Biotite
    for key, value in atoms.arrays.items():
        if key in ("numbers", "positions"):  # Skip default ASE arrays
            continue
        name = key
        if formal_charges and key == "initial_charges" and np.all(value == np.round(value)):
            name, value = "charge", value.astype(np.int8)
        try:
            array.set_annotation(name, value)
        except ValueError as e:
            logger.debug(f"Could not add annotation '{name}': {e}")

    # ... transfer metadata from atoms.info to array._info

    # NOTE: Biotite's AtomArray doesn't provide a public API for arbitrary metadata.
    # Using _info (private attribute) is necessary for complete data transfer in
    # round-trip conversions.
    if hasattr(atoms, "info") and atoms.info:
        info = dict(atoms.info)
        bonds = info.pop(_BONDS, None)
        if bonds is not None:
            array.bonds = _bonds_for_these_atoms(bonds, atoms.arrays.get("atom_id"), len(array))
        array._info = info

    return array


def atom_array_to_ase(array: AtomArray) -> Atoms:
    """Convert AtomArray to ASE Atoms object.

    The formal ``charge`` goes to ``initial_charges``, and the bonds to ``info["atomworks_bonds"]`` as
    ``(atom_id, atom_id, BondType)`` rows, preserving Biotite's bond-type codes, including aromatic
    and coordination types. IDs start at 0 if the array has no ``atom_id``. Bonds survive in memory
    only: ASE trajectory and database files drop the per-atom IDs needed to restore them.

    See Also:
        :py:func:`ase_to_atom_array`: Convert ASE Atoms to Biotite format
    """
    if not isinstance(array, AtomArray):
        raise TypeError(f"Expected Biotite AtomArray, got {type(array).__name__}")

    # Extract required attributes
    symbols = np.array([elem.capitalize() for elem in array.element])
    positions = array.coord
    box = getattr(array, "box", None)

    # Determine PBC from stored value (or infer from box)
    if hasattr(array, "_pbc"):
        pbc = array._pbc
    else:
        # If no PBC stored, set True for all dimensions if box exists
        pbc = [box is not None] * 3

    # Create ASE Atoms object
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=box,
        pbc=pbc,
    )

    # ... transfer any additional arrays (annotations) from AtomArray to ASE Atoms

    # NOTE: Biotite stores annotations in _annot dict. While private, this is
    # the standard pattern for bulk annotation access (similar to how atoms.arrays works in ASE).
    annotations = dict(array._annot)
    charges = annotations.pop("charge", None)
    atoms.arrays.update(annotations)
    if charges is not None:
        atoms.set_initial_charges(np.asarray(charges, dtype=float))

    # Transfer metadata, if available
    if hasattr(array, "_info"):
        atoms.info.update(array._info)
    else:
        logger.debug("No _info attribute found in the AtomArray.")

    # After ``_info``, so the array's current bonds replace any carried table
    if array.bonds is not None:
        if "atom_id" not in atoms.arrays:
            atoms.arrays["atom_id"] = np.arange(array.array_length())
        bonds = array.bonds.as_array().astype(np.int64)
        # Remap only endpoints; the third column retains the Biotite bond type.
        bonds[:, :2] = atoms.arrays["atom_id"][bonds[:, :2]]
        atoms.info[_BONDS] = bonds

    return atoms
