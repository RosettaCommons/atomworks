"""Conversion utilities between ASE Atoms and Biotite AtomArray objects."""

import logging

import numpy as np
from ase import Atoms
from biotite.structure import AtomArray

logger = logging.getLogger(__name__)


def ase_to_atom_array(atoms: Atoms) -> AtomArray:
    """Convert ASE Atoms object to AtomArray.

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
        if key not in ("numbers", "positions"):  # Skip default ASE arrays
            try:
                array.set_annotation(key, value)
            except ValueError as e:
                logger.debug(f"Could not add annotation '{key}': {e}")

    # ... transfer metadata from atoms.info to array._info

    # NOTE: Biotite's AtomArray doesn't provide a public API for arbitrary metadata.
    # Using _info (private attribute) is necessary for complete data transfer in
    # round-trip conversions.
    if hasattr(atoms, "info") and atoms.info:
        array._info = dict(atoms.info)

    return array


def atom_array_to_ase(array: AtomArray) -> Atoms:
    """Convert AtomArray to ASE Atoms object.

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
    atoms.arrays.update(array._annot)

    # Transfer metadata, if available
    if hasattr(array, "_info"):
        atoms.info.update(array._info)
    else:
        logger.debug("No _info attribute found in the AtomArray.")

    return atoms
