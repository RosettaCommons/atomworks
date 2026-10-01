"""Loader for ASE datasets (e.g., from XYZ files) into AtomWorks format."""

import functools
import logging
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from biotite.structure import BondType

from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.rdkit import atom_array_from_rdkit, atom_array_to_rdkit
from atomworks.io.transforms.atom_array import (
    get_coarse_graph_as_nodes_and_edges,
    get_connected_nodes,
)
from atomworks.io.utils.ase_conversions import ase_to_atom_array
from atomworks.io.utils.atom_array import annotate_and_remove_hydrogens
from atomworks.io.utils.chain import create_chain_id_generator

logger = logging.getLogger(__name__)


def _ase_loader_function(
    raw_data: tuple,
    per_atom_properties: list[str],
    global_properties: list[str],
    hydrogen_policy: Literal["keep", "remove"],
) -> dict[str, Any]:
    """ASE loader function (picklable when used with functools.partial)."""
    # Unpack the tuple (metadata_row available but unused for now)
    atoms_row, global_idx, _metadata_row = raw_data

    # Convert ASE row to atoms object
    atoms = atoms_row.toatoms()

    # Update atoms.info with any data from the row
    if isinstance(atoms_row.data, dict):
        atoms.info.update(atoms_row.data)

    # Convert to Biotite AtomArray
    atom_array = ase_to_atom_array(atoms)

    # Add bonds, hybridization, chirality, etc. via RDKit
    mol = atom_array_to_rdkit(
        atom_array,
        infer_bonds=True,
        timeout_seconds=2,
        hydrogen_policy="keep",
        system_charge=atoms.info.get("charge", 0),
    )
    atom_array = atom_array_from_rdkit(mol, remove_hydrogens=False)

    if hydrogen_policy == "remove":
        # Set nhyd annotation and remove hydrogens
        atom_array = annotate_and_remove_hydrogens(atom_array, increment=False)

    # Create unique atom IDs and assign chain IDs and residue IDs based on connectivity
    atom_array.set_annotation("atom_id", np.arange(atom_array.array_length()))
    # Exclude coordination bonds so metal ions and ligands get separate chains
    connected_atoms = get_connected_nodes(
        *get_coarse_graph_as_nodes_and_edges(atom_array, "atom_id", exclude_bond_types={BondType.COORDINATION})
    )

    # Assign chain IDs based on connected components
    chain_id_gen = create_chain_id_generator()
    for connected_atom in connected_atoms:
        chain_letter = next(chain_id_gen)
        res_number = 1
        element_counts = {}

        for atom_id in connected_atom:
            atom_array.chain_id[atom_id] = chain_letter
            atom_array.res_id[atom_id] = res_number
            atom_array.res_name[atom_id] = f"{chain_letter}:{res_number}"

            element = atom_array.element[atom_id]
            element_counts[element] = element_counts.get(element, 0) + 1
            atom_name = f"{element}{element_counts[element]}"
            atom_array.atom_name[atom_id] = atom_name

    # Extract per-atom properties BEFORE parse_atom_array
    for prop in per_atom_properties:
        if prop in ("numbers", "positions"):
            continue

        if prop in atoms.arrays:
            if not hasattr(atom_array, prop):
                prop_data = atoms.arrays[prop]
                if len(prop_data) == atom_array.array_length():
                    atom_array.set_annotation(prop, prop_data)
                else:
                    logger.warning(
                        f"Property '{prop}' found in atoms.arrays but length mismatch "
                        f"({len(prop_data)} vs {atom_array.array_length()} atoms). "
                        f"Skipping — this can happen when hydrogens are removed."
                    )
        elif prop in atoms.info:
            prop_data = atoms.info[prop]
            if not hasattr(atom_array, prop):
                if hasattr(prop_data, "__len__") and len(prop_data) == atom_array.array_length():
                    atom_array.set_annotation(prop, np.array(prop_data))
                else:
                    logger.warning(
                        f"Property '{prop}' found in atoms.info but not compatible as per-atom "
                        f"(length {len(prop_data) if hasattr(prop_data, '__len__') else 'scalar'} vs {atom_array.array_length()} atoms)"
                    )
        else:
            logger.warning(
                f"Requested per-atom property '{prop}' not found for example {global_idx}. "
                f"Available in atoms.arrays: {list(atoms.arrays.keys())}, "
                f"available in atoms.info: {list(atoms.info.keys())}"
            )

    # Parse atom array
    data = parse_atom_array(
        atom_array,
        add_missing_atoms=False,
        remove_waters=False,
        remove_ccds=None,
        hydrogen_policy="keep",
    )

    # Extract the processed AtomArray
    atom_array = data["assemblies"]["1"][0]
    data["atom_array"] = atom_array

    # Extract global properties
    for prop in global_properties:
        if prop in atoms.info:
            data["extra_info"][prop] = atoms.info[prop]
        elif prop in atoms_row:
            data["extra_info"][prop] = atoms_row[prop]
        else:
            logger.warning(
                f"Requested global property '{prop}' not found in atoms.info for example {global_idx}. "
                f"Available properties: {list(atoms.info.keys())}"
            )

    return data


def create_ase_loader(
    per_atom_properties: list[str] | None = None,
    global_properties: list[str] | None = None,
    hydrogen_policy: Literal["keep", "remove"] = "keep",
) -> Callable:
    """Factory function that creates a picklable loader for ASE LMDB datasets.

    The loader processes ASE atoms_row objects into AtomWorks-compatible dictionaries
    with atom_array and extra_info fields. The ``example_id`` is set by the dataset's
    ``__getitem__`` method from the :py:class:`~atomworks.ml.datasets.metadata.MetadataIndex`.

    Args:
        per_atom_properties: List of per-atom properties to extract from atoms.arrays
            as AtomArray annotations (e.g., forces, charges, spins)
        global_properties: List of global properties to extract from atoms.info
            and/or atoms_row into extra_info dict (e.g., energy, charge, spin)
        hydrogen_policy: Whether to keep or remove explicit hydrogens. When "remove",
            explicit hydrogens are removed and their count is stored in the ``nhyd``
            annotation.

    Returns:
        A picklable loader function (via functools.partial) for multiprocessing.
    """
    return functools.partial(
        _ase_loader_function,
        per_atom_properties=per_atom_properties or [],
        global_properties=global_properties or [],
        hydrogen_policy=hydrogen_policy,
    )
