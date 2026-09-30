"""
An extension of atomworks.io.utils.standard_annotations.annotator.
This module includes annotators that rely on functionality in atomworks.ml
"""

import functools
import logging
from collections.abc import Generator

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from jaxtyping import Bool, Float, Int

from atomworks.constants import MASKED
from atomworks.io.tools.rdkit import atom_array_to_rdkit, ccd_code_to_rdkit
from atomworks.io.utils.annotator import *  # noqa: F403 # re-export all annotators to preserve imports
from atomworks.io.utils.annotator import _register_lazy_annotator, _requires_annotations
from atomworks.io.utils.ccd import _standard_ccd_only_cache
from atomworks.io.utils.scatter import apply_and_spread_segment_wise
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.transforms.atom_array import get_within_group_res_idx, get_within_group_source_res_idx
from atomworks.ml.transforms.rdkit_utils import get_stereochemistry
from atomworks.ml.utils.token import get_token_starts

logger = logging.getLogger(__name__)

Array = np.ndarray
"""Alias for numpy.ndarray"""


@_register_lazy_annotator("is_token_start")
def is_token_start(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F405
    """Mark the first atom of each token."""
    token_starts = get_token_starts(atom_array, add_exclusive_stop=False)
    is_token_start = np.zeros(atom_array.array_length(), dtype=bool)
    is_token_start[token_starts] = True
    return is_token_start


@_register_lazy_annotator("within_chain_res_idx")
def within_chain_res_idx(atom_array: AtomArray) -> Int[Array, "n_atoms"]:  # noqa: F405
    """Get the within-chain residue index for the atom array."""
    annotations = atom_array.get_annotation_categories()
    chain_group = "chain_iid" if "chain_iid" in annotations else "chain_id"
    return get_within_group_res_idx(atom_array, group_by=chain_group)


@_register_lazy_annotator("within_chain_source_res_idx")
def within_chain_source_res_idx(atom_array: AtomArray) -> Int[Array, "n_atoms"]:  # noqa: F405
    """Gets the within chain res idx, but relies on the res_id field to preserve gaps that may exist in the original sequence"""
    annotations = atom_array.get_annotation_categories()
    chain_group = "chain_iid" if "chain_iid" in annotations else "chain_id"
    return get_within_group_source_res_idx(atom_array, group_by=chain_group)


@_register_lazy_annotator("token_min_occupancy")
@_requires_annotations("is_token_start")
def token_min_occupancy(atom_array: AtomArray) -> Float[Array, "n_atoms"]:  # noqa: F405
    """Calculate minimum occupancy for each token."""
    is_token_start = atom_array.get_annotation("is_token_start")
    token_start_idxs = np.where(is_token_start)[0]
    token_segments = np.concatenate([token_start_idxs, [atom_array.array_length()]])
    return apply_and_spread_segment_wise(token_segments, atom_array.occupancy, np.min)


@_register_lazy_annotator("token_id")
@_requires_annotations("is_token_start")
def token_id(atom_array: AtomArray) -> Int[Array, "n_atoms"]:  # noqa: F405
    """Assign a unique ID to each token."""
    is_token_start = atom_array.get_annotation("is_token_start")
    token_start_idxs = np.where(is_token_start)[0]
    token_id = np.arange(sum(is_token_start))
    token_segments = np.concatenate([token_start_idxs, [atom_array.array_length()]])
    return struc.segments.spread_segment_wise(token_segments, token_id)


@_standard_ccd_only_cache(functools.lru_cache(maxsize=1000))
def _get_stereochemistry_for_ccd_code(ccd_code: str) -> dict[str, list[dict]]:
    """Get both stereo readouts for a CCD code via one cached RDKit assignment."""
    mol = ccd_code_to_rdkit(ccd_code)
    return get_stereochemistry(mol)


def _iter_residue_stereochemistry(atom_array: AtomArray) -> Generator[tuple[int, int, dict], None, None]:
    """Yield residue bounds and CCD reference stereochemistry, falling back to its atoms."""
    res_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    starts, stops = res_starts[:-1], res_starts[1:]

    for start, stop in zip(starts, stops, strict=False):
        res_name = atom_array.res_name[start]

        if res_name == MASKED:
            continue  # skip masked residues with unknown chemistry

        try:
            # Cached lookup for both stereo readouts based on CCD code, if available
            stereo = _get_stereochemistry_for_ccd_code(str(res_name))
        except Exception as e:
            # Fallback to RDKit-based computation if CCD code lookup fails (e.g., non-standard residues)
            logger.debug("CCD stereochemistry lookup failed for %s: %s", res_name, e, exc_info=True)
            try:
                mol = atom_array_to_rdkit(atom_array[start:stop], set_coord=True)
                stereo = get_stereochemistry(mol)
            except Exception as e:
                logger.debug("RDKit stereochemistry fallback failed for %s: %s", res_name, e, exc_info=True)
                continue

        yield int(start), int(stop), stereo


def _iter_chiral_centers(atom_array: AtomArray) -> list[tuple[int, int, int, dict]]:
    """Return residue bounds, global center indices, and reference center metadata."""
    results = []
    for start, stop, stereo in _iter_residue_stereochemistry(atom_array):
        res_atom_names = atom_array.atom_name[start:stop]
        for center_info in stereo["chiral_centers"]:
            center_name = center_info.get("chiral_center_atom_name")
            if center_name is None:
                continue

            # Find the index of the chiral center atom within the residue
            matches = np.where(res_atom_names == center_name)[0]
            if len(matches) > 0:
                # Add the global index of the chiral center atom and its metadata to the results
                results.append((int(start), int(stop), start + matches[0], center_info))

    return results


@_register_lazy_annotator("stereogenic_bond_atoms")
def get_stereogenic_bond_atoms(atom_array: AtomArray) -> Bool[Array, "n_atoms"]:  # noqa: F405
    """Annotate only the two endpoints of each assigned stereogenic double bond.

    Adds a boolean ``stereogenic_bond_atoms`` annotation in place if absent.
    Reference substituents need not be present; both endpoints and their double
    bond must be present. Cross-residue stereo perception is not supported.
    """
    mask = np.zeros(len(atom_array), dtype=bool)
    if atom_array.bonds is None:
        return mask
    for start, stop, stereo in _iter_residue_stereochemistry(atom_array):
        if not stereo["double_bonds"]:
            continue
        by_name = dict(zip(atom_array.atom_name[start:stop], range(start, stop), strict=True))
        if len(by_name) != stop - start:
            raise ValueError("Double-bond stereo requires unique atom names within each residue")
        for bond in stereo["double_bonds"]:
            names = bond["atom_names"][:2]
            if any(name not in by_name for name in names):
                continue
            left, right = (by_name[name] for name in names)
            neighbors, types = atom_array.bonds.get_bonds(left)
            if np.any((neighbors == right) & (types == struc.BondType.DOUBLE)):
                mask[[left, right]] = True
    return mask


@_register_lazy_annotator("chiral_type")
def chiral_type(atom_array: AtomArray) -> Int[Array, "n_atoms"]:  # noqa: F405
    """Compute ``chiral_type`` annotation (int8) for ``atom_array``.

    Values are :class:`~atomworks.enums.ChiralType` int values; ``0`` = not a chiral center.
    """
    types = np.zeros(atom_array.array_length(), dtype=np.int8)
    for _, _, idx, info in _iter_chiral_centers(atom_array):
        types[idx] = int(info.get("chiral_type", 0))
    return types
