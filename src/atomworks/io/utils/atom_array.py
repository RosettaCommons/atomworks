import logging
from collections.abc import Iterable

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.constants import BIOTITE_BOND_TYPE_TO_BOND_ORDER, HYDROGEN_LIKE_SYMBOLS, WATER_LIKE_CCDS

# Backwards compatability: imported here to avoid breaking changes, but should be imported from atomworks.io.utils.scatter in the future.
from atomworks.io.utils.scatter import apply_and_spread_segment_wise as apply_and_spread  # noqa: F401

logger = logging.getLogger(__name__)


def _count_explicit_h_neighbors(atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
    """Per-atom count of explicit hydrogen atoms directly bonded to each atom."""
    counts = np.zeros(atom_array.array_length(), dtype=np.int8)
    if atom_array.bonds is None:
        return counts
    bond_arr = atom_array.bonds.as_array()
    if len(bond_arr) == 0:
        return counts
    is_h = np.isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS)
    i_atoms, j_atoms = bond_arr[:, 0], bond_arr[:, 1]
    # Bonds from H (i) to non-H (j): increment on j; and the reverse orientation.
    np.add.at(counts, j_atoms[is_h[i_atoms] & ~is_h[j_atoms]], 1)
    np.add.at(counts, i_atoms[is_h[j_atoms] & ~is_h[i_atoms]], 1)
    return counts


def _assert_no_nhyd_with_explicit_h(atom_array: AtomArray | AtomArrayStack) -> None:
    """Assert the ``nhyd`` invariant: no atom carries both ``nhyd > 0`` and an explicit bonded H."""
    if "nhyd" not in atom_array.get_annotation_categories():
        return
    has_nhyd = atom_array.nhyd > 0
    if not has_nhyd.any():
        return
    has_explicit_h = _count_explicit_h_neighbors(atom_array) > 0
    assert not (has_nhyd & has_explicit_h).any(), (
        "Atom(s) carry both `nhyd > 0` and an explicit bonded hydrogen; hydrogens would be "
        "double-counted. Call `remove_hydrogens` after `annotate_hydrogens` "
        "(or use `annotate_and_remove_hydrogens`)."
    )


def _bonds_to_dict(atom_array: struc.AtomArray) -> dict[tuple[str, str], int] | None:
    """Convert AtomArray bonds to Biotite-comptabile dictionary format."""
    if atom_array.bonds is None or len(atom_array.bonds.as_array()) == 0:
        return None

    bond_dict = {}
    for bond in atom_array.bonds.as_array():
        # Convert numpy strings to Python strings (biotite requirement)
        atom1 = str(atom_array.atom_name[bond[0]])
        atom2 = str(atom_array.atom_name[bond[1]])
        bond_type = int(bond[2])
        # Ensure consistent ordering (biotite accepts both, but we normalize)
        key = tuple(sorted([atom1, atom2]))
        bond_dict[key] = bond_type

    return bond_dict


def remove_components(
    atoms: AtomArray | AtomArrayStack,
    *,
    remove_waters: bool = True,
    remove_ccds: Iterable[str] = (),
) -> AtomArray | AtomArrayStack:
    """Remove excluded components unless covalently connected to a non-excluded residue.

    Args:
        atoms: The atom array or stack to filter.
        remove_waters: Remove water-like components regardless of connectivity.
        remove_ccds: Removal candidates. Supplied bonds preserve whole connected residues,
            excluding coordination bonds; molecules made entirely of candidates are removed.
    """
    if not remove_ccds and not remove_waters:
        return atoms

    remove_set = set(map(str.upper, remove_ccds))
    if remove_waters:
        remove_set.update(WATER_LIKE_CCDS)

    keep = ~np.isin(atoms.res_name, list(remove_set))
    if remove_ccds and keep.any() and not keep.all() and atoms.bonds is not None:
        residue_index = struc.get_all_residue_positions(atoms)
        keep_residue = keep[struc.get_residue_starts(atoms)]

        # Build covalent connectivity between whole residues.
        bonds = atoms.bonds.as_array()
        pairs = bonds[bonds[:, 2] != struc.BondType.COORDINATION, :2]
        graph = struc.BondList(len(keep_residue), residue_index[pairs])

        # Keep connected groups containing any non-excluded residue.
        for component in struc.get_molecule_indices(graph):
            keep_residue[component] = keep_residue[component].any()
        keep = keep_residue[residue_index]

        if remove_waters:
            keep &= ~np.isin(atoms.res_name, WATER_LIKE_CCDS)

    if isinstance(atoms, AtomArrayStack):
        return atoms[:, keep]

    return atoms[keep]


def subset_to_first_transformation(atom_array: AtomArray | AtomArrayStack) -> AtomArray:
    """Subset atom array to first transformation_id if present.

    Args:
        atom_array: Atom array or stack to subset.

    Returns:
        Atom array subsetted to first transformation_id, or original if no transformation_id annotation.
    """
    if "transformation_id" in atom_array.get_annotation_categories():
        first_transform = atom_array.transformation_id[0]
        return atom_array[atom_array.transformation_id == first_transform]

    return atom_array


def annotate_hydrogens(
    template: AtomArray | AtomArrayStack,
    *,
    increment: bool = False,
) -> AtomArray | AtomArrayStack:
    """Set the ``nhyd`` annotation: count of H atoms directly bonded to each heavy atom.

    Args:
        template: The atom array to annotate.
        increment: If ``True``, add the newly counted H atoms to any existing ``nhyd``
            values rather than overwriting them. Useful when some Hs are already counted
            implicitly and additional explicit Hs remain. Defaults to ``False`` (overwrite).

    .. note::
       To maintain the invariant that ``nhyd`` is present only when no explicit H atoms exist,
       this function should always be followed by :py:func:`remove_hydrogens`.
    """
    nhyd_delta = _count_explicit_h_neighbors(template)

    if increment and "nhyd" in template.get_annotation_categories():
        nhyd_delta += template.nhyd

    template.set_annotation("nhyd", nhyd_delta)

    return template


def remove_hydrogens(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Remove hydrogen atoms from an AtomArray or AtomArrayStack."""
    keep = ~np.isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS)
    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, keep]
    return atom_array[keep]


def annotate_and_remove_hydrogens(
    atom_array: AtomArray | AtomArrayStack,
    *,
    increment: bool = False,
) -> AtomArray | AtomArrayStack:
    """Convenience wrapper around :py:func:`annotate_hydrogens` followed by :py:func:`remove_hydrogens`.

    Args:
        atom_array: The atom array to annotate and remove hydrogens from.
        increment: Passed through to :py:func:`annotate_hydrogens`. Defaults to ``False``.
    """
    return remove_hydrogens(annotate_hydrogens(atom_array, increment=increment))


def _get_bond_neighbors(bonds_arr: np.ndarray, atom_idx: int) -> np.ndarray:
    """Return indices of all atoms bonded to ``atom_idx``."""
    mask0 = bonds_arr[:, 0] == atom_idx
    mask1 = bonds_arr[:, 1] == atom_idx
    return np.concatenate([bonds_arr[mask0, 1], bonds_arr[mask1, 0]])


def _find_bonded_hydrogens(atom_array: AtomArray, atom_idx: int) -> np.ndarray:
    """Return indices of hydrogen atoms directly bonded to ``atom_idx``."""
    bonds_array = atom_array.bonds.as_array()
    mask_col0 = bonds_array[:, 0] == atom_idx
    mask_col1 = bonds_array[:, 1] == atom_idx
    neighbors = np.concatenate([bonds_array[mask_col0, 1], bonds_array[mask_col1, 0]])
    is_h = np.isin(atom_array.element[neighbors], HYDROGEN_LIKE_SYMBOLS)
    return neighbors[is_h]


def count_bonded_hydrogens(atom_array: AtomArray, atom_idx: int, include_implicit: bool = False) -> int:
    """Return the count of hydrogen atoms bonded to ``atom_idx``.

    Args:
        include_implicit: If ``True`` and an ``nhyd`` annotation is present,
            return the implicit hydrogen count instead of searching explicit bonds.
    """
    if include_implicit and "nhyd" in atom_array.get_annotation_categories():
        _assert_no_nhyd_with_explicit_h(atom_array)
        return int(atom_array.nhyd[atom_idx])
    return len(_find_bonded_hydrogens(atom_array, atom_idx))


def has_annotation(arr: AtomArray, annotation: str | list[str]) -> bool:
    """Check if an AtomArray has an annotation.

    Args:
        arr: AtomArray to check.
        annotation: Annotation(s) to check for.
    """
    existing_annotations = frozenset(["coord", *arr.get_annotation_categories()])
    if isinstance(annotation, str):
        return annotation in existing_annotations
    else:
        return set(annotation).issubset(existing_annotations)


def chain_identifier(atom_array: AtomArray) -> np.ndarray:
    """Return ``chain_iid`` if present, else ``chain_id``."""
    if "chain_iid" in atom_array.get_annotation_categories():
        return atom_array.chain_iid
    return atom_array.chain_id


def get_bond_degree_per_atom(atom_array: AtomArray) -> np.ndarray:
    """Returns the total degree (= sum of bond orders) for each atom."""
    # Count both ends of each edge
    edge_list = atom_array.bonds._bonds[:, :2]
    bond_types = atom_array.bonds._bonds[:, -1]
    weights = np.array([BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(struc.BondType(bt), bt) for bt in bond_types])

    degree = np.bincount(edge_list.ravel(), weights=np.repeat(weights, 2))

    # ... pad in case of unbonded atoms
    if len(degree) <= atom_array.array_length():
        degree = np.pad(degree, (0, atom_array.array_length() - len(degree)))

    # ... add implicit hydrogens if nhyd annotation is present
    if "nhyd" in atom_array.get_annotation_categories():
        _assert_no_nhyd_with_explicit_h(atom_array)
        degree += atom_array.nhyd

    return degree
