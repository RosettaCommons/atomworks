"""Connectivity, leaving-group, and charge rules for inter-residue links."""

import contextlib
import functools
import logging
import os
from collections import defaultdict, deque
from typing import Literal

import biotite.structure as struc
import networkx as nx
import numpy as np
from biotite.structure import AtomArray

from atomworks.constants import (
    BIOTITE_BOND_TYPE_TO_BOND_ORDER,
    BOND_DISTANCE_THRESHOLD_CHNO,
    BOND_DISTANCE_THRESHOLD_CHNOPS,
    BOND_DISTANCE_THRESHOLD_OTHER,
    CCD_MIRROR_PATH,
    CHNO_ELEMENTS,
    CHNOPS_ELEMENTS,
    DEFAULT_VALENCE,
    DO_NOT_MATCH_CCD,
    HYDROGEN_LIKE_SYMBOLS,
)
from atomworks.io.utils.atom_array import (
    _find_bonded_hydrogens,
    _get_bond_neighbors,
    get_bond_degree_per_atom,
)
from atomworks.io.utils.ccd import _standard_ccd_only_cache, atom_array_from_ccd_code, get_polymerization_atoms

logger = logging.getLogger("atomworks.io.utils.leaving_atoms")
bond_logger = logging.getLogger("atomworks.io")


_GLYCOSYLATION_ROLES = (
    "c-mannosylation",
    "n-glycosylation",
    "o-glycosylation",
    "s-glycosylation",
)
_MAX_GLYCOSYLATION_DISTANCE = 2.4


def filter_link_distances(
    atom_array: AtomArray,
    filtered: dict[str, np.ndarray],
    idx1: np.ndarray,
    idx2: np.ndarray,
    valid: np.ndarray,
    distance_policy: str,
) -> np.ndarray:
    """Filter or diagnose candidate links using their authored distances."""
    n_rows = len(idx1)
    conn_type_arr = filtered["conn_type_id"]
    chains = [filtered["ptnr1_label_asym_id"], filtered["ptnr2_label_asym_id"]]
    res_names_q = [filtered["ptnr1_label_comp_id"], filtered["ptnr2_label_comp_id"]]
    atom_ids = [filtered["ptnr1_label_atom_id"], filtered["ptnr2_label_atom_id"]]
    # --- Distance validation using pdbx_dist_value from struct_conn ---
    pdbx_dist_value = filtered.get("pdbx_dist_value")
    if distance_policy != "keep" and valid.any() and pdbx_dist_value is not None:
        dists = np.full(n_rows, np.nan)
        for i in range(n_rows):
            with contextlib.suppress(ValueError, TypeError):
                dists[i] = float(pdbx_dist_value[i])

        expected_thresholds = _element_distance_thresholds(atom_array.element[idx1], atom_array.element[idx2])
        roles = filtered.get("pdbx_role")
        is_glycosylation = (
            np.isin(np.char.lower(np.char.strip(roles.astype(str))), _GLYCOSYLATION_ROLES)
            if roles is not None
            else np.zeros(n_rows, dtype=bool)
        )
        is_carbohydrate = struc.filter_carbohydrates(atom_array)
        is_glycosylation |= (conn_type_arr == "covale") & is_carbohydrate[idx1] & is_carbohydrate[idx2]
        expected_thresholds[is_glycosylation] = BOND_DISTANCE_THRESHOLD_CHNO
        allowed_thresholds = expected_thresholds.copy()
        allowed_thresholds[is_glycosylation] = _MAX_GLYCOSYLATION_DISTANCE

        is_long = valid & ~np.isnan(dists) & (dists > expected_thresholds)
        exceeds_allowed = is_long & (dists > allowed_thresholds)
        if np.any(is_long):
            base = distance_policy.split("_")[0] if "_" in distance_policy else distance_policy
            if base == "filter":
                valid[exceeds_allowed] = False

            for k in np.where(is_long)[0]:
                msg = (
                    f"struct_conn bond {chains[0][k]}/{res_names_q[0][k]}/{atom_ids[0][k]}"
                    f" — {chains[1][k]}/{res_names_q[1][k]}/{atom_ids[1][k]}:"
                    f" pdbx_dist_value {dists[k]:.3f} A exceeds {expected_thresholds[k]:.1f} A threshold"
                )
                if not exceeds_allowed[k]:
                    bond_logger.warning(
                        "Long explicit glycosylation %s (keeping through %.1f A)",
                        msg,
                        allowed_thresholds[k],
                    )
                elif base == "raise":
                    raise ValueError(msg)
                elif base == "filter":
                    bond_logger.warning("Skipping %s", msg)
                else:
                    bond_logger.warning("Long struct_conn bond %s (keeping)", msg)

    return valid


def infer_link_orders(filtered: dict[str, np.ndarray], bond_orders: np.ndarray) -> None:
    """Infer unspecified struct_conn bond orders in-place from CCD leaving groups."""
    pdbx_value_order = filtered["pdbx_value_order"]
    # For ambiguous "?" bond orders, infer from the CCD leaving atom bond type.
    # Example: See PLP in 1AHO, where the Schiff base bond is unknown but should be a double bond
    ambiguous = np.array([str(s) == "?" for s in pdbx_value_order])
    if ambiguous.any():
        p1_res = filtered["ptnr1_label_comp_id"]
        p2_res = filtered["ptnr2_label_comp_id"]
        p1_atom = filtered["ptnr1_label_atom_id"]
        p2_atom = filtered["ptnr2_label_atom_id"]
        for i in np.where(ambiguous)[0]:
            candidates = [
                bt
                for res, atom in ((p1_res, p1_atom), (p2_res, p2_atom))
                if (bt := get_leaving_atom_bond_type(str(res[i]), str(atom[i]))) is not None
            ]
            if candidates:
                bond_orders[i] = max(int(bt) for bt in candidates)


def _element_distance_thresholds(
    elements1: np.ndarray,
    elements2: np.ndarray,
    chno_threshold: float = BOND_DISTANCE_THRESHOLD_CHNO,
    chnops_threshold: float = BOND_DISTANCE_THRESHOLD_CHNOPS,
    other_threshold: float = BOND_DISTANCE_THRESHOLD_OTHER,
) -> np.ndarray:
    """Compute per-bond distance thresholds based on element pairs."""
    e1 = np.char.upper(elements1.astype(str))
    e2 = np.char.upper(elements2.astype(str))
    thresholds = np.full(len(e1), other_threshold)
    is_chno = np.isin(e1, list(CHNO_ELEMENTS)) & np.isin(e2, list(CHNO_ELEMENTS))
    thresholds[is_chno] = chno_threshold
    is_chnops = np.isin(e1, list(CHNOPS_ELEMENTS)) & np.isin(e2, list(CHNOPS_ELEMENTS))
    thresholds[is_chnops & ~is_chno] = chnops_threshold
    return thresholds


def _add_polymer_inter_residue_bonds(atom_array: AtomArray) -> AtomArray:
    """Add inter-residue bonds for consecutive polymer residues.

    Avoids adding bonds when already present (e.g. from ``struct_conn``)
    to prevent incorrect bonding in cases (e.g., PDB ID ``1xvk``),
    where a non-canonical bond (from ``struct_conn``) overrides the default polymer bond.
    """
    if atom_array.bonds is None or "chain_type" not in atom_array.get_annotation_categories():
        return atom_array

    has_tid = "transformation_id" in atom_array.get_annotation_categories()
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)

    # Pre-compute atom→residue mapping...
    bonds_arr = atom_array.bonds.as_array()[:, :2]
    sizes = np.diff(res_starts)
    atom_to_res = np.repeat(np.arange(len(res_starts) - 1, dtype=np.intp), sizes)

    # ... and identify which bonds are cross-residue
    res_a = atom_to_res[bonds_arr[:, 0]]
    res_b = atom_to_res[bonds_arr[:, 1]]
    cross_mask = res_a != res_b
    bonded_residue_pairs = set(zip(res_a[cross_mask].tolist(), res_b[cross_mask].tolist(), strict=False))

    def _find_atom(start: int, stop: int, name: str) -> int | None:
        return next((start + j for j in range(stop - start) if atom_array.atom_name[start + j] == name), None)

    new_bonds = []
    for i in range(len(res_starts) - 2):
        s1, e1, s2, e2 = res_starts[i], res_starts[i + 1], res_starts[i + 1], res_starts[i + 2]
        if atom_array.chain_id[s1] != atom_array.chain_id[s2]:
            continue
        if has_tid and atom_array.transformation_id[s1] != atom_array.transformation_id[s2]:
            continue
        # Skip over genuine chain-break gaps (missing loops)
        if int(atom_array.res_id[s2]) - int(atom_array.res_id[s1]) > 1:
            continue

        atoms_1 = get_polymerization_atoms(atom_array.res_name[s1])
        atoms_2 = get_polymerization_atoms(atom_array.res_name[s2])
        if atoms_1[0] is None or atoms_2[1] is None:
            continue

        idx_in = _find_atom(s2, e2, atoms_2[1])
        if idx_in is None:
            continue

        # Skip if ANY bond already exists between the two residues (e.g. ester bond via struct_conn)
        if (i, i + 1) in bonded_residue_pairs or (i + 1, i) in bonded_residue_pairs:
            continue

        idx_out = _find_atom(s1, e1, atoms_1[0])
        if idx_out is not None:
            new_bonds.append((idx_out, idx_in, int(struc.BondType.SINGLE)))

    if new_bonds:
        new_bond_arr = np.array(new_bonds, dtype=np.uint32)
        atom_array.bonds = atom_array.bonds.merge(struc.BondList(atom_array.array_length(), new_bond_arr))

    return atom_array


def _find_connected_components_after_removal(graph: nx.Graph, node_to_remove: int) -> list[list[int]]:
    """Identifies connected components that would form after removing a node from a graph.

    Args:
        graph: The input graph.
        node_to_remove: The node to hypothetically remove.

    Returns:
        List of lists containing node indices in each new component.
    """
    # Only the removed atom's neighbours seed a traversal, and treating it as
    # already seen keeps it out of them, so no subgraph has to be built per atom.
    unvisited = set(graph.neighbors(node_to_remove))
    components = []
    while unvisited:
        start = unvisited.pop()
        seen = {node_to_remove, start}
        queue = deque([start])
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in graph[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
        unvisited -= set(component)

    return components


@_standard_ccd_only_cache(functools.cache)
def _get_chem_comp_leaving_atom_groups_cached(
    ccd_code: str, ccd_mirror_path_str: str, mode: Literal["warn", "raise"] = "warn"
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Internal cached implementation. Do not call directly."""
    # Skip CCD lookup for codes that shouldn't be matched against CCD (e.g., UNL, water-like codes)
    if ccd_code in DO_NOT_MATCH_CCD:
        if mode == "warn":
            logger.debug(f"Skipping CCD lookup for `{ccd_code}` as it's in DO_NOT_MATCH_CCD")
        return {}

    try:
        chem_comp = atom_array_from_ccd_code(ccd_code, ccd_mirror_path_str, coords=None)
    except (ValueError, AttributeError) as e:
        if mode == "warn":
            logger.warning(f"Failed to compute leaving groups for `{ccd_code}`: {e}")
        elif mode == "raise":
            raise ValueError(f"Failed to compute leaving groups for `{ccd_code}`: {e}") from e
        return {}

    if "is_leaving_atom" not in chem_comp.get_annotation_categories():
        if mode == "warn":
            logger.warning(
                f"No 'is_leaving_atom' annotation found for `{ccd_code}`. "
                "Cannot compute leaving groups, returning empty dictionary. "
                "Check if your CCD mirror is up to date."
            )
        elif mode == "raise":
            raise ValueError(
                f"No 'is_leaving_atom' annotation found for `{ccd_code}`. "
                "Cannot compute leaving groups. Check if your CCD mirror is up to date."
            )
        return {}

    # ... initialize output
    leaving_atom_groups = defaultdict(list)

    # ... get relevant annotations
    is_leaving_atom = chem_comp.get_annotation("is_leaving_atom")
    atom_name = chem_comp.get_annotation("atom_name")
    element = chem_comp.get_annotation("element")

    # ... skip if no atoms are annotated as leaving atoms (majority of CCD entries)
    if not any(is_leaving_atom):
        return {}

    # ... compute the leaving groups based on the bond graph and annotation
    bond_graph = chem_comp.bonds.as_graph()
    for atom_idx in range(chem_comp.array_length()):
        # ... find the connected groups of atoms if the current atom were removed
        connected_groups = _find_connected_components_after_removal(bond_graph, atom_idx)

        # ... check if all atoms in the connected group are flagged as leaving atoms
        #     by the CCD entry
        for connected_group in connected_groups:
            heavy_atoms: list[int] = list(filter(lambda x: element[x] != "H", connected_group))
            is_leaving_group = (
                all(is_leaving_atom[heavy_atoms]) if len(heavy_atoms) > 0 else all(is_leaving_atom[connected_group])
            )

            if is_leaving_group:
                leaving_atom_groups[atom_name[atom_idx]].append(sorted(connected_group))

    # ... order members and groups by CCD declaration index so the result is stable
    return {
        parent: tuple(tuple(atom_name[idx] for idx in group) for group in sorted(groups, key=min))
        for parent, groups in leaving_atom_groups.items()
    }


def get_chem_comp_leaving_atom_groups(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, mode: Literal["warn", "raise"] = "warn"
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Compute the leaving atom groups for a given CCD entry.

    The returned dictionary maps the name of an atom to the groups of atoms that would
    become disconnected if the atom were removed. Each unit of inter-residue bond order
    displaces one group.

    Example:
        >>> get_chem_comp_leaving_atom_groups("ALA")
        {'N': (('H2',),), 'C': (('OXT', 'HXT'),), 'OXT': (('HXT',),)}
    """
    # Normalize path and delegate to cached implementation (use empty string for None)
    cache_key = os.fspath(ccd_mirror_path) if ccd_mirror_path else ""
    return dict(_get_chem_comp_leaving_atom_groups_cached(ccd_code, cache_key, mode))


def get_leaving_atom_bond_type(ccd_code: str, atom_name: str) -> struc.BondType | None:
    """Return the bond type between ``atom_name`` and its CCD leaving atoms."""
    leaving_groups = get_chem_comp_leaving_atom_groups(ccd_code)
    if atom_name not in leaving_groups:
        return None

    try:
        chem_comp = atom_array_from_ccd_code(ccd_code)
    except (ValueError, AttributeError):
        return None

    names = chem_comp.atom_name
    matches = np.where(names == atom_name)[0]
    if not len(matches):
        return None

    atom_idx = int(matches[0])
    leaving_set = {name for group in leaving_groups[atom_name] for name in group}
    rows = chem_comp.bonds.as_array()
    rows = rows[(rows[:, 0] == atom_idx) | (rows[:, 1] == atom_idx)]
    for a1, a2, bt in rows:
        nb = a2 if a1 == atom_idx else a1
        if names[nb] in leaving_set and (bond_type := struc.BondType(int(bt))) != struc.BondType.SINGLE:
            return bond_type
    return None


def _check_bonds_in_direction(
    bonds_array: np.ndarray,
    leaving_atom_mask: np.ndarray,
    parent_atom_mask: np.ndarray,
    leaving_col: int,
    other_col: int,
    atom_array: struc.AtomArray,
) -> None:
    """Check that leaving atoms at ``leaving_col`` only bond to valid atoms at ``other_col``."""
    is_leaving = leaving_atom_mask[bonds_array[:, leaving_col]]
    bonds_with_leaving = bonds_array[is_leaving]

    if len(bonds_with_leaving) == 0:
        return

    other_atoms = bonds_with_leaving[:, other_col]
    # Valid bonds: to parent atoms OR to other leaving atoms
    is_other_valid = parent_atom_mask[other_atoms] | leaving_atom_mask[other_atoms]
    invalid_mask = ~is_other_valid

    if np.any(invalid_mask):
        # Find first invalid bond for error message
        invalid_bond = bonds_with_leaving[invalid_mask][0]
        leaving_idx = invalid_bond[leaving_col]
        other_idx = invalid_bond[other_col]
        raise ValueError(
            f"Leaving atom '{atom_array.atom_name[leaving_idx]}' in residue "
            f"{atom_array.res_name[leaving_idx]} (index {leaving_idx}) is bonded to "
            f"'{atom_array.atom_name[other_idx]}' (index {other_idx}), which is neither "
            f"a parent atom making inter-residue bonds nor another leaving atom. "
            f"Cannot safely remove."
        )


def _validate_leaving_atoms_have_no_unexpected_bonds(
    atom_array: struc.AtomArray,
    leaving_atom_mask: np.ndarray,
    atoms_with_inter_bonds: np.ndarray,
) -> None:
    """Validate that atoms marked for removal don't have unexpected bonds."""
    if atom_array.bonds is None or atom_array.bonds.get_bond_count() == 0:
        return

    bonds_array = atom_array.bonds.as_array()
    parent_atom_mask = np.zeros(len(atom_array), dtype=bool)
    parent_atom_mask[atoms_with_inter_bonds] = True

    # Check both directions
    _check_bonds_in_direction(bonds_array, leaving_atom_mask, parent_atom_mask, 0, 1, atom_array)
    _check_bonds_in_direction(bonds_array, leaving_atom_mask, parent_atom_mask, 1, 0, atom_array)


def _get_inter_residue_bonds(atom_array: struc.AtomArray) -> np.ndarray:
    """Return inter-residue bonds excluding coordination, as ``[atom1, atom2, bond_type]`` rows."""
    if atom_array.bonds is None or atom_array.bonds.get_bond_count() == 0:
        return np.empty((0, 3), dtype=int)

    bonds_array = atom_array.bonds.as_array()

    bonds_array = bonds_array[bonds_array[:, 2] != struc.BondType.COORDINATION]

    # Define residue identity
    group_by = ["chain_id", "res_id", "ins_code"]
    if "transformation_id" in atom_array.get_annotation_categories():
        group_by.append("transformation_id")

    is_inter_bond = np.zeros(len(bonds_array), dtype=bool)
    for field in group_by:
        values = atom_array.get_annotation(field)
        is_inter_bond |= values[bonds_array[:, 0]] != values[bonds_array[:, 1]]

    return bonds_array[is_inter_bond]


def get_inter_residue_atom_mask(atom_array: struc.AtomArray) -> np.ndarray:
    """Get boolean mask indicating which atoms are involved in inter-residue bonds."""
    inter_bonds = _get_inter_residue_bonds(atom_array)

    if len(inter_bonds) == 0:
        return np.zeros(len(atom_array), dtype=bool)

    # Get unique atoms involved in inter-residue bonds
    atoms_with_inter_bonds = np.unique(inter_bonds[:, :2])

    # Create atom mask
    atom_mask = np.zeros(atom_array.array_length(), dtype=bool)
    atom_mask[atoms_with_inter_bonds] = True

    return atom_mask


def _maybe_fix_overvalent_carbon(atom_array: struc.AtomArray, atom_idx: int) -> list[int]:
    """Fix an overvalent carbon atom.

    Tries multiple strategies in order and returns after the first that applies:

    1. **Bonded H removal**: remove a directly bonded H (e.g., a CCD ghost H on a
       non-polymer cap such as ACE/1j8z that is displaced when the covalent bond
       forms).
    2. **C=O decrement**: decrement a C=O double bond to single and set the oxygen
       charge to -1 (nucleophilic addition pushes lone pair to O; only applies when
       no H is available to remove — e.g., 4PA/CAI in 1tqh where SER-94 OG attacks
       the carbonyl carbon).

    Returns:
        List of atom indices to remove (caller must apply). Empty list means the fix
        was applied in-place (e.g., C=O decrement).
    """
    bonds_arr = atom_array.bonds.as_array()
    neighbors = _get_bond_neighbors(bonds_arr, atom_idx)

    o_neighbors = neighbors[atom_array.element[neighbors] == "O"]

    # --- Strategy 1: remove bonded H (explicit or implicit) ---
    has_nhyd = "nhyd" in atom_array.get_annotation_categories()

    def _log_h_removal() -> None:
        logger.warning(
            f"Removed H from overvalent {atom_array.res_name[atom_idx]}/{atom_array.atom_name[atom_idx]} "
            f"(chain={atom_array.chain_id[atom_idx]}, res_id={atom_array.res_id[atom_idx]})"
        )

    # ... implicit path
    if has_nhyd and atom_array.nhyd[atom_idx] > 0:
        atom_array.nhyd[atom_idx] -= 1
        _log_h_removal()
        return []

    # ... explicit path
    h_neighbors = neighbors[atom_array.element[neighbors] == "H"]
    if len(h_neighbors) > 0:
        _log_h_removal()
        return [int(h_neighbors[0])]

    # --- Strategy 2: C=O decrement ---
    for o_idx in o_neighbors:
        bond_mask = ((bonds_arr[:, 0] == atom_idx) & (bonds_arr[:, 1] == o_idx)) | (
            (bonds_arr[:, 0] == o_idx) & (bonds_arr[:, 1] == atom_idx)
        )
        if bond_mask.any() and bonds_arr[bond_mask][0, 2] == int(struc.BondType.DOUBLE):
            atom_array.bonds.remove_bond(atom_idx, int(o_idx))
            atom_array.bonds.add_bond(atom_idx, int(o_idx), struc.BondType.SINGLE)
            atom_array.charge[o_idx] = -1
            logger.warning(
                f"Decremented C=O bond order for {atom_array.res_name[atom_idx]}/{atom_array.atom_name[atom_idx]} "
                f"(chain={atom_array.chain_id[atom_idx]}, res_id={atom_array.res_id[atom_idx]})"
            )
            return []

    return []


def _remove_ccd_leaving_atoms(atom_array: struc.AtomArray, inter_bonds: np.ndarray) -> struc.AtomArray:
    """Remove CCD-defined leaving atoms for atoms that form inter-residue bonds.

    Each bond sheds as many leaving groups as its order
    """
    leaving_atom_mask = np.zeros(len(atom_array), dtype=bool)
    has_nhyd = "nhyd" in atom_array.get_annotation_categories()
    _h_first = frozenset(s[0] for s in HYDROGEN_LIKE_SYMBOLS)
    atoms_with_inter_bonds = np.unique(inter_bonds[:, :2])

    def is_implicit_h(group: tuple[str, ...], present: set[str]) -> bool:
        """Identify absent hydrogen-only groups tracked by nhyd, excluding explicit hydrogens."""
        return has_nhyd and present.isdisjoint(group) and all(n[0] in _h_first for n in group if n)

    # Pre-fetch CCD leaving info for every residue that makes an inter-residue bond.
    unique_res_names = np.unique(atom_array.res_name[atoms_with_inter_bonds])
    leaving_cache = {rn: get_chem_comp_leaving_atom_groups(rn) for rn in unique_res_names}
    # Plain aromatic bonds have no well-defined order; they displace one group.
    orders = np.array([BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(struc.BondType(bt), 1) for bt in inter_bonds[:, 2]])
    n_displaceable = np.bincount(inter_bonds[:, :2].ravel(), weights=np.repeat(orders, 2), minlength=len(atom_array))

    # Build per-atom residue slices for O(1) lookup inside the loop.
    _rs = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    sizes = np.diff(_rs)
    atom_to_res_start = np.repeat(_rs[:-1], sizes)
    atom_to_res_stop = np.repeat(_rs[1:], sizes)

    for atom_idx in atoms_with_inter_bonds:
        groups = leaving_cache[atom_array.res_name[atom_idx]].get(atom_array.atom_name[atom_idx], ())
        if not groups:
            continue

        res_start = int(atom_to_res_start[atom_idx])
        res_stop = int(atom_to_res_stop[atom_idx])
        res_atom_names = atom_array.atom_name[res_start:res_stop]
        present = set(res_atom_names)
        observed = set(res_atom_names[np.isfinite(atom_array.coord[res_start:res_stop]).all(axis=-1)])
        still_attached = [g for g in groups if is_implicit_h(g, present) or not present.isdisjoint(g)]
        still_attached.sort(key=lambda group: not observed.isdisjoint(group))
        budget = max(0, int(n_displaceable[atom_idx]) - (len(groups) - len(still_attached)))

        for group in still_attached[:budget]:
            leaving_atom_mask[res_start:res_stop] |= np.isin(res_atom_names, list(group))
            # Implicit H are counted in nhyd rather than present as atoms; keep the
            # counter consistent with the removal.
            if is_implicit_h(group, present):
                atom_array.nhyd[atom_idx] = max(0, int(atom_array.nhyd[atom_idx]) - len(group))

    if np.any(leaving_atom_mask):
        # Sanity check: leaving atoms must only bond to their parent or each other.
        _validate_leaving_atoms_have_no_unexpected_bonds(atom_array, leaving_atom_mask, atoms_with_inter_bonds)
        atom_array = atom_array[~leaving_atom_mask]

    return atom_array


def _resolve_overvalent_atoms(atom_array: struc.AtomArray) -> struc.AtomArray:
    """Fix atoms that remain over-valent after CCD leaving atom removal."""
    inter_bonds = _get_inter_residue_bonds(atom_array)
    if len(inter_bonds) == 0:
        return atom_array

    inter_atoms = np.unique(inter_bonds[:, :2])
    degree = get_bond_degree_per_atom(atom_array)
    has_nhyd = "nhyd" in atom_array.get_annotation_categories()
    # Collect removals and apply in one batch — removing inside the loop would
    # invalidate the pre-computed inter_atoms index array.
    atoms_to_remove: list[int] = []

    for atom_idx in inter_atoms:
        expected = DEFAULT_VALENCE.get(atom_array.element[atom_idx])
        if expected is not None and atom_array.element[atom_idx] != "C":
            # Charge adjusts N/O valence but must not permit five bonds on carbon.
            expected += atom_array.charge[atom_idx]
        if expected is None or degree[atom_idx] <= expected:
            continue

        elem = atom_array.element[atom_idx]
        if elem == "C":
            atoms_to_remove.extend(_maybe_fix_overvalent_carbon(atom_array, atom_idx))
        elif elem == "N" or elem == "O":
            # N/N+: bonded to one extra heavy atom via inter-residue bond; shed H
            #     to restore valence (e.g. LYS NZ→PLP Schiff base, DLY NZ→PLP).
            # O:  formed a covalent bond (ester/ether) but CCD did not mark its
            #     hydroxyl H as a leaving atom; remove that H here.
            excess = int(degree[atom_idx] - expected)
            if has_nhyd and atom_array.nhyd[atom_idx] > 0:
                # Implicit H — in-place decrement, no index change needed.
                n_to_remove = min(excess, atom_array.nhyd[atom_idx])
                atom_array.nhyd[atom_idx] -= n_to_remove
                logger.warning(
                    f"Removed H from overvalent {atom_array.res_name[atom_idx]}/"
                    f"{atom_array.atom_name[atom_idx]} "
                    f"(chain={atom_array.chain_id[atom_idx]}, res_id={atom_array.res_id[atom_idx]})"
                )
            else:
                bonded_h = _find_bonded_hydrogens(atom_array, atom_idx)
                atoms_to_remove.extend([int(h) for h in bonded_h[:excess]])
                if bonded_h[:excess].size:
                    logger.warning(
                        f"Removed H from overvalent {atom_array.res_name[atom_idx]}/"
                        f"{atom_array.atom_name[atom_idx]} "
                        f"(chain={atom_array.chain_id[atom_idx]}, res_id={atom_array.res_id[atom_idx]})"
                    )

    if atoms_to_remove:
        keep = np.ones(len(atom_array), dtype=bool)
        keep[atoms_to_remove] = False
        atom_array = atom_array[keep]

    return atom_array


def resolve_leaving_atoms(atom_array: struc.AtomArray) -> tuple[struc.AtomArray, np.ndarray]:
    """Remove leaving atoms and fix valence for atoms involved in inter-residue bonds.

    Returns:
        ``(atom_array, impacted_mask)`` where ``impacted_mask`` marks atoms
        involved in inter-residue bonds, suitable for charge-correction.

    Raises:
        ValueError: If a leaving atom is bonded to atoms other than its parent
            or fellow leaving atoms.
    """
    inter_bonds = _get_inter_residue_bonds(atom_array)
    if len(inter_bonds) == 0:
        return atom_array, np.zeros(len(atom_array), dtype=bool)

    atom_array = _remove_ccd_leaving_atoms(atom_array, inter_bonds)
    atom_array = _resolve_overvalent_atoms(atom_array)
    return atom_array, get_inter_residue_atom_mask(atom_array)


def correct_formal_charges_for_specified_atoms(atom_array: struc.AtomArray, to_update: np.ndarray) -> struc.AtomArray:
    """Fix formal charges for atoms in an AtomArray based on valence rules and current bonding pattern.

    Args:
        atom_array (AtomArray): The AtomArray to fix.
        to_update (np.ndarray): A boolean mask of atoms whose formal charges should be fixed.
            These are normally the atoms for which bonds were manually added or modified (e.g., inter-residue bonds).

    Returns:
        AtomArray: The AtomArray with fixed formal charges.
    """
    # ... check that the AtomArray has hydrogens or nhyd annotation (which we need to fix formal charges based on valence)
    has_nhyd = "nhyd" in atom_array.get_annotation_categories()
    has_h = np.isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS).any()
    if not has_nhyd and not has_h:
        bond_logger.warning("Neither hydrogens nor nhyd annotation present. Cannot fix formal charges.")
        return atom_array

    # ... get valences (masked for elements with no default valence)
    _invalid = -10
    default_valence = np.array([DEFAULT_VALENCE.get(elt, _invalid) for elt in atom_array.element[to_update]])

    # ... compute total number of bonds per atom
    degree = get_bond_degree_per_atom(atom_array)[to_update]

    # ... compute formal charge
    formal_charge = degree - default_valence

    # Sanity check: atoms with only inter-residue bonds likely lack CCD templates
    # Example: 8cuy UNL, a lipid with inter-residue bonds but no bond information
    lone = degree == 1
    if lone.any():
        global_lone = np.where(to_update)[0][lone]
        has_inter = get_inter_residue_atom_mask(atom_array)
        no_intra_mask = has_inter[global_lone]
        if no_intra_mask.any():
            offending = global_lone[no_intra_mask]
            # Single-atom residues can't have intra-residue bonds by definition — skip them
            _res_start_ends = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
            _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]
            res_sizes = _res_ends - _res_starts
            res_indices = np.searchsorted(_res_starts, offending, side="right") - 1
            multi_atom = offending[res_sizes[res_indices] > 1]
            if len(multi_atom) > 0:
                idx = multi_atom[0]
                raise ValueError(
                    f"Atom {idx} ({atom_array.element[idx]}, atom_name={atom_array.atom_name[idx]}, "
                    f"res_name={atom_array.res_name[idx]}, chain_id={atom_array.chain_id[idx]}, "
                    f"res_id={atom_array.res_id[idx]}) has inter-residue bonds but no "
                    f"intra-residue bonds — usually a missing CCD template. Ensure the residue is not an "
                    f"unknown ligand (UNL) or otherwise lacking bonding information."
                )

    valid = default_valence != _invalid

    # ... convert local indices to global indices
    global_idxs = np.arange(atom_array.array_length())[to_update]
    atom_array.charge[global_idxs[valid]] = formal_charge[valid]
    return atom_array


def _has_amide_bond(atom_array: AtomArray, n_idx: int, bonds_arr: np.ndarray) -> bool:
    """Check if nitrogen at ``n_idx`` is part of an N-C(=O) amide pattern."""
    n_neighbors = _get_bond_neighbors(bonds_arr, n_idx)
    for c_idx in n_neighbors[atom_array.element[n_neighbors] == "C"]:
        c_neighbors = _get_bond_neighbors(bonds_arr, c_idx)
        for o_idx in c_neighbors[atom_array.element[c_neighbors] == "O"]:
            bond_mask = ((bonds_arr[:, 0] == c_idx) & (bonds_arr[:, 1] == o_idx)) | (
                (bonds_arr[:, 0] == o_idx) & (bonds_arr[:, 1] == c_idx)
            )
            if bond_mask.any() and bonds_arr[bond_mask][0, 2] == int(struc.BondType.DOUBLE):
                return True
    return False


def correct_charged_amide_nitrogens(
    atom_array: struc.AtomArray,
    to_update: np.ndarray | None = None,
) -> struc.AtomArray:
    """Neutralize charged nitrogens that are part of an amide pattern.

    When a charged nitrogen (charge=+1) is bonded to a carbon that is also bonded
    to an oxygen (N-C=O / N-C-O amide pattern), the nitrogen should be neutral.

    Example: PDB ID 1qfe Lysine NZ becomes a component of an amide bond to small molecule DHS, and should NOT be charged.

    Args:
        atom_array: The AtomArray to fix.
        to_update: Boolean mask of atoms to consider. If ``None``, defaults to
            atoms involved in inter-residue bonds.

    Returns:
        The AtomArray with amide nitrogens corrected (modified in-place).
    """
    if to_update is None:
        to_update = get_inter_residue_atom_mask(atom_array)
    if not np.any(to_update):
        return atom_array

    bonds_arr = atom_array.bonds.as_array()
    n_mask = (atom_array.element == "N") & (atom_array.charge == 1) & to_update

    h_to_remove: list[int] = []
    for n_idx in np.where(n_mask)[0]:
        if not _has_amide_bond(atom_array, n_idx, bonds_arr):
            continue
        if "nhyd" in atom_array.get_annotation_categories():
            if atom_array.nhyd[n_idx] > 0:
                atom_array.nhyd[n_idx] -= 1
        else:
            bonded_h = _find_bonded_hydrogens(atom_array, n_idx)
            if len(bonded_h) > 0:
                h_to_remove.append(bonded_h[0])
        atom_array.charge[n_idx] -= 1

    if h_to_remove:
        keep = np.ones(len(atom_array), dtype=bool)
        keep[h_to_remove] = False
        atom_array = atom_array[keep]

    return atom_array
