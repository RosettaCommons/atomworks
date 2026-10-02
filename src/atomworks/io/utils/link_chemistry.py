"""Link chemistry: propose connectivity, then resolve and validate linked residues."""

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
from rdkit.Chem import GetPeriodicTable

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
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array import (
    _find_bonded_hydrogens,
    _get_bond_neighbors,
    get_bond_degree_per_atom,
)
from atomworks.io.utils.ccd import _standard_ccd_only_cache, atom_array_from_ccd_code, get_polymerization_atoms

logger = logging.getLogger(__name__)


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
        # Allow authored glycosylation through 2.4 A, e.g. the 2.211 A ASN-NAG link in PDB 2ODP.
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

        # Longer competing links can belong to another conformer, e.g. the PLP-LYS link in PDB 1RCQ.
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
                    logger.warning(
                        "Long explicit glycosylation %s (keeping through %.1f A)",
                        msg,
                        allowed_thresholds[k],
                    )
                elif base == "raise":
                    raise ValueError(msg)
                elif base == "filter":
                    logger.warning("Skipping %s", msg)
                else:
                    logger.warning("Long struct_conn bond %s (keeping)", msg)

    return valid


def _valence_with_neutralization(element: str, reference_valence: int) -> int:
    """Allow the reference or neutral default valence, without selecting a higher hypervalence.

    For example, O(-) with reference valence 1 can form a second bond and become neutral ester oxygen.
    """
    return max(reference_valence, GetPeriodicTable().GetDefaultValence(element.title()))


def infer_link_order(atom_array: AtomArray, atom1: int, atom2: int) -> int:
    """Infer an unspecified link order; the caller preserves recognized explicit orders.

    Reuse the CCD order for an existing bond within the same residue. Otherwise,
    start with the higher CCD leaving-bond order of the two partners (single if
    neither declares one), then cap it by both partners' available capacity:
    hydrogen/leaving-group displacement plus unused valence, including neutralization.
    Thus O3'-P stays single in 1DPN, while unbonded selenium can accept a bond in 7ZCY.

    For single-bond proposals, two PDB compatibility heuristics also permit:
    - Carbonyl addition, with downstream C=O reduction to C-O(-), as in 1TQH.
    - Addition to neutral nitrogen with bond-order sum 3, producing four-coordinate N(+).
    These are candidates only; downstream chemistry must still validate the product.

    Raises:
        ValueError: Required CCD template/atom/bond data are missing, or either
            partner cannot support a bond under these rules.
    """
    partners = [(str(atom_array.res_name[i]), str(atom_array.atom_name[i])) for i in (atom1, atom2)]
    same_residue = all(
        atom_array.get_annotation(key)[atom1] == atom_array.get_annotation(key)[atom2]
        for key in ("chain_id", "res_id", "ins_code", "res_name", "transformation_id")
        if key in atom_array.get_annotation_categories()
    )
    # A leaving C=O can support a new C=N, e.g. the PLP-DLY link in PDB 1RCQ.
    order = max((int(get_leaving_atom_bond_type(*p) or 1) for p in partners), default=1)
    single_addition = order == 1
    for res_name, atom_name in partners:
        try:
            template = atom_array_from_ccd_code(res_name, coords=None)
        except ValueError as error:
            raise ValueError(f"Cannot infer link order for {partners}: missing CCD template for {res_name}") from error
        matches = np.flatnonzero(template.atom_name == atom_name)
        if len(matches) != 1 or template.bonds is None:
            raise ValueError(
                f"Cannot infer link order for {partners}: missing CCD atom/bonds for {res_name}/{atom_name}"
            )
        neighbors, types = template.bonds.get_bonds(int(matches[0]))
        # PDB 1FYL repeats the intra-residue BRU C5-BR bond in struct_conn; it needs no additional valence.
        if same_residue:
            existing = types[template.atom_name[neighbors] == partners[1][1]]
            if len(existing):
                return int(existing[0])
        orders = np.array([BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(struc.BondType(t), 1) for t in types])
        leaving = {n for group in get_leaving_atom_groups(template).get(atom_name, ()) for n in group}
        retained = ~np.isin(template.element[neighbors], HYDROGEN_LIKE_SYMBOLS) & ~np.isin(
            template.atom_name[neighbors], list(leaving)
        )
        # Both partners must support the order: in PDB 1DPN, O3' gives up one H, so O3'-P stays single.
        # Include unused valence: an unbonded atom need not displace anything, e.g. selenium in 7ZCY.
        capacity = _valence_with_neutralization(template.element[matches[0]], int(orders.sum())) - int(
            orders[retained].sum()
        )

        # Compatibility heuristics for under-annotated PDB links: the CCD may describe the unlinked reactant.
        # Allow these single-bond candidates despite insufficient capacity; downstream chemistry must still validate.
        if single_addition and capacity < 1:
            element = template.element[matches[0]]
            # Carbonyl carbon: downstream cleanup can reduce C=O to C-O(-), preserving 1TQH behavior.
            allows_carbonyl_addition = element == "C" and np.any((orders == 2) & (template.element[neighbors] == "O"))
            # Neutral N with bond-order sum 3: an added bond can produce four-coordinate N(+) without displacement.
            allows_nitrogen_cation = element == "N" and orders.sum() == 3 and template.charge[matches[0]] == 0
            if allows_carbonyl_addition or allows_nitrogen_cation:
                capacity = 1

        order = min(order, capacity)
    if order < 1:
        raise ValueError(
            f"Cannot infer link order for {partners}: retained heavy-atom bonds leave no valence; provide explicit product connectivity"
        )
    return order


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


def _has_existing_polymer_attachment(atom_array: AtomArray, atom1: int, atom2: int, atom_to_res: np.ndarray) -> bool:
    """Respect attachments to the adjacent residue; reject competing attachments elsewhere."""
    has_other_link = False
    for atom, other in ((atom1, atom2), (atom2, atom1)):
        neighbors, types = atom_array.bonds.get_bonds(atom)
        residues = atom_to_res[neighbors[types != struc.BondType.COORDINATION]]
        # Includes the proposed bond itself and noncanonical attachments, e.g. 1XVK.
        if atom_to_res[other] in residues:
            return True
        has_other_link |= np.any(residues != atom_to_res[atom])
    if has_other_link:
        raise ValueError(
            f"Conflicting polymer attachment at {atom_array.chain_id[atom1]}/"
            f"{atom_array.res_id[atom1]}:{atom_array.atom_name[atom1]}-"
            f"{atom_array.res_id[atom2]}:{atom_array.atom_name[atom2]}"
        )
    return False


def _validate_inferred_polymer_bond(
    atom_array: AtomArray, atom1: int, atom2: int, degree: np.ndarray, leaving_groups: dict[str, dict]
) -> None:
    """Check whether a proposed polymer bond can be added without internal bond rearrangement.

    Each endpoint must have free valence, an explicit/implicit H, or a CCD-declared
    leaving group. Sequence alone cannot justify rearranging a saturated backbone
    (e.g. SNN in 7P1D). Existing attachments are checked separately by the caller.
    ``leaving_groups`` caches CCD lookups within the caller's operation only.

    Raises:
        ValueError: An endpoint has neither free valence nor a hydrogen/declared
            leaving group to displace.
    """
    # Sequence does not authorize rearranging an already saturated backbone, e.g. SNN in 7P1D.
    for idx in (atom1, atom2):
        res_name = atom_array.res_name[idx]
        if res_name not in leaving_groups:
            leaving_groups[res_name] = get_chem_comp_leaving_atom_groups(res_name)
        groups = leaving_groups[res_name]
        expected = DEFAULT_VALENCE.get(atom_array.element[idx], 0) + atom_array.charge[idx]
        has_h = ("nhyd" in atom_array.get_annotation_categories() and atom_array.nhyd[idx] > 0) or len(
            _find_bonded_hydrogens(atom_array, int(idx))
        )
        if degree[idx] >= expected and not groups.get(atom_array.atom_name[idx]) and not has_h:
            raise ValueError(
                f"Cannot infer polymer bond at {atom_array.res_name[idx]}/{atom_array.atom_name[idx]}: "
                "no free valence or declared leaving group"
            )


def add_polymer_bonds(atom_array: AtomArray) -> AtomArray:
    """Add single bonds between consecutive polymer attachment sites.

    Leave arrays without bonds or chain-type annotations unchanged. Consider adjacent
    residues only within the same chain/transformation, skipping residue-number gaps
    greater than one and pairs whose attachment atoms cannot be identified or found.

    For each candidate pair:
    - Preserve an existing attachment from either backbone site to the adjacent residue,
      including noncanonical attachments (1XVK); do not add a second backbone link.
    - Allow side-chain crosslinks to coexist with the backbone (4AAH), but reject a
      backbone site already attached to another residue outside the pair.
    - Require free valence, H, or a declared leaving group at both sites, including
      sequence-defined caps (VAL-NH2 in 3N95 and unresolved ACE-SER in 7RCU).

    Merge accepted bonds into the input array and return it. Leaving-atom removal
    and charge correction are handled later, not by this function.

    Raises:
        ValueError: A backbone attachment conflicts or an endpoint has no capacity
            for the inferred bond.
    """
    if atom_array.bonds is None or "chain_type" not in atom_array.get_annotation_categories():
        return atom_array

    has_tid = "transformation_id" in atom_array.get_annotation_categories()
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)

    # Pre-compute atom→residue mapping and valence for attachment checks.
    sizes = np.diff(res_starts)
    atom_to_res = np.repeat(np.arange(len(res_starts) - 1, dtype=np.intp), sizes)
    degree = get_bond_degree_per_atom(atom_array)

    atom_names = atom_array.atom_name.tolist()
    res_names = atom_array.res_name
    chain_ids = atom_array.chain_id
    res_ids = atom_array.res_id
    transformation_ids = atom_array.transformation_id if has_tid else None

    def _find_atom(start: int, stop: int, name: str) -> int | None:
        try:
            return atom_names.index(name, start, stop)
        except ValueError:
            return None

    new_bonds = []
    # Keep reuse local: custom CCD entries can change between calls.
    polymerization_atoms = functools.cache(get_polymerization_atoms)
    leaving_groups = {}
    for i in range(len(res_starts) - 2):
        # Find candidate attachment atoms in consecutive residues of the same chain instance.
        s1, e1, s2, e2 = res_starts[i], res_starts[i + 1], res_starts[i + 1], res_starts[i + 2]
        if chain_ids[s1] != chain_ids[s2]:
            continue
        if has_tid and transformation_ids[s1] != transformation_ids[s2]:
            continue
        # Skip over genuine chain-break gaps (missing loops)
        if int(res_ids[s2]) - int(res_ids[s1]) > 1:
            continue

        atoms_1 = polymerization_atoms(res_names[s1], ChainType.as_enum(atom_array.chain_type[s1]))
        atoms_2 = polymerization_atoms(res_names[s2], ChainType.as_enum(atom_array.chain_type[s2]))
        if atoms_1[0] is None or atoms_2[1] is None:
            continue

        idx_in = _find_atom(s2, e2, atoms_2[1])
        idx_out = _find_atom(s1, e1, atoms_1[0])
        if idx_in is None or idx_out is None:
            continue

        # Existing backbone attachments take precedence; side-chain links do not (4AAH).
        if _has_existing_polymer_attachment(atom_array, idx_out, idx_in, atom_to_res):
            continue

        # Validate capacity for ordinary H/leaving-group removal.
        _validate_inferred_polymer_bond(atom_array, idx_out, idx_in, degree, leaving_groups)
        new_bonds.append((idx_out, idx_in, int(struc.BondType.SINGLE)))

    if new_bonds:
        new_bond_arr = np.array(new_bonds, dtype=np.uint32)
        atom_array.bonds = atom_array.bonds.merge(struc.BondList(atom_array.array_length(), new_bond_arr))

    return atom_array


def _find_connected_components_after_removal(graph: nx.Graph, node_to_remove: int) -> list[list[int]]:
    """Identifies connected components that would form after removing a node from a graph.

    The graph is not modified. Only components reachable from neighbors of the
    removed node are returned; unrelated components are not included.

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
    return get_leaving_atom_groups(chem_comp)


def get_leaving_atom_groups(chem_comp: struc.AtomArray) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Leaving atom groups of a component template (none without ``is_leaving_atom`` flags)."""
    if "is_leaving_atom" not in chem_comp.get_annotation_categories():
        return {}
    is_leaving_atom = chem_comp.get_annotation("is_leaving_atom")
    if not any(is_leaving_atom):
        return {}

    return dict(
        _leaving_groups_from_topology(
            tuple(chem_comp.atom_name),
            tuple(chem_comp.element),
            tuple(is_leaving_atom),
            tuple(map(tuple, chem_comp.bonds.as_array()[:, :2].tolist())),
        )
    )


@functools.lru_cache(maxsize=2048)
def _leaving_groups_from_topology(
    atom_name: tuple[str, ...],
    element: tuple[str, ...],
    leaving_flags: tuple[bool, ...],
    bond_endpoints: tuple[tuple[int, int], ...],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Reuse leaving groups by chemical content, including for changing registry overrides.

    Bond orders and coordinates do not affect connectivity or leaving-group membership.
    Callers must copy the returned dictionary before exposing it for mutation.
    """
    leaving_atom_groups = defaultdict(list)
    is_leaving_atom = np.asarray(leaving_flags)
    bond_graph = nx.Graph()
    bond_graph.add_edges_from(bond_endpoints)
    for atom_idx in range(len(atom_name)):
        # ... find the connected groups of atoms if the current atom were removed
        connected_groups = _find_connected_components_after_removal(bond_graph, atom_idx)

        # A heavy leaving group carries its attached hydrogens with it.
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


def _get_inter_residue_bonds(atom_array: AtomArray) -> np.ndarray:
    """Return inter-residue bonds excluding coordination."""
    if atom_array.bonds is None:
        return np.empty((0, 3), dtype=int)
    bonds = atom_array.bonds.as_array()
    bonds = bonds[bonds[:, 2] != struc.BondType.COORDINATION]
    identity = ["chain_id", "res_id", "ins_code"]
    if "transformation_id" in atom_array.get_annotation_categories():
        identity.append("transformation_id")
    is_inter = np.zeros(len(bonds), dtype=bool)
    for field in identity:
        values = atom_array.get_annotation(field)
        is_inter |= values[bonds[:, 0]] != values[bonds[:, 1]]
    return bonds[is_inter]


def get_inter_residue_atom_mask(atom_array: struc.AtomArray) -> np.ndarray:
    """Get boolean mask indicating which atoms are involved in inter-residue bonds."""
    inter_bonds = _get_inter_residue_bonds(atom_array)
    atom_mask = np.zeros(len(atom_array), dtype=bool)
    atom_mask[np.unique(inter_bonds[:, :2])] = True
    return atom_mask


def _is_overvalent(atom_array: struc.AtomArray, atom_idx: int, degree: np.ndarray) -> bool:
    expected = DEFAULT_VALENCE.get(atom_array.element[atom_idx])
    if expected is not None and atom_array.element[atom_idx] != "C":
        # Charge adjusts N/O valence but must not permit five bonds on carbon.
        expected += atom_array.charge[atom_idx]
    return expected is not None and degree[atom_idx] > expected


def _displace_hydrogens(atom_array: AtomArray, atom_idx: int, count: int, removed: set[int]) -> int:
    """Detach up to ``count`` H, updating implicit counts and collecting explicit atoms for deletion."""
    implicit = 0
    if "nhyd" in atom_array.get_annotation_categories():
        implicit = min(count, int(atom_array.nhyd[atom_idx]))
        atom_array.nhyd[atom_idx] -= implicit
    explicit = _find_bonded_hydrogens(atom_array, atom_idx)[: count - implicit]
    for hydrogen in explicit:
        atom_array.bonds.remove_bond(atom_idx, int(hydrogen))
        removed.add(int(hydrogen))
    return implicit + len(explicit)


def _maybe_fix_overvalent_carbon(
    atom_array: AtomArray, atom_idx: int, degree: np.ndarray, removed: set[int], impacted: np.ndarray
) -> None:
    """Try to resolve an overvalent carbon involved in an inter-residue link.

    First reject an intra-residue C=C/C=N addition when the carbon has no removable
    H or its multiple-bond partner is also overvalent. Removing H in these cases
    could hide an unsupported reaction product (e.g. 1N4E and 8QIA).

    Otherwise, try these strategies in order, returning after the first applies:

    1. **Hydrogen displacement:** remove one implicit or explicit bonded H, such
       as the template H displaced from the ACE cap in 1J8Z. Explicit H bonds are
       detached immediately, but the caller must delete the collected H atoms.
    2. **Carbonyl addition:** when no H can be removed, reduce a C=O bond to C-O
       and set the oxygen charge to -1 (e.g. SER OG attacking 4PA CAI in 1TQH).
       Reject this inference if the carbon retains a single-bonded nitrogen in
       its own residue: addition versus amide substitution needs an explicit
       product definition, regardless of coordinates (e.g. ASN CG in 6N0A).

    If neither strategy applies, leave the carbon unresolved for the caller's
    final valence validation. No atoms are deleted from the array here.

    Args:
        atom_array: Structure whose bonds, hydrogen counts, and charges may be
            modified in place.
        atom_idx: Index of the overvalent carbon to examine.
        degree: Per-atom bond-order sums, including hydrogens; updated in place.
        removed: Set to which displaced explicit H indices are added for later deletion.
        impacted: Boolean mask updated to include oxygen changed by carbonyl addition.

    Raises:
        ValueError: If resolving the link requires an unsupported multiple-bond
            addition or an ambiguous amide-carbonyl reaction.
    """
    partners, types = atom_array.bonds.get_bonds(atom_idx)
    residue = struc.get_all_residue_positions(atom_array)
    same_residue = residue[partners] == residue[atom_idx]

    # --- Guard: reject unsupported additions across C/C or C/N multiple bonds ---
    # Do not disguise C=C/C=N additions as H substitution or invent proton transfers (1N4E, 8QIA).
    multiple_bonded_partners = partners[
        same_residue
        & np.isin(atom_array.element[partners], ("C", "N"))
        & np.array([BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(struc.BondType(t), 1) > 1 for t in types])
    ]
    has_h = ("nhyd" in atom_array.get_annotation_categories() and atom_array.nhyd[atom_idx] > 0) or len(
        _find_bonded_hydrogens(atom_array, atom_idx)
    ) > 0
    if len(multiple_bonded_partners) and (
        not has_h or any(_is_overvalent(atom_array, int(i), degree) for i in multiple_bonded_partners)
    ):
        raise ValueError(
            f"Unsupported link chemistry at {atom_array.res_name[atom_idx]}/{atom_array.atom_name[atom_idx]}: "
            "C=C/C=N addition requires explicit product bond orders and protonation"
        )

    # --- Strategy 1: displace one bonded H (implicit or explicit) ---
    # An added link can displace a template H, e.g. on the ACE cap in PDB 1J8Z.
    if _displace_hydrogens(atom_array, atom_idx, 1, removed):
        degree[atom_idx] -= 1
        return

    # --- Strategy 2: carbonyl addition, only when no H can be displaced ---
    # C=O becomes C-O(-), e.g. SER OG attacking the 4PA carbonyl in PDB 1TQH.
    carbonyl_oxygens = partners[(atom_array.element[partners] == "O") & (types == struc.BondType.DOUBLE)]
    if len(carbonyl_oxygens) == 0:
        return  # No supported correction; the caller's final valence validation must reject unresolved carbon.

    # Amide exception: retained N leaves substitution vs. addition ambiguous (ASN in 6N0A).
    # Neither missing nor present N coordinates specify which product to build.
    if np.any(same_residue & (atom_array.element[partners] == "N") & (types == struc.BondType.SINGLE)):
        raise ValueError(
            f"Ambiguous link chemistry at {atom_array.res_name[atom_idx]}/{atom_array.atom_name[atom_idx]}: "
            "amide carbonyl retains nitrogen; provide an explicit product definition"
        )

    oxygen = int(carbonyl_oxygens[0])
    atom_array.bonds.remove_bond(atom_idx, oxygen)
    atom_array.bonds.add_bond(atom_idx, oxygen, struc.BondType.SINGLE)
    atom_array.charge[oxygen] = -1
    degree[[atom_idx, oxygen]] -= 1
    impacted[oxygen] = True


def _validate_leaving_bond_orders(atom_array: AtomArray) -> None:
    """Reject unsupported bond-order deficits after CCD leaving atoms have been removed.

    For each linked atom, compare its CCD leaving-bond order with the sum of its
    inter-residue bond orders (excluding coordination). If the new links replace
    less bond order than the leaving bond supplied, and the atom's current total
    bond-order sum is below its CCD reference, require explicit product bond orders.

    For example, TAF P in 1DPN loses a double-bonded oxygen but gains a single P-O
    link. The link alone does not specify which remaining P-O bond should become
    double, so we reject the ambiguity rather than choose a bond to promote.

    Raises:
        ValueError: A linked atom meets both deficit conditions above.
    """
    inter_bonds = _get_inter_residue_bonds(atom_array)
    orders = [BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(struc.BondType(bt), 1) for bt in inter_bonds[:, 2]]
    gained = np.bincount(inter_bonds[:, :2].ravel(), weights=np.repeat(orders, 2), minlength=len(atom_array))
    degree = get_bond_degree_per_atom(atom_array)
    # Repeated attachment sites share CCD chemistry only for this validation call.
    leaving_bond_type = functools.cache(get_leaving_atom_bond_type)
    for atom_idx in np.flatnonzero(gained):
        key = (atom_array.res_name[atom_idx], atom_array.atom_name[atom_idx])
        lost = BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(leaving_bond_type(*key), 0)
        if lost > gained[atom_idx]:
            template = atom_array_from_ccd_code(key[0], coords=None)
            expected = get_bond_degree_per_atom(template)[template.atom_name == key[1]].item()
            # The incoming link does not determine which internal bond to promote, e.g. TAF in 1DPN.
            if degree[atom_idx] < expected:
                raise ValueError(
                    f"Unsupported bond-order rearrangement at {key[0]}/{key[1]}: "
                    "provide explicit product bond orders instead of inferring a replacement double bond"
                )


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
    # Each unit of link order displaces one leaving group.
    orders = np.array([BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(struc.BondType(bt), 1) for bt in inter_bonds[:, 2]])
    n_displaceable = np.bincount(inter_bonds[:, :2].ravel(), weights=np.repeat(orders, 2), minlength=len(atom_array))

    # Build per-atom residue slices for O(1) lookup inside the loop.
    _rs = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    sizes = np.diff(_rs)
    atom_to_res_start = np.repeat(_rs[:-1], sizes)
    atom_to_res_stop = np.repeat(_rs[1:], sizes)
    observed_mask = np.isfinite(atom_array.coord).all(axis=-1)
    current_res_start = -1

    for atom_idx in atoms_with_inter_bonds:
        groups = leaving_cache[atom_array.res_name[atom_idx]].get(atom_array.atom_name[atom_idx], ())
        if not groups:
            continue

        res_start = int(atom_to_res_start[atom_idx])
        res_stop = int(atom_to_res_stop[atom_idx])
        # Inter-bond atoms are sorted: all parents in a residue can share these sets.
        if res_start != current_res_start:
            res_atom_names = atom_array.atom_name[res_start:res_stop]
            present = set(res_atom_names)
            observed = set(res_atom_names[observed_mask[res_start:res_stop]])
            current_res_start = res_start
        still_attached = [g for g in groups if is_implicit_h(g, present) or not present.isdisjoint(g)]
        still_attached.sort(key=lambda group: not observed.isdisjoint(group))
        budget = max(0, int(n_displaceable[atom_idx]) - (len(groups) - len(still_attached)))

        for group in still_attached[:budget]:
            leaving_atom_mask[res_start:res_stop] |= (
                res_atom_names == group[0] if len(group) == 1 else np.isin(res_atom_names, list(group))
            )
            # Implicit H are counted in nhyd rather than present as atoms; keep the
            # counter consistent with the removal.
            if is_implicit_h(group, present):
                atom_array.nhyd[atom_idx] = max(0, int(atom_array.nhyd[atom_idx]) - len(group))

    if np.any(leaving_atom_mask):
        # Sanity check: leaving atoms must only bond to their parent or each other.
        _validate_leaving_atoms_have_no_unexpected_bonds(atom_array, leaving_atom_mask, atoms_with_inter_bonds)
        atom_array = atom_array[~leaving_atom_mask]

    return atom_array


def _resolve_overvalent_atoms(atom_array: AtomArray) -> tuple[AtomArray, np.ndarray]:
    """Apply supported link corrections without letting atom order change the valence bookkeeping."""
    impacted = get_inter_residue_atom_mask(atom_array)
    inter_atoms = np.flatnonzero(impacted)
    carbons = inter_atoms[atom_array.element[inter_atoms] == "C"]
    other_atoms = inter_atoms[atom_array.element[inter_atoms] != "C"]
    degree = get_bond_degree_per_atom(atom_array)
    removed = set()
    # Validate carbon chemistry before donor H cleanup can hide an unsupported addition.
    for idx in [*carbons, *other_atoms]:
        if idx in removed or not _is_overvalent(atom_array, idx, degree):
            continue
        element = atom_array.element[idx]
        if element == "C":
            _maybe_fix_overvalent_carbon(atom_array, idx, degree, removed, impacted)
        elif element in ("N", "O"):
            # Displace only enough H to accommodate the added bond order.
            excess = int(degree[idx] - DEFAULT_VALENCE[element] - atom_array.charge[idx])
            _displace_hydrogens(atom_array, int(idx), excess, removed)
        degree = get_bond_degree_per_atom(atom_array)
    if not removed:
        return atom_array, impacted
    keep = ~np.isin(np.arange(len(atom_array)), list(removed))
    return atom_array[keep], impacted[keep]


def resolve_leaving_atoms(atom_array: struc.AtomArray) -> tuple[struc.AtomArray, np.ndarray]:
    """Remove leaving atoms and fix valence for atoms involved in inter-residue bonds.

    Returns:
        ``(atom_array, impacted_mask)`` where ``impacted_mask`` marks atoms
        involved in inter-residue bonds or changed by their repairs, suitable for charge-correction.

    Raises:
        ValueError: If a leaving atom is bonded to atoms other than its parent
            or fellow leaving atoms, or supported corrections leave an atom overvalent.
    """
    inter_bonds = _get_inter_residue_bonds(atom_array)
    if len(inter_bonds) == 0:
        return atom_array, np.zeros(len(atom_array), dtype=bool)

    atom_array = _remove_ccd_leaving_atoms(atom_array, inter_bonds)
    _validate_leaving_bond_orders(atom_array)
    atom_array, impacted = _resolve_overvalent_atoms(atom_array)
    _validate_link_valence(atom_array, impacted)
    return atom_array, impacted


def correct_formal_charges_for_specified_atoms(atom_array: AtomArray, to_update: np.ndarray) -> AtomArray:
    """Update selected atoms' formal charges from completed valence and hydrogen counts."""
    if (
        "nhyd" not in atom_array.get_annotation_categories()
        and not np.isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS).any()
    ):
        logger.warning("Neither hydrogens nor nhyd annotation present. Cannot fix formal charges.")
        return atom_array

    indices = np.flatnonzero(to_update)
    degree = get_bond_degree_per_atom(atom_array)[indices]
    # An isolated link endpoint can indicate missing CCD connectivity, e.g. the UNL lipid in PDB 8CUY.
    lone = indices[(degree == 1) & get_inter_residue_atom_mask(atom_array)[indices]]
    if len(lone):
        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        sizes = np.diff(starts)
        multi_atom = lone[sizes[np.searchsorted(starts, lone, side="right") - 1] > 1]
        if len(multi_atom):
            idx = multi_atom[0]
            raise ValueError(
                f"Atom {idx} ({atom_array.element[idx]}, atom_name={atom_array.atom_name[idx]}, "
                f"res_name={atom_array.res_name[idx]}, chain_id={atom_array.chain_id[idx]}, "
                f"res_id={atom_array.res_id[idx]}) has inter-residue bonds but no "
                "intra-residue bonds — usually a missing CCD template. Ensure the residue is not an "
                "unknown ligand (UNL) or otherwise lacking bonding information."
            )

    default_valence = np.array([DEFAULT_VALENCE.get(elt, -10) for elt in atom_array.element[indices]])
    valid = default_valence != -10
    atom_array.charge[indices[valid]] = (degree - default_valence)[valid]
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
    """Neutralize linked amide nitrogens, removing the excess H."""
    # Amide formation removes the excess N-H, e.g. LYS NZ linked to DHS in PDB 1QFE.
    if to_update is None:
        to_update = get_inter_residue_atom_mask(atom_array)
    if not np.any(to_update):
        return atom_array

    bonds_arr = atom_array.bonds.as_array()
    n_mask = (atom_array.element == "N") & (atom_array.charge == 1) & to_update

    h_to_remove = set()
    for n_idx in np.where(n_mask)[0]:
        if not _has_amide_bond(atom_array, n_idx, bonds_arr):
            continue
        # Neutralization requires an actual removable H, not just a charge adjustment.
        if not _displace_hydrogens(atom_array, int(n_idx), 1, h_to_remove):
            raise ValueError(f"Cannot neutralize amide nitrogen {n_idx}: no removable hydrogen")
        atom_array.charge[n_idx] -= 1

    if h_to_remove:
        keep = np.ones(len(atom_array), dtype=bool)
        keep[list(h_to_remove)] = False
        atom_array = atom_array[keep]

    return atom_array


def _validate_link_valence(atom_array: AtomArray, impacted: np.ndarray, *, complete: bool = False) -> None:
    """Validate link endpoints and repaired neighbors before and after charge normalization."""
    degree = get_bond_degree_per_atom(atom_array)
    # Reject unresolved valence, e.g. the three-bond O3' from conflicting authored links in PDB 4V4S.
    for idx in np.flatnonzero(impacted):
        element, charge = atom_array.element[idx], atom_array.charge[idx]
        expected = DEFAULT_VALENCE.get(element)
        expected = expected if element == "C" or expected is None else expected + charge
        incomplete = complete and expected is not None and degree[idx] != expected
        # Neutral N can become N(+) upon alkylation; do not similarly hide a third oxygen bond with a charge.
        nitrogen_cation = element == "N" and degree[idx] == 4 and charge in (0, 1)
        # Validate connectivity before charge correction; a linked anion may become neutral (e.g. ester O).
        overvalent = expected is not None and degree[idx] > _valence_with_neutralization(element, expected)
        if incomplete or (overvalent and not nitrogen_cation):
            raise ValueError(
                f"Unresolved link valence at {atom_array.chain_id[idx]}/{atom_array.res_id[idx]}/"
                f"{atom_array.res_name[idx]}/{atom_array.atom_name[idx]} "
                f"(element={atom_array.element[idx]}, charge={atom_array.charge[idx]}, bond-order sum={degree[idx]:g})"
            )


def resolve_link_chemistry(atom_array: AtomArray) -> AtomArray:
    """Resolve supported link chemistry and charges; raise if valence remains inconsistent."""
    atom_array, impacted = resolve_leaving_atoms(atom_array)
    if not impacted.any():
        return atom_array
    atom_array = correct_formal_charges_for_specified_atoms(atom_array, impacted)
    _validate_link_valence(atom_array, impacted, complete=True)
    return correct_charged_amide_nitrogens(atom_array, impacted)
