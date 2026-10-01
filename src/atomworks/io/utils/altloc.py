"""Clash-aware alternate conformer selection that respects multiple-occupancy correlations across chains and residues."""

__all__ = ["has_multiple_altlocs_per_residue", "select_altlocs_clash_aware"]

import logging
import random
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from scipy.spatial import cKDTree

from atomworks.constants import ALTLOC_DEFAULT_IDS
from atomworks.io.utils.ccd import (
    get_parent_comp_id,
    get_parent_template_atom_names,
    get_polymerization_atoms,
    is_parent_strict_atom_subset,
)
from atomworks.io.utils.selection import get_annotation

logger = logging.getLogger(__name__)

AtomKey = tuple[str, str, str]  # (chain_id, res_id, atom_name) for struct_conn lookup
ResidueKey = tuple[str, int]  # (chain_id, res_id)


def has_multiple_altlocs_per_residue(atom_array: AtomArray | AtomArrayStack) -> bool:
    """Check if any residue has more than one altloc letter."""
    if "altloc_id" not in atom_array.get_annotation_categories():
        return False

    alt_id = get_annotation(atom_array, "altloc_id")
    non_default = ~np.isin(alt_id, ALTLOC_DEFAULT_IDS)
    if not np.any(non_default):
        return False

    nd_idx = np.where(non_default)[0]
    chains = atom_array.chain_id[nd_idx]
    res_ids = atom_array.res_id[nd_idx]
    alts = alt_id[nd_idx]

    has_trans = "transformation_id" in atom_array.get_annotation_categories()
    if has_trans:
        trans = atom_array.transformation_id[nd_idx]
        residue_keys = set(zip(chains, res_ids, trans, strict=True))
        residue_alt_keys = set(zip(chains, res_ids, trans, alts, strict=True))
    else:
        residue_keys = set(zip(chains, res_ids, strict=True))
        residue_alt_keys = set(zip(chains, res_ids, alts, strict=True))

    return len(residue_alt_keys) > len(residue_keys)


def select_altlocs_clash_aware(
    atom_array: AtomArray | AtomArrayStack,
    clash_distance: float = 1.5,
    proximity_distance: float = 5.0,
    seed: int | None = None,
    unresolved_clash_policy: Literal["raise", "warn"] = "raise",
    cif_block: Any = None,
) -> AtomArray | AtomArrayStack:
    """Select one altloc per residue avoiding steric clashes, and return the subsetted structure.

    Considers:
    - **Intra-chain clashes**: avoids selecting altloc combinations that overlap spatially (``1CBN``).
    - **Cross-chain correlation**: proximal residues sharing altloc letters get the same letter (``6LZB``, ``6QHP``).
    - **Bonded atom exclusion**: struct_conn and polymer bonds are not counted as clashes (``1XVK``, ``1RCQ``).
    - **Parent fallback**: omitted modified residues fall back to their CCD parent (``6QHP``: ASB -> ASP).

    Args:
        atom_array: Structure with all altlocs loaded. If an ``AtomArrayStack``,
            coordinates from the first model are used.
        clash_distance: Distance threshold (Angstroms) below which two atoms from
            different altloc groups are considered clashing. Defaults to ``1.5``.
        proximity_distance: Distance threshold (Angstroms) for detecting cross-chain
            correlated altlocs. Defaults to ``5.0``.
        seed: Random seed for deterministic selection. ``None`` for non-deterministic.
        unresolved_clash_policy: What to do when no fully clash-free assignment exists.
            ``"raise"`` raises a ``ValueError``, ``"warn"`` logs a warning and
            returns the best-effort assignment. Defaults to ``"raise"``.
        cif_block: Optional CIF block for parsing ``struct_conn`` bonds to exclude
            from clash detection. ``None`` (default) when no CIF is available.
    """
    arr = atom_array[0] if isinstance(atom_array, AtomArrayStack) else atom_array

    altloc_ids = get_annotation(atom_array, "altloc_id")
    if altloc_ids is None:
        return atom_array

    defaults = np.isin(altloc_ids, ALTLOC_DEFAULT_IDS)
    altloc_groups = _identify_altloc_groups(arr, altloc_ids, defaults)

    if not altloc_groups:
        return atom_array

    struct_conn_bonds = _parse_struct_conn_bonds(cif_block)
    clashes, preferences = _build_constraints(arr, altloc_groups, clash_distance, proximity_distance, struct_conn_bonds)

    # Remove clashes between preference-linked residues (not real clashes)
    if preferences:
        pref_pairs = set(preferences) | {(r2, r1) for r1, r2 in preferences}
        clashes = [(ri, li, rj, lj) for ri, li, rj, lj in clashes if (ri, rj) not in pref_pairs]

    rng = random.Random(seed)
    assignment, omitted = _solve_greedy(altloc_groups, clashes, preferences, rng, unresolved_clash_policy)

    mask = defaults.copy()
    for res_key, letter in assignment.items():
        mask[altloc_groups[res_key][letter]] = True

    parent_fallbacks = _apply_parent_fallbacks(arr, mask, altloc_groups, omitted, clash_distance, struct_conn_bonds)

    result = atom_array[..., mask]
    for (chain_id, res_id), parent_code in parent_fallbacks.items():
        fb_mask = (result.chain_id == chain_id) & (result.res_id == res_id)
        result.res_name[fb_mask] = parent_code

    return result


def _apply_parent_fallbacks(
    arr: AtomArray,
    mask: np.ndarray,
    altloc_groups: dict[ResidueKey, dict[str, np.ndarray]],
    omitted: set[ResidueKey],
    clash_distance: float,
    struct_conn_bonds: set[tuple[AtomKey, AtomKey]],
) -> dict[ResidueKey, str]:
    """Fall back omitted modified residues to their canonical CCD parent.

    Only applies when the rest of the chain is present (single-residue omission,
    not whole-chain altlocs). Mutates ``mask`` in place.

    Always raises on clash regardless of ``unresolved_clash_policy`` — the fallback
    is deterministic, so a clash indicates a data issue rather than bad luck.
    """
    if not omitted:
        return {}

    # Build KD-tree of selected atoms once for all fallback checks
    selected_indices = np.where(mask)[0]
    selected_coords = arr.coord[selected_indices]
    valid_sel = ~np.any(np.isnan(selected_coords), axis=1)
    sel_tree = cKDTree(selected_coords[valid_sel]) if np.any(valid_sel) else None
    sel_valid_indices = selected_indices[valid_sel]

    parent_fallbacks: dict[ResidueKey, str] = {}
    for res_key in omitted:
        chain_id = res_key[0]
        if not np.any(mask & (arr.chain_id == chain_id)):
            continue

        any_indices = next(iter(altloc_groups[res_key].values()))
        res_name = arr.res_name[any_indices[0]]

        parent = get_parent_comp_id(res_name)
        if parent is None or not is_parent_strict_atom_subset(res_name, parent):
            continue

        parent_atoms = get_parent_template_atom_names(parent)
        first_letter = next(iter(altloc_groups[res_key]))
        indices = altloc_groups[res_key][first_letter]
        keep = np.array([arr.atom_name[i] in parent_atoms for i in indices])
        fallback_indices = indices[keep]

        # Check fallback atoms don't clash with selected atoms
        clashing_idx = None
        if sel_tree is not None and len(fallback_indices) > 0:
            fb_coords = arr.coord[fallback_indices]
            valid_fb = ~np.any(np.isnan(fb_coords), axis=1)
            for ci, coord in zip(fallback_indices[valid_fb], fb_coords[valid_fb], strict=False):
                for nb in sel_tree.query_ball_point(coord, r=clash_distance):
                    si = sel_valid_indices[nb]
                    if not _is_bonded_pair(arr, ci, si, struct_conn_bonds):
                        clashing_idx = si
                        break
                if clashing_idx is not None:
                    break

        if clashing_idx is not None:
            raise ValueError(
                f"Parent fallback for chain {chain_id} res {res_key[1]} "
                f"({res_name} -> {parent}) clashes with "
                f"chain {arr.chain_id[clashing_idx]} res {arr.res_id[clashing_idx]} "
                f"atom {arr.atom_name[clashing_idx]}."
            )

        mask[fallback_indices] = True
        parent_fallbacks[res_key] = parent
        logger.info(
            "Altloc fallback: %s res %d (%s) -> parent %s (%d/%d atoms kept)",
            res_key[0],
            res_key[1],
            res_name,
            parent,
            int(keep.sum()),
            len(indices),
        )

    return parent_fallbacks


def _is_bonded_pair(
    arr: AtomArray,
    idx_a: int,
    idx_b: int,
    struct_conn_bonds: set[tuple[AtomKey, AtomKey]],
) -> bool:
    """Check if two atoms are bonded via polymer bond or struct_conn."""
    name_a, name_b = arr.atom_name[idx_a], arr.atom_name[idx_b]

    out_a, in_a = get_polymerization_atoms(arr.res_name[idx_a])
    out_b, in_b = get_polymerization_atoms(arr.res_name[idx_b])

    if (name_a == out_a and name_b == in_b) or (name_b == out_b and name_a == in_a):
        return True

    key_a = (arr.chain_id[idx_a], str(arr.res_id[idx_a]), name_a)
    key_b = (arr.chain_id[idx_b], str(arr.res_id[idx_b]), name_b)
    return (key_a, key_b) in struct_conn_bonds


def _identify_altloc_groups(
    atom_array: AtomArray,
    altloc_ids: np.ndarray,
    defaults: np.ndarray,
) -> dict[ResidueKey, dict[str, np.ndarray]]:
    """Map each residue with altlocs to ``{letter: atom_indices}``."""
    non_default = ~defaults
    if not np.any(non_default):
        return {}

    groups: dict[ResidueKey, dict[str, list]] = {}
    for idx in np.where(non_default)[0]:
        key = (atom_array.chain_id[idx], int(atom_array.res_id[idx]))
        if key not in groups:
            groups[key] = defaultdict(list)
        groups[key][altloc_ids[idx]].append(idx)

    return {
        key: {letter: np.array(indices, dtype=int) for letter, indices in letters.items()}
        for key, letters in groups.items()
    }


def _parse_struct_conn_bonds(cif_block: Any) -> set[tuple[AtomKey, AtomKey]]:
    """Extract bonded atom pairs from the ``struct_conn`` CIF category."""
    bonds: set[tuple[AtomKey, AtomKey]] = set()
    if cif_block is None or "struct_conn" not in cif_block:
        return bonds

    sc = cif_block["struct_conn"]
    chains_1 = sc["ptnr1_label_asym_id"].as_array()
    seq_ids_1 = sc["ptnr1_label_seq_id"].as_array()
    atoms_1 = sc["ptnr1_label_atom_id"].as_array()
    chains_2 = sc["ptnr2_label_asym_id"].as_array()
    seq_ids_2 = sc["ptnr2_label_seq_id"].as_array()
    atoms_2 = sc["ptnr2_label_atom_id"].as_array()

    # Non-polymer residues have label_seq_id="."; fall back to auth_seq_id
    # so keys match arr.res_id (which uses auth_seq_id for non-polymers).
    for partner, seq_ids in [("1", seq_ids_1), ("2", seq_ids_2)]:
        try:
            auth = sc[f"ptnr{partner}_auth_seq_id"].as_array()
            seq_ids[:] = np.where(seq_ids == ".", auth, seq_ids)
        except KeyError:
            pass

    for c1, s1, a1, c2, s2, a2 in zip(chains_1, seq_ids_1, atoms_1, chains_2, seq_ids_2, atoms_2, strict=False):
        key1 = (str(c1), str(s1), str(a1))
        key2 = (str(c2), str(s2), str(a2))
        bonds.add((key1, key2))
        bonds.add((key2, key1))

    return bonds


def _build_constraints(
    atom_array: AtomArray,
    altloc_groups: dict[ResidueKey, dict[str, np.ndarray]],
    clash_distance: float,
    proximity_distance: float,
    struct_conn_bonds: set[tuple[AtomKey, AtomKey]] | None = None,
) -> tuple[list[tuple[ResidueKey, str, ResidueKey, str]], list[tuple[ResidueKey, ResidueKey]]]:
    """Build hard clash constraints and soft proximity preferences in one pass.

    Bonded atom pairs (polymer bonds, struct_conn) are excluded from clash detection.
    Altloc residues within ``proximity_distance`` are preference-linked.
    """
    if struct_conn_bonds is None:
        struct_conn_bonds = set()

    # Build parallel arrays for each altloc atom
    all_indices: list[int] = []
    all_res_keys: list[ResidueKey] = []
    all_letters: list[str] = []
    for res_key, letters in altloc_groups.items():
        for letter, indices in letters.items():
            all_indices.extend(indices)
            all_res_keys.extend([res_key] * len(indices))
            all_letters.extend([letter] * len(indices))

    all_indices_arr = np.array(all_indices, dtype=int)
    if len(all_indices_arr) < 2:
        return [], []

    coords = atom_array.coord[all_indices_arr]
    valid = ~np.any(np.isnan(coords), axis=1)
    valid_mask_idx = np.where(valid)[0]
    valid_indices = all_indices_arr[valid_mask_idx]
    valid_coords = coords[valid_mask_idx]
    if len(valid_coords) < 2:
        return [], []

    valid_res_keys = [all_res_keys[i] for i in valid_mask_idx]
    valid_letters_arr = [all_letters[i] for i in valid_mask_idx]

    tree = cKDTree(valid_coords)
    pairs = tree.query_pairs(r=max(clash_distance, proximity_distance), output_type="ndarray")
    if len(pairs) == 0:
        return [], []

    dists = np.linalg.norm(valid_coords[pairs[:, 0]] - valid_coords[pairs[:, 1]], axis=1)

    clash_set: set[tuple[ResidueKey, str, ResidueKey, str]] = set()
    pref_set: set[tuple[ResidueKey, ResidueKey]] = set()

    for k in range(len(pairs)):
        pi, pj = pairs[k, 0], pairs[k, 1]
        rk_i, rk_j = valid_res_keys[pi], valid_res_keys[pj]
        if rk_i == rk_j:
            continue

        dist = dists[k]

        if dist <= clash_distance:
            gi, gj = valid_indices[pi], valid_indices[pj]
            if not _is_bonded_pair(atom_array, gi, gj, struct_conn_bonds):
                lt_i, lt_j = valid_letters_arr[pi], valid_letters_arr[pj]
                edge = (rk_i, lt_i, rk_j, lt_j)
                if (rk_j, lt_j, rk_i, lt_i) not in clash_set:
                    clash_set.add(edge)

        # Soft preference: any different residues within proximity_distance get linked
        if dist <= proximity_distance:
            pair = (rk_i, rk_j) if rk_i < rk_j else (rk_j, rk_i)
            pref_set.add(pair)

    return list(clash_set), list(pref_set)


def _solve_greedy(
    altloc_groups: dict[ResidueKey, dict[str, np.ndarray]],
    clashes: list[tuple[ResidueKey, str, ResidueKey, str]],
    preferences: list[tuple[ResidueKey, ResidueKey]],
    rng: random.Random,
    unresolved_clash_policy: Literal["raise", "warn"] = "raise",
) -> tuple[dict[ResidueKey, str], set[ResidueKey]]:
    """Greedy altloc assignment minimizing clashes and respecting preference links.

    Preference-linked residues are grouped via union-find and assigned the same letter.
    """
    residues = list(altloc_groups.keys())
    options = {res: list(altloc_groups[res].keys()) for res in residues}
    for letters in options.values():
        rng.shuffle(letters)

    hard: dict[tuple[ResidueKey, str], set[tuple[ResidueKey, str]]] = defaultdict(set)
    for res_i, let_i, res_j, let_j in clashes:
        hard[(res_i, let_i)].add((res_j, let_j))
        hard[(res_j, let_j)].add((res_i, let_i))

    # Union-find for preference-linked residues
    uf_parent: dict[ResidueKey, ResidueKey] = {r: r for r in residues}

    def find(x: ResidueKey) -> ResidueKey:
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    for r_i, r_j in preferences:
        if r_i in uf_parent and r_j in uf_parent:
            uf_parent[find(r_i)] = find(r_j)

    group_map: dict[ResidueKey, list[ResidueKey]] = defaultdict(list)
    for r in residues:
        group_map[find(r)].append(r)
    groups = list(group_map.values())
    rng.shuffle(groups)

    assignment: dict[ResidueKey, str] = {}
    omitted: set[ResidueKey] = set()

    def count_clashes(res: ResidueKey, letter: str) -> int:
        return sum(1 for o_r, o_l in hard.get((res, letter), set()) if assignment.get(o_r) == o_l)

    for group in groups:
        rng.shuffle(group)
        if len(group) == 1:
            res = group[0]
            assignment[res] = min(options[res], key=lambda lt: count_clashes(res, lt))
        else:
            all_letters: set[str] = set()
            for res in group:
                all_letters.update(options[res])
            candidates = sorted(all_letters)
            rng.shuffle(candidates)

            def group_score(letter: str, grp: list[ResidueKey] = group) -> tuple[int, int]:
                supporters = sum(1 for r in grp if letter in options[r])
                clash_total = sum(count_clashes(r, letter) for r in grp if letter in options[r])
                return (-supporters, clash_total)

            best = min(candidates, key=group_score)
            for res in group:
                if best in options[res]:
                    assignment[res] = best
                else:
                    omitted.add(res)

    total = sum(
        1 for r, lt in assignment.items() for o_r, o_l in hard.get((r, lt), set()) if assignment.get(o_r) == o_l
    )
    if total > 0:
        msg = (
            f"No fully clash-free altloc assignment found for {len(residues)} residues. "
            f"{total // 2} unresolved clashes remain."
        )
        if unresolved_clash_policy == "raise":
            raise ValueError(msg)
        logger.warning("%s Using best-effort assignment.", msg)

    return assignment, omitted
