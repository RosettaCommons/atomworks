"""pH-aware hydrogen addition via RDKit + Dimorphite-DL.

Adds explicit hydrogens to biomolecular structures at a target pH.  Standard
residues are protonated per-residue; non-standard residues and covalent
modifications are protonated as connected bond subgraphs.  Boundary atoms
(bonded outside their component) are re-protonated against the fully assembled
geometry so cross-links and termini are placed correctly.  Each H ends up directly
after the heavy atoms of the residue it is bonded to.  See :func:`ensure_hydrogens`
for the full step-by-step.
"""

import contextlib
import itertools
import logging
from collections import defaultdict

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from rdkit import Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from atomworks.constants import (
    CHAIN_LEVEL_ANNOTATIONS,
    HYDROGEN_LIKE_SYMBOLS,
    METAL_ELEMENTS,
    RESIDUE_LEVEL_ANNOTATIONS,
)
from atomworks.external.dimorphite_dl import protonate_mol_variants
from atomworks.io.tools.rdkit import atom_array_to_rdkit, suppress_rdkit_warnings
from atomworks.io.transforms.atomize import compute_standard_atomize_mask
from atomworks.io.utils.annotator import ANNOTATOR_REGISTRY, _resolve_standard_annotation, ensure_annotations
from atomworks.io.utils.atom_array import annotate_and_remove_hydrogens
from atomworks.io.utils.atom_array_plus import concatenate_atom_array_plus
from atomworks.io.utils.standard_annotations.base import Level

logger = logging.getLogger("atomworks.ml")


_NEUTRAL_VALENCE: dict[str, int] = {"N": 3, "O": 2, "S": 2, "P": 3, "C": 4}

# Explicit defaults for certain common atom-level annotations.
# Applied only to new H atoms; heavy-atom values are left unchanged.
_H_ATOM_ANNOTATION_DEFAULTS: dict[str, object] = {
    "occupancy": 1.0,
    "b_factor": np.nan,
    "stereo": "N",
    "is_aromatic": False,
    "is_backbone_atom": False,
}

# Annotations where H atoms inherit the value from their bonded heavy atom.
# Propagated after bonds are set, so bond connectivity is available.
_H_ATOM_INHERIT_FROM_PARENT: frozenset[str] = frozenset(
    {
        "molecule_entity",
        "label_entity_id",
        "molecule_id",
        "pn_unit_entity",
        "pn_unit_id",
        "chain_iid",
        "pn_unit_iid",
        "molecule_iid",
        "is_covalent_modification",
        "chain_type",
        "transformation_id",
    }
)


# ---------------------------------------------------------------------------
# Charge adjustment helpers
# ---------------------------------------------------------------------------


def _adjust_charges_for_pruned_h(
    result: AtomArray,
    remove: set[int],
    bonded_pairs: set[tuple[int, int]],
) -> None:
    """Adjust formal charges on heavy atoms whose bonded H will be pruned.

    When an H is removed from a charged group (e.g. NH3+ losing one H),
    the parent heavy atom's formal charge must be decremented to remain
    chemically consistent.  Modifies ``result.charge`` in place.

    Args:
        result: The full assembled AtomArray (before pruning).
        remove: Indices of H atoms that will be removed.
        bonded_pairs: Set of ``(i, j)`` index pairs representing bonds.
    """
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in bonded_pairs:
        adj[a].append(b)

    for h_idx in remove:
        parent_indices = [b for b in adj[h_idx] if result.element[b] != "H"]

        for parent_idx in parent_indices:
            current_charge = int(result.charge[parent_idx])
            elem = result.element[parent_idx].upper()
            neutral_val = _NEUTRAL_VALENCE.get(elem)
            if neutral_val is None:
                result.charge[parent_idx] = current_charge - 1
                logger.debug(
                    f"Adjusted charge on {result.atom_name[parent_idx]} (idx {parent_idx}) "
                    f"from {current_charge:+d} to {current_charge - 1:+d} after H pruning "
                    f"(no known neutral valence for {elem}, assumed proton loss)"
                )
                continue

            neighbors = adj[parent_idx]
            surviving_h = sum(1 for b in neighbors if result.element[b] == "H" and b not in remove)
            total_non_h_bonds = sum(1 for b in neighbors if result.element[b] != "H")
            actual_valence = total_non_h_bonds + surviving_h
            new_charge = actual_valence - neutral_val
            if new_charge != current_charge:
                result.charge[parent_idx] = new_charge
                logger.debug(
                    f"Adjusted charge on {result.atom_name[parent_idx]} (idx {parent_idx}) "
                    f"from {current_charge:+d} to {new_charge:+d} after H pruning"
                )


# ---------------------------------------------------------------------------
# Subgraph partitioning
# ---------------------------------------------------------------------------


def _partition_atomized_subgraphs(
    heavy_array: AtomArray,
    atomize_mask: np.ndarray,
) -> list[np.ndarray]:
    """Partition atomized atoms into connected subgraphs via bond adjacency.

    Returns a list of arrays of *global* indices into *heavy_array*, one per
    connected component in the bond graph restricted to atomized atoms.
    """
    global_indices = np.where(atomize_mask)[0]
    n_atomized = len(global_indices)
    if n_atomized == 0:
        return []

    local_of_arr = np.full(heavy_array.array_length(), -1, dtype=np.intp)
    local_of_arr[global_indices] = np.arange(n_atomized)

    bond_arr = heavy_array.bonds.as_array()
    a_idx = bond_arr[:, 0].astype(int)
    b_idx = bond_arr[:, 1].astype(int)
    la = local_of_arr[a_idx]
    lb = local_of_arr[b_idx]
    both_in = (la >= 0) & (lb >= 0)

    rows = np.concatenate([la[both_in], lb[both_in]])
    cols = np.concatenate([lb[both_in], la[both_in]])
    data = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(n_atomized, n_atomized))
    n_components, labels = connected_components(adj, directed=False)
    component_indices = [global_indices[labels == c] for c in range(n_components)]

    return component_indices


def _find_boundary_local_indices(
    global_indices: np.ndarray,
    heavy_array: AtomArray,
) -> set[int]:
    """Return *local* indices (within global_indices) of boundary atoms.

    A boundary atom has at least one bond to an atom outside global_indices.
    """
    n = heavy_array.array_length()
    local_of_arr = np.full(n, -1, dtype=np.intp)
    local_of_arr[global_indices] = np.arange(len(global_indices))

    bond_arr = heavy_array.bonds.as_array()
    a_idx = bond_arr[:, 0].astype(int)
    b_idx = bond_arr[:, 1].astype(int)

    la = local_of_arr[a_idx]
    lb = local_of_arr[b_idx]

    boundary_local_a = la[(la >= 0) & (lb < 0)]
    boundary_local_b = lb[(lb >= 0) & (la < 0)]
    if len(boundary_local_a) == 0 and len(boundary_local_b) == 0:
        return set()

    boundary_local_indices = np.unique(np.concatenate([boundary_local_a, boundary_local_b]))
    return {int(x) for x in boundary_local_indices}


# ---------------------------------------------------------------------------
# Core protonation
# ---------------------------------------------------------------------------


def _make_h_block(
    sub: AtomArray,
    mol_with_h: Chem.Mol,
    only_bonded_to: int | None = None,
) -> AtomArray | None:
    """Extract new H atoms from *mol_with_h* as an AtomArray.

    Assumes the first ``len(sub)`` atoms in *mol_with_h* correspond to *sub*
    (heavy atoms) and the remainder are new H.  Each H takes its residue/chain
    annotations from the heavy atom it is bonded to, which matters whenever a
    component spans more than one residue (e.g. covalently linked ``MSE``).

    Args:
        sub: Heavy-atom AtomArray whose annotations serve as the template.
        mol_with_h: RDKit Mol returned by ``Chem.AddHs``.
        only_bonded_to: If set, only include H atoms bonded to this RDKit atom
            index.  Used when placing boundary H to discard neighbour H that
            were added by ``AddHs`` but should not be copied back.

    Returns:
        AtomArray of new H atoms, or ``None`` if none were added.
    """
    n_heavy = len(sub)
    if mol_with_h.GetNumConformers() == 0:
        return None

    if only_bonded_to is not None:
        h_rdkit_indices = [
            a.GetIdx() for a in mol_with_h.GetAtomWithIdx(only_bonded_to).GetNeighbors() if a.GetIdx() >= n_heavy
        ]
    else:
        h_rdkit_indices = list(range(n_heavy, mol_with_h.GetNumAtoms()))

    n_new_h = len(h_rdkit_indices)
    if n_new_h == 0:
        return None

    conf = mol_with_h.GetConformer()
    h_block = struc.AtomArray(n_new_h)
    h_block.coord = np.array(
        [
            [conf.GetAtomPosition(idx).x, conf.GetAtomPosition(idx).y, conf.GetAtomPosition(idx).z]
            for idx in h_rdkit_indices
        ]
    )
    # AddHs gives every new H exactly one neighbour: the heavy atom it was placed on.
    parent_of_h = np.array([mol_with_h.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx() for idx in h_rdkit_indices])

    h_block.element[:] = "H"
    h_block.chain_id[:] = sub.chain_id[parent_of_h]
    h_block.res_id[:] = sub.res_id[parent_of_h]
    h_block.res_name[:] = sub.res_name[parent_of_h]

    h_block.atom_name = np.array([f"H{idx - n_heavy}" for idx in h_rdkit_indices])

    _residue_and_chain = frozenset(RESIDUE_LEVEL_ANNOTATIONS + CHAIN_LEVEL_ANNOTATIONS)
    for annot in sub.get_annotation_categories():
        if annot in ("chain_id", "res_id", "res_name", "atom_name", "element", "coord"):
            continue
        if annot == "charge":
            h_block.set_annotation("charge", np.zeros(n_new_h, dtype=sub.get_annotation(annot).dtype))
        elif annot == "atomic_number":
            h_block.set_annotation("atomic_number", np.full(n_new_h, 1, dtype=sub.get_annotation(annot).dtype))
        elif annot in _residue_and_chain:
            with contextlib.suppress(Exception):
                h_block.set_annotation(annot, sub.get_annotation(annot)[parent_of_h])
        else:
            logger.debug("Skipping atom-level annotation '%s' for new H atoms", annot)

    return h_block


def _add_h_to_mol(
    sub: AtomArray,
    mol_3d: Chem.Mol,
) -> AtomArray:
    """Place H atoms in 3D and append them to *sub*, returning the combined array.

    Args:
        sub: Heavy-atom AtomArray for the component.
        mol_3d: RDKit Mol (heavy atoms only, with 3D conformer).
    """
    mol_3d_h = Chem.AddHs(mol_3d, addCoords=True)
    h_block = _make_h_block(sub, mol_3d_h)
    if h_block is None:
        return sub

    n_heavy = mol_3d.GetNumAtoms()
    sub_h = concatenate_atom_array_plus([sub, h_block], on_annotation_mismatch_policy="drop")

    bond_tuples = []
    if sub.bonds is not None:
        for row in sub.bonds.as_array():
            bond_tuples.append((int(row[0]), int(row[1]), int(row[2])))
    n_sub = len(sub)
    for bond in mol_3d_h.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a >= n_heavy or b >= n_heavy:
            mapped_a = a if a < n_heavy else n_sub + (a - n_heavy)
            mapped_b = b if b < n_heavy else n_sub + (b - n_heavy)
            bond_tuples.append((mapped_a, mapped_b, int(struc.BondType.SINGLE)))
    if bond_tuples:
        sub_h.bonds = struc.BondList(len(sub_h), np.array(bond_tuples))

    return sub_h


def _protonate_component(
    sub: AtomArray,
    ph: float,
    silence_rdkit_warnings: bool = False,
) -> AtomArray:
    """Protonate a component (residue or subgraph) via Dimorphite-DL + RDKit.

    Charges are assigned by Dimorphite-DL for the whole component.  Boundary
    correction (resetting charges and H count at atoms bonded outside this
    component) is applied later, after all components are assembled and clashing
    H removed.

    Args:
        sub: Heavy-atom AtomArray for the component.
        ph: Target pH.
        silence_rdkit_warnings: If ``True``, silence Python-level RDKit warnings
            during molecule conversion. Defaults to ``False``.

    Raises:
        ValueError: If bonds are missing on *sub*.
    """
    if sub.bonds is None:
        raise ValueError(
            f"Bonds missing on component {sub.chain_id[0]}:{sub.res_id[0]} — "
            "bonds must be present before protonation."
        )

    ctx = suppress_rdkit_warnings() if silence_rdkit_warnings else contextlib.nullcontext()
    with ctx:
        mol = atom_array_to_rdkit(
            sub,
            set_coord=True,
            hydrogen_policy="remove",
            sanitize=True,
            attempt_fixing_corrupted_molecules=False,
        )
    if mol is None:
        return sub

    variants = protonate_mol_variants(
        mol,
        min_ph=ph,
        max_ph=ph,
        pka_precision=0.1,
        max_variants=1,
        silent=True,
    )
    mol_3d = variants[0] if variants else mol

    # Dimorphite returns atoms in input order; assignment also enforces equal lengths.
    if "charge" in sub.get_annotation_categories():
        sub.charge[:] = [atom.GetFormalCharge() for atom in mol_3d.GetAtoms()]

    return _add_h_to_mol(sub, mol_3d)


# ---------------------------------------------------------------------------
# Hydrogen clash removal
# ---------------------------------------------------------------------------


def _order_hydrogens_by_residue(result: AtomArray, n_heavy: int) -> AtomArray:
    """Give every added H its parent's residue identity, a unique name, and a slot in that residue.

    Args:
        result: Protonated structure, heavy atoms first (in input order), then the new H.
        n_heavy: Number of heavy atoms, i.e. the length of that leading block.
    """
    n_atoms = result.array_length()
    if result.bonds is None or n_heavy >= n_atoms:
        return result

    is_h = np.isin(result.element, HYDROGEN_LIKE_SYMBOLS)
    bond_arr = result.bonds.as_array()
    a_idx, b_idx = bond_arr[:, 0].astype(int), bond_arr[:, 1].astype(int)
    parent = np.full(n_atoms, -1, dtype=np.intp)
    h_a, h_b = is_h[a_idx] & ~is_h[b_idx], ~is_h[a_idx] & is_h[b_idx]
    parent[a_idx[h_a]] = b_idx[h_a]
    parent[b_idx[h_b]] = a_idx[h_b]
    has_parent = is_h & (parent >= 0)

    # Sort each H into its own residue, keeping heavy atoms (and H among themselves) in order.
    res_starts = struc.get_residue_starts(result[:n_heavy], add_exclusive_stop=True)
    residue_rank = np.full(n_atoms, len(res_starts), dtype=np.intp)  # H without a parent sort last
    residue_rank[:n_heavy] = np.repeat(np.arange(len(res_starts) - 1), np.diff(res_starts))
    residue_rank[has_parent] = residue_rank[parent[has_parent]]
    result = result[np.lexsort((np.arange(n_atoms), is_h, residue_rank))]

    # Number the H per residue rather than per component, so names stay unique within a residue.
    is_h = np.isin(result.element, HYDROGEN_LIKE_SYMBOLS)
    atom_names = result.atom_name.copy()
    res_start_stops = struc.get_residue_starts(result, add_exclusive_stop=True)
    for start, stop in itertools.pairwise(res_start_stops):
        h_indices = np.where(is_h[start:stop])[0] + start
        atom_names[h_indices] = [f"H{number}" for number in range(len(h_indices))]
    result.set_annotation("atom_name", atom_names)

    if "atom_id" in result.get_annotation_categories():
        result.set_annotation("atom_id", np.arange(n_atoms, dtype=result.atom_id.dtype))

    return result


def _remove_clashing_hydrogens(result: AtomArray) -> AtomArray:
    """Remove H atoms that sterically clash with non-bonded neighbours.

    Removes H-H pairs closer than 0.7 A and H-heavy pairs closer than
    1.0 A (across residue boundaries).  Adjusts formal charges on parent
    heavy atoms when their bonded H is pruned.

    Atoms with NaN coordinates (disordered heavy atoms skipped during
    protonation) are excluded from the spatial index but never removed.
    """
    h_mask = result.element == "H"
    if not h_mask.any():
        return result

    h_indices = np.where(h_mask)[0]
    remove: set[int] = set()

    bonded_pairs: set[tuple[int, int]] = set()
    if result.bonds is not None:
        for row in result.bonds.as_array():
            bonded_pairs.add((int(row[0]), int(row[1])))
            bonded_pairs.add((int(row[1]), int(row[0])))

    # NaN-coord atoms cannot go into a KD-tree; H atoms from RDKit always
    # have valid coordinates.
    valid_coord_mask = ~np.isnan(result.coord).any(axis=-1)
    valid_global = np.where(valid_coord_mask)[0]
    if len(valid_global) == 0:
        return result

    all_tree = cKDTree(result.coord[valid_coord_mask])

    for hi in h_indices:
        neighbor_local = all_tree.query_ball_point(result.coord[hi], r=1.0)
        for nl in neighbor_local:
            ni = int(valid_global[nl])
            if ni == hi or (hi, ni) in bonded_pairs:
                continue
            dist = np.linalg.norm(result.coord[hi] - result.coord[ni])
            if result.element[ni] == "H" and dist < 0.7:
                remove.add(hi)
                break
            if result.element[ni] != "H" and dist < 1.0:
                remove.add(hi)
                break

    if remove:
        if "charge" in result.get_annotation_categories():
            _adjust_charges_for_pruned_h(result, remove, bonded_pairs)
        keep = np.ones(len(result), dtype=bool)
        keep[list(remove)] = False
        result = result[keep]
        logger.debug("Removed %d clashing H atoms", len(remove))

    return result


# ---------------------------------------------------------------------------
# Boundary correction (applied after full assembly and clash removal)
# ---------------------------------------------------------------------------


def _place_h_on_boundary_atom(
    result: AtomArray,
    center_idx: int,
    nhyd: int,
) -> AtomArray | None:
    """Place exactly ``nhyd`` H on a boundary heavy atom using RDKit 3D geometry.

    Builds a local subarray (boundary atom at index 0, plus all bonded heavy
    neighbors) and converts it to an RDKit mol via
    :func:`~atomworks.io.tools.rdkit.atom_array_to_rdkit`.  The fragment is
    sanitized with neighbours at their natural valence (so hybridisation is
    correct for geometry) and the boundary atom's H count is overridden *after*
    sanitisation.  :func:`rdkit.Chem.AddHs` places H on all atoms; only those
    bonded to the boundary atom are kept.

    Args:
        result: AtomArray containing all heavy atoms with their bonds (no H present).
        center_idx: Index of the boundary heavy atom in ``result``.
        nhyd: Exact number of H atoms to place.

    Returns:
        An :class:`~biotite.structure.AtomArray` of the new H atoms, or ``None``
        if RDKit placement failed.
    """
    if result.bonds is None:
        return None

    nan_coord = np.isnan(result.coord).any(axis=-1)

    bonded_nbrs, _ = result.bonds.get_bonds(center_idx)
    # Exclude NaN-coord neighbors: their coordinates corrupt RDKit's bond-vector
    # geometry, causing the "zero perpendicular" assertion in AddHs(addCoords=True).
    heavy_neighbor_indices = [
        int(n) for n in bonded_nbrs if result.element[n] not in HYDROGEN_LIKE_SYMBOLS and not nan_coord[n]
    ]

    local_indices = np.array([center_idx, *heavy_neighbor_indices])
    local_sub = result[local_indices]  # bonds auto-filtered and reindexed by biotite

    mol = atom_array_to_rdkit(
        local_sub,
        set_coord=True,
        hydrogen_policy="remove",
        sanitize=False,
        attempt_fixing_corrupted_molecules=False,
    )

    # Sanitize with neighbours at natural valence so the fragment is chemically
    # valid and hybridisation is correct for geometry.  H count on the boundary
    # atom is overridden *after* sanitisation so the valence check never sees it.
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        logger.warning(f"Sanitization failed for boundary H at idx {center_idx}: {e}")

    # Set exact H count on the boundary atom after sanitisation.
    center_rdatom = mol.GetAtomWithIdx(0)
    center_rdatom.SetNumExplicitHs(nhyd)
    center_rdatom.SetNoImplicit(True)

    try:
        mol_with_h = Chem.AddHs(mol, addCoords=True)
    except Exception as e:
        logger.warning(f"Chem.AddHs failed for boundary atom idx {center_idx}: {e}")
        return None

    # Neighbours also receive H from AddHs; discard them via only_bonded_to.
    return _make_h_block(local_sub, mol_with_h, only_bonded_to=0)


# ---------------------------------------------------------------------------
# Main hydrogen addition
# ---------------------------------------------------------------------------


def ensure_hydrogens(
    atom_array: AtomArray,
    ph: float = 7.4,
    silence_rdkit_warnings: bool = False,
    silence_lost_annotation_warnings: bool = False,
) -> AtomArray:
    """Add hydrogens with pH-aware protonation via RDKit + Dimorphite-DL.

    Standard residues are processed per-residue.  Non-standard residues and
    covalently-modified polymer residues are processed as connected subgraphs
    so that the only boundary atoms are either in or directly bonded to
    standard polymer residues.

    After all components are assembled and clashing H removed, boundary atoms
    (bonded to atoms outside their component) have all Dimorphite-placed H
    removed, charges reset to the input ``charge`` annotation, and exactly
    ``nhyd`` H re-added via RDKit using the full assembled geometry.  This
    ensures correct H positions at covalent cross-links and polymer termini.

    1. Strip existing H for a clean slate.
    2. Bonds must be present; raises ``ValueError`` if absent.
    3. Determine atomized atoms (non-standard residues, covalent mods) via
       :func:`~atomworks.io.utils.atomize.compute_standard_atomize_mask`.
    4. Validate that no residue is partially atomized.
    5. Non-atomized residues → per-residue protonation (NaN-coord atoms skipped
       but preserved in output at their original positions).
    6. Atomized atoms → partition into connected subgraphs; protonate each.
    7. Charge updates are applied to the original heavy-atom array in-place.
    8. New H atoms are appended after all heavy atoms; bonds are remapped to
       global indices.
    9. Remove clashing H (H-H < 0.7 A, H-heavy < 1.0 A across residues).
    10. Reset boundary atom charges and H counts to match input annotations.
    11. Move each H next to the residue it is bonded to and rename it uniquely
        within that residue.

    Args:
        atom_array: Input structure.
        ph: Target pH for protonation.  Default 7.4.
        silence_rdkit_warnings: If ``True``, silence Python-level RDKit warnings
            during molecule conversion. Defaults to ``False``.
        silence_lost_annotation_warnings: If ``True``, silence warnings about lost annotations
            during hydrogen addition.  Defaults to ``False``.

    Returns:
        Structure with H added at the target pH.  Heavy atoms appear in their
        original order (including NaN-coord atoms), with each residue's H directly
        after that residue's heavy atoms, so chains stay contiguous and ``res_id``
        non-decreasing.  ``nhyd`` -- the *implicit* H count -- is zero on output,
        except on heavy atoms that were never protonated (NaN coords), which keep
        theirs.  H are renamed ``H0``, ``H1``, ... within their residue.

    Raises:
        ValueError: If bonds are absent, or if a partially-atomized residue
            is detected (indicates a bug in the atomization logic).

    NOTE: Free H atoms or ions would be lost in this workflow. This could be
    addressed if there is a use-case in which it becomes limiting.
    """
    if isinstance(atom_array, AtomArrayStack):
        raise TypeError(
            "ensure_hydrogens does not support AtomArrayStack as input. "
            "Extract a single model first (e.g. atom_array[0])."
        )

    if (np.isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS)).any():
        logger.warning("Input atom_array contains hydrogen-like atoms; these will be removed before protonation.")

    # ``increment=True``: ``nhyd`` counts *implicit* H (H that exist chemically but are not
    # present as atoms), so adding the explicit count to it and then removing explicity hydrogens is correct.
    heavy_array = annotate_and_remove_hydrogens(atom_array, increment=True)

    if heavy_array.bonds is None:
        raise ValueError("Input atom_array has no bonds. Bonds must be assigned before calling ensure_hydrogens. ")
    if "charge" not in heavy_array.get_annotation_categories():
        raise ValueError(
            "Input atom_array must have a 'charge' annotation for heavy atoms before calling ensure_hydrogens. "
            "This is used as a fallback for boundary atoms between standard residues and atomized regions. "
        )

    atomize_mask = compute_standard_atomize_mask(heavy_array)
    nan_mask = np.isnan(heavy_array.coord).any(axis=-1)

    # result_heavy is our working copy; charge updates from protonation are
    # applied here in-place.
    result_heavy = heavy_array.copy()
    result_heavy.set_annotation("atom_id", np.arange(len(heavy_array), dtype=np.intp))

    new_h_arrays: list[AtomArray] = []
    all_new_bonds: list[tuple[int, int, int]] = []
    h_offset = 0  # cumulative H atoms collected so far

    # -----------------------------------------------------------------------
    # Build the list of components to protonate.
    # canonical residues first, then atomized subgraphs.
    # -----------------------------------------------------------------------
    components: list[np.ndarray] = []

    res_starts_stops = struc.get_residue_starts(heavy_array, add_exclusive_stop=True)
    res_starts = res_starts_stops[:-1]
    res_stops = res_starts_stops[1:]

    for s, e in zip(res_starts, res_stops, strict=False):
        indices = np.arange(s, e)
        res_atomize = atomize_mask[indices]
        if res_atomize.all():
            continue  # handled by the subgraph path
        if res_atomize.any():
            raise ValueError(
                f"Partially-atomized residue detected at {heavy_array.chain_id[s]}:{heavy_array.res_id[s]} "
                f"({heavy_array.res_name[s]}): {res_atomize.sum()}/{len(indices)} atoms flagged. "
            )
        valid = indices[~nan_mask[indices]]
        if len(valid) > 0:
            components.append(valid)

    for sg_global in _partition_atomized_subgraphs(heavy_array, atomize_mask):
        valid = sg_global[~nan_mask[sg_global]]
        if len(valid) > 0:
            components.append(valid)

    # -----------------------------------------------------------------------
    # Process each component: protonate, update charges, collect new H atoms.
    # -----------------------------------------------------------------------
    for valid_global_indices in components:
        sub = heavy_array[valid_global_indices]
        if all(elem.upper() in METAL_ELEMENTS for elem in sub.element):
            continue  # metals pass through unchanged in result_heavy

        boundary_local = _find_boundary_local_indices(valid_global_indices, heavy_array)
        protonated = _protonate_component(sub, ph, silence_rdkit_warnings=silence_rdkit_warnings)
        n_valid = len(valid_global_indices)

        # Update charges for non-boundary heavy atoms only; boundary atoms keep
        # their original charge from heavy_array.
        non_boundary_mask = ~np.isin(np.arange(n_valid), list(boundary_local))
        protonated_heavy = protonated[:n_valid]
        result_heavy.charge[valid_global_indices[non_boundary_mask]] = protonated_heavy.charge[non_boundary_mask]

        n_new_h = len(protonated) - n_valid
        if n_new_h > 0:
            # keep_h: indicates non-boundary hydrogens added by Dimorphite
            keep_h = np.zeros(protonated.array_length(), dtype=bool)
            keep_h[n_valid:] = True
            if protonated.bonds is not None:
                for boundary_local_idx in boundary_local:
                    bonded_local_indices, _ = protonated.bonds.get_bonds(boundary_local_idx)
                    bonded_h = bonded_local_indices[
                        np.isin(protonated.element[bonded_local_indices], HYDROGEN_LIKE_SYMBOLS)
                    ]
                    keep_h[bonded_h] = False

            included_h = protonated[keep_h]
            if len(included_h) > 0:
                new_h_arrays.append(included_h)

                # Map every component-local index → global result index.
                local_to_global = np.full(protonated.array_length(), -1, dtype=np.intp)
                local_to_global[:n_valid] = valid_global_indices
                local_to_global[keep_h] = len(heavy_array) + h_offset + np.arange(keep_h.sum())

                if protonated.bonds is not None:
                    for row in protonated.bonds.as_array():
                        a, b, bt = int(row[0]), int(row[1]), int(row[2])
                        if a < n_valid and b < n_valid:
                            continue  # heavy-heavy; already in result_heavy.bonds
                        ga, gb = int(local_to_global[a]), int(local_to_global[b])
                        if ga < 0 or gb < 0:
                            continue  # boundary H excluded
                        all_new_bonds.append((ga, gb, bt))

                h_offset += len(included_h)

        # Place RDKit H for boundary atoms using result_heavy (has all heavy-heavy bonds).
        for loc in boundary_local:
            global_index = int(valid_global_indices[loc])
            nhyd = int(heavy_array.nhyd[global_index])
            if nhyd <= 0:
                continue
            h_block = _place_h_on_boundary_atom(result_heavy, global_index, nhyd)
            if h_block is None or len(h_block) == 0:
                continue
            new_h_arrays.append(h_block)
            for j in range(len(h_block)):
                all_new_bonds.append((global_index, len(heavy_array) + h_offset + j, int(struc.BondType.SINGLE)))
            h_offset += len(h_block)

    # -----------------------------------------------------------------------
    # Assemble: heavy atoms in original order, then all new H appended.
    # -----------------------------------------------------------------------
    to_inherit_from_parent: set[str] = set()

    if new_h_arrays:
        all_h = concatenate_atom_array_plus(new_h_arrays, on_annotation_mismatch_policy="drop")
        result = concatenate_atom_array_plus([result_heavy.copy(), all_h], on_annotation_mismatch_policy="drop")

        updated_atom_id = np.arange(len(result), dtype=np.intp)
        n_heavy = result_heavy.array_length()
        updated_atom_id[:n_heavy] = result_heavy.atom_id
        result.set_annotation("atom_id", updated_atom_id)

        # Add any annotations with defaults that were present in the original AtomArray.
        # Use defaults for new H atoms and preserve their original values for heavy atoms
        output_annots = result.get_annotation_categories()
        lost_annots = set()
        annotations_to_add = set()
        for annot in result_heavy.get_annotation_categories():
            if annot in output_annots:
                continue
            elif _resolve_standard_annotation(annot) is not None:
                sa_cls = _resolve_standard_annotation(annot)[0]
                if sa_cls.level == Level.ATOM:
                    annotations_to_add.add(annot)
                else:
                    to_inherit_from_parent.add(annot)
            elif annot in ANNOTATOR_REGISTRY:
                annotations_to_add.add(annot)
            elif annot in _H_ATOM_ANNOTATION_DEFAULTS:
                original_annot = result_heavy.get_annotation(annot)
                vals = np.full(len(result), _H_ATOM_ANNOTATION_DEFAULTS[annot], dtype=original_annot.dtype)
                vals[:n_heavy] = original_annot
                result.set_annotation(annot, vals)
            elif annot in _H_ATOM_INHERIT_FROM_PARENT:
                to_inherit_from_parent.add(annot)
            elif annot == "nhyd":
                pass  # recomputed from actual bonded H count after clash removal
            else:
                lost_annots.add(annot)

        ensure_annotations(result, *list(annotations_to_add))
        for annot in annotations_to_add:
            result_annot = result.get_annotation(annot)
            result_annot[:n_heavy] = result_heavy.get_annotation(annot)
            result.set_annotation(annot, result_annot)

        if lost_annots and not silence_lost_annotation_warnings:
            logger.warning(
                f"The following annotations without defaults were lost due to protonation: {', '.join(lost_annots)}. "
            )

    else:
        result = result_heavy

    # concatenate_atom_array_plus carries result_heavy's heavy-heavy bonds into result unchanged
    # Extend with the H-heavy bonds computed during protonation.
    if all_new_bonds:
        existing = result.bonds.as_array() if result.bonds is not None else np.empty((0, 3), dtype=np.intp)
        combined = np.vstack([existing, np.array(all_new_bonds)]) if len(existing) else np.array(all_new_bonds)
        result.bonds = struc.BondList(len(result), combined)

    # Propagate per-residue annotations from each heavy atom to its bonded H atoms.
    # Done after bond extension so H-heavy bonds are available.
    if to_inherit_from_parent and result.bonds is not None:
        n_heavy = result_heavy.array_length()
        bond_arr = result.bonds.as_array()
        a_idx = bond_arr[:, 0].astype(int)
        b_idx = bond_arr[:, 1].astype(int)
        is_h = np.isin(result.element, HYDROGEN_LIKE_SYMBOLS)

        # H atoms will all have exactly one heavy atom parent
        parent_of = np.full(len(result), -1, dtype=np.intp)
        h_a = is_h[a_idx] & ~is_h[b_idx]
        h_b = ~is_h[a_idx] & is_h[b_idx]
        parent_of[a_idx[h_a]] = b_idx[h_a]
        parent_of[b_idx[h_b]] = a_idx[h_b]

        h_idx = np.where(is_h)[0]
        parents = parent_of[h_idx]
        for annot in to_inherit_from_parent:
            src = result_heavy.get_annotation(annot)
            vals = np.empty(len(result), dtype=src.dtype)
            vals[:n_heavy] = src
            vals[h_idx] = src[parents]
            result.set_annotation(annot, vals)

    result = _remove_clashing_hydrogens(result)

    # ``nhyd`` counts *implicit* H only.  Every H placed above is an explicit atom, so the
    # remaining implicit count is zero -- except on heavy atoms we never protonated
    # (e.g. those with NaN coords)
    if "nhyd" in heavy_array.get_annotation_categories():
        n_heavy_atoms = result_heavy.array_length()
        nhyd_arr = np.zeros(len(result), dtype=heavy_array.nhyd.dtype)
        never_protonated = np.isnan(result_heavy.coord).any(axis=-1)
        nhyd_arr[:n_heavy_atoms][never_protonated] = heavy_array.nhyd[never_protonated]
        result.set_annotation("nhyd", nhyd_arr)

    result = _order_hydrogens_by_residue(result, result_heavy.array_length())

    logger.debug("Added H (pH %s): %d -> %d atoms", ph, len(heavy_array), len(result))
    return result
