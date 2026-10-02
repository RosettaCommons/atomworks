"""Non-covalent interaction annotation for biomolecular structures (PLIP-aligned).

Detects H-bonds, hydrophobic contacts, pi-stacking, pi-cation, salt bridges,
halogen bonds, and metal coordination between chains.  The
:class:`AnnotateInteractions` transform adds pH-aware hydrogens
(:mod:`atomworks.experimental.protonation`) if missing, writes 1D/2D annotations, then strips temporary H.
Two H-bond geometry models are available, selected via ``hbond_model``:
``"rosetta"`` (O'Meara et al. 2015) and ``"plip"`` (Hubbard & Haider 2001).
"""

import contextlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import cKDTree

from atomworks.constants import METAL_ELEMENTS, STANDARD_DNA, STANDARD_POLYMER_RESIDUES, STANDARD_RNA
from atomworks.experimental.protonation import add_hydrogens
from atomworks.io.tools.rdkit import atom_array_to_rdkit, suppress_rdkit_warnings
from atomworks.io.utils.atom_array import chain_identifier
from atomworks.io.utils.atom_array_plus import AnnotationList2D, as_atom_array_plus
from atomworks.io.utils.scatter import safe_scatter
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.ml.transforms._checks import check_atom_array_has_bonds, check_contains_keys, check_is_instance
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.geometry import (
    angle_between_vectors,
    dihedral_angle,
    in_plane_offset,
    is_planar,
    plane_normal,
)

logger = logging.getLogger("atomworks.ml")


def _atom_keys(aa: AtomArray) -> list[tuple[str, int, str]]:
    """Per-atom identity tuple ``(chain_identifier, res_id, atom_name)``."""
    return list(zip(chain_identifier(aa), aa.res_id, aa.atom_name, strict=False))


def build_atom_index_mapping(src: AtomArray, dst: AtomArray) -> np.ndarray:
    """Map each atom in *src* to its counterpart in *dst* by ``(chain_identifier, res_id, atom_name)``.

    Returns:
        Int array of length ``len(src)``.  ``result[i] = j`` means
        ``src[i]`` corresponds to ``dst[j]``.  ``-1`` for unmatched atoms.
        If a key appears multiple times in *dst*, the first occurrence wins.
    """
    dst_lookup: dict[tuple, int] = {}
    for i, k in enumerate(_atom_keys(dst)):
        dst_lookup.setdefault(k, i)

    mapping = np.full(len(src), -1, dtype=np.intp)
    for i, k in enumerate(_atom_keys(src)):
        if k in dst_lookup:
            mapping[i] = dst_lookup[k]

    n_matched = int((mapping >= 0).sum())
    if n_matched < len(src) or len(src) != len(dst):
        logger.warning(
            "Annotation transfer: matched %d/%d src atoms to %d dst atoms.",
            n_matched,
            len(src),
            len(dst),
        )
    return mapping


_EMPTY_PAIRS = np.empty((0, 2), dtype=np.int32)
_EMPTY_VALUES = np.empty(0, dtype=np.int32)

_INTERACTION_1D = (
    "interaction_hbond_role",
    "interaction_hydrophobic",
    "interaction_aromatic",
    "interaction_pistacking",
    "interaction_pication_role",
    "interaction_charged_role",
    "interaction_metal_role",
    "interaction_halogen",
)
_ANNOTATION_NAMES_2D: dict[str, str] = {
    "hbond": "interaction_hbond",
    "hydrophobic": "interaction_hydrophobic",
    "pistacking": "interaction_pistacking",
    "pication": "interaction_pication",
    "saltbridge": "interaction_saltbridge",
    "halogen": "interaction_halogen",
    "metal": "interaction_metal",
}


def _transfer_interaction_annotations(src: AtomArray, dst: AtomArray) -> None:
    """Transfer all interaction annotations from *src* to *dst*, matching atoms by identity.

    Handles 1D (per-atom role) and 2D (pairwise interaction) annotations.
    Unmatched atoms get zero-filled defaults.
    """
    idx = build_atom_index_mapping(src, dst)

    for name in _INTERACTION_1D:
        if name not in src.get_annotation_categories():
            continue
        src_vals = src.get_annotation(name)
        dst.set_annotation(name, safe_scatter(src_vals, idx, np.zeros(len(dst), dtype=src_vals.dtype)))

    for name in _ANNOTATION_NAMES_2D.values():
        if name not in get_annotation_categories(src, n_body=2):
            continue
        ann2d = src.get_annotation(name, n_body=2)
        if len(ann2d.pairs) == 0:
            dst.set_annotation(name, AnnotationList2D(len(dst), _EMPTY_PAIRS, _EMPTY_VALUES), n_body=2)
            continue
        a_mapped = idx[ann2d.pairs[:, 0]]
        b_mapped = idx[ann2d.pairs[:, 1]]
        keep = (a_mapped >= 0) & (b_mapped >= 0)
        if keep.any():
            dst.set_annotation(
                name,
                AnnotationList2D(
                    len(dst), np.column_stack([a_mapped[keep], b_mapped[keep]]).astype(np.int32), ann2d.values[keep]
                ),
                n_body=2,
            )
        else:
            dst.set_annotation(name, AnnotationList2D(len(dst), _EMPTY_PAIRS, _EMPTY_VALUES), n_body=2)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InteractionType(IntEnum):
    """Integer codes stored as ``AnnotationList2D`` values."""

    HBOND = 1
    HYDROPHOBIC = 2
    PISTACKING_P = 3
    PISTACKING_T = 4
    PICATION = 5
    SALTBRIDGE = 6
    HALOGEN = 7
    METAL_COORD = 8


class HBondRole(IntEnum):
    """Per-atom H-bond role.  Supports bitwise OR for dual-role atoms."""

    NONE = 0
    DONOR = 1
    ACCEPTOR = 2
    BOTH = 3


class ChargedRole(IntEnum):
    """Per-atom charge role in salt bridges."""

    NONE = 0
    POSITIVE = 1
    NEGATIVE = 2


class MetalRole(IntEnum):
    """Per-atom role in metal coordination."""

    NONE = 0
    METAL = 1
    COORDINATING = 2


class PiCationRole(IntEnum):
    """Per-atom role in pi-cation interactions.  Supports bitwise OR for dual-role atoms."""

    NONE = 0
    AROMATIC = 1
    CATION = 2
    BOTH = 3


# ---------------------------------------------------------------------------
# Atom-typing lookup tables
# ---------------------------------------------------------------------------

_DONOR_ATOMS_AA: frozenset[tuple[str, str]] = frozenset(
    {
        ("*", "N"),
        ("SER", "OG"),
        ("THR", "OG1"),
        ("TYR", "OH"),
        ("ASN", "ND2"),
        ("GLN", "NE2"),
        ("LYS", "NZ"),
        ("ARG", "NH1"),
        ("ARG", "NH2"),
        ("ARG", "NE"),
        ("HIS", "ND1"),
        ("HIS", "NE2"),
        ("TRP", "NE1"),
        ("CYS", "SG"),
    }
)

_ACCEPTOR_TABLE_AA: frozenset[tuple[str, str]] = frozenset(
    {
        ("*", "O"),
        ("SER", "OG"),
        ("THR", "OG1"),
        ("TYR", "OH"),
        ("ASN", "OD1"),
        ("GLN", "OE1"),
        ("ASP", "OD1"),
        ("ASP", "OD2"),
        ("GLU", "OE1"),
        ("GLU", "OE2"),
        ("HIS", "ND1"),
        ("HIS", "NE2"),
        ("MET", "SD"),
    }
)

_AROMATIC_RINGS: dict[str, list[list[str]]] = {
    "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TRP": [["CG", "CD1", "NE1", "CE2", "CD2"], ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"]],
    "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],
    "DA": [["N9", "C8", "N7", "C5", "C4"], ["C4", "C5", "C6", "N1", "C2", "N3"]],
    "DG": [["N9", "C8", "N7", "C5", "C4"], ["C4", "C5", "C6", "N1", "C2", "N3"]],
    "DC": [["N1", "C2", "N3", "C4", "C5", "C6"]],
    "DT": [["N1", "C2", "N3", "C4", "C5", "C6"]],
    "A": [["N9", "C8", "N7", "C5", "C4"], ["C4", "C5", "C6", "N1", "C2", "N3"]],
    "G": [["N9", "C8", "N7", "C5", "C4"], ["C4", "C5", "C6", "N1", "C2", "N3"]],
    "C": [["N1", "C2", "N3", "C4", "C5", "C6"]],
    "U": [["N1", "C2", "N3", "C4", "C5", "C6"]],
}

_POSITIVE_CHARGED_AA: frozenset[tuple[str, str]] = frozenset(
    {
        ("ARG", "CZ"),
        ("ARG", "NH1"),
        ("ARG", "NH2"),
        ("ARG", "NE"),
        ("LYS", "NZ"),
        ("HIS", "CE1"),
        ("HIS", "ND1"),
        ("HIS", "NE2"),
    }
)

_NEGATIVE_CHARGED_AA: frozenset[tuple[str, str]] = frozenset(
    {
        ("ASP", "CG"),
        ("ASP", "OD1"),
        ("ASP", "OD2"),
        ("GLU", "CD"),
        ("GLU", "OE1"),
        ("GLU", "OE2"),
    }
)


_SP2_ACCEPTORS: frozenset[tuple[str, str]] = frozenset(
    {
        ("*", "O"),
        ("ASN", "OD1"),
        ("GLN", "OE1"),
        ("ASP", "OD1"),
        ("ASP", "OD2"),
        ("GLU", "OE1"),
        ("GLU", "OE2"),
    }
)


def _build_na_tables() -> tuple[frozenset[tuple[str, str]], frozenset[tuple[str, str]], frozenset[tuple[str, str]]]:
    """Build donor/acceptor/negative-charge sets extended with DNA/RNA entries."""
    donor: set[tuple[str, str]] = set(_DONOR_ATOMS_AA)
    acceptor: set[tuple[str, str]] = set(_ACCEPTOR_TABLE_AA)
    negative: set[tuple[str, str]] = set(_NEGATIVE_CHARGED_AA)
    base_acceptors = {
        "A": ("N1", "N3", "N7"),
        "G": ("N3", "N7", "O6"),
        "C": ("N3", "O2"),
        "T": ("O2", "O4"),
        "U": ("O2", "O4"),
    }
    for na_res in (*STANDARD_DNA, *STANDARD_RNA):
        acceptor.update((na_res, atom) for atom in base_acceptors[na_res.removeprefix("D")])
        for oa in ("OP1", "OP2", "O3'", "O5'", "O4'", "O2'"):
            acceptor.add((na_res, oa))
        negative.add((na_res, "OP1"))
        negative.add((na_res, "OP2"))
        if na_res in ("DA", "A"):
            donor.add((na_res, "N6"))
        elif na_res in ("DG", "G"):
            donor.add((na_res, "N1"))
            donor.add((na_res, "N2"))
        elif na_res in ("DC", "C"):
            donor.add((na_res, "N4"))
        elif na_res in ("U", "DT"):
            donor.add((na_res, "N3"))
    return frozenset(donor), frozenset(acceptor), frozenset(negative)


_DONOR_ATOMS: frozenset[tuple[str, str]]
_ACCEPTOR_TABLE: frozenset[tuple[str, str]]
_NEGATIVE_CHARGED: frozenset[tuple[str, str]]
_DONOR_ATOMS, _ACCEPTOR_TABLE, _NEGATIVE_CHARGED = _build_na_tables()

_ALL_STANDARD_RESIDUES = frozenset(STANDARD_POLYMER_RESIDUES)


# ---------------------------------------------------------------------------
# AtomTyping dataclass
# ---------------------------------------------------------------------------


@dataclass
class AtomTyping:
    """Per-atom boolean masks and ring indices for each chemical role.

    All boolean arrays have shape ``(n_atoms,)`` matching the parent
    ``AtomArray``.

    Attributes:
        hba: H-bond acceptor mask.
        hbd_heavy: H-bond donor heavy-atom mask.
        hbd_h: H-bond donor hydrogen mask.
        hydrophobic: Hydrophobic carbon mask (C with only C/H neighbors).
        cation: Positively charged atom mask.
        anion: Negatively charged atom mask.
        elements: Element symbol array.
        res_idx: Per-atom integer residue index (from ``get_residue_starts``).
            ``res_idx[a] == res_idx[b]`` iff atoms a, b are in the same residue.
        chain_idx: Per-atom integer chain index. ``chain_idx[a] == chain_idx[b]``
            iff atoms a, b are in the same chain (uses ``chain_iid`` when available).
        aromatic_rings: List of index arrays, one per aromatic ring.
        ring_res_idx: Integer residue index for each aromatic ring.
        xb_donor: Halogen bond donor mask (Cl/Br/I bonded to C).
        xb_acceptor: Halogen bond acceptor mask (O/N/S with C/P/N/S neighbor).
        metal: Metal ion mask.
        sp2_acceptor: sp2-hybridised acceptor mask (for rosetta torsion check).
        adj: Bond adjacency dict ``{atom_idx: [neighbor_idxs]}``.
        donor_h_map: ``{donor_heavy_idx: [h_idxs]}``.
        acceptor_base: ``{acceptor_idx: base_heavy_neighbor_idx}``.
        xb_acceptor_base: ``{acceptor_idx: Y_neighbor_idx}`` for halogen
            acceptor angle check (Y-A...X).
    """

    hba: np.ndarray
    hbd_heavy: np.ndarray
    hbd_h: np.ndarray
    hydrophobic: np.ndarray
    cation: np.ndarray
    anion: np.ndarray
    elements: np.ndarray
    res_idx: np.ndarray
    chain_idx: np.ndarray
    aromatic_rings: list[np.ndarray]
    ring_res_idx: list[int]
    xb_donor: np.ndarray
    xb_acceptor: np.ndarray
    metal: np.ndarray
    sp2_acceptor: np.ndarray
    adj: dict[int, list[int]]
    donor_h_map: dict[int, list[int]]
    acceptor_base: dict[int, int]
    xb_acceptor_base: dict[int, int]


# ---------------------------------------------------------------------------
# Atom-typing implementation
# ---------------------------------------------------------------------------


def _build_adjacency(bond_arr: np.ndarray) -> dict[int, list[int]]:
    """Build adjacency dict from a bond array for O(1) neighbor lookups.

    Args:
        bond_arr: ``(n_bonds, 3)`` array with columns ``[idx_a, idx_b, type]``.

    Returns:
        Dict mapping each atom index to a list of bonded neighbor indices.
    """
    adj: dict[int, list[int]] = defaultdict(list)
    for row in bond_arr:
        a, b = int(row[0]), int(row[1])
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _detect_ligand_rings(
    atom_array: AtomArray,
    elements: np.ndarray,
    res_starts: np.ndarray,
    aromatic_rings: list[np.ndarray],
    ring_res_idx: list[int],
) -> None:
    """Detect aromatic rings in non-standard residues via RDKit SSSR.

    For each non-standard residue, converts to an RDKit Mol (preserving
    CCD bond orders) and uses RDKit's ring perception + aromaticity
    detection.  Falls back to a planarity check for rings that RDKit
    does not flag as aromatic.  Appends detected rings in-place.

    Args:
        atom_array: Full structure.
        elements: Element symbols.
        res_starts: Residue start indices (from ``get_residue_starts``).
        aromatic_rings: List to append ring index arrays to.
        ring_res_idx: List to append residue indices to.
    """
    for r in range(len(res_starts) - 1):
        s, e = res_starts[r], res_starts[r + 1]
        rn = atom_array.res_name[s]
        if rn in _ALL_STANDARD_RESIDUES:
            continue

        local_el = elements[s:e]
        valid = ~np.isin(local_el, [*METAL_ELEMENTS, "H"])
        keep = np.arange(s, e)[valid]
        if len(keep) < 5:
            continue

        sub_array = atom_array[keep]
        if sub_array.bonds is None:
            raise ValueError(
                f"No bonds on sub-array for residue {rn} (atoms {s}:{e}). "
                "Bonds must be assigned before interaction detection (e.g. via connect_via_residue_names)."
            )

        mol = atom_array_to_rdkit(
            sub_array,
            set_coord=True,
            hydrogen_policy="keep",
            sanitize=True,
            attempt_fixing_corrupted_molecules=False,
        )
        if mol is None:
            continue

        ri = mol.GetRingInfo()
        atom_rings = list(ri.AtomRings())

        for ring_atom_local in atom_rings:
            if not (4 < len(ring_atom_local) <= 6):
                continue

            is_aromatic = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring_atom_local)
            if not is_aromatic:
                ring_coords = sub_array.coord[list(ring_atom_local)]
                is_aromatic = is_planar(ring_coords)

            if is_aromatic:
                global_indices = np.array([keep[j] for j in ring_atom_local])
                aromatic_rings.append(global_indices)
                ring_res_idx.append(r)


def _type_atoms_lookup(atom_array: AtomArray) -> AtomTyping:
    """Assign chemical roles using residue lookup tables and formal charges.

    Standard residues (amino acids, DNA, RNA) use ``_DONOR_ATOMS`` /
    ``_ACCEPTOR_TABLE`` to identify donor and acceptor heavy atoms.
    Donor hydrogen atoms are always found via the bond graph (bonded H
    neighbors).  This works regardless of H naming convention.

    Non-standard residues (ligands) fall back to element-based heuristics
    matching PLIP:

    - **Hydrophobic**: C with only C/H neighbors.
    - **H-bond acceptor**: N, O, S.
    - **H-bond donor**: N or O bonded to at least one H.
    - **Halogen bond acceptor**: O/N/S with a C/P/N/S neighbor.

    Nonzero values of the optional ``charge`` annotation are respected if given.
    When the charge is zero on a standard residue atom that is known to be charged
    (e.g. LYS CZ), the known charge will applied.

    Args:
        atom_array: Structure with optional ``bonds`` attribute.

    Returns:
        Populated ``AtomTyping`` dataclass.
    """
    n = len(atom_array)
    res_names = atom_array.res_name
    atom_names = np.char.strip(atom_array.atom_name.astype(str))
    elements = np.char.upper(atom_array.element)

    hba = np.zeros(n, dtype=bool)
    hbd_heavy = np.zeros(n, dtype=bool)
    hbd_h = np.zeros(n, dtype=bool)
    hydrophobic = np.zeros(n, dtype=bool)
    cation = np.zeros(n, dtype=bool)
    anion = np.zeros(n, dtype=bool)
    sp2_acceptor = np.zeros(n, dtype=bool)
    metal = np.zeros(n, dtype=bool)
    xb_donor = np.zeros(n, dtype=bool)
    xb_acceptor = np.zeros(n, dtype=bool)

    donor_h_map: dict[int, list[int]] = {}
    acceptor_base: dict[int, int] = {}
    xb_acceptor_base: dict[int, int] = {}
    aromatic_rings: list[np.ndarray] = []

    if atom_array.bonds is None:
        raise ValueError("Bonds are required for interaction type detection. ")

    bond_arr = atom_array.bonds.as_array()
    adj = _build_adjacency(bond_arr)

    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    res_lengths = np.diff(res_starts)
    res_idx = np.repeat(np.arange(len(res_lengths), dtype=np.int32), res_lengths)

    _chain_arr = chain_identifier(atom_array)
    _, chain_idx = np.unique(_chain_arr, return_inverse=True)

    # --- Main typing loop (standard residues) ---
    for i in range(n):
        rn, an, el = res_names[i], atom_names[i], elements[i]

        if el in METAL_ELEMENTS:
            metal[i] = True
            continue

        if el in ("CL", "BR", "I"):
            xb_donor[i] = True

        # Halogen bond acceptor: O/N/S with a C/P/N/S neighbor (PLIP definition)
        if el in ("O", "N", "S"):
            for partner in adj.get(i, []):
                if elements[partner] in ("C", "P", "N", "S"):
                    xb_acceptor[i] = True
                    xb_acceptor_base[i] = partner
                    break

        # Hydrophobic: C with only C/H neighbors (PLIP: same rule for all residues)
        if (
            el == "C"
            and (rn, an) not in _POSITIVE_CHARGED_AA
            and (rn, an) not in _NEGATIVE_CHARGED_AA
            and all(elements[p] in ("C", "H") for p in adj.get(i, []))
        ):
            hydrophobic[i] = True

        if (rn, an) in _POSITIVE_CHARGED_AA:
            cation[i] = True
        if (rn, an) in _NEGATIVE_CHARGED_AA:
            anion[i] = True

        is_acc = (rn, an) in _ACCEPTOR_TABLE or ("*", an) in _ACCEPTOR_TABLE
        if is_acc:
            hba[i] = True
            if (rn, an) in _SP2_ACCEPTORS or ("*", an) in _SP2_ACCEPTORS:
                sp2_acceptor[i] = True

        # Donor heavy atoms from lookup table, bonded H found via adjacency
        is_donor = (rn, an) in _DONOR_ATOMS or ("*", an) in _DONOR_ATOMS
        if is_donor:
            hbd_heavy[i] = True
            h_indices = [p for p in adj.get(i, []) if elements[p] == "H"] if adj else []
            for h_idx in h_indices:
                hbd_h[h_idx] = True
            if h_indices:
                donor_h_map[i] = h_indices

    # --- Acceptor base atoms (heavy neighbor for BAH angle) ---
    for i in np.where(hba)[0]:
        for partner in adj.get(int(i), []):
            if elements[partner] != "H":
                acceptor_base[int(i)] = partner
                break

    # --- Aromatic rings from lookup tables (standard residues) ---
    ring_res_idx: list[int] = []
    for r in range(len(res_starts) - 1):
        s, e = res_starts[r], res_starts[r + 1]
        rn = res_names[s]
        if rn not in _AROMATIC_RINGS:
            continue
        local_names = atom_names[s:e]
        for ring_atom_names in _AROMATIC_RINGS[rn]:
            indices = []
            for ra in ring_atom_names:
                matches = np.where(local_names == ra)[0]
                if len(matches) > 0:
                    indices.append(s + int(matches[0]))
            if len(indices) >= 3:
                aromatic_rings.append(np.array(indices))
                ring_res_idx.append(r)

    # --- Aromatic rings for non-standard residues (RDKit SSSR) ---
    _detect_ligand_rings(
        atom_array,
        elements,
        res_starts,
        aromatic_rings,
        ring_res_idx,
    )

    # --- Ligand heuristics (non-standard residues) ---
    for i in range(n):
        rn, el = res_names[i], elements[i]
        if rn in _ALL_STANDARD_RESIDUES or metal[i]:
            continue
        if el in ("N", "O", "S") and not hba[i]:
            hba[i] = True
        if el in ("N", "O") and not hbd_heavy[i] and adj:
            for partner in adj.get(i, []):
                if elements[partner] == "H":
                    hbd_heavy[i] = True
                    hbd_h[partner] = True
                    donor_h_map.setdefault(i, []).append(partner)
        if el == "C" and not hydrophobic[i] and adj and all(elements[p] in ("C", "H") for p in adj.get(i, [])):
            hydrophobic[i] = True

    if "charge" in atom_array.get_annotation_categories():
        charge = atom_array.charge
        charged = (charge > 0) | (charge < 0)
        cation[charged] = charge[charged] > 0
        anion[charged] = charge[charged] < 0
        hydrophobic[charged] = False

    # --- Exclude atoms with NaN coordinates (cKDTree crashes on NaN) ---
    has_nan = np.isnan(atom_array.coord).any(axis=-1)
    if has_nan.any():
        valid = ~has_nan
        hba &= valid
        hbd_heavy &= valid
        hbd_h &= valid
        hydrophobic &= valid
        cation &= valid
        anion &= valid
        xb_donor &= valid
        xb_acceptor &= valid
        metal &= valid
        sp2_acceptor &= valid
        donor_h_map = {k: [h for h in v if valid[h]] for k, v in donor_h_map.items() if valid[k]}
        donor_h_map = {k: v for k, v in donor_h_map.items() if v}
        acceptor_base = {k: v for k, v in acceptor_base.items() if valid[k] and valid[v]}
        xb_acceptor_base = {k: v for k, v in xb_acceptor_base.items() if valid[k] and valid[v]}
        valid_ring_mask = [valid[r].all() for r in aromatic_rings]
        aromatic_rings = [r for r, v in zip(aromatic_rings, valid_ring_mask, strict=False) if v]
        ring_res_idx = [ri for ri, v in zip(ring_res_idx, valid_ring_mask, strict=False) if v]

    return AtomTyping(
        hba=hba,
        hbd_heavy=hbd_heavy,
        hbd_h=hbd_h,
        hydrophobic=hydrophobic,
        cation=cation,
        anion=anion,
        elements=elements,
        res_idx=res_idx,
        chain_idx=chain_idx,
        aromatic_rings=aromatic_rings,
        ring_res_idx=ring_res_idx,
        xb_donor=xb_donor,
        xb_acceptor=xb_acceptor,
        metal=metal,
        sp2_acceptor=sp2_acceptor,
        adj=adj,
        donor_h_map=donor_h_map,
        acceptor_base=acceptor_base,
        xb_acceptor_base=xb_acceptor_base,
    )


# ---------------------------------------------------------------------------
# H-bond geometry kernels
# ---------------------------------------------------------------------------


def _find_hbonds_rosetta(
    coords: np.ndarray,
    typing: AtomTyping,
    inter_chain_only: bool = True,
    dist_ha_max: float = 3.0,
    ahd_angle_min: float = 120.0,
    bah_angle_min: float = 90.0,
    sp2_torsion_max: float = 60.0,
) -> list[tuple[int, int]]:
    """Rosetta-style H-bond detection (O'Meara et al. 2015).

    Checks H...A distance, A-H-D angle, B-A-H angle, and for sp2 acceptors
    the B-A-H-chi torsion ensuring approach in the lone-pair plane.  Requires
    explicit H atoms.

    Args:
        coords: Coordinate array ``(n_atoms, 3)``.
        typing: Pre-computed atom typing (includes ``chain_idx``).
        inter_chain_only: Only detect inter-chain interactions.
        dist_ha_max: Max H...A distance (A).
        ahd_angle_min: Min A-H-D angle (deg).
        bah_angle_min: Min B-A-H angle (deg).
        sp2_torsion_max: Max chi torsion for sp2 acceptors (deg).

    Returns:
        List of ``(donor_heavy_idx, acceptor_idx)`` pairs.
    """
    donor_heavy_idx = np.where(typing.hbd_heavy)[0]
    acceptor_idx = np.where(typing.hba)[0]
    if len(donor_heavy_idx) == 0 or len(acceptor_idx) == 0:
        return []

    results: list[tuple[int, int]] = []
    acc_tree = cKDTree(coords[acceptor_idx])

    for d_heavy in donor_heavy_idx:
        for h_idx in typing.donor_h_map.get(d_heavy, []):
            h_pos = coords[h_idx]
            for j in acc_tree.query_ball_point(h_pos, dist_ha_max):
                a_idx = acceptor_idx[j]
                if a_idx in (d_heavy, h_idx):
                    continue
                if inter_chain_only and typing.chain_idx[d_heavy] == typing.chain_idx[a_idx]:
                    continue

                a_pos, d_pos = coords[a_idx], coords[d_heavy]
                if angle_between_vectors(a_pos - h_pos, d_pos - h_pos) < ahd_angle_min:
                    continue

                base_idx = typing.acceptor_base.get(a_idx)
                if base_idx is not None:
                    b_pos = coords[base_idx]
                    if angle_between_vectors(b_pos - a_pos, h_pos - a_pos) < bah_angle_min:
                        continue
                    if typing.sp2_acceptor[a_idx]:
                        torsion = dihedral_angle(b_pos, a_pos, h_pos, d_pos)
                        if min(torsion, abs(180 - torsion)) > sp2_torsion_max:
                            continue

                results.append((d_heavy, a_idx))

    return results


def _find_hbonds_plip(
    coords: np.ndarray,
    typing: AtomTyping,
    inter_chain_only: bool = True,
    dist_da_max: float = 4.1,
    dist_min: float = 0.5,
    dha_angle_min: float = 100.0,
) -> list[tuple[int, int]]:
    """PLIP-style H-bond detection (Hubbard & Haider 2001).

    Checks D...A distance and D-H-A angle.  Thresholds match PLIP v3.0.0
    defaults: ``HBOND_DIST_MAX = 4.1``, ``HBOND_DON_ANGLE_MIN = 100``,
    ``MIN_DIST = 0.5``.  Requires explicit H atoms.

    Args:
        coords: Coordinate array ``(n_atoms, 3)``.
        typing: Pre-computed atom typing (includes ``chain_idx``).
        inter_chain_only: Only detect inter-chain interactions.
        dist_da_max: Max D-A distance (A).  PLIP default: 4.1.
        dist_min: Min D-A distance (A).  PLIP default: 0.5.
        dha_angle_min: Min D-H-A angle (deg).  PLIP default: 100.

    Returns:
        List of ``(donor_heavy_idx, acceptor_idx)`` pairs.
    """
    donor_heavy_idx = np.where(typing.hbd_heavy)[0]
    acceptor_idx = np.where(typing.hba)[0]
    if len(donor_heavy_idx) == 0 or len(acceptor_idx) == 0:
        return []

    results: list[tuple[int, int]] = []
    acc_tree = cKDTree(coords[acceptor_idx])

    for d_heavy in donor_heavy_idx:
        h_list = typing.donor_h_map.get(d_heavy, [])
        for j in acc_tree.query_ball_point(coords[d_heavy], dist_da_max):
            a_idx = acceptor_idx[j]
            if a_idx == d_heavy:
                continue
            da_dist = float(np.linalg.norm(coords[d_heavy] - coords[a_idx]))
            if da_dist < dist_min:
                continue
            if inter_chain_only and typing.chain_idx[d_heavy] == typing.chain_idx[a_idx]:
                continue
            if h_list and any(
                angle_between_vectors(coords[d_heavy] - coords[h], coords[a_idx] - coords[h]) >= dha_angle_min
                for h in h_list
            ):
                results.append((d_heavy, a_idx))

    return results


# ---------------------------------------------------------------------------
# H-bond refinement (PLIP post-processing)
# ---------------------------------------------------------------------------


def _refine_hbonds(
    hbonds: list[tuple[int, int]],
    saltbridges: list[tuple[int, int]],
    coords: np.ndarray,
    typing: AtomTyping,
) -> list[tuple[int, int]]:
    """Apply PLIP-style H-bond refinement.

    1. Remove H-bonds between atoms already connected by a salt bridge.
    2. One-H-one-hbond rule: per donor H atom, keep only the H-bond with the
       DHA angle closest to 180 deg.  Donor heavy atoms that carry multiple H
       (e.g. ASN ND2 with HD21 + HD22) may therefore contribute more than one
       H-bond — one per H atom — matching PLIP's per-hydrogen semantics.

    Args:
        hbonds: Raw H-bond pairs ``(donor_heavy, acceptor)``.
        saltbridges: Salt bridge pairs ``(cation, anion)``.
        coords: Coordinate array.
        typing: Atom typing (for donor H positions).

    Returns:
        Refined list of ``(donor_heavy, acceptor)`` pairs.
    """
    if not hbonds:
        return hbonds

    # Step 1: remove H-bonds overlapping with salt bridges
    sb_atoms = set()
    for cat, ani in saltbridges:
        sb_atoms.add(cat)
        sb_atoms.add(ani)

    filtered = [(d, a) for d, a in hbonds if not (d in sb_atoms and a in sb_atoms)]

    # Step 2: one H atom -> one H-bond (keep best DHA angle per H).
    # Key is (donor_heavy, h_idx) so bidentate NH2 donors (ASN ND2, GLN NE2,
    # nucleic acid NH2 groups) can each H donate independently.
    best_per_h: dict[tuple[int, int], tuple[float, int, int]] = {}
    for d, a in filtered:
        h_list = typing.donor_h_map.get(d, [])
        if h_list:
            for h in h_list:
                angle = angle_between_vectors(coords[d] - coords[h], coords[a] - coords[h])
                key = (d, h)
                if key not in best_per_h or angle > best_per_h[key][0]:
                    best_per_h[key] = (angle, d, a)
        else:
            # No explicit H recorded; fall back to keying on donor heavy atom
            key = (d, -1)
            if key not in best_per_h or best_per_h[key][0] < 180.0:
                best_per_h[key] = (180.0, d, a)

    return [(d, a) for _, d, a in best_per_h.values()]


# ---------------------------------------------------------------------------
# Other interaction geometry kernels
# ---------------------------------------------------------------------------


def _find_hydrophobic(
    coords: np.ndarray,
    typing: AtomTyping,
    inter_chain_only: bool = True,
    cutoff: float = 4.0,
) -> list[tuple[int, int]]:
    """Find hydrophobic contacts (PLIP: ``HYDROPH_DIST_MAX = 4.0``).

    Hydrophobic atoms are carbons with only carbon or hydrogen neighbors.
    Same-chain same-residue pairs are always excluded.

    Args:
        coords: Coordinate array.
        typing: Pre-computed atom typing.
        inter_chain_only: Only detect inter-chain interactions.
        cutoff: Distance cutoff (A).  PLIP default: 4.0.

    Returns:
        List of ``(atom_i, atom_j)`` pairs.
    """
    hp_idx = np.where(typing.hydrophobic)[0]
    if len(hp_idx) < 2:
        return []
    tree = cKDTree(coords[hp_idx])
    results: list[tuple[int, int]] = []
    for i, j in tree.query_pairs(cutoff):
        ai, aj = hp_idx[i], hp_idx[j]
        if inter_chain_only:
            if typing.chain_idx[ai] == typing.chain_idx[aj]:
                continue
        elif typing.res_idx[ai] == typing.res_idx[aj]:
            continue
        results.append((ai, aj))
    return results


def _refine_hydrophobic(
    hydrophobic: list[tuple[int, int]],
    pistacking_atoms: set[int],
    typing: AtomTyping,
    coords: np.ndarray,
) -> list[tuple[int, int]]:
    """Apply PLIP-style hydrophobic contact reduction.

    1. Remove pairs where both atoms participate in pi-stacking.
    2. Per-residue dedup: keep only the closest contact per residue pair.

    Args:
        hydrophobic: Raw hydrophobic pairs.
        pistacking_atoms: Set of atom indices involved in pi-stacking.
        typing: Pre-computed atom typing (for ``res_idx``).
        coords: Coordinate array.

    Returns:
        Reduced list of hydrophobic pairs.
    """
    if not hydrophobic:
        return hydrophobic

    # Step 1: exclude pairs overlapping with pi-stacking rings
    filtered = [(a, b) for a, b in hydrophobic if not (a in pistacking_atoms and b in pistacking_atoms)]

    # Step 2: per-residue-pair, keep only closest
    best: dict[tuple[int, int], tuple[float, int, int]] = {}
    for a, b in filtered:
        ra, rb = int(typing.res_idx[a]), int(typing.res_idx[b])
        key = (min(ra, rb), max(ra, rb))
        dist = float(np.linalg.norm(coords[a] - coords[b]))
        if key not in best or dist < best[key][0]:
            best[key] = (dist, a, b)

    return [(a, b) for _, a, b in best.values()]


def _find_pistacking(
    coords: np.ndarray,
    typing: AtomTyping,
    inter_chain_only: bool = True,
    dist_max: float = 5.5,
    ang_dev: float = 30.0,
    offset_max: float = 2.0,
) -> list[tuple[int, int, int]]:
    """Find pi-stacking interactions (PLIP-aligned).

    PLIP thresholds: ``PISTACK_DIST_MAX = 5.5``, ``PISTACK_ANG_DEV = 30``,
    ``PISTACK_OFFSET_MAX = 2.0``.  P-stacking (parallel): inter-normal angle
    0-30 deg.  T-stacking (perpendicular): 60-120 deg.  The ring-center
    offset check projects each center onto the opposite ring plane.

    Args:
        coords: Coordinate array.
        typing: Pre-computed atom typing.
        inter_chain_only: Only detect inter-chain interactions.
        dist_max: Max center-center distance.  PLIP default: 5.5.
        ang_dev: Max angular deviation from ideal (0 or 90 deg).
        offset_max: Max ring-center offset.  PLIP default: 2.0.

    Returns:
        List of ``(atom_a, atom_b, InteractionType)`` tuples, one per pair of
        atoms across the two interacting rings (full cross-product).
    """
    rings = typing.aromatic_rings
    if len(rings) < 2:
        return []

    centroids = np.array([coords[r].mean(axis=0) for r in rings])
    normals = [plane_normal(coords[r[:3]]) for r in rings]
    tree = cKDTree(centroids)
    results: list[tuple[int, int, int]] = []

    for i, j in tree.query_pairs(dist_max):
        ri_a, ri_b = typing.ring_res_idx[i], typing.ring_res_idx[j]
        if ri_a == ri_b:
            continue
        if inter_chain_only and typing.chain_idx[rings[i][0]] == typing.chain_idx[rings[j][0]]:
            continue

        normal_angle = angle_between_vectors(normals[i], normals[j])
        normal_angle = min(normal_angle, 180.0 - normal_angle)

        # Offset check (PLIP: project each center onto the other ring plane)
        offset = min(
            in_plane_offset(normals[j], centroids[j], centroids[i]),
            in_plane_offset(normals[i], centroids[i], centroids[j]),
        )
        if offset > offset_max:
            continue

        if normal_angle <= ang_dev:
            itype: InteractionType | None = InteractionType.PISTACKING_P
        elif (90.0 - ang_dev) <= normal_angle <= (90.0 + ang_dev):
            itype = InteractionType.PISTACKING_T
        else:
            itype = None

        if itype is not None:
            for ai in rings[i]:
                for aj in rings[j]:
                    results.append((int(ai), int(aj), itype))

    return results


def _find_pication(
    coords: np.ndarray,
    typing: AtomTyping,
    inter_chain_only: bool = True,
    dist_cutoff: float = 6.0,
    offset_max: float = 2.0,
) -> list[tuple[int, int]]:
    """Find pi-cation interactions (PLIP-aligned).

    PLIP thresholds: ``PICATION_DIST_MAX = 6.0``,
    ``PISTACK_OFFSET_MAX = 2.0``.

    Args:
        coords: Coordinate array.
        typing: Pre-computed atom typing.
        inter_chain_only: Only detect inter-chain interactions.
        dist_cutoff: Max cation-ring-center distance.  PLIP default: 6.0.
        offset_max: Max offset of cation projection onto ring plane.

    Returns:
        List of ``(ring_atom, cation_atom)`` pairs, one per ring atom.
    """
    cat_idx = np.where(typing.cation)[0]
    rings = typing.aromatic_rings
    if len(cat_idx) == 0 or len(rings) == 0:
        return []

    results: list[tuple[int, int]] = []
    centroids = [coords[r].mean(axis=0) for r in rings]
    normals = [plane_normal(coords[r[:3]]) for r in rings]

    for ci in cat_idx:
        for ri, ring in enumerate(rings):
            if ci in ring:
                continue
            if inter_chain_only and typing.chain_idx[ci] == typing.chain_idx[ring[0]]:
                continue
            dist = float(np.linalg.norm(coords[ci] - centroids[ri]))
            if dist > dist_cutoff:
                continue
            offset = in_plane_offset(normals[ri], centroids[ri], coords[ci])
            if offset > offset_max:
                continue
            for ra in ring:
                results.append((int(ra), int(ci)))

    return results


def _find_saltbridges(
    coords: np.ndarray,
    typing: AtomTyping,
    inter_chain_only: bool = True,
    cutoff: float = 5.5,
) -> list[tuple[int, int]]:
    """Find salt bridges (PLIP: ``SALTBRIDGE_DIST_MAX = 5.5``).

    Args:
        coords: Coordinate array.
        typing: Pre-computed atom typing.
        inter_chain_only: Only detect inter-chain interactions.
        cutoff: Max distance between opposite charges.  PLIP default: 5.5.

    Returns:
        List of ``(cation_atom, anion_atom)`` pairs.
    """
    cat_idx = np.where(typing.cation)[0]
    ani_idx = np.where(typing.anion)[0]
    if len(cat_idx) == 0 or len(ani_idx) == 0:
        return []

    results: list[tuple[int, int]] = []
    tree = cKDTree(coords[ani_idx])
    for ci in cat_idx:
        for j in tree.query_ball_point(coords[ci], cutoff):
            ai = ani_idx[j]
            if inter_chain_only:
                if typing.chain_idx[ci] == typing.chain_idx[ai]:
                    continue
            elif typing.res_idx[ci] == typing.res_idx[ai]:
                continue
            results.append((ci, ai))
    return results


def _find_halogen(
    coords: np.ndarray,
    typing: AtomTyping,
    inter_chain_only: bool = True,
    dist_max: float = 4.0,
    don_angle_center: float = 165.0,
    don_angle_dev: float = 30.0,
    acc_angle_center: float = 120.0,
    acc_angle_dev: float = 30.0,
) -> list[tuple[int, int]]:
    """Find halogen bonds (PLIP-aligned).

    PLIP checks both donor angle (C-X...A at 165 +/- 30 deg) and acceptor
    angle (Y-A...X at 120 +/- 30 deg).

    Args:
        coords: Coordinate array.
        typing: Pre-computed atom typing.
        inter_chain_only: Only detect inter-chain interactions.
        dist_max: Max X...A distance.  PLIP default: 4.0.
        don_angle_center: Ideal C-X...A angle.  PLIP: 165.
        don_angle_dev: Allowed deviation from ideal donor angle.  PLIP: 30.
        acc_angle_center: Ideal Y-A...X angle.  PLIP: 120.
        acc_angle_dev: Allowed deviation from ideal acceptor angle.  PLIP: 30.

    Returns:
        List of ``(halogen_idx, acceptor_idx)`` pairs.
    """
    xb_d = np.where(typing.xb_donor)[0]
    xb_a = np.where(typing.xb_acceptor)[0]
    if len(xb_d) == 0 or len(xb_a) == 0:
        return []

    x_to_c: dict[int, int] = {}
    for xi in xb_d:
        for partner in typing.adj.get(int(xi), []):
            if typing.elements[partner] == "C":
                x_to_c[int(xi)] = partner
                break

    results: list[tuple[int, int]] = []
    tree = cKDTree(coords[xb_a])
    for xi in xb_d:
        c_idx = x_to_c.get(int(xi))
        if c_idx is None:
            continue  # free halide ion
        for j in tree.query_ball_point(coords[xi], dist_max):
            ai = xb_a[j]
            if ai == xi:
                continue
            if inter_chain_only and typing.chain_idx[xi] == typing.chain_idx[ai]:
                continue
            # Donor angle: C-X...A
            don_angle = angle_between_vectors(coords[c_idx] - coords[xi], coords[ai] - coords[xi])
            if abs(don_angle - don_angle_center) > don_angle_dev:
                continue
            # Acceptor angle: Y-A...X (Y is the heavy neighbor of A)
            y_idx = typing.xb_acceptor_base.get(int(ai))
            if y_idx is not None:
                acc_angle = angle_between_vectors(coords[y_idx] - coords[ai], coords[xi] - coords[ai])
                if abs(acc_angle - acc_angle_center) > acc_angle_dev:
                    continue
            results.append((int(xi), int(ai)))

    return results


def _find_metal_coordination(
    coords: np.ndarray,
    typing: AtomTyping,
    inter_chain_only: bool = True,
    cutoff: float = 3.0,
) -> list[tuple[int, int]]:
    """Find metal coordination (PLIP: ``METAL_DIST_MAX = 3.0``).

    Coordinating atoms are N, O, S per PLIP 3.0 and Harding (2001).
    Phosphorus is excluded: phosphate groups coordinate metals through
    their oxygen atoms (OP1/OP2), not through P directly.

    Args:
        coords: Coordinate array.
        typing: Pre-computed atom typing.
        inter_chain_only: Only detect inter-chain interactions.
        cutoff: Max metal-target distance.  PLIP default: 3.0.

    Returns:
        List of ``(metal_idx, coordinating_idx)`` pairs.
    """
    metal_idx = np.where(typing.metal)[0]
    if len(metal_idx) == 0:
        return []

    has_valid_coords = ~np.isnan(coords).any(axis=-1)
    coord_mask = np.isin(typing.elements, ["N", "O", "S"]) & ~typing.metal & has_valid_coords
    coord_idx = np.where(coord_mask)[0]
    if len(coord_idx) == 0:
        return []

    results: list[tuple[int, int]] = []
    tree = cKDTree(coords[coord_idx])
    for mi in metal_idx:
        for j in tree.query_ball_point(coords[mi], cutoff):
            ci = coord_idx[j]
            if inter_chain_only and typing.chain_idx[mi] == typing.chain_idx[ci]:
                continue
            results.append((int(mi), int(ci)))

    return results


# ---------------------------------------------------------------------------
# 1D role derivation from 2D pairs
# ---------------------------------------------------------------------------


def _derive_1d_roles(
    atom_array: AtomArray,
    results: dict[str, list[tuple[int, int]]],
) -> dict[str, np.ndarray]:
    """Derive per-atom 1D role annotations from detected 2D pairs.

    Args:
        atom_array: Structure (used only for length).
        results: Interaction results dict from ``annotate_interactions``.

    Returns:
        Dict mapping annotation names to arrays of length ``len(atom_array)``.
    """
    n = len(atom_array)

    hbond_role = np.full(n, HBondRole.NONE, dtype=np.int32)
    for d, a in results.get("hbond", []):
        hbond_role[d] |= HBondRole.DONOR
        hbond_role[a] |= HBondRole.ACCEPTOR

    is_hydrophobic = np.zeros(n, dtype=bool)
    for ai, aj in results.get("hydrophobic", []):
        is_hydrophobic[ai] = True
        is_hydrophobic[aj] = True

    is_aromatic = np.zeros(n, dtype=bool)
    for ai, aj in results.get("pistacking", []):
        is_aromatic[ai] = True
        is_aromatic[aj] = True
    for ai, _aj in results.get("pication", []):
        is_aromatic[ai] = True

    is_pistacking = np.zeros(n, dtype=bool)
    for ai, aj in results.get("pistacking", []):
        is_pistacking[ai] = True
        is_pistacking[aj] = True

    pication_role = np.full(n, PiCationRole.NONE, dtype=np.int32)
    for ai, ci in results.get("pication", []):
        pication_role[ai] |= PiCationRole.AROMATIC
        pication_role[ci] |= PiCationRole.CATION

    charged_role = np.full(n, ChargedRole.NONE, dtype=np.int32)
    for ci, ai in results.get("saltbridge", []):
        charged_role[ci] = ChargedRole.POSITIVE
        charged_role[ai] = ChargedRole.NEGATIVE

    metal_role = np.full(n, MetalRole.NONE, dtype=np.int32)
    for mi, ci in results.get("metal", []):
        metal_role[mi] = MetalRole.METAL
        metal_role[ci] = MetalRole.COORDINATING

    is_halogen = np.zeros(n, dtype=bool)
    for xi, ai in results.get("halogen", []):
        is_halogen[xi] = True
        is_halogen[ai] = True

    return {
        "interaction_hbond_role": hbond_role,
        "interaction_hydrophobic": is_hydrophobic,
        "interaction_aromatic": is_aromatic,
        "interaction_pistacking": is_pistacking,
        "interaction_pication_role": pication_role,
        "interaction_charged_role": charged_role,
        "interaction_metal_role": metal_role,
        "interaction_halogen": is_halogen,
    }


# ---------------------------------------------------------------------------
# Top-level annotation function
# ---------------------------------------------------------------------------


def annotate_interactions(
    atom_array: AtomArray,
    interaction_types: tuple[str, ...] = (
        "hbond",
        "hydrophobic",
        "pistacking",
        "pication",
        "saltbridge",
        "halogen",
        "metal",
    ),
    hbond_model: Literal["rosetta", "plip"] = "rosetta",
    inter_chain_only: bool = True,
) -> dict[str, Any]:
    """Detect non-covalent interactions in a structure.

    For accurate H-bond detection the input should contain explicit
    hydrogen atoms.  Add them with :func:`~atomworks.experimental.protonation.assign_hydrogens`
    and :func:`~atomworks.experimental.protonation.add_hydrogens`, or call through :class:`AnnotateInteractions`
    which handles the H lifecycle automatically.

    All default thresholds match PLIP v3.0.0.

    Salt bridges use amino-acid charge-group lookup supplemented by the optional
    formal ``charge`` annotation. Nonzero charges take precedence over lookup.

    Args:
        atom_array: Structure with bonds (via ``connect_via_residue_names``).
            Should contain explicit H for accurate H-bond detection.
            Chain disambiguation uses ``chain_iid`` when present, falling
            back to ``chain_id``.
        interaction_types: Which interaction types to detect.
        hbond_model: ``"rosetta"`` (O'Meara 2015) or ``"plip"`` (PLIP defaults).
        inter_chain_only: If ``True``, only detect inter-chain interactions.

    Returns:
        Dict mapping interaction name to list of ``(idx_a, idx_b)`` pairs.
        Indices refer to *atom_array*.  A special ``"_pistacking_types"``
        key holds the ``InteractionType`` code for each pi-stack pair.
    """
    typing = _type_atoms_lookup(atom_array)
    coords = atom_array.coord
    results: dict[str, list[tuple[int, int]]] = {}
    results["_aromatic_rings"] = typing.aromatic_rings

    # Salt bridges first (needed for H-bond refinement)
    if "saltbridge" in interaction_types or "hbond" in interaction_types:
        saltbridges = _find_saltbridges(coords, typing, inter_chain_only)
        if "saltbridge" in interaction_types:
            results["saltbridge"] = saltbridges
    else:
        saltbridges = []

    if "hbond" in interaction_types:
        fn = _find_hbonds_rosetta if hbond_model == "rosetta" else _find_hbonds_plip
        raw_hbonds = fn(coords, typing, inter_chain_only)
        results["hbond"] = _refine_hbonds(raw_hbonds, saltbridges, coords, typing)

    if "pistacking" in interaction_types:
        raw = _find_pistacking(coords, typing, inter_chain_only)
        results["pistacking"] = [(a, b) for a, b, _ in raw]
        results["_pistacking_types"] = [t for _, _, t in raw]

    if "hydrophobic" in interaction_types:
        raw_hp = _find_hydrophobic(coords, typing, inter_chain_only)
        pistack_atoms: set[int] = set()
        for a, b in results.get("pistacking", []):
            pistack_atoms.add(a)
            pistack_atoms.add(b)
        results["hydrophobic"] = _refine_hydrophobic(raw_hp, pistack_atoms, typing, coords)

    if "pication" in interaction_types:
        results["pication"] = _find_pication(coords, typing, inter_chain_only)

    if "halogen" in interaction_types:
        results["halogen"] = _find_halogen(coords, typing, inter_chain_only)

    if "metal" in interaction_types:
        results["metal"] = _find_metal_coordination(coords, typing, inter_chain_only)

    return results


# ---------------------------------------------------------------------------
# Transform class
# ---------------------------------------------------------------------------


class AnnotateInteractions(Transform):
    """Annotate non-covalent interactions on ``AtomArrayPlus``.

    Adds hydrogens via RDKit with pH-aware protonation (Dimorphite-DL) if
    missing, detects interactions on the H-augmented array, writes 2D + 1D
    annotations, then strips temporary H.  ``AtomArrayPlus`` boolean-mask
    slicing auto-remaps 2D pair indices.

    2D annotations (``AnnotationList2D``):
        ``interaction_hbond``, ``interaction_hydrophobic``,
        ``interaction_pistacking``, ``interaction_pication``,
        ``interaction_saltbridge``, ``interaction_halogen``,
        ``interaction_metal``.

    1D annotations (per-atom arrays):
        ``interaction_hbond_role``, ``interaction_hydrophobic``,
        ``interaction_aromatic``, ``interaction_pistacking``,
        ``interaction_pication_role``, ``interaction_charged_role``,
        ``interaction_metal_role``, ``interaction_halogen``.

    Args:
        interaction_types: Which interaction types to detect.
        hbond_model: ``"rosetta"`` or ``"plip"``.
        inter_chain_only: Only detect inter-chain interactions.
    """

    def __init__(
        self,
        interaction_types: tuple[str, ...] = (
            "hbond",
            "hydrophobic",
            "pistacking",
            "pication",
            "saltbridge",
            "halogen",
            "metal",
        ),
        hbond_model: Literal["rosetta", "plip"] = "rosetta",
        inter_chain_only: bool = True,
        silence_rdkit_warnings: bool = False,
    ) -> None:
        self.interaction_types = interaction_types
        self.hbond_model = hbond_model
        self.inter_chain_only = inter_chain_only
        self.silence_rdkit_warnings = silence_rdkit_warnings

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_has_bonds(data)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        aap = as_atom_array_plus(data["atom_array"])

        ctx = suppress_rdkit_warnings() if self.silence_rdkit_warnings else contextlib.nullcontext()
        with ctx:
            # Work on a copy: add H, detect interactions, then transfer annotations
            # back to the original array. This ensures no original annotations are
            # lost during hydrogen addition / concatenation.
            aap_h = as_atom_array_plus(add_hydrogens(aap))
            added_h = len(aap_h) > len(aap)

            results = annotate_interactions(
                aap_h,
                interaction_types=self.interaction_types,
                hbond_model=self.hbond_model,
                inter_chain_only=self.inter_chain_only,
            )

            pistack_types = results.pop("_pistacking_types", [])
            # Drop the aromatic-ring index lists so the 2D-annotation loop below only
            # iterates real interaction types; the values themselves are not used here.
            results.pop("_aromatic_rings", None)

            # Write 2D + 1D interaction annotations onto the H-added copy
            for int_type, pairs_list in results.items():
                annot_name = _ANNOTATION_NAMES_2D[int_type]
                if pairs_list:
                    pairs = np.array(pairs_list, dtype=np.int32)
                    if int_type == "pistacking" and pistack_types:
                        values = np.array(pistack_types, dtype=np.int32)
                    else:
                        values = np.ones(len(pairs_list), dtype=np.int32)
                    aap_h.set_annotation(annot_name, AnnotationList2D(len(aap_h), pairs, values), n_body=2)
                else:
                    aap_h.set_annotation(
                        annot_name, AnnotationList2D(len(aap_h), _EMPTY_PAIRS, _EMPTY_VALUES), n_body=2
                    )

            for annot_name, annot_values in _derive_1d_roles(aap_h, results).items():
                aap_h.set_annotation(annot_name, annot_values)

            # Strip H -- AtomArrayPlus boolean slicing auto-remaps 2D indices
            if added_h:
                aap_h = aap_h[aap_h.element != "H"]

            # Transfer interaction annotations from the copy back to the original.
            # Match by (chain_id, res_id, atom_name) to handle cases where
            # protonation changed heavy atom count (e.g. input had partial H).
            _transfer_interaction_annotations(src=aap_h, dst=aap)

        data["atom_array"] = aap
        return data
