"""Hydrogen assignment: every heavy atom's charge and how many hydrogens it carries, from the chemical
component dictionary, the bonds, the metals and Dimorphite-DL."""

from __future__ import annotations

import functools
import itertools
import logging
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from rdkit import Chem, rdBase

from atomworks.constants import BIOTITE_BOND_TYPE_TO_BOND_ORDER, HYDROGEN_LIKE_SYMBOLS, METAL_ELEMENTS
from atomworks.enums import ChainType
from atomworks.experimental.protonation._titration import (
    bonds_within_components,
    imidazole_rings,
    partition_atomized_subgraphs,
    titrate_component,
)
from atomworks.io.transforms.atomize import compute_standard_atomize_mask
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array import annotate_hydrogens, chain_identifier
from atomworks.io.utils.ccd import atom_array_from_ccd_code, get_custom_ccd_entries, get_polymerization_atoms

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Component titrations shared across calls, least recently used first; the arrays are read-only.
_titrations: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
_MAX_TITRATIONS = 4096
_UNTITRATED = object()
_TITRATIONS_LOCK = threading.Lock()


def _cache_ccd_lookup(func: Callable[..., T]) -> Callable[..., T]:
    """Cache *func* by its arguments, the first a CCD code, except for codes the custom CCD registry defines,
    which may change. The bound limits each worker's working set, not supported CCD codes;
    evicted entries are recomputed."""
    cached = functools.lru_cache(maxsize=8192)(func)

    @functools.wraps(func)
    def wrapper(res_name: str, *args: Any) -> T:
        return func(res_name, *args) if res_name.upper() in get_custom_ccd_entries() else cached(res_name, *args)

    return wrapper


@_cache_ccd_lookup
def _ccd_template(res_name: str) -> AtomArray | None:
    """The dictionary's copy of a component with its bonds (a one-atom component with none); None without one.

    A ligand under a placeholder code is the ordinary case rather than an error.
    """
    try:
        template = atom_array_from_ccd_code(res_name)
    except (AttributeError, ValueError, KeyError):
        return None
    if template.bonds is None and template.array_length() == 1:
        template = template.copy()
        template.bonds = struc.BondList(1)
    return template if template.bonds is not None else None


# What the dictionary gives each atom of a component: formal charge; valence (bond orders, metal bonds left out;
# -1 on a metal); heavy neighbours other than metals; those that do not leave on polymerization; hydrogens; and
# hydrogens that stay on polymerization (-1 where the component flags no leaving atoms or the atom has none).
_COLUMNS = ("charge", "valence", "heavy_degree", "kept_degree", "hydrogens", "linked_hydrogens")


@_cache_ccd_lookup
def _ccd_atom_properties(res_name: str) -> dict[str, tuple]:
    """Per atom name of a component, its :data:`_COLUMNS` values and its hydrogens' names; empty without one."""
    template = _ccd_template(res_name)
    if template is None:
        return {}
    n = template.array_length()
    element = np.char.upper(template.element.astype(str))
    is_h = np.isin(element, HYDROGEN_LIKE_SYMBOLS)
    metal = np.isin(element, sorted(METAL_ELEMENTS))
    flagged = "is_leaving_atom" in template.get_annotation_categories()
    leaving = template.is_leaving_atom if flagged else np.zeros(n, dtype=bool)
    bonds = template.bonds.as_array()
    covalent = bonds[~metal[bonds[:, :2]].any(axis=1)]
    weights = [BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(struc.BondType(t), 1) for t in covalent[:, 2]]
    valence = np.where(metal, -1, np.bincount(covalent[:, :2].ravel(), np.repeat(weights, 2), minlength=n))
    counted = ~is_h & ~metal

    def degree(mask: np.ndarray) -> np.ndarray:
        return np.bincount(bonds[mask[bonds[:, :2]].all(axis=1), :2].ravel(), minlength=n)

    heavy_degree, kept_degree = degree(counted), degree(counted & ~leaving)
    rows = {}
    for i, name in enumerate(template.atom_name.tolist()):
        partners = template.bonds.get_bonds(i)[0] if not is_h[i] else np.empty(0, dtype=int)
        hydrogens = template.atom_name[partners[is_h[partners]]].tolist()
        linked = sum(not leaving[j] for j in partners[is_h[partners]]) if flagged and hydrogens else -1
        rows[name] = (
            int(template.charge[i]),
            int(valence[i]),
            int(heavy_degree[i]),
            int(kept_degree[i]),
            len(hydrogens),
            int(linked),
            tuple(hydrogens),
        )
    return rows


def template_hydrogen_names(res_name: str) -> dict[str, tuple[str, ...]]:
    """Each atom of a component that carries hydrogens, mapped to their names; empty without a template."""
    return {name: row[-1] for name, row in _ccd_atom_properties(res_name).items() if row[-1]}


def _lookup_ccd_atom_property(codes: np.ndarray, names: np.ndarray, column: str, default: int) -> np.ndarray:
    """Each atom's :data:`_COLUMNS` *column* value in its component's template, looked up once per distinct
    (code, name); *default* where the template has no such atom."""
    i = _COLUMNS.index(column)
    distinct, of_code = np.unique(np.asarray(codes, dtype=str), return_inverse=True)
    atom_names, of_name = np.unique(np.asarray(names, dtype=str), return_inverse=True)
    pairs, of_pair = np.unique(of_code * len(atom_names) + of_name, return_inverse=True)
    templates = {code: _ccd_atom_properties(code) for code in distinct.tolist()}
    keys = zip(distinct[pairs // len(atom_names)].tolist(), atom_names[pairs % len(atom_names)].tolist(), strict=True)
    values = [templates[code][name][i] if name in templates[code] else default for code, name in keys]
    return np.array(values, dtype=int)[of_pair]


# pKw, the pKa of a free water on the activity scale that aqua-ion pKa values share.
WATER_PKA: float = 14.0
# pKa of M(H2O)n by element, or element and formal charge where it differs by oxidation state (Baes &
# Mesmer, The Hydrolysis of Cations, 1976; Smith & Martell, Critical Stability Constants).
AQUA_PKA: dict[str, float] = {
    "LI": 13.6, "NA": 14.2, "K": 14.5, "RB": 14.5, "CS": 14.5,  # Rb and Cs as K
    "MG": 11.4, "CA": 12.8, "SR": 13.3, "BA": 13.5, "MN": 10.6, "FE": 9.5, "FE3": 2.2, "CO": 9.7, "NI": 9.9,
    "CU": 8.0, "ZN": 9.0, "CD": 10.1, "HG": 3.4, "PB": 7.7, "AL": 5.0, "GA": 2.6, "IN": 4.0, "CR": 4.0,
    "AG": 12.0, "Y": 7.7, "LA": 8.5, "SM": 7.9, "EU": 7.8, "GD": 8.0, "TB": 7.9, "YB": 7.7, "LU": 7.6,
}  # fmt: skip
# An unlisted metal takes the median of the listed divalent d-block ions (Mn to Zn, Cd).
DEFAULT_AQUA_PKA: float = 9.7
# 5CQO THR 59 OG1, 2.98 A from Hg: past the covalent radii (Cordero 2008) by more than this, in A, a metal bond
# carries under 0.15 valence units (s = exp(-excess / 0.37), Brown & Altermatt 1985), a contact, not coordination.
COORDINATION_TOLERANCE: float = 0.7


def is_metal(atom_array: AtomArray, elements: frozenset[str] = METAL_ELEMENTS) -> np.ndarray:
    """Mask of the atoms whose element is one of *elements*."""
    symbols, of_symbol = np.unique(atom_array.element.astype(str), return_inverse=True)
    return np.isin(np.char.upper(symbols), sorted(elements))[of_symbol]


def extract_inter_residue_metal_bonds(atom_array: AtomArray) -> np.ndarray:
    """Remove the bonds to a metal in another residue from *atom_array* and return them.

    Such a bond is a coordination bond whatever its type says, as parsing retypes it: it
    fills no valence of the donor, so titration runs without it. One inside a component
    (HC0's iron-carbon bonds, 1dn8's NCO) is that component's own connectivity.
    """
    bond_array = atom_array.bonds.as_array()
    metal = is_metal(atom_array)
    residue = struc.get_all_residue_positions(atom_array)
    left, right = bond_array[:, 0].astype(int), bond_array[:, 1].astype(int)
    take = (metal[left] | metal[right]) & (residue[left] != residue[right])
    if take.any():
        atom_array.bonds = struc.BondList(atom_array.array_length(), bond_array[~take])
    return bond_array[take]


def atoms_bonded_to(mask: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    """Mask of the atoms that a row of *bonds* joins to an atom of *mask*."""
    left, right = bonds[:, 0].astype(int), bonds[:, 1].astype(int)
    found = np.zeros(len(mask), dtype=bool)
    found[left[mask[right]]] = True
    found[right[mask[left]]] = True
    return found


def _covalent_radius(element: str) -> float:
    """RDKit's covalent radius of *element*, in A; NaN for a symbol it does not know."""
    try:
        return Chem.GetPeriodicTable().GetRcovalent(element.capitalize())
    except RuntimeError:
        return np.nan


def effective_donor_ph(
    atom_array: AtomArray, coordination_bonds: np.ndarray, metal: np.ndarray, donor: np.ndarray, ph: float
) -> np.ndarray:
    """Per atom, the pH at which a *donor* bonded to a *metal* titrates; NaN for the other atoms.

    A metal that lowers a bound water's pKa from :data:`WATER_PKA` to its :data:`AQUA_PKA` stabilises
    any donor's conjugate base as much, so a donor titrates at *ph* plus ``WATER_PKA - AQUA_PKA`` summed
    over its metals (1HZY's water bridging two Zn is a hydroxide), and never below *ph*. A primary carboxamide N
    binds through its O's resonance, so a metal shifts it nothing (7URH ASN ND2). A bond longer than
    :data:`COORDINATION_TOLERANCE` past the pair's covalent radii shifts nothing; one without positions, or to
    an element without a radius (1JQK UNX's ``X``), counts.
    """
    element = np.char.upper(atom_array.element.astype(str))
    covalent = atom_array.bonds.as_array()
    carbonyl = atoms_bonded_to(element == "O", covalent[covalent[:, 2] == struc.BondType.DOUBLE]) & (element == "C")
    heavy_degree = np.bincount(
        covalent[~np.isin(element[covalent[:, :2]], ("H", "D")).any(axis=1), :2].ravel(), minlength=len(element)
    )
    donor = donor & ~((element == "N") & (heavy_degree == 1) & atoms_bonded_to(carbonyl, covalent))
    pairs = coordination_bonds[:, :2].astype(int)
    pairs = np.vstack([pairs, pairs[:, ::-1]])
    pairs = pairs[metal[pairs[:, 0]] & donor[pairs[:, 1]]]
    radius = {e: _covalent_radius(e) for e in np.unique(atom_array.element[pairs.ravel()].astype(str))}
    reach = [radius[m] + radius[d] + COORDINATION_TOLERANCE for m, d in atom_array.element[pairs].astype(str)]
    length = np.linalg.norm(atom_array.coord[pairs[:, 0]] - atom_array.coord[pairs[:, 1]], axis=-1)
    pairs = pairs[~(length > np.asarray(reach, dtype=float))]
    symbols = np.char.upper(atom_array.element[pairs[:, 0]].astype(str))
    charges = atom_array.charge[pairs[:, 0]]
    pka = [AQUA_PKA.get(f"{s}{q}", AQUA_PKA.get(s, DEFAULT_AQUA_PKA)) for s, q in zip(symbols, charges, strict=True)]
    shift = np.zeros(atom_array.array_length())
    np.add.at(shift, pairs[:, 1], WATER_PKA - np.asarray(pka, dtype=float))
    return np.where(donor, ph + np.maximum(shift, 0.0), np.nan)


def metal_imidazole_hydrogen_counts(
    atom_array: AtomArray, donates_to_metal: np.ndarray, acidified: np.ndarray
) -> np.ndarray:
    """Ring-N hydrogen counts for the imidazoles, unsubstituted on N, that coordinate a metal; -1 where unset.

    The nitrogen on the metal gives up its lone pair and carries no hydrogen, so the tautomer moves
    to its partner (1VMJ HIS NE2), which no pH rule expresses. A ring on metals through both
    nitrogens is the imidazolate where each is *acidified* (its metals lower a water's pKa; 1SPD HIS 63).
    """
    targets = np.full(atom_array.array_length(), -1, dtype=np.int8)
    if not donates_to_metal.any():
        return targets
    rings = imidazole_rings(atom_array)
    for pair in rings.n[rings.open]:
        if donates_to_metal[pair].any() and (acidified[pair].all() or not donates_to_metal[pair].all()):
            targets[pair] = ~donates_to_metal[pair]
        elif donates_to_metal[pair].all() and acidified[pair].any():
            # One N on an acidifying metal, the other on one that is not (an alkali): the proton is on the latter.
            targets[pair] = ~acidified[pair]
    return targets


def valence_capacity(element: np.ndarray, charge: np.ndarray, *, lowest: bool = True) -> np.ndarray:
    """Bonds, hydrogens included, an atom holds: the lowest (or highest) valence of the element it is
    isoelectronic with at its charge (RDKit's; N+ as C, O- as F, B- as C); -1 for a metal or an unknown element.
    """
    element = np.char.upper(np.asarray(element).astype(str))
    keys = np.rec.fromarrays([element, np.broadcast_to(np.asarray(charge, dtype=int), element.shape)])
    distinct, of_key = np.unique(keys, return_inverse=True)
    table, capacities = Chem.GetPeriodicTable(), []
    for symbol, q in distinct.tolist():
        try:
            number = table.GetAtomicNumber(symbol.capitalize())
        except RuntimeError:
            number = 0
        if not number or -1 in table.GetValenceList(number):
            capacities.append(-1)
            continue
        valences = sorted(table.GetValenceList(number - q)) if 0 < number - q < 119 else [0]
        capacities.append(max(valences[0] if lowest else valences[-1], 0))
    return np.array(capacities, dtype=int)[of_key]


def covalent_bond_order_sums(atom_array: AtomArray, metal: np.ndarray) -> np.ndarray:
    """Each atom's sum of bond orders to atoms other than a metal, which fills no valence."""
    bonds = atom_array.bonds.as_array()
    bonds = bonds[~metal[bonds[:, :2]].any(axis=1)]
    weights = [BIOTITE_BOND_TYPE_TO_BOND_ORDER.get(struc.BondType(t), 1) for t in bonds[:, 2]]
    return np.bincount(bonds[:, :2].ravel(), np.repeat(weights, 2), minlength=atom_array.array_length()).astype(int)


def _template_neighbor_deficit(heavy: AtomArray, codes: np.ndarray, include_leaving_atoms: bool = True) -> np.ndarray:
    """CCD non-metal neighbour count minus the structure's retained heavy-bond count.

    Call after removing inter-residue metal bonds. Component-internal metal bonds still count as
    occupied connections; negative values therefore mean extra covalent or internal metal neighbours.
    Without leaving atoms, count only intra-residue bonds: polymer links replace leaving atoms.
    """
    template_degree = _lookup_ccd_atom_property(
        codes, heavy.atom_name, "heavy_degree" if include_leaving_atoms else "kept_degree", 0
    )
    bonds = heavy.bonds.as_array()[:, :2]
    if not include_leaving_atoms:
        residue = struc.get_all_residue_positions(heavy)
        bonds = bonds[residue[bonds[:, 0]] == residue[bonds[:, 1]]]
    observed_degree = np.bincount(bonds.ravel(), minlength=heavy.array_length())
    return template_degree - observed_degree


def _polymer_entry_atoms_after_gaps(heavy: AtomArray) -> np.ndarray:
    """Entering atoms (N, P) of residues that follow a lower-numbered residue of their chain that no bond
    joins them to."""
    starts = struc.get_residue_starts(heavy)
    residue = struc.get_all_residue_positions(heavy)
    bonds = heavy.bonds.as_array()[:, :2]
    joined = np.sort(residue[bonds], axis=1) @ [len(starts), 1]
    firsts = heavy[starts]
    ensure_annotations(firsts, "chain_type")
    entering = np.array(
        [
            get_polymerization_atoms(str(code), ChainType.as_enum(chain_type))[1] or ""
            for code, chain_type in zip(firsts.res_name, firsts.chain_type, strict=True)
        ],
        dtype=str,
    )
    # 1AWD assembly TYR 1 N: assembly copies share chain IDs; chain_iid tells them apart.
    chain_of = chain_identifier(firsts).astype(str)
    in_sequence = np.lexsort([firsts.ins_code, firsts.res_id, chain_of])
    continues = np.zeros(len(starts), dtype=bool)
    chain = chain_of[in_sequence][None]
    neighbours = np.sort(np.c_[in_sequence[:-1], in_sequence[1:]], axis=1) @ [len(starts), 1]
    continues[in_sequence[1:]] = (chain[:, 1:] == chain[:, :-1]).all(axis=0) & ~np.isin(neighbours, joined)
    return continues[residue] & (heavy.atom_name == entering[residue])


def _format_atom_examples(heavy: AtomArray, indices: np.ndarray, limit: int = 5) -> str:
    """``count`` atoms as ``chain:res_id:res_name/atom``, the first *limit* of them named."""
    named = ", ".join(
        f"{heavy.chain_id[i]}:{heavy.res_id[i]}:{heavy.res_name[i]}/{heavy.atom_name[i]}" for i in indices[:limit]
    )
    return f"{len(indices)} atoms ({named}{', ...' if len(indices) > limit else ''})"


def _select_input_hydrogens(
    atom_array: AtomArray, is_h: np.ndarray, heavy_index: np.ndarray, count: np.ndarray, protonated: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The input hydrogens the assignment keeps, and the heavy atom (in heavy numbering) each is on.

    A hydrogen is kept where it has a position, one bond, to a heavy atom being protonated
    with a position, and that atom carries at least as many hydrogens as the input states.
    """
    finite = np.isfinite(atom_array.coord).all(axis=-1)
    bonds = atom_array.bonds.as_array()
    left, right = bonds[:, 0].astype(np.intp), bonds[:, 1].astype(np.intp)
    n_bonds = np.bincount(bonds[:, :2].ravel(), minlength=len(atom_array))
    pair = np.concatenate([np.stack([left, right], 1), np.stack([right, left], 1)])
    pair = pair[is_h[pair[:, 0]] & ~is_h[pair[:, 1]]]
    parent = np.full(len(atom_array), -1, dtype=np.intp)
    parent[pair[:, 0]] = pair[:, 1]
    candidate = is_h & finite & (n_bonds == 1) & (parent >= 0)
    candidate[candidate] &= finite[parent[candidate]]
    hydrogens = np.flatnonzero(candidate)
    on = heavy_index[parent[hydrogens]]
    stated = np.bincount(on, minlength=len(count))
    # Rebuild a smaller target count rather than arbitrarily retaining a subset of input positions.
    keep = protonated[on] & (count[on] >= stated[on])
    return hydrogens[keep], on[keep]


def _validate_declared_hydrogen_counts(hydrogens: np.ndarray | None, is_input_h: np.ndarray) -> np.ndarray:
    """The declared hydrogen counts of the heavy atoms, -1 where none is declared."""
    if hydrogens is None:
        return np.full(np.count_nonzero(~is_input_h), -1, dtype=np.int64)
    counts = np.asarray(hydrogens)
    if counts.shape != is_input_h.shape or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(f"hydrogens must hold one integer per atom ({len(is_input_h)}), got {counts.shape}.")
    if (counts < -1).any() or (counts[is_input_h] >= 0).any():
        raise ValueError("hydrogens declares a count below -1 or on a hydrogen; declare it on the heavy atom.")
    return counts[~is_input_h].astype(np.int64)


def assign_hydrogens(
    atom_array: AtomArray,
    *,
    ph: float = 7.4,
    hydrogens: np.ndarray | None = None,
) -> AtomArray:
    """Decide every heavy atom's charge and hydrogen count at a pH, without placing hydrogens.

    Standard residues are titrated one at a time with Dimorphite-DL, atomized atoms as connected
    bond subgraphs. Boundary atoms keep their input charge and ``nhyd``, metal donors take their
    donating state, and stated histidine tautomers and declared counts stand. A residue drawn with
    neither hydrogens nor a nonzero ``nhyd`` takes the dictionary's counts, and an input charge that
    leaves an atom too little room for its heavy bonds is ignored. Input hydrogens with a position and one bond are kept where their heavy
    atom carries at least that many; the rest are dropped.

    For a monatomic component, an undeclared state uses its CCD/custom charge when the input is zero
    (also the parser's value for missing charge). Declare ``hydrogens`` to distinguish, e.g., HCl from chloride.

    Args:
        atom_array: Structure with bonds and ``charge`` and ``pn_unit_iid`` (or ``pn_unit_id``), as
            :func:`~atomworks.io.parse` returns it. A donor is on a metal only through a bond (parse
            with ``"metalc"`` in ``add_bond_types_from_struct_conn``).
        ph: Target pH.
        hydrogens: Declared hydrogen count per input atom, -1 where none; it outranks the input's
            hydrogens, a metal and the pH, and one on both ring nitrogens sets a histidine's tautomer.

    Returns:
        The input atoms in input order, less the input hydrogens it drops, with ``atom_id`` (numbered
        from 0 if absent), ``charge`` (formal), ``nhyd`` (hydrogens each heavy atom still needs),
        ``skip_hydrogen_placement`` (input hydrogens, unresolved or unbonded heavy atoms, and metals;
        their remaining ``nhyd`` counts are not placed) and
        ``tautomer_free`` (histidines nothing decides; a neutral one takes NE2-H).

    Raises:
        TypeError: If *atom_array* is an ``AtomArrayStack``.
        ValueError: If bonds, ``charge`` or ``pn_unit_iid`` are absent, a bond is ``ANY`` or ``AROMATIC``,
            *ph* is not finite, a residue
            is partially atomized, or *hydrogens* has invalid entries or exceeds chemical valence.
    """
    if isinstance(atom_array, AtomArrayStack):
        raise TypeError("assign_hydrogens takes an AtomArray; extract a single model first (e.g. atom_array[0]).")
    if not np.isfinite(ph):
        raise ValueError(f"ph must be a finite number, got {ph}.")
    if atom_array.bonds is None:
        raise ValueError("Input atom_array has no bonds. Bonds must be assigned before protonation.")
    if "charge" not in atom_array.get_annotation_categories():
        raise ValueError("Input atom_array must have a 'charge' annotation for heavy atoms before protonation.")
    if np.isin(atom_array.bonds.as_array()[:, 2], (struc.BondType.ANY, struc.BondType.AROMATIC)).any():
        raise ValueError("Bonds of unstated order (BondType.ANY or AROMATIC) cannot be protonated; state them.")

    is_input_h = np.isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS)
    declared_hydrogens = _validate_declared_hydrogen_counts(hydrogens, is_input_h)
    # 3C3G HIS: a ring N's hydrogen with a position states the tautomer (the dictionary's HIS has HD1 and HE2,
    # unplaced, so only a placed one says anything).
    imidazole_targets = np.full(len(atom_array), -1, dtype=np.int8)
    placed_h = is_input_h & np.isfinite(atom_array.coord).all(axis=-1)
    has_placed_h = atoms_bonded_to(placed_h, atom_array.bonds.as_array())
    for ring_n in imidazole_rings(atom_array).n:
        if has_placed_h[ring_n].any():
            imidazole_targets[ring_n] = has_placed_h[ring_n]
    imidazole_targets = imidazole_targets[~is_input_h]
    input_tautomer = imidazole_targets >= 0
    has_input_hydrogen_counts = "nhyd" in atom_array.get_annotation_categories()
    # 3KS3: a residue with placed hydrogens or nonzero nhyd states its counts; hydrogens in other residues
    # say nothing about it. A parse without missing atoms leaves zero counts, so it takes the dictionary's.
    input_residue = struc.get_all_residue_positions(atom_array)
    residue_has_placed_hydrogens = (np.bincount(input_residue, weights=placed_h) > 0)[input_residue]
    residue_has_hydrogen_counts = np.zeros(len(atom_array), dtype=bool)
    if has_input_hydrogen_counts:
        residue_has_hydrogen_counts = (np.bincount(input_residue, weights=atom_array.nhyd) > 0)[input_residue]
    counts_unspecified = ~(residue_has_placed_hydrogens | residue_has_hydrogen_counts)[~is_input_h]
    annotated = annotate_hydrogens(atom_array.copy(), increment=True)
    heavy = annotated[~is_input_h]
    declared_ring = np.zeros(len(heavy), dtype=bool)
    for ring_n in imidazole_rings(heavy).n:
        declared_ring[ring_n] = (declared_hydrogens[ring_n] >= 0).all()
    declared_atom = (declared_hydrogens >= 0) & ~declared_ring
    input_tautomer |= declared_ring
    ccd_codes = heavy.res_name.astype(str)
    needs_ccd_hydrogens = (heavy.nhyd == 0) & (counts_unspecified | (not has_input_hydrogen_counts))
    heavy.nhyd[needs_ccd_hydrogens] = _lookup_ccd_atom_property(
        ccd_codes[needs_ccd_hydrogens], heavy.atom_name[needs_ccd_hydrogens], "hydrogens", 0
    )
    if has_input_hydrogen_counts:
        # 8HBF NO N, 7YPY PER O1: zero counts do state a residue the dictionary gives no hydrogens.
        residue = struc.get_all_residue_positions(heavy)
        counts_unspecified &= (np.bincount(residue, weights=heavy.nhyd) > 0)[residue]
    hydrogens_stated = ~counts_unspecified

    coordination_bonds = extract_inter_residue_metal_bonds(heavy)
    template_neighbor_deficit = _template_neighbor_deficit(heavy, ccd_codes)
    # 5ARK LYS C: a leaving atom without density (an unresolved OXT) is missing, not replaced by a hydrogen; the
    # atom it leaves keeps the dictionary's hydrogens.
    missing_nonleaving_neighbors = np.maximum(
        _template_neighbor_deficit(heavy, ccd_codes, include_leaving_atoms=False), 0
    )
    missing_leaving_atoms = template_neighbor_deficit > missing_nonleaving_neighbors
    component_bonds = heavy.bonds.as_array()
    isolated = np.bincount(component_bonds[:, :2].ravel(), minlength=len(heavy)) == 0
    metal = is_metal(heavy)
    # 1FDN SF4 S: valence cannot count the bonds of a metal or of the atoms bonded to one, so they keep their counts.
    within_metal_component = metal | atoms_bonded_to(metal, component_bonds)
    preserve_metal_component_state = within_metal_component & ~isolated & hydrogens_stated
    # 9EWF SIA O1A, 1YTJ PPN N1: an input charge the dictionary does not draw, at which an atom off a metal cannot
    # hold its heavy bonds, becomes the nearest to the dictionary's at which it can (a boronate adduct's B -1).
    bond_order_sums = covalent_bond_order_sums(heavy, metal)
    ccd_charge = _lookup_ccd_atom_property(ccd_codes, heavy.atom_name, "charge", 0)
    has_known_valence = valence_capacity(heavy.element, 0) >= 0
    overbonded = (
        (bond_order_sums > valence_capacity(heavy.element, heavy.charge, lowest=False))
        & has_known_valence
        & ~within_metal_component
    )
    corrected_charge_indices = []
    for index in np.flatnonzero(overbonded & (ccd_charge != heavy.charge)):
        ccd_charge_value, element = ccd_charge[index], heavy.element[[index]]
        compatible_charges = [
            q
            for q in (ccd_charge_value, ccd_charge_value - 1, ccd_charge_value + 1)
            if valence_capacity(element, q, lowest=False)[0] >= bond_order_sums[index]
        ]
        if compatible_charges:
            heavy.charge[index] = compatible_charges[0]
            corrected_charge_indices.append(index)
    if corrected_charge_indices:
        logger.warning(
            "Replaced incompatible charges of %s with bond-compatible charges using the CCD as reference.",
            _format_atom_examples(heavy, np.array(corrected_charge_indices)),
        )
    # 6dmz_mod_l CYS SG, 2YAK OSV S16: a bond to a neighbour takes the room of a hydrogen; an atom holds its element's
    # lowest valence, or the higher one the dictionary draws it with.
    charge_matches_ccd = ccd_charge == heavy.charge
    ccd_valences = _lookup_ccd_atom_property(ccd_codes, heavy.atom_name, "valence", -1)
    valence_limit = np.maximum(
        valence_capacity(heavy.element, heavy.charge), np.where(charge_matches_ccd, ccd_valences, -1)
    )
    heavy.nhyd = np.where(
        valence_limit >= 0, np.clip(valence_limit - bond_order_sums, 0, heavy.nhyd), heavy.nhyd
    ).astype(heavy.nhyd.dtype)

    atomize_mask = compute_standard_atomize_mask(heavy)
    missing_coordinates = ~np.isfinite(heavy.coord).all(axis=-1)
    sizes = np.diff(struc.get_residue_starts(heavy, add_exclusive_stop=True))
    # 2AJ6 UNL: an atom no bond reaches, in a residue of several, keeps the charge and count it is given.
    unbonded_ligand_atoms = isolated & np.repeat(sizes > 1, sizes) & atomize_mask & (declared_hydrogens < 0)
    missing_nonleaving_neighbors[unbonded_ligand_atoms] = 0

    # 4EJK ALA N: a chain continues across a gap, so the atom entering it keeps its linked residue's hydrogens.
    after_a_gap = _polymer_entry_atoms_after_gaps(heavy) & ~missing_coordinates & (ccd_codes != "")
    linked_hydrogen_counts = _lookup_ccd_atom_property(
        ccd_codes[after_a_gap], heavy.atom_name[after_a_gap], "linked_hydrogens", -1
    )
    heavy.nhyd[after_a_gap] = np.where(linked_hydrogen_counts >= 0, linked_hydrogen_counts, heavy.nhyd[after_a_gap])
    missing_nonleaving_neighbors[after_a_gap] = 0
    # 7NO8 PRO N: an atom short of a heavy neighbour the dictionary gives it takes a hydrogen in its place.
    capped_atoms = np.flatnonzero(missing_nonleaving_neighbors)
    heavy.nhyd[capped_atoms] = (
        _lookup_ccd_atom_property(ccd_codes[capped_atoms], heavy.atom_name[capped_atoms], "hydrogens", 0)
        + missing_nonleaving_neighbors[capped_atoms]
    )
    if len(capped_atoms):
        logger.warning(
            "Capped %s with hydrogen: the dictionary gives them a heavy neighbour the structure does "
            "not. Supply the missing atoms, or parse with add_missing_atoms=True, for the real chemistry.",
            _format_atom_examples(heavy, capped_atoms),
        )

    bonded = heavy.copy()
    for left, right, bond_type in coordination_bonds:
        bonded.bonds.add_bond(int(left), int(right), int(bond_type))

    components: list[np.ndarray] = []
    for start, stop in itertools.pairwise(struc.get_residue_starts(heavy, add_exclusive_stop=True)):
        res_atomize = atomize_mask[start:stop]
        if res_atomize.all():
            continue
        if res_atomize.any():
            raise ValueError(
                f"Partially-atomized residue detected at {heavy.chain_id[start]}:{heavy.res_id[start]} "
                f"({heavy.res_name[start]}): {res_atomize.sum()}/{stop - start} atoms flagged."
            )
        components.append(np.arange(start, stop))
    components += partition_atomized_subgraphs(heavy, atomize_mask)
    # 1hxq U5P: an unplaced atom ending a branch of its residue at a placed atom holding more covalent bonds than
    # the dictionary gives it is a leaving atom that bond displaced; the others are titrated with their component.
    of_residue = struc.get_all_residue_positions(heavy)
    own_bonds = component_bonds[of_residue[component_bonds[:, 0]] == of_residue[component_bonds[:, 1]]]
    ends_a_branch = np.bincount(own_bonds[:, :2].ravel(), minlength=len(heavy)) == 1
    coordination_degree = np.bincount(coordination_bonds[:, :2].ravel(), minlength=len(heavy))
    displaces_leaving_atoms = (
        (template_neighbor_deficit + coordination_degree < 0) & ~within_metal_component & ~missing_coordinates
    )
    displaced_leaving_atoms = missing_coordinates & ends_a_branch & atoms_bonded_to(displaces_leaving_atoms, own_bonds)
    components = [indices[~displaced_leaving_atoms[indices]] for indices in components]
    components = [indices for indices in components if len(indices)]

    metal_donors = atoms_bonded_to(metal, coordination_bonds) & ~metal & (declared_hydrogens < 0)
    donor_ph = effective_donor_ph(heavy, coordination_bonds, metal, metal_donors, ph)
    metal_targets = metal_imidazole_hydrogen_counts(heavy, metal_donors, donor_ph > ph)
    # 8HBF NO N: coordination only takes protons; a donor keeps at most the hydrogens its input states.
    max_donor_hydrogens = np.where(metal_donors & hydrogens_stated, heavy.nhyd, -1)
    imidazole_targets = np.where(imidazole_targets < 0, metal_targets, imidazole_targets)
    imidazole_targets = np.where(declared_ring, declared_hydrogens, imidazole_targets)

    component_local_bonds, on_boundary, links = bonds_within_components(heavy, components)
    boundary = (
        on_boundary | (missing_nonleaving_neighbors > 0) | missing_leaving_atoms | within_metal_component | after_a_gap
    )
    hydrogen_counts = np.zeros(len(heavy), dtype=np.int64)
    ring_bonds: list[np.ndarray] = []
    unbonded_components: list[int] = []
    template_charge_indices: list[int] = []
    preserved_hydrogen_counts = np.where(
        declared_atom, declared_hydrogens, np.where(preserve_metal_component_state, heavy.nhyd, -1)
    )
    # Each component takes its own bonds, so no slice filters the whole structure's bond list.
    heavy.bonds = None
    for number, indices in enumerate(components):
        if metal[indices].all():
            continue
        template = _ccd_template(ccd_codes[indices[0]]) if len(indices) == 1 else None
        if template is not None and template.array_length() == 1 and declared_hydrogens[indices[0]] < 0:
            # An undeclared monatomic state follows its component definition; explicit counts take precedence.
            if bonded.charge[indices[0]] == 0 and template.charge[0] != 0:
                bonded.charge[indices] = template.charge[0]
                template_charge_indices.append(int(indices[0]))
            continue
        if unbonded_ligand_atoms[indices].all():
            unbonded_components += indices.tolist()
            continue
        targets, coordinating, stated = (
            imidazole_targets[indices],
            max_donor_hydrogens[indices],
            preserved_hydrogen_counts[indices],
        )
        inputs = (
            heavy.element[indices],
            heavy.charge[indices],
            component_local_bonds[number],
            heavy.res_id[indices] - heavy.res_id[indices[0]],
            targets,
            coordinating,
            donor_ph[indices],
            stated,
            links[number],
        )
        key = (ph, *((array.dtype.str, array.tobytes()) for array in inputs))
        with _TITRATIONS_LOCK:
            titrated = _titrations.pop(key, _UNTITRATED)
        if titrated is _UNTITRATED:
            sub = heavy[indices]
            sub.bonds = struc.BondList(len(indices), component_local_bonds[number])
            with rdBase.BlockLogs():
                titrated = titrate_component(sub, ph, targets, coordinating, stated, donor_ph[indices], links[number])
            for array in titrated or ():
                array.setflags(write=False)
        with _TITRATIONS_LOCK:
            _titrations[key] = titrated
            if len(_titrations) > _MAX_TITRATIONS:
                _titrations.pop(next(iter(_titrations)), None)
        titrated_interior = ~boundary[indices] & (titrated is not None)
        if titrated is not None:
            charge, hydrogens, rekekulized = titrated
            bonded.charge[indices[titrated_interior]] = charge[titrated_interior]
            hydrogen_counts[indices[titrated_interior]] = hydrogens[titrated_interior]
            if len(rekekulized):
                ring_bonds.append(np.column_stack([indices[rekekulized[:, :2]], rekekulized[:, 2]]))
        hydrogen_counts[indices[~titrated_interior]] = np.maximum(heavy.nhyd[indices[~titrated_interior]], 0)
    if template_charge_indices:
        logger.warning(
            "Used CCD/custom charges for %s with zero input charge and no declared hydrogen count. "
            "Zero also represents missing charge; declare hydrogens or supply a custom component for another state.",
            _format_atom_examples(heavy, np.array(template_charge_indices)),
        )
    hydrogen_counts = np.where(declared_hydrogens >= 0, declared_hydrogens, hydrogen_counts)
    if ring_bonds:
        ring, bonds = np.vstack(ring_bonds), bonded.bonds.as_array()
        ring[:, :2].sort(axis=1)
        keys = bonds[:, 0].astype(np.int64) * len(bonded) + bonds[:, 1]
        order = np.argsort(keys)
        bonds[order[np.searchsorted(keys, ring[:, 0] * len(bonded) + ring[:, 1], sorter=order)], 2] = ring[:, 2]
        bonded.bonds = struc.BondList(len(bonded), bonds)
    # Chemical valence limits hydrogen counts; placement geometry must not change the assigned state.
    bond_order_sums = covalent_bond_order_sums(bonded, metal)
    valence_limit = valence_capacity(bonded.element, bonded.charge, lowest=False)
    hydrogen_capacity = np.where(valence_limit < 0, hydrogen_counts, np.maximum(valence_limit - bond_order_sums, 0))
    # 10GS LYS NZ: a declared count stands; the atom takes the charge nearest its own at which it holds its
    # bonds and those hydrogens (an NZ declared with three H at charge 0 is the ammonium).
    for index in np.flatnonzero((declared_hydrogens >= 0) & (hydrogen_counts > hydrogen_capacity)):
        charge, element = int(bonded.charge[index]), bonded.element[[index]]
        for q in sorted(range(charge - 2, charge + 3), key=lambda q: abs(q - charge)):
            if hydrogen_counts[index] <= valence_capacity(element, q, lowest=False)[0] - bond_order_sums[index]:
                bonded.charge[index], hydrogen_capacity[index] = q, hydrogen_counts[index]
                break
        else:
            raise ValueError(
                f"Declared hydrogen count exceeds chemical valence for {_format_atom_examples(heavy, [index])}."
            )
    overvalent_atoms = np.flatnonzero(hydrogen_counts > hydrogen_capacity)
    if len(overvalent_atoms):
        logger.warning(
            "Gave %s only the hydrogens their bonds leave room for.", _format_atom_examples(heavy, overvalent_atoms)
        )
        hydrogen_counts = np.minimum(hydrogen_counts, hydrogen_capacity)
    if unbonded_components:
        logger.warning(
            "No bond is stated for %s, so their charges and hydrogen counts are left as given. Supply the bonds "
            "(CONECT records, a struct_conn category, or a chemical component definition) to have them protonated.",
            _format_atom_examples(heavy, np.array(unbonded_components)),
        )

    # Metal hydrides are not protons; retain their counts without inventing placement geometry.
    skip_placement = missing_coordinates | unbonded_ligand_atoms | metal
    metal_hydrides = np.flatnonzero(metal & (hydrogen_counts > 0))
    if len(metal_hydrides):
        logger.warning(
            "Metal-bound hydrogens on %s remain as nhyd counts: hydride placement is unsupported.",
            _format_atom_examples(heavy, metal_hydrides),
        )
    heavy_index = np.cumsum(~is_input_h) - 1
    kept_hydrogen_indices, kept_hydrogen_parents = _select_input_hydrogens(
        annotated, is_input_h, heavy_index, hydrogen_counts, ~skip_placement
    )
    if placed_h.sum() > len(kept_hydrogen_indices):
        logger.warning(
            "Dropped %d placed input hydrogens that cannot be retained; assigned hydrogen counts are preserved.",
            placed_h.sum() - len(kept_hydrogen_indices),
        )
    kept_atoms = ~is_input_h
    kept_atoms[kept_hydrogen_indices] = True
    assigned = annotated[kept_atoms]
    output_indices = np.cumsum(kept_atoms) - 1
    heavy_output_indices = output_indices[~is_input_h]
    bonds = bonded.bonds.as_array()
    bonds[:, :2] = heavy_output_indices[bonds[:, :2]]
    h_bonds = np.stack(
        [
            heavy_output_indices[kept_hydrogen_parents],
            output_indices[kept_hydrogen_indices],
            np.full(len(kept_hydrogen_indices), 1),
        ],
        axis=1,
    )
    assigned.bonds = struc.BondList(len(assigned), np.vstack([bonds, h_bonds]).astype(np.int64))

    nhyd = np.zeros(len(assigned), dtype=heavy.nhyd.dtype)
    nhyd[heavy_output_indices] = np.where(
        unbonded_ligand_atoms, heavy.nhyd, hydrogen_counts - np.bincount(kept_hydrogen_parents, minlength=len(heavy))
    )
    charge = assigned.charge.copy()
    charge[heavy_output_indices] = bonded.charge
    skip_hydrogen_placement = np.ones(len(assigned), dtype=bool)
    skip_hydrogen_placement[heavy_output_indices] = skip_placement
    # A neutral imidazole is free where nothing states its ring hydrogen and neither N bonds a third heavy atom.
    free = np.zeros(len(assigned), dtype=bool)
    ring_h = np.where(skip_placement, 0, hydrogen_counts)
    degree = np.bincount(bonded.bonds.as_array()[:, :2].ravel(), minlength=len(bonded))
    for ring_n in imidazole_rings(bonded).n:
        if not input_tautomer[ring_n].any() and ring_h[ring_n].sum() == 1 and (degree[ring_n] == 2).all():
            free[heavy_output_indices[ring_n]] = True
    assigned.set_annotation("charge", charge)
    assigned.set_annotation("nhyd", nhyd)
    assigned.set_annotation("skip_hydrogen_placement", skip_hydrogen_placement)
    assigned.set_annotation("tautomer_free", free)
    if "atom_id" not in assigned.get_annotation_categories():
        assigned.set_annotation("atom_id", np.arange(len(assigned), dtype=np.intp))
    return assigned
