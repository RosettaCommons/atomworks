"""Hydrogen placement: the plan (TMol's internal coordinates from its table, a rule otherwise), building it on
coordinates, and adding the hydrogens to an AtomArray."""

from __future__ import annotations

import functools
import itertools
import json
import logging
import math
from collections import defaultdict
from collections.abc import Container, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.constants import (
    CHAIN_LEVEL_ANNOTATIONS,
    ELEMENT_NAME_TO_ATOMIC_NUMBER,
    HYDROGEN_LIKE_SYMBOLS,
    RESIDUE_LEVEL_ANNOTATIONS,
)
from atomworks.experimental.protonation._assign import (
    _cache_ccd_lookup,
    _ccd_template,
    _format_atom_examples,
    assign_hydrogens,
    is_metal,
    template_hydrogen_names,
)
from atomworks.experimental.protonation.geometry import (
    _measure_icoors,
    _numbered,
    _perpendicular,
    _unit,
    build_coordinates,
    names_from_parent,
)
from atomworks.io.utils.annotator import ANNOTATOR_REGISTRY, ensure_annotations
from atomworks.io.utils.atom_array_plus import AnnotationList2D, AtomArrayPlus, concatenate_atom_array_plus
from atomworks.io.utils.ccd import get_custom_ccd_entries, get_polymerization_atoms
from atomworks.io.utils.selection import get_annotation, get_annotation_categories
from atomworks.io.utils.standard_annotations.base import STANDARD_ANNOTATIONS, Level

# --- table ---
_TABLE_PATH = Path(__file__).parent / "hydrogen_icoors.json"

Row = tuple[str, str, str, str, float, float, float]


@functools.cache
def entries_by_code() -> dict[str, tuple[tuple[str, frozenset[str], tuple[Row, ...]], ...]]:
    """The table's entries grouped by the code they describe, as ``(entry, names, rows)``."""
    table = json.loads(_TABLE_PATH.read_text())
    raw = {k: v for k, v in table.items() if not k.startswith("_")}
    for terminus in table["_termini"]:
        for entry, body in [(k, v) for k, v in raw.items() if v["code"] in terminus["codes"]]:
            patch = {k: terminus[k] for k in ("without", "hydrogens") if k in terminus}
            raw[entry + terminus["suffix"]] = {"code": body["code"], "extends": entry, **patch}

    def rows_for(entry: str) -> list[list]:
        body = raw[entry]
        inherited = rows_for(body["extends"]) if "extends" in body else []
        dropped = set(body.get("without", ()))
        return [row for row in inherited if row[0] not in dropped] + (body.get("hydrogens") or [])

    grouped: dict[str, list] = defaultdict(list)
    for entry, body in raw.items():
        rows = tuple(
            (name, parent, gp, ggp, d, math.radians(theta), math.radians(phi))
            for name, parent, gp, ggp, phi, theta, d in rows_for(entry)
        )
        grouped[body["code"]].append((entry, frozenset(row[0] for row in rows), rows))
    return {code: tuple(entries) for code, entries in grouped.items()}


@functools.lru_cache(maxsize=1024)
def select_entry(code: str, hydrogen_names: frozenset[str]) -> tuple[Row, ...] | None:
    """The entry describing a residue with these hydrogens, if one does.

    An entry applies when it names every hydrogen the table knows for the code (a
    protonated aspartate's HD2 is in no entry); the one naming fewest extra wins, which
    picks the tautomer built, and a remaining tie goes against the ``_D`` (ND1) histidine.
    """
    entries = entries_by_code().get(code, ())
    wanted = hydrogen_names & frozenset().union(*(names for _, names, _ in entries))
    candidates = [(entry, rows) for entry, names, rows in entries if wanted and wanted <= names]
    return min(candidates, key=lambda c: (len(c[1]), "_D" in c[0], c[0]))[1] if candidates else None


@functools.lru_cache(maxsize=1024)
def names_on_parent(code: str, parent: str, count: int) -> tuple[str, ...] | None:
    """The names an entry gives *count* hydrogens on one atom (``H1 H2 H3`` on a terminal N)."""
    for _, _, rows in entries_by_code().get(code, ()):
        names = tuple(row[0] for row in rows if row[1] == parent)
        if len(names) == count:
            return names
    return None


# --- names ---
def residue_hydrogen_names(
    res_name: str, heavy_names: frozenset[str], parents: tuple[tuple[str, str, int], ...]
) -> tuple[tuple[str, ...], ...]:
    """The names one residue's hydrogens take.

    Args:
        res_name: The residue's code.
        heavy_names: Names of its heavy atoms.
        parents: ``(name, element, count)`` of each heavy atom carrying hydrogens, in order.

    Returns:
        Per parent, the names of its hydrogens.
    """
    by_parent = template_hydrogen_names(res_name)
    taken = set(heavy_names)
    # 6Q9T QUK: a generated name avoids every name the component declares, not just those handed out.
    declared_names = {name for names in by_parent.values() for name in names}
    result = []
    for heavy_name, element, count in parents:
        declared = tuple(by_parent.get(heavy_name, ()))
        blocked = taken | declared_names.difference(declared)
        if not declared:
            names = names_from_parent(heavy_name, element, count, blocked)
        elif count > len(declared):
            # More hydrogens than names: a protonated amino terminus, numbered as such.
            names = [name for name in names_on_parent(res_name, heavy_name, count) or () if name not in blocked][:count]
            names += _numbered(declared[0].rstrip("0123456789") or "H", count - len(names), blocked | set(names))
        else:
            names = [name for name in declared if name not in blocked][:count]
            names += itertools.islice(_continued_names(declared, blocked | set(names)), count - len(names))
        taken.update(names)
        result.append(tuple(names))
    return tuple(result)


def _continued_names(declared: tuple[str, ...], taken: Container[str]) -> Iterator[str]:
    """Names continuing a parent's *declared* hydrogen series, less *taken* (``H``, ``H2`` -> ``H3``)."""
    prefix = declared[0].rstrip("0123456789")
    highest = max(int(name[len(prefix) :]) if name[len(prefix) :].isdigit() else 1 for name in declared)
    return (name for number in itertools.count(highest + 1) if (name := f"{prefix}{number}") not in taken)


# --- rule ---
_BOND_LENGTH = {"C": 1.09, "N": 1.01, "O": 0.97, "S": 1.34, "P": 1.42, "B": 1.19, "SE": 1.47, "SI": 1.48}
_TETRAHEDRAL = float(np.arccos(-1.0 / 3.0))
# Cosine to its neighbours below which an apex hydrogen has no side of its own (1/3 when tetrahedral).
_FLAT = 1.0 / 6.0
_PI_BONDS = np.array([int(struc.BondType[t]) for t in ("DOUBLE", "AROMATIC_SINGLE", "AROMATIC_DOUBLE", "AROMATIC")])
_TRIPLE_BONDS = np.array([int(struc.BondType.TRIPLE), int(struc.BondType.AROMATIC_TRIPLE)])
_TETRA, _PLANAR, _LINEAR, _ON_METAL, _SYN, _CONJUGATED_ON_METAL = 0, 1, 2, 3, 4, 5
# (theta, phi) of each free direction about one known bond, and in a lab frame; an O on a metal is trigonal
# (M-O-H 125.3 deg), and a hydroxyl on a pi-bonded atom lies in its plane, syn to an acid's C=O, away from a metal.
_ONE_KNOWN = {
    _TETRA: [(np.pi - _TETRAHEDRAL, np.pi + k * 2 * np.pi / 3) for k in range(3)],
    _PLANAR: [(np.pi / 3, np.pi), (np.pi / 3, 0.0)],
    _LINEAR: [(0.0, 0.0)],
    _ON_METAL: [(_TETRAHEDRAL / 2, np.pi), (_TETRAHEDRAL / 2, 0.0)],
    _SYN: [(np.pi - _TETRAHEDRAL, 0.0)],
}
_NONE_KNOWN = {
    _TETRA: [(0.0, 0.0)] + [(_TETRAHEDRAL, k * 2 * np.pi / 3) for k in range(3)],
    _PLANAR: [(0.0, 0.0), (2 * np.pi / 3, 0.0), (2 * np.pi / 3, np.pi)],
    _LINEAR: [(0.0, 0.0), (np.pi, 0.0)],
}
_NONE_KNOWN[_ON_METAL] = _NONE_KNOWN[_CONJUGATED_ON_METAL] = _NONE_KNOWN[_TETRA]
_ONE_KNOWN[_CONJUGATED_ON_METAL] = _ONE_KNOWN[_TETRA]


class Topology(NamedTuple):
    """The atom array as the rule reads it; *pairs* holds every bond both ways and
    *name_rank* orders atoms by name, so ties do not depend on the atom order."""

    coord: np.ndarray
    element: np.ndarray
    is_h: np.ndarray
    finite: np.ndarray
    pairs: np.ndarray
    bond_type: np.ndarray
    residue: np.ndarray
    metal: np.ndarray
    name_rank: np.ndarray


def _sorted_neighbours(topo: Topology, pairs: np.ndarray, *keys: np.ndarray) -> np.ndarray:
    """``[n, k]`` partners of each atom sorted by *keys* (last key most significant), then by
    name (5HS6 J3Z HO3), ``-1`` padded."""
    n = len(topo.coord)
    pairs = pairs[np.lexsort((pairs[:, 1], topo.name_rank[pairs[:, 1]], *keys, pairs[:, 0]))]
    degree = np.bincount(pairs[:, 0], minlength=n)
    table = np.full((n, max(int(degree.max(initial=0)), 1)), -1, dtype=np.intp)
    table[pairs[:, 0], np.arange(len(pairs)) - np.repeat(np.cumsum(degree) - degree, degree)] = pairs[:, 1]
    return table


def _first(candidates: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per row, the first candidate marked *valid*, else -1."""
    return np.where(valid.any(axis=1), candidates[np.arange(len(candidates)), valid.argmax(axis=1)], -1)


def _or(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(v).all(axis=-1, keepdims=True), v, fallback)


def _hydrogen_parent_geometries(topo: Topology, atoms: np.ndarray, m: np.ndarray, n_build: np.ndarray) -> np.ndarray:
    """Each parent's shape: linear (a triple or two double bonds), planar (a pi bond, or an N
    conjugated to one with three neighbours), trigonal about a metal (an O bonded to one, with
    three neighbours at most; in the plane of a pi bond it is conjugated to, off the metal), syn
    (a hydroxyl conjugated to a pi bond), else tetrahedral; tetrahedral if the shape is full."""
    n, (a, t) = len(topo.coord), (topo.pairs[:, 0], topo.bond_type)
    double = np.bincount(a, weights=t == int(struc.BondType.DOUBLE), minlength=n)
    linear = (np.bincount(a, weights=np.isin(t, _TRIPLE_BONDS), minlength=n) > 0) | (double >= 2)
    planar = np.bincount(a, weights=np.isin(t, _PI_BONDS), minlength=n) > 0
    heavy = ~topo.is_h[topo.pairs[:, 1]]
    conjugated = np.bincount(a[heavy], weights=planar[topo.pairs[heavy, 1]], minlength=n) > 0
    on_metal = np.bincount(a, weights=topo.metal[topo.pairs[:, 1]], minlength=n) > 0

    steric = m + n_build
    # 6dmz_mod_d: an N with three neighbours conjugated to a pi bond (an amide N) is planar.
    amine = (topo.element[atoms] == "N") & conjugated[atoms] & (steric == 3)
    shape = np.where((planar[atoms] | amine) & (steric <= 3), _PLANAR, _TETRA)
    oxygen = topo.element[atoms] == "O"
    # ACY: a hydroxyl conjugated to a pi bond lies syn to it.
    hydroxyl = oxygen & conjugated[atoms] & ~planar[atoms] & (m == 1) & (n_build == 1)
    shape = np.where(hydroxyl, _SYN, shape)
    # 1HZY: an O on a metal is trigonal.
    metal_oxygen = oxygen & on_metal[atoms] & (steric <= 3)
    shape = np.where(metal_oxygen, np.where(conjugated[atoms], _CONJUGATED_ON_METAL, _ON_METAL), shape)
    shape = np.where(linear[atoms] & (steric <= 2), _LINEAR, shape)
    crowded = np.flatnonzero(steric > 4)
    if len(crowded):
        atom = atoms[crowded[0]]
        raise ValueError(f"Atom {atom} ({topo.element[atom]}) has more hydrogens than free directions.")
    return shape


def _measured_slots(
    coord: np.ndarray,
    shape: np.ndarray,
    m: np.ndarray,
    x: np.ndarray,
    known: np.ndarray,
    d: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """``[k, 2, 3]`` ``(d, theta, phi)`` of the free directions of parents with two or more known;
    *reference* is the torsion reference of each first known atom (-1 where none)."""
    ua, ub, uc = (_unit(coord[known[:, i]] - coord[x]) for i in range(3))
    bisector = _or(-_unit(ua + ub), _perpendicular(ua))
    normal = _or(_unit(np.cross(ua, ub)), _unit(np.cross(ua, bisector)))
    half = _TETRAHEDRAL / 2
    first = np.cos(half) * bisector + np.sin(half) * normal
    second = np.cos(half) * bisector - np.sin(half) * normal
    first = np.where(((shape == _TETRA) & (m == 2))[:, None], first, bisector)
    in_plane = _unit(np.cross(np.cross(ua, coord[np.maximum(reference, 0)] - coord[known[:, 0]]), ua))
    syn, anti = (np.cos(_TETRAHEDRAL) * ua + sign * np.sin(_TETRAHEDRAL) * in_plane for sign in (1, -1))
    syn_nearer_metal = np.einsum("ij,ij->i", syn, ub) > np.einsum("ij,ij->i", anti, ub)
    off_metal = np.where(syn_nearer_metal[:, None], anti, syn)
    first = np.where(((shape == _CONJUGATED_ON_METAL) & (m == 2) & (reference >= 0))[:, None], off_metal, first)
    # With three known, the apex is equally far from each; a flat centre offers both sides.
    tips = _or(_unit(np.cross(ub - ua, uc - ua)), bisector)
    side = np.einsum("ij,ij->i", tips, ua)
    apex = np.where((side > 0)[:, None], -tips, tips)
    three = (m >= 3)[:, None]
    first = np.where(three, apex, first)
    second = np.where(three, np.where((np.abs(side) < _FLAT)[:, None], -apex, apex), second)
    directions = np.stack([first, second], axis=1)
    position = coord[x][:, None] + d[:, None, None] * directions
    frame = [coord[v][:, None] for v in (x, known[:, 0], known[:, 1])]
    return _measure_icoors(position, *frame)


def _torsion_reference(topo: Topology, x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The heaviest heavy atom bonded to each *a* other than *x*, in *a*'s residue first; -1 where none."""
    pairs = topo.pairs
    heavy = ~topo.is_h[pairs[:, 1]] & topo.finite[pairs[:, 1]]
    symbols, of_symbol = np.unique(topo.element, return_inverse=True)
    number = np.array([ELEMENT_NAME_TO_ATOMIC_NUMBER.get(str(e), 0) for e in symbols])[of_symbol]
    outside = topo.residue[pairs[:, 1]] != topo.residue[pairs[:, 0]]
    table = _sorted_neighbours(topo, pairs[heavy], outside[heavy], -number[pairs[heavy, 1]])
    candidates = table[np.maximum(a, 0)]
    return _first(candidates, (candidates >= 0) & (candidates != x[:, None]) & (a[:, None] >= 0))


def _name_by_template(
    topo: Topology,
    slots: np.ndarray,
    anchor: np.ndarray,
    n_build: np.ndarray,
    m: np.ndarray,
    shape: np.ndarray,
    template: np.ndarray,
    names_at: np.ndarray,
) -> np.ndarray:
    """``[n_parents, 4]`` the slot each hydrogen takes, the one nearest its name's dictionary
    dihedral about the parent's bond, where the dictionary places the frame in one residue.

    Where a parent's hydrogens fill its free slots this only names them; a lone hydrogen on
    a tetrahedral atom with two known neighbours, or three lying flat, takes the side that
    keeps the dictionary's chirality (7N5V DT H1'). *names_at* is ``[n_parents, 3, 3]``, the hydrogens at their dictionary positions.
    """
    choice = np.tile(np.arange(4), (len(anchor), 1))
    residue = topo.residue
    cases = ((1, 2, (m >= 2) & (shape == _TETRA)), (2, 2, (m == 1) | (m == 2)), (3, 3, (m == 1) | (m == 2)))
    for count, among, where in cases:
        group = np.flatnonzero((n_build == count) & where)
        if not len(group):
            continue
        x, a, ref = anchor[group].T
        same = (ref >= 0) & (residue[np.maximum(a, 0)] == residue[x]) & (residue[np.maximum(ref, 0)] == residue[x])
        frame = (template[v][:, None] for v in (x, np.maximum(a, 0), np.maximum(ref, 0)))
        wanted = -_measure_icoors(names_at[group, :count], *frame)[..., 2]
        ok = same & np.isfinite(wanted).all(axis=1)
        if not ok.any():
            continue
        perms = np.array(list(itertools.permutations(range(among), count)))
        gap = np.abs(np.angle(np.exp(1j * (wanted[ok, None, :] + slots[group[ok]][:, perms, 2]))))
        choice[group[ok], :count] = perms[gap.sum(axis=-1).argmin(axis=1)]
    return choice


def plan_by_rule(
    topo: Topology,
    rows: np.ndarray,
    parent: np.ndarray,
    ancestors: np.ndarray,
    geometry: np.ndarray,
    template: np.ndarray,
    template_row: np.ndarray,
) -> None:
    """Fill in ``ancestors`` and ``geometry`` of the plan rows marked *rows*.

    Which free direction each hydrogen takes is fixed by the rule; the dictionary's
    positions (*template* per atom, *template_row* per row) decide which name goes where
    and, at a stereocentre missing a substituent, which side its hydrogen is on.
    """
    rows = np.flatnonzero(rows)
    if not len(rows):
        return
    coord, pairs = topo.coord, topo.pairs
    rows = rows[np.argsort(parent[rows], kind="stable")]
    atoms, first, n_build = np.unique(parent[rows], return_index=True, return_counts=True)
    rank = np.arange(len(rows)) - np.repeat(first, n_build)
    of_row = np.repeat(np.arange(len(atoms)), n_build)

    within = topo.residue[pairs[:, 1]] == topo.residue[pairs[:, 0]]
    kind = np.where(topo.metal[pairs[:, 1]], 3, np.where(topo.is_h[pairs[:, 1]], 2, np.where(within, 0, 1)))
    usable = topo.finite[pairs[:, 1]]
    known = _sorted_neighbours(topo, pairs[usable], kind[usable])[atoms]
    m = (known >= 0).sum(axis=1)
    known = np.pad(known, ((0, 0), (0, 3)), constant_values=-1)[:, :3]
    shape = _hydrogen_parent_geometries(topo, atoms, m, n_build)
    symbols, of_symbol = np.unique(topo.element[atoms], return_inverse=True)
    d = np.array([_BOND_LENGTH.get(str(e), 1.0) for e in symbols])[of_symbol]

    slots = np.full((len(atoms), 4, 3), np.nan)
    measured = m >= 2
    if measured.any():
        reference = _torsion_reference(topo, atoms[measured], known[measured, 0])
        slots[measured, :2] = _measured_slots(
            coord, shape[measured], m[measured], atoms[measured], known[measured], d[measured], reference
        )
    for known_count, directions_by_shape in ((1, _ONE_KNOWN), (0, _NONE_KNOWN)):
        for shape_kind, directions in directions_by_shape.items():
            select = (m == known_count) & (shape == shape_kind)
            for k, (theta, phi) in enumerate(directions):
                slots[select, k] = np.stack([d[select], np.full(select.sum(), theta), np.full(select.sum(), phi)], -1)

    anchor = np.stack([atoms, known[:, 0], known[:, 1]], axis=1)
    one = m == 1
    anchor[one, 2] = _torsion_reference(topo, atoms[one], known[one, 0])
    anchor[m == 0, 1:] = -1

    names_at = np.full((len(atoms), 3, 3), np.nan)
    listed = rank < 3
    names_at[of_row[listed], rank[listed]] = template_row[rows[listed]]
    choice = _name_by_template(topo, slots, anchor, n_build, m, shape, template, names_at)
    ancestors[rows] = anchor[of_row]
    geometry[rows] = slots[of_row, choice[of_row, rank]]


def backups(
    topo: Topology, built: np.ndarray, ancestors: np.ndarray, placed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A second frame for each built hydrogen, avoiding its great-grandparent, and its geometry there.

    The frame swaps the great-grandparent for the first other heavy atom bonded to the
    parent or grandparent; the geometry is measured on the hydrogen as the plan places it.
    """
    n = len(topo.coord)
    backup = np.full_like(ancestors, -1)
    geometry = np.zeros((len(ancestors), 3))
    x, a, b = ancestors.T
    rows = np.flatnonzero(built & (a >= 0) & (a < n))
    if not len(rows):
        return backup, geometry
    pairs = topo.pairs
    heavy = ~topo.is_h[pairs[:, 1]] & topo.finite[pairs[:, 1]]
    table = _sorted_neighbours(topo, pairs[heavy], topo.residue[pairs[heavy, 1]] != topo.residue[pairs[heavy, 0]])
    candidates = np.concatenate([table[x[rows]], table[a[rows]]], axis=1)
    others = (candidates != x[rows, None]) & (candidates != a[rows, None]) & (candidates != b[rows, None])
    c = _first(candidates, (candidates >= 0) & others)
    rows, c = rows[c >= 0], c[c >= 0]
    backup[rows] = np.stack([x[rows], a[rows], c], axis=1)
    geometry[rows] = _measure_icoors(placed[rows], topo.coord[x[rows]], topo.coord[a[rows]], topo.coord[c])
    return backup, geometry


# --- place ---
_LAB_X = np.array([1.0, 0.0, 0.0])
# Rows of one level are independent; building them this many (hydrogens x samples) at a
# time bounds the temporaries without changing any value.
_CHUNK = 1 << 15


def _frame(known: np.ndarray, ancestors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ancestor positions ``[..., rows, 3]``; a missing grandparent is a lab axis off the parent,
    and a missing great-grandparent the grandparent itself, which leaves the dihedral free."""
    parent = known[..., np.maximum(ancestors[:, 0], 0), :]
    grandparent = np.where(
        (ancestors[:, 1] >= 0)[:, None], known[..., np.maximum(ancestors[:, 1], 0), :], parent - _LAB_X
    )
    great = np.where((ancestors[:, 2] >= 0)[:, None], known[..., np.maximum(ancestors[:, 2], 0), :], grandparent)
    return parent, grandparent, great


def build_levels(
    coord: np.ndarray,
    ancestors: np.ndarray,
    geometry: np.ndarray,
    level: np.ndarray,
    kept_index: np.ndarray,
    backup_ancestors: np.ndarray | None = None,
    backup_geometry: np.ndarray | None = None,
) -> np.ndarray:
    """Hydrogen positions ``[..., n_H, 3]`` from atom positions ``[..., n_atoms, 3]``, one dependency level at a
    time; a row whose frame lacks a position is built from its backup frame, else left NaN."""
    n = coord.shape[-2]
    lead = coord.shape[:-2]
    known = np.empty((*lead, n + len(level), 3))
    known[..., :n, :], known[..., n:, :] = coord, np.nan
    kept = np.flatnonzero(kept_index >= 0)
    known[..., n + kept, :] = coord[..., kept_index[kept], :]
    step = max(1, _CHUNK // max(1, int(np.prod(lead))))
    by_level = (np.flatnonzero(level == depth) for depth in range(1, int(level.max(initial=0)) + 1))
    for rows in (at[i : i + step] for at in by_level for i in range(0, len(at), step)):
        placed = build_coordinates(*_frame(known, ancestors[rows]), geometry[rows])
        if backup_ancestors is not None:
            usable = backup_ancestors[rows, 0] >= 0
            missing = ~np.isfinite(placed).all(axis=-1) & usable
            if missing.any():
                backup = build_coordinates(*_frame(known, backup_ancestors[rows]), backup_geometry[rows])
                placed = np.where(missing[..., None], backup, placed)
        known[..., n + rows, :] = placed
    return known[..., n:, :]


# --- plan ---
# A table frame atom is a heavy atom of the residue, one of its hydrogens, or the previous
# residue's connection atom.
_HEAVY, _ROW, _DOWN = 0, 1, 2


@dataclass(frozen=True)
class _Recipe:
    """One residue's rows, the same for every residue with its code, heavy atoms and counts."""

    parent: np.ndarray  # [r] heavy position in the residue
    name: np.ndarray  # [r]
    template_heavy: np.ndarray  # [n_heavy, 3] dictionary positions, NaN where it has none
    template_row: np.ndarray  # [r, 3]
    table: np.ndarray  # [r] the table describes the row's parent as it is
    frame: np.ndarray  # [r, 2, 2] (kind, index) of grandparent and great-grandparent
    geometry: np.ndarray  # [r, 3]


@_cache_ccd_lookup
def _recipe(res_name: str, heavy_names: tuple, elements: tuple, totals: tuple, use_table: bool) -> _Recipe:
    """The rows of a residue whose heavy atom ``i`` is named ``heavy_names[i]`` and carries ``totals[i]``."""
    parents = [(i, n, e, t) for i, (n, e, t) in enumerate(zip(heavy_names, elements, totals, strict=True)) if t]
    groups = residue_hydrogen_names(res_name, frozenset(heavy_names), tuple(p[1:] for p in parents))
    parent = np.array([p[0] for p, names in zip(parents, groups, strict=True) for _ in names], dtype=np.intp)
    name = [n for names in groups for n in names]
    template, nan = _ccd_template(res_name), np.full(3, np.nan)
    tpl = {}
    if template is not None:
        finite = np.isfinite(template.coord).all(axis=-1)
        tpl = dict(zip(template.atom_name[finite], template.coord[finite], strict=True))
    r = len(name)
    table, frame, geometry = np.zeros(r, bool), np.zeros((r, 2, 2), np.intp), np.zeros((r, 3))
    entry = select_entry(res_name, frozenset(name)) if use_table else None
    if entry is not None:
        rows = {row[0]: row for row in entry}
        heavy_pos = {n: i for i, n in enumerate(heavy_names)}
        row_pos = {n: i for i, n in enumerate(name)}
        for i, (p, n) in enumerate(zip(parent, name, strict=True)):
            row = rows.get(n)
            if row is None or row[1] != heavy_names[p]:
                continue
            refs = [(_HEAVY, heavy_pos[a]) if a in heavy_pos else (_ROW, row_pos.get(a, -1)) for a in row[2:4]]
            refs = [(_DOWN, 0) if a == "down" else ref for a, ref in zip(row[2:4], refs, strict=True)]
            if all(index >= 0 for _, index in refs):
                table[i], frame[i], geometry[i] = True, refs, row[4:7]
    return _Recipe(
        parent=parent,
        name=np.array(name, dtype=str),
        template_heavy=np.array([tpl.get(n, nan) for n in heavy_names]).reshape(-1, 3),
        template_row=np.array([tpl.get(n, nan) for n in name]).reshape(-1, 3),
        table=table,
        frame=frame,
        geometry=geometry,
    )


def _polymer_atoms(atom_array: AtomArray, heavy_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks of each residue's entering and leaving polymer atoms (N and C, P and O3')."""
    codes, of_code = np.unique(atom_array.res_name, return_inverse=True)
    in_heavy = np.isin(codes, atom_array.res_name[heavy_idx])
    names = [get_polymerization_atoms(str(c)) if h else (None, None) for c, h in zip(codes, in_heavy, strict=True)]
    leaving, entering = np.array(names, dtype=object).reshape(-1, 2)[of_code].T
    return atom_array.atom_name == entering, atom_array.atom_name == leaving


def _prepare_hydrogen_coordinates(atom_array: AtomArray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every hydrogen of a protonation state: its name, its heavy atom, the input hydrogen it keeps (-1 where it is
    built) and its position.

    Each heavy atom not ``skip_hydrogen_placement`` gets ``nhyd`` hydrogens besides those the atom array keeps. A standard
    residue takes TMol's table where its frame atoms are present and its parent is drawn as the template
    draws it; any other hydrogen takes the free direction its parent's hybridization leaves, a rotatable
    group staggered. Names follow the dictionary.
    """
    if isinstance(atom_array, AtomArrayStack):
        raise TypeError("add_hydrogens takes one protonation state (an AtomArray), e.g. stack[0].")
    missing = {"nhyd", "skip_hydrogen_placement"} - set(atom_array.get_annotation_categories())
    if missing:
        raise ValueError(
            f"Not a protonation state: no {', '.join(sorted(missing))} annotation. Pass the array through "
            "assign_hydrogens first, or call add_hydrogens(atom_array)."
        )
    n = atom_array.array_length()
    coord = atom_array.coord.astype(np.float64)
    symbols, of_symbol = np.unique(atom_array.element.astype(str), return_inverse=True)
    element = np.char.upper(symbols)[of_symbol]
    is_h = np.isin(element, HYDROGEN_LIKE_SYMBOLS)
    bonds = atom_array.bonds.as_array().astype(np.intp)
    pairs = np.concatenate([bonds[:, :2], bonds[:, 1::-1]])

    heavy_idx = np.flatnonzero(~is_h)
    starts = struc.get_residue_starts(atom_array[heavy_idx], add_exclusive_stop=True)
    codes = atom_array.res_name[heavy_idx].astype(str)
    residue = np.full(n, -1, dtype=np.intp)
    residue[heavy_idx] = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    kept, kept_on = pairs[is_h[pairs[:, 0]] & ~is_h[pairs[:, 1]]].T
    residue[kept] = residue[kept_on]
    total = np.where(~is_h & ~atom_array.skip_hydrogen_placement, atom_array.nhyd, 0).astype(np.intp)
    total += np.bincount(kept_on, minlength=n)

    # Four covalent neighbours leave no direction for a metal contact. Keep the recorded bond,
    # but do not use it to place hydrogens or to disqualify the residue's covalent template.
    metal = is_metal(atom_array)
    covalent_heavy = (np.tile(bonds[:, 2], 2) != struc.BondType.COORDINATION) & ~is_h[pairs[:, 1]]
    covalent_neighbours = np.bincount(pairs[covalent_heavy, 0], minlength=n) + total
    saturated = (covalent_neighbours == 4) & (total > 0)
    ignore_contact = (bonds[:, 2] == struc.BondType.COORDINATION) & np.any(
        saturated[bonds[:, :2]] & metal[bonds[:, 1::-1]], axis=1
    )
    if ignore_contact.any():
        affected = np.unique(bonds[ignore_contact, :2])
        logger.warning(
            "Ignored metal contacts as hydrogen-placement constraints for %s: four covalent neighbours "
            "already fill their geometry. Assigned hydrogens, charges and recorded bonds are unchanged.",
            _format_atom_examples(atom_array, affected[saturated[affected] & ~metal[affected]]),
        )
        bonds = bonds[~ignore_contact]
        pairs = np.concatenate([bonds[:, :2], bonds[:, 1::-1]])

    custom = get_custom_ccd_entries()
    has_rows = np.bincount(residue[heavy_idx], weights=total[heavy_idx] != 0, minlength=max(len(starts) - 1, 0)) > 0
    with_rows = np.flatnonzero(has_rows)
    # A residue's code and the bytes of its (name, element, total) rows identify its recipe.
    records = np.rec.fromarrays([atom_array.atom_name[heavy_idx], element[heavy_idx], total[heavy_idx]]).tobytes()
    width, bounds = len(records) // max(len(heavy_idx), 1), starts.tolist()
    codes = codes[starts[:-1]].tolist()
    recipes, by_key = [], {}
    for r in with_rows.tolist():
        start, stop = bounds[r], bounds[r + 1]
        key = (codes[r], records[start * width : stop * width])
        if key not in by_key:
            idx = heavy_idx[start:stop]
            names, elements, totals = (tuple(a[idx].tolist()) for a in (atom_array.atom_name, element, total))
            by_key[key] = _recipe(key[0], names, elements, totals, key[0] not in custom)
        recipes.append(by_key[key])
    counts = np.array([len(rec.name) for rec in recipes], dtype=np.intp)
    row_residue = np.repeat(with_rows, counts)
    row_base = np.repeat(np.cumsum(counts) - counts, counts)

    def cat(field: str, width: int = 0) -> np.ndarray:
        shape = (0, width) if width else (0,)
        return np.concatenate([getattr(rec, field) for rec in recipes]) if recipes else np.zeros(shape)

    first_heavy = starts[row_residue]
    parent = heavy_idx[first_heavy + cat("parent").astype(np.intp)]
    name = cat("name").astype(str)
    template = np.full((n, 3), np.nan)
    if recipes:
        template[heavy_idx[np.repeat(has_rows, np.diff(starts))]] = cat("template_heavy", 3)
    n_rows = len(name)

    kept_index = np.full(n_rows, -1, dtype=np.intp)
    # A kept hydrogen takes the row of its name where it can, else the next free row on its atom.
    if len(kept):
        row_of = {key: i for i, key in enumerate(zip(parent.tolist(), name.tolist(), strict=True))}
        free: dict[int, list[int]] = {}
        for i, p in enumerate(parent.tolist()):
            free.setdefault(p, []).append(i)
        pending = []
        for h, p in sorted(zip(kept.tolist(), kept_on.tolist(), strict=True)):
            i = row_of.get((p, str(atom_array.atom_name[h])))
            if i is None or kept_index[i] >= 0:
                pending.append((h, p))
                continue
            kept_index[i] = h
            free[p].remove(i)
        for h, p in pending:
            kept_index[free[p].pop(0)] = h
    fixed = kept_index >= 0
    ancestors = np.full((n_rows, 3), -1, dtype=np.intp)
    ancestors[:, 0] = parent

    # The previous residue's connection atom is the partner of this residue's entering atom
    # that is its own residue's leaving atom, whatever the atom order.
    enter, leave = _polymer_atoms(atom_array, heavy_idx)
    link = pairs[enter[pairs[:, 0]] & leave[pairs[:, 1]] & (residue[pairs[:, 0]] != residue[pairs[:, 1]])]
    down = np.full(len(starts), -1, dtype=np.intp)
    down[residue[link[:, 0]]] = link[:, 1]

    # 1CS4 (metalc): a parent bonded to a metal or to another residue other than through the polymer
    # link is not the residue the table describes.
    heavy_pairs = pairs[~is_h[pairs[:, 0]] & ~is_h[pairs[:, 1]]]
    linked = (enter[heavy_pairs[:, 0]] & leave[heavy_pairs[:, 1]]) | (
        leave[heavy_pairs[:, 0]] & enter[heavy_pairs[:, 1]]
    )
    foreign = (metal[heavy_pairs[:, 1]] | (residue[heavy_pairs[:, 0]] != residue[heavy_pairs[:, 1]])) & ~linked
    has_foreign = np.bincount(heavy_pairs[foreign, 0], minlength=n) > 0

    # 2CDB: a centre drawn with the other hand than its template's takes the rule, which reads it off the structure.
    inner = heavy_pairs[residue[heavy_pairs[:, 0]] == residue[heavy_pairs[:, 1]]]
    centre = np.flatnonzero(np.bincount(inner[:, 0], minlength=n) == 3)
    arms = inner[np.isin(inner[:, 0], centre)]
    arms = arms[np.lexsort((arms[:, 1], arms[:, 0])), 1].reshape(-1, 3)

    def hand(xyz: np.ndarray) -> np.ndarray:
        a, b, c = (xyz[arms[:, k]] - xyz[centre] for k in range(3))
        return np.einsum("ij,ij->i", np.cross(a, b), c)

    mirrored = np.zeros(n, dtype=bool)
    mirrored[centre] = hand(coord) * hand(template) < 0

    finite = np.isfinite(coord).all(axis=-1)
    table = cat("table").astype(bool) & ~fixed & finite[parent] & ~has_foreign[parent] & ~mirrored[parent]
    frame = cat("frame", 2).astype(np.intp).reshape(-1, 2, 2)
    for col in range(2):
        kind, index = frame[:, col, 0], frame[:, col, 1]
        heavy_ref = heavy_idx[np.minimum(first_heavy + index, len(heavy_idx) - 1)]
        ref = np.where(kind == _HEAVY, heavy_ref, np.where(kind == _ROW, n + row_base + index, down[row_residue]))
        atom_ref = (ref < n) & (kind != _ROW)
        table &= ~atom_ref | ((ref >= 0) & finite[np.clip(ref, 0, n - 1)])
        ancestors[:, col + 1] = np.where(table, ref, ancestors[:, col + 1])
    table &= ~(np.bincount(parent[~table], minlength=n) > 0)[parent]
    geometry = np.where(table[:, None], cat("geometry", 3).reshape(-1, 3), 0.0)
    ancestors[~table & ~fixed, 1:] = -1

    name_rank = np.unique(atom_array.atom_name.astype(str), return_inverse=True)[1]
    topology = Topology(coord, element, is_h, finite, pairs, np.tile(bonds[:, 2], 2), residue, metal, name_rank)
    plan_by_rule(topology, ~fixed & ~table, parent, ancestors, geometry, template, cat("template_row", 3))

    level = np.where(fixed, 0, 1)
    for _ in range(n_rows + 1):
        refs = ancestors - n
        deepest = np.where(refs >= 0, level[np.clip(refs, 0, max(n_rows - 1, 0))], 0).max(axis=1, initial=0)
        updated = np.where(fixed, 0, deepest + 1)
        if (updated == level).all():
            break
        level = updated
    else:
        raise ValueError("The hydrogen plan has a cycle among its ancestors.")

    placed = build_levels(coord, ancestors, geometry, level, kept_index)
    backup_ancestors, backup_geometry = backups(topology, ~fixed, ancestors, placed)
    return (
        name,
        parent,
        kept_index,
        build_levels(coord, ancestors, geometry, level, kept_index, backup_ancestors, backup_geometry),
    )


# --- insert ---
logger = logging.getLogger(__name__)

# Values a new hydrogen takes for these atom-level annotations.
_H_DEFAULTS: dict[str, object] = {
    "b_factor": np.nan,
    "stereo": "N",
    "is_aromatic": False,
    "is_backbone_atom": False,
    "skip_hydrogen_placement": False,
    "tautomer_free": False,
}
_RESIDUE_AND_CHAIN = frozenset(RESIDUE_LEVEL_ANNOTATIONS + CHAIN_LEVEL_ANNOTATIONS)


def _append_hydrogens(atoms: AtomArray, parent: np.ndarray, names: np.ndarray) -> tuple[AtomArray, list[str]]:
    """*atoms* followed by hydrogens singly bonded to ``atoms[parent]``, and the annotations to recompute.

    A hydrogen takes charge 0, a default from :data:`_H_DEFAULTS`, and its parent's value for
    any annotation AtomWorks does not compute. Those it computes are left out of the appended.
    The hydrogens have no position.
    """
    n_atoms, n_h = atoms.array_length(), len(parent)
    block = AtomArrayPlus(n_h)
    block.element[:] = "H"
    block.chain_id[:] = atoms.chain_id[parent]
    block.res_id[:] = atoms.res_id[parent]
    block.res_name[:] = atoms.res_name[parent]
    block.atom_name = names
    for annot in atoms.get_annotation_categories():
        values = atoms.get_annotation(annot)
        if annot == "charge":
            block.set_annotation(annot, np.zeros(n_h, dtype=values.dtype))
        elif annot in _RESIDUE_AND_CHAIN and annot not in ("chain_id", "res_id", "res_name"):
            block.set_annotation(annot, values[parent])
    for name in get_annotation_categories(atoms, n_body=2):
        annotation = get_annotation(atoms, name, n_body=2)
        block.set_annotation(name, AnnotationList2D(n_h, [], annotation.values[:0]), n_body=2)
    appended = concatenate_atom_array_plus([atoms.copy(), block], on_annotation_mismatch_policy="drop")

    recompute, inherit = [], []
    for annot in atoms.get_annotation_categories():
        if annot in appended.get_annotation_categories() or annot in ("nhyd", "atom_id"):
            continue
        standard = next((a for a in STANDARD_ANNOTATIONS if annot in (a.name, a.full_name, *a.aliases)), None)
        if standard is not None:
            (recompute if standard.level == Level.ATOM else inherit).append(annot)
        elif annot in ANNOTATOR_REGISTRY:
            recompute.append(annot)
        elif annot in _H_DEFAULTS:
            values = atoms.get_annotation(annot)
            appended.set_annotation(annot, np.concatenate([values, np.full(n_h, _H_DEFAULTS[annot], values.dtype)]))
        else:
            inherit.append(annot)
    new_bonds = np.stack([parent, n_atoms + np.arange(n_h), np.full(n_h, int(struc.BondType.SINGLE))], axis=1)
    appended.bonds = struc.BondList(len(appended), np.vstack([appended.bonds.as_array(), new_bonds]))
    for annot in inherit:
        values = atoms.get_annotation(annot)
        appended.set_annotation(annot, np.concatenate([values, values[parent]]))
    return appended, recompute


def place_hydrogens(atom_array: AtomArray) -> AtomArray:
    """Name, position and insert the hydrogens :func:`assign_hydrogens` decided, built on ``atom_array.coord``.

    Args:
        atom_array: A protonation state, as :func:`assign_hydrogens` returns it.

    Returns:
        The atom array with every hydrogen explicit, named and bonded after its residue's heavy atoms (a
        kept hydrogen joins its parent's residue). ``nhyd`` is
        zero except on ``skip_hydrogen_placement`` heavy atoms. Its atoms keep their ``atom_id`` and
        annotations; added hydrogens are numbered on from the largest ``atom_id`` and take their
        parent's value of any annotation AtomWorks does not compute.

    Raises:
        TypeError: If *atom_array* is an ``AtomArrayStack``.
        ValueError: If *atom_array* is not a protonation state or needs unsupported placement geometry.
    """
    names, parents, kept_index, h_coord = _prepare_hydrogen_coordinates(atom_array)
    n = atom_array.array_length()
    coord = atom_array.coord
    built = kept_index < 0
    unbuilt = built & ~np.isfinite(h_coord).all(axis=-1) & np.isfinite(coord[parents]).all(axis=-1)
    # 3UQ: coincident frame atoms leave a hydrogen without a position.
    if unbuilt.any():
        logger.warning(
            "Left the hydrogens of %s without a position: two atoms of their frame coincide or lack a position.",
            _format_atom_examples(atom_array, np.unique(parents[unbuilt])),
        )
    protonated, recompute = _append_hydrogens(atom_array, parents[built], names[built])
    hydrogen_index = kept_index.copy()
    hydrogen_index[built] = np.arange(n, protonated.array_length())
    protonated.atom_name[hydrogen_index] = names
    for annot in _RESIDUE_AND_CHAIN.intersection(protonated.get_annotation_categories()):
        values = protonated.get_annotation(annot)
        values[kept_index[~built]] = values[parents[~built]]

    is_h = np.isin(protonated.element, HYDROGEN_LIKE_SYMBOLS)
    nhyd = np.zeros(protonated.array_length(), dtype=atom_array.nhyd.dtype)
    nhyd[:n] = np.where(atom_array.skip_hydrogen_placement & ~is_h[:n], atom_array.nhyd, 0)
    protonated.set_annotation("nhyd", nhyd)
    rank = np.zeros(protonated.array_length(), dtype=np.intp)
    rank[~is_h] = struc.get_all_residue_positions(protonated[~is_h])
    rank[hydrogen_index] = rank[parents]
    within_residue = np.arange(protonated.array_length())
    within_residue[hydrogen_index] = np.arange(len(parents))
    order = np.lexsort((within_residue, is_h, rank))
    protonated = protonated[order]
    from_state = order < n
    state_ids = get_annotation(atom_array, "atom_id", default=np.arange(n))
    atom_id = np.zeros(protonated.array_length(), dtype=np.result_type(state_ids, np.intp))
    atom_id[from_state] = state_ids[order[from_state]]
    atom_id[~from_state] = state_ids.max(initial=-1) + 1 + np.arange(np.count_nonzero(~from_state))
    protonated.set_annotation("atom_id", atom_id)
    protonated.coord = np.concatenate([coord, h_coord[built]])[order]
    ensure_annotations(protonated, *recompute)
    for annot in recompute:
        values = protonated.get_annotation(annot)
        values[from_state] = atom_array.get_annotation(annot)[order[from_state]]
        protonated.set_annotation(annot, values)
    return protonated


def add_hydrogens(atom_array: AtomArray, *, ph: float = 7.4) -> AtomArray:
    """Protonate *atom_array* at *ph*: :func:`assign_hydrogens`, then :func:`place_hydrogens`."""
    return place_hydrogens(assign_hydrogens(atom_array, ph=ph))
