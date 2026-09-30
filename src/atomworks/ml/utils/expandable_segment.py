"""Expandable segment sentinels for AtomArrayPlus structures.

An *expandable segment sentinel* is a single-atom placeholder residue that
represents a stretch of unknown length between two anchoring
residues.  Sentinels can be inserted, realized into concrete (but
coordinate-free) atom stubs, and collapsed back into sentinels.

Sentinel identification uses the mask derived from ``annotation_expsegmin_1_residue``:

* ``mask == False``  →  regular residue (not an expandable segment)
* ``mask == True``, ``res_name != MASKED``  →  unrealized sentinel (e.g. ``"<5-15>"``)
* ``mask == True``, ``res_name == MASKED``   →  realized segment atom stub

Public API
----------
- :py:func:`insert_expandable_segment`
- :py:func:`realize_expandable_segments`
- :py:func:`collapse_expandable_segments`
"""

import logging

import biotite.structure as struc
import numpy as np

from atomworks.constants import (
    DEFAULT_ALTLOC_ID,
    DEFAULT_B_FACTOR,
    DEFAULT_CHARGE,
    DEFAULT_INS_CODE,
    DEFAULT_OCCUPANCY,
    MASKED,
    MOLECULE_LEVEL_ANNOTATIONS,
    PN_UNIT_LEVEL_ANNOTATIONS,
    UNKNOWN_ATOM_NAME,
    UNKNOWN_COORD_VALUE,
    UNKNOWN_ELEMENT,
)
from atomworks.enums import ChainType, ChainTypeInfo
from atomworks.io.transforms import atom_array as io_ta
from atomworks.io.utils.annotator import ANNOTATOR_REGISTRY, ensure_annotations
from atomworks.io.utils.atom_array_plus import (
    AtomArrayPlus,
    as_atom_array_plus,
    concatenate_atom_array_plus,
    insert_atoms,
)
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.standard_annotations import S_SEGMAX, S_SEGMIN
from atomworks.io.utils.standard_annotations.base import STANDARD_ANNOTATIONS
from atomworks.ml.conditions import C_CTR, C_NTR
from atomworks.ml.transforms.atom_array import (
    add_global_atom_id_annotation,
    add_global_token_id_annotation,
    get_within_group_res_idx,
    get_within_group_source_res_idx,
    get_within_poly_res_idx,
)

logger = logging.getLogger("atomworks.ml")

# Internal annotation used to uniquely identify otherwise-identical expandable segment sentinels.
_SEG_UID_ANNOT = "_expseg_uid"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reset_present_standard_annotations_on_sentinel(
    sentinel: AtomArrayPlus,
    existing_annots: set[str],
) -> set[str]:
    """Reset all present StandardAnnotations on sentinel to defaults (except expandable segment annotations)."""
    handled: set[str] = set()
    for sa_cls in STANDARD_ANNOTATIONS:
        if sa_cls in {S_SEGMIN, S_SEGMAX}:
            continue

        field_names = (sa_cls.full_name, *sa_cls.aliases)
        if any(name in existing_annots for name in field_names):
            # Delete the copied value before generating a fresh structure-dependent default.
            present = sentinel.get_annotation_categories(n_body=sa_cls.n_body)
            for name in field_names:
                if name in present:
                    sentinel.del_annotation(name, n_body=sa_cls.n_body)
            sa_cls.set_annotation(sentinel, sa_cls.default_annotation(sentinel))
            handled.add(sa_cls.full_name)

    return handled


def _ensure_seg_uids(array: AtomArrayPlus, is_sentinel: np.ndarray) -> None:
    """Assign a fresh, unique ``_SEG_UID_ANNOT`` to any sentinel atom missing one."""
    if _SEG_UID_ANNOT not in array.get_annotation_categories():
        array.set_annotation(_SEG_UID_ANNOT, np.full(array.array_length(), -1, dtype=np.int64))
    uid_ann = array.get_annotation(_SEG_UID_ANNOT)
    missing = is_sentinel & (uid_ann == -1)
    if np.any(missing):
        next_uid = int(uid_ann.max()) + 1
        for idx in np.where(missing)[0]:
            uid_ann[idx] = next_uid
            next_uid += 1
        array.set_annotation(_SEG_UID_ANNOT, uid_ann)


def _make_expandable_segment_sentinel_from_copied_atom(
    array: AtomArrayPlus,
    source_atom_idx: int,
    seg_min: int,
    seg_max: int,
    chain_id: str,
    sentinel_res_id: int,
    seg_uid: int,
    chain_type: ChainType | None = None,
) -> tuple[AtomArrayPlus, set[str]]:
    """Create a 1-atom sentinel by copying an existing atom and normalizing fields."""
    sentinel = as_atom_array_plus(array[source_atom_idx : source_atom_idx + 1].copy())
    existing_annots = set(array.get_annotation_categories())
    handled: set[str] = set()

    # Core identity fields.
    sentinel.chain_id[:] = chain_id
    seg_name = f"{seg_min}-{seg_max}"
    _ensure_res_name_width(sentinel, max(9, len(seg_name)))
    sentinel.res_name[:] = seg_name
    sentinel.atom_name[:] = UNKNOWN_ATOM_NAME
    sentinel.element[:] = UNKNOWN_ELEMENT
    sentinel.coord[:] = UNKNOWN_COORD_VALUE
    sentinel.hetero[:] = True
    sentinel.res_id[:] = int(sentinel_res_id)
    handled |= {"chain_id", "res_name", "atom_name", "element", "coord", "hetero", "res_id"}

    # Common parse-level optional fields.
    if "ins_code" in existing_annots:
        sentinel.ins_code[:] = DEFAULT_INS_CODE
        handled.add("ins_code")
    if "occupancy" in existing_annots:
        sentinel.occupancy[:] = DEFAULT_OCCUPANCY
        handled.add("occupancy")
    if "b_factor" in existing_annots:
        sentinel.b_factor[:] = DEFAULT_B_FACTOR
        handled.add("b_factor")
    if "altloc_id" in existing_annots:
        sentinel.altloc_id[:] = DEFAULT_ALTLOC_ID
        handled.add("altloc_id")
    if "charge" in existing_annots:
        sentinel.charge[:] = DEFAULT_CHARGE
        handled.add("charge")

    if chain_type is not None:
        if "chain_type" in existing_annots:
            sentinel.chain_type[:] = np.int8(chain_type)
            handled.add("chain_type")
        if "is_polymer" in existing_annots:
            sentinel.is_polymer[:] = chain_type.is_polymer()
            handled.add("is_polymer")

    # Update the entity id if creating a new chain. Use the global max (not the
    # source atom's own entity id) so the new id is guaranteed unused regardless
    # of whether entity ids happen to increase monotonically with array/chain order.
    if "label_entity_id" in existing_annots and chain_id != array.chain_id[source_atom_idx]:
        new_entity_id = int(array.label_entity_id.max()) + 1
        sentinel.label_entity_id[:] = new_entity_id

    # Set expandable segment annotations BEFORE the standard-annotation reset, so that
    # Atomize.default_annotation can see SEGMIN/SEGMAX and treat this sentinel as a standard polymer (not atomized).
    S_SEGMIN.set_annotation(sentinel, np.array([seg_min], dtype=int))
    S_SEGMAX.set_annotation(sentinel, np.array([seg_max], dtype=int))
    sentinel.set_annotation(_SEG_UID_ANNOT, np.array([seg_uid], dtype=np.int64))
    handled |= {S_SEGMIN.full_name, S_SEGMAX.full_name, _SEG_UID_ANNOT}

    # Reset all standard annotations that are present on source array.
    handled |= _reset_present_standard_annotations_on_sentinel(sentinel, existing_annots)

    return sentinel, handled


def _compute_insert_res_id(
    array: AtomArrayPlus,
    *,
    insert_position: int,
    chain_identifier_key: str,
    chain_identifier_value: str | int | np.integer,
) -> tuple[int, bool]:
    """Compute sentinel ``res_id`` and whether downstream shifting is needed.

    Returns:
        ``(sentinel_res_id, needs_downstream_shift)``.  When a gap exists between
        the flanking residues the sentinel is placed in the gap (``next_res_id - 1``)
        and no shifting is required.  Otherwise ``prev_res_id + 1`` is used and the
        caller must shift downstream residues up by one.
    """
    chain_mask = array.get_annotation(chain_identifier_key) == chain_identifier_value
    chain_indices = np.where(chain_mask)[0]
    if len(chain_indices) == 0:
        raise ValueError("Cannot compute insert res_id: selected chain instance has no atoms.")

    # Assert res_ids are monotonically increasing within chain
    assert np.all(np.diff(array.res_id[chain_indices]) >= 0), "res_ids must be monotonically increasing within chain"

    n_chain_atoms_before_insert = int(np.sum(chain_indices < insert_position))
    n_chain_atoms_at_or_after = len(chain_indices) - n_chain_atoms_before_insert

    if n_chain_atoms_before_insert > 0:
        prev_atom_idx = int(chain_indices[n_chain_atoms_before_insert - 1])
        prev_res_id = int(array.res_id[prev_atom_idx])
        if n_chain_atoms_at_or_after > 0:
            next_atom_idx = int(chain_indices[n_chain_atoms_before_insert])
            next_res_id = int(array.res_id[next_atom_idx])
            if next_res_id > prev_res_id + 1:
                # Gap between flanking residues: fit sentinel in without shifting downstream.
                return next_res_id - 1, False
        return prev_res_id + 1, True

    # Inserting before the first residue of this chain instance.
    first_res_id = int(array.res_id[chain_indices[0]])
    if first_res_id > 0:
        return first_res_id - 1, False
    return first_res_id, True


def _shift_chain_res_ids_after_insert(
    result: AtomArrayPlus,
    *,
    sentinel_idx: int,
    sentinel_res_id: int,
    chain_identifier_key: str,
    chain_identifier_value: str | int | np.integer,
) -> None:
    """Shift downstream residues in the same chain instance by +1."""
    chain_mask = result.get_annotation(chain_identifier_key) == chain_identifier_value
    downstream_mask = np.arange(result.array_length()) > sentinel_idx
    to_shift = chain_mask & downstream_mask & (result.res_id >= sentinel_res_id)
    result.res_id[to_shift] = result.res_id[to_shift] + 1


def _ensure_res_name_width(array: AtomArrayPlus, min_width: int = 9) -> None:
    """Widen the ``res_name`` annotation dtype in-place if it is narrower than *min_width* characters."""
    current_width = array.res_name.dtype.itemsize // 4
    if current_width < min_width:
        array.set_annotation("res_name", array.res_name.astype(f"<U{min_width}"))


def _get_blindly_copied_annotations(
    original_annots: set[str],
    explicitly_handled: set[str],
) -> list[str]:
    """Get a list of annotations copied from source atom without explicit recomputation logic."""
    rederived_or_registry = {
        "atom_id",
        "token_id",
        "within_chain_res_idx",
        "within_chain_source_res_idx",
        "within_poly_res_idx",
        "chain_entity",
        "chain_iid",
        *MOLECULE_LEVEL_ANNOTATIONS,
        *PN_UNIT_LEVEL_ANNOTATIONS,
        *ANNOTATOR_REGISTRY.keys(),
    }
    standard_annotation_names = set(STANDARD_ANNOTATIONS.get_field_names(include_aliases=True))
    known_safe = explicitly_handled | rederived_or_registry | standard_annotation_names
    blindly_copied = sorted(original_annots - known_safe)

    return blindly_copied


def _find_flanking_bond_atom_idxs(
    array: AtomArrayPlus,
    sentinel_idx: int,
    chain_key: str,
) -> tuple[int | None, int | None]:
    """Return the global indices of the polymer-bond endpoints flanking a sentinel.

    A neighbouring expandable-segment placeholder stands in for its missing
    polymerisation atom. This keeps separately named adjacent segments connected.

    Args:
        array: Structure containing the sentinel.
        sentinel_idx: Index of the sentinel atom.
        chain_key: Annotation name used to identify chain instances (``"chain_iid"`` or ``"chain_id"``).

    Returns:
        ``(prev_terminal_idx, next_start_idx)`` — either may be ``None`` if the sentinel
        is at a chain boundary or the chain type is not polymer.
    """
    if "chain_type" not in array.get_annotation_categories():
        return None, None
    sentinel_chain_type = ChainType(int(array.chain_type[sentinel_idx]))
    bond_atom_names = ChainTypeInfo.ATOMS_AT_POLYMER_BOND.get(sentinel_chain_type)
    if bond_atom_names is None:
        return None, None
    term_name, start_name = bond_atom_names  # e.g. ("C", "N") for polypeptides

    chain_val = array.get_annotation(chain_key)[sentinel_idx]
    chain_vals = array.get_annotation(chain_key)

    res_bounds = get_residue_starts(array, add_exclusive_stop=True)
    res_starts_arr, res_stops_arr = res_bounds[:-1], res_bounds[1:]
    sentinel_res_idx = int(np.searchsorted(res_starts_arr, sentinel_idx, side="right")) - 1
    is_expandable = S_SEGMIN.mask(array)

    prev_terminal_idx = None
    if sentinel_res_idx > 0:
        ps, pe = int(res_starts_arr[sentinel_res_idx - 1]), int(res_stops_arr[sentinel_res_idx - 1])
        if chain_vals[ps] == chain_val:
            for ai in range(ps, pe):
                if array.atom_name[ai] == term_name or (pe - ps == 1 and is_expandable[ai]):
                    prev_terminal_idx = ai
                    break

    next_start_idx = None
    if sentinel_res_idx < len(res_starts_arr) - 1:
        ns, ne = int(res_starts_arr[sentinel_res_idx + 1]), int(res_stops_arr[sentinel_res_idx + 1])
        if chain_vals[ns] == chain_val:
            for ai in range(ns, ne):
                if array.atom_name[ai] == start_name or (ne - ns == 1 and is_expandable[ai]):
                    next_start_idx = ai
                    break

    return prev_terminal_idx, next_start_idx


def _install_sentinel_bonds(
    array: AtomArrayPlus,
    sentinel_idx: int,
    chain_key: str,
) -> None:
    """Wire a sentinel into the polymer bond graph.

    Removes the existing inter-residue bond between the flanking residues (if
    present) and ensures single bonds exist from the sentinel to each flanking
    polymer-bond atom.  Safe to call multiple times — existing sentinel bonds
    are not duplicated.
    """
    prev_idx, next_idx = _find_flanking_bond_atom_idxs(array, sentinel_idx, chain_key)
    if prev_idx is None and next_idx is None:
        return

    existing = {(min(int(b[0]), int(b[1])), max(int(b[0]), int(b[1]))) for b in array.bonds.as_array()}

    # Remove existing bond between flanking residues
    if prev_idx is not None and next_idx is not None:
        direct = (min(prev_idx, next_idx), max(prev_idx, next_idx))
        if direct in existing:
            array.bonds.remove_bond(prev_idx, next_idx)

    # Add new bonds from sentinel to flanking residues
    if prev_idx is not None and (min(prev_idx, sentinel_idx), max(prev_idx, sentinel_idx)) not in existing:
        array.bonds.add_bond(prev_idx, sentinel_idx, struc.BondType.SINGLE)
    if next_idx is not None and (min(sentinel_idx, next_idx), max(sentinel_idx, next_idx)) not in existing:
        array.bonds.add_bond(sentinel_idx, next_idx, struc.BondType.SINGLE)


def _post_insert_rederive_common_annotations(
    array: AtomArrayPlus,
    original_annots: set[str],
) -> tuple[AtomArrayPlus, list[str]]:
    """Re-derive common parser/feature annotations after expandable segment insertion."""
    # Recompute connectivity-derived IDs/entities if they existed.
    entity_annots = {"pn_unit_id", "molecule_id", "chain_entity", "pn_unit_entity", "molecule_entity"}
    if entity_annots & original_annots:
        array = io_ta.add_id_and_entity_annotations(array, overwrite=True)

    iid_annots = {"chain_iid", "pn_unit_iid", "molecule_iid"}
    if "transformation_id" in array.get_annotation_categories() and (iid_annots & original_annots):
        array = io_ta.add_iid_annotations(array, overwrite=True)

    # Recompute global/index IDs through one canonical helper.
    _renumber_ids_and_termini(array)

    # Re-derive all lazily computed annotations that were present originally.
    # The chain type is not re-derived since we set it manually for the sentinel
    recomputed_annots = []
    for annot_name in ANNOTATOR_REGISTRY:
        if annot_name in original_annots and annot_name != "chain_type":
            recomputed_annots.append(annot_name)
            if annot_name in array.get_annotation_categories():
                array.del_annotation(annot_name)
            ensure_annotations(array, annot_name)

    return array, recomputed_annots


def _renumber_ids_and_termini(array: AtomArrayPlus) -> None:
    """
    Renumber ``atom_id`` and ``token_id`` (if present) in-place.

    ``atom_id`` and ``token_id`` are renumbered sequentially (1-based) across the whole array.
    ``within_chain_res_idx`` is renumbered sequentially (0-based) within each chain instance (or chain).
    ``within_poly_res_idx`` is renumbered sequentially (0-based) within each polymer chain.

    Args:
        array: Array to renumber in-place.
    """
    annots = set(array.get_annotation_categories())

    if "atom_id" in annots:
        add_global_atom_id_annotation(array)
    if "token_id" in annots:
        # Recompute token boundaries from residue identity and atomization.
        array.del_annotation("token_id")
        add_global_token_id_annotation(array)
    if "within_chain_res_idx" in annots:
        group_by = "chain_iid" if "chain_iid" in array.get_annotation_categories() else "chain_id"
        array.set_annotation("within_chain_res_idx", get_within_group_res_idx(array, group_by=group_by))
    if "within_chain_source_res_idx" in annots:
        group_by = "chain_iid" if "chain_iid" in array.get_annotation_categories() else "chain_id"
        array.set_annotation("within_chain_source_res_idx", get_within_group_source_res_idx(array, group_by=group_by))
    if "within_poly_res_idx" in annots:
        array.set_annotation("within_poly_res_idx", get_within_poly_res_idx(array))
    if "is_chain_start" in annots:
        ensure_annotations(array, "is_chain_start")
    if C_CTR.full_name in annots:
        C_CTR.set_annotation(array, C_CTR.default_annotation(array))
    if C_NTR.full_name in annots:
        C_NTR.set_annotation(array, C_NTR.default_annotation(array))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def insert_expandable_segment(
    array: AtomArrayPlus,
    seg_min: int,
    seg_max: int,
    atom_idx: int | None = None,
    insert_before: bool = True,
    new_chain: bool = False,
    new_chain_id: str | None = None,
    new_chain_type: ChainType | str | int | None = None,
) -> AtomArrayPlus:
    """Insert an expandable segment sentinel into an existing chain or as a new chain.

    Args:
        array: The structure to insert the sentinel into.
        seg_min: Minimum realized segment length (≥ 0).
        seg_max: Maximum realized segment length (≥ seg_min).
        atom_idx: Reference atom index for insertion into an existing chain.
            Required when ``new_chain=False``.
        insert_before: If ``True`` (default), insert before ``atom_idx``.
            If ``False``, insert after ``atom_idx``.
        new_chain: If ``True``, append sentinel as a new chain.
        new_chain_id: Required when ``new_chain=True``. New chain ID.
        new_chain_type: Required when ``new_chain=True``. Must be a valid ``ChainType``.

    Returns:
        New :py:class:`~atomworks.io.utils.atom_array_plus.AtomArrayPlus` with
        the sentinel inserted.

    Raises:
        ValueError: If arguments are invalid or insertion index is out of range.
    """
    array = as_atom_array_plus(array)
    n = array.array_length()
    if n == 0:
        raise ValueError("Cannot insert an expandable segment into an empty array.")
    if seg_min < 0:
        raise ValueError(f"seg_min must be >= 0, got {seg_min}.")
    if seg_max < seg_min:
        raise ValueError(f"seg_max ({seg_max}) must be >= seg_min ({seg_min}).")

    if new_chain and atom_idx is not None:
        raise ValueError("Specify either atom_idx (existing-chain insertion) OR new_chain=True, not both.")
    if not new_chain and atom_idx is None:
        raise ValueError("atom_idx is required when new_chain=False.")

    original_annots = set(array.get_annotation_categories())

    if new_chain:
        if not new_chain_id:
            raise ValueError("new_chain_id is required when new_chain=True.")
        if new_chain_type is None:
            raise ValueError("new_chain_type is required when new_chain=True.")
        chain_type = ChainType.as_enum(new_chain_type)
        insert_position = n
        source_atom_idx = n - 1
        chain_id = new_chain_id
        sentinel_res_id = 1
        chain_identifier_key = "chain_iid" if "chain_iid" in array.get_annotation_categories() else "chain_id"
        chain_identifier_value: str | int | np.integer = chain_id
    else:
        assert atom_idx is not None  # for type narrowing
        if not (0 <= atom_idx < n):
            raise ValueError(f"atom_idx={atom_idx} is out of range [0, {n}).")
        res_bounds = get_residue_starts(array, add_exclusive_stop=True)
        res_starts, res_stops = res_bounds[:-1], res_bounds[1:]
        res_idx = int(np.searchsorted(res_starts, atom_idx, side="right")) - 1
        residue_start, residue_stop = int(res_starts[res_idx]), int(res_stops[res_idx])
        if insert_before and atom_idx != residue_start:
            raise ValueError(
                f"atom_idx={atom_idx} is not the first atom of its residue "
                f"(residue spans [{residue_start}, {residue_stop})) -- insert_before=True "
                "requires atom_idx to be the residue's first atom, or the sentinel would "
                "be spliced into the middle of that residue."
            )
        if not insert_before and atom_idx != residue_stop - 1:
            raise ValueError(
                f"atom_idx={atom_idx} is not the last atom of its residue "
                f"(residue spans [{residue_start}, {residue_stop})) -- insert_before=False "
                "requires atom_idx to be the residue's last atom, or the sentinel would "
                "be spliced into the middle of that residue."
            )
        insert_position = atom_idx if insert_before else atom_idx + 1
        source_atom_idx = atom_idx
        chain_id = str(array.chain_id[atom_idx])
        chain_identifier_key = "chain_iid" if "chain_iid" in array.get_annotation_categories() else "chain_id"
        chain_identifier_value = array.get_annotation(chain_identifier_key)[atom_idx]
        sentinel_res_id, needs_shift = _compute_insert_res_id(
            array,
            insert_position=insert_position,
            chain_identifier_key=chain_identifier_key,
            chain_identifier_value=chain_identifier_value,
        )
        chain_type = None

    # Ensure expandable segment annotations exist on source array.
    if not S_SEGMIN.has_annotation(array):
        S_SEGMIN.set_annotation(array, S_SEGMIN.default_annotation(array))
    if not S_SEGMAX.has_annotation(array):
        S_SEGMAX.set_annotation(array, S_SEGMAX.default_annotation(array))
    if _SEG_UID_ANNOT not in array.get_annotation_categories():
        array.set_annotation(_SEG_UID_ANNOT, np.full(n, -1, dtype=np.int64))
    seg_uid = int(array.get_annotation(_SEG_UID_ANNOT).max()) + 1

    sentinel, handled = _make_expandable_segment_sentinel_from_copied_atom(
        array=array,
        source_atom_idx=source_atom_idx,
        seg_min=seg_min,
        seg_max=seg_max,
        chain_id=chain_id,
        sentinel_res_id=sentinel_res_id,
        seg_uid=seg_uid,
        chain_type=chain_type,
    )
    blindly_copied = _get_blindly_copied_annotations(original_annots, handled)

    result = insert_atoms(
        array,
        [sentinel],
        [insert_position],
        on_annotation_mismatch_policy="drop",
        fill_missing_standard_annotations=True,
    )
    if not new_chain:
        if needs_shift:
            # Shift res_ids BEFORE installing bonds so that get_residue_starts
            # sees the sentinel as its own residue group (otherwise the sentinel
            # shares res_id with the downstream residue and they merge).
            _shift_chain_res_ids_after_insert(
                result,
                sentinel_idx=insert_position,
                sentinel_res_id=sentinel_res_id,
                chain_identifier_key=chain_identifier_key,
                chain_identifier_value=chain_identifier_value,
            )
        _install_sentinel_bonds(result, sentinel_idx=insert_position, chain_key=chain_identifier_key)

    result, recomputed_annots = _post_insert_rederive_common_annotations(result, original_annots=original_annots)

    blindly_copied = [annot for annot in blindly_copied if annot not in recomputed_annots]
    if blindly_copied:
        logger.warning(
            "Expandable segment sentinel insertion copied annotation values without explicit re-derivation for source atom_idx=%d: %s",
            source_atom_idx,
            blindly_copied,
        )

    return result


def realize_expandable_segments(
    array: AtomArrayPlus,
    rng: np.random.Generator | None = None,
) -> AtomArrayPlus:
    """Replace each expandable segment sentinel with N concrete (coordinate-free) atom stubs.

    For each sentinel (identified by ``S_SEGMIN.mask(array) == True`` and
    ``res_name != MASKED``), a random length N is sampled uniformly from
    ``[seg_min, seg_max]`` using *rng*.  N single-atom residues with
    ``res_name = MASKED``, ``element = "X"``, ``occupancy = 0``, and
    ``coord = NaN`` are inserted in place of the sentinel.

    The segment mask and ``expsegmin``/``expsegmax`` annotations are preserved
    on the realized stubs so that :py:func:`collapse_expandable_segments` can
    reconstruct the original sentinel range.

    Args:
        array: Input structure, possibly containing expandable segment sentinels.
        rng: NumPy random generator. ``None`` seeds one from the global numpy stream.

    Returns:
        New array with all sentinels realized.

    Examples:
        >>> realized = realize_expandable_segments(arr, rng=np.random.default_rng(42))
        >>> assert MASKED in realized.res_name
    """
    array = as_atom_array_plus(array)

    if not S_SEGMIN.has_annotation(array):
        return array

    # Sentinels are atoms where the segment mask is True and res_name is not yet realized (not MASKED stubs)
    is_sentinel = S_SEGMIN.mask(array) & (array.res_name != MASKED)
    assert np.array_equal(
        S_SEGMIN.mask(array), S_SEGMAX.mask(array)
    ), "S_SEGMIN and S_SEGMAX masks must agree on every atom."

    if not np.any(is_sentinel):
        return array

    _ensure_seg_uids(array, is_sentinel)

    if rng is None:
        # Seeded past both early returns so arrays with nothing to realize draw no randomness.
        rng = np.random.default_rng(np.random.randint(0, 2**32))

    segmin_ann = S_SEGMIN.annotation(array)
    segmax_ann = S_SEGMAX.annotation(array)

    sentinel_indices = np.where(is_sentinel)[0]
    n_realized_by_idx = {
        int(idx): int(rng.integers(int(segmin_ann[idx]), int(segmax_ann[idx]) + 1)) for idx in sentinel_indices
    }

    result = array
    chain_key = "chain_iid" if "chain_iid" in result.get_annotation_categories() else "chain_id"

    # Process right-to-left so indices of yet-to-process sentinels remain valid.
    for idx in sorted(n_realized_by_idx.keys(), reverse=True):
        n_realized = n_realized_by_idx[idx]
        sentinel_res_id = int(result.res_id[idx])
        chain_value = result.get_annotation(chain_key)[idx]

        if n_realized == 0:  # Edge case where we sample 0-length segments
            # Capture flanking bond atoms before removing the sentinel.
            prev_idx, next_idx = _find_flanking_bond_atom_idxs(result, idx, chain_key)

            # Remove sentinel entirely; downstream residues shift by -1.
            keep = np.ones(result.array_length(), dtype=bool)
            keep[idx] = False
            result = result[keep]
            # Biotite drops all sentinel bonds automatically; next_idx shifts down by 1.
            if prev_idx is not None and next_idx is not None:
                result.bonds.add_bond(prev_idx, next_idx - 1, struc.BondType.SINGLE)

            downstream_mask = np.arange(result.array_length()) >= int(idx)
            same_chain_mask = result.get_annotation(chain_key) == chain_value
            shift_mask = downstream_mask & same_chain_mask
            result.res_id[shift_mask] = result.res_id[shift_mask] - 1
            continue

        # Keep the original sentinel atom as first realized stub (guarantees no leftover "<x-y>").
        result.res_name[idx] = MASKED
        result.element[idx] = "X"
        result.res_id[idx] = sentinel_res_id

        # Insert the remaining realized stubs (n_realized - 1) immediately after idx.
        extra = n_realized - 1
        if extra > 0:
            stubs = _make_realized_stubs(result, int(idx), extra)
            stubs.res_id = np.arange(
                sentinel_res_id + 1,
                sentinel_res_id + 1 + extra,
                dtype=result.res_id.dtype,
            )
            result = insert_atoms(result, [stubs], [int(idx + 1)])
            # Biotite remaps all bond indices through insert_atoms.  The sentinel's
            # bond to next_start now points to next_start+extra.  Move it to the
            # last stub.  _make_realized_stubs already wired stub-to-stub bonds,
            # so we only need to bridge sentinel→first-stub and last-stub→next_start.
            bonds_arr = result.bonds.as_array()
            sentinel_bond_mask = (bonds_arr[:, 0] == idx) | (bonds_arr[:, 1] == idx)
            next_start_new = None
            for b in bonds_arr[sentinel_bond_mask]:
                partner = int(b[1]) if int(b[0]) == idx else int(b[0])
                if partner > idx + extra:
                    next_start_new = partner
                    break
            if next_start_new is not None:
                result.bonds.remove_bond(idx, next_start_new)
                result.bonds.add_bond(idx + extra, next_start_new, struc.BondType.SINGLE)
            result.bonds.add_bond(idx, idx + 1, struc.BondType.SINGLE)

            downstream_mask = np.arange(result.array_length()) >= int(idx + 1 + extra)
            same_chain_mask = result.get_annotation(chain_key) == chain_value
            shift_mask = downstream_mask & same_chain_mask
            result.res_id[shift_mask] = result.res_id[shift_mask] + extra

    _renumber_ids_and_termini(result)
    return result


def _make_realized_stubs(
    template: AtomArrayPlus,
    sentinel_idx: int,
    n: int,
) -> AtomArrayPlus:
    """Build *n* MASKED stub atoms from the sentinel at *sentinel_idx*.

    Args:
        template: Full array (used for annotation schema and chain info).
        sentinel_idx: Index of the sentinel in *template*.
        n: Number of stubs to produce.

    Returns:
        AtomArrayPlus with *n* realized MASKED stubs.
    """
    # Slice the sentinel atom to inherit all its annotations (including segment mask=True).
    # Use a range slice (not integer index) so the result is an AtomArrayPlus, not an Atom.
    stub = template[sentinel_idx : sentinel_idx + 1]

    # Override identity fields for a realized stub but only change what we need
    stub.res_name = np.array([MASKED])
    stub.element = np.array(["X"])

    stubs = concatenate_atom_array_plus([stub] * n)
    for i in range(n - 1):
        stubs.bonds.add_bond(i, i + 1, struc.BondType.SINGLE)
    return stubs


def collapse_expandable_segments(array: AtomArrayPlus) -> AtomArrayPlus:
    """Collapse runs of realized segment stubs back into single sentinels.

    Atoms with ``S_SEGMIN.mask(array) == True`` and ``res_name == MASKED``
    are treated as realized expandable segment stubs.  Contiguous runs sharing
    the same ``(expsegmin, expsegmax)`` range are collapsed by:

    1. Keeping only the first atom of each run (via slicing, preserving all
       annotations and 2D pairs).
    2. Renaming that atom's ``res_name`` back to ``"<seg_min-seg_max>"``.

    Args:
        array: Array potentially containing realized expandable segment stubs.

    Returns:
        New array with all realized segment runs collapsed into sentinels.

    Examples:
        >>> collapsed = collapse_expandable_segments(realized)
        >>> assert not np.any(collapsed.res_name == MASKED)
    """
    array = as_atom_array_plus(array)

    if not S_SEGMIN.has_annotation(array):
        return array

    assert np.array_equal(
        S_SEGMIN.mask(array), S_SEGMAX.mask(array)
    ), "S_SEGMIN and S_SEGMAX masks must agree on every atom."

    is_segment = S_SEGMIN.mask(array) & (array.res_name == MASKED)

    if not np.any(is_segment):
        return array

    segmin_ann = S_SEGMIN.annotation(array)
    segmax_ann = S_SEGMAX.annotation(array)
    chain_key = "chain_iid" if "chain_iid" in array.get_annotation_categories() else "chain_id"
    if _SEG_UID_ANNOT in array.get_annotation_categories():
        seg_uid_ann = array.get_annotation(_SEG_UID_ANNOT)
    else:
        seg_uid_ann = None
        logger.warning(
            "collapse_expandable_segments: no %r annotation found -- falling back to grouping "
            "realized stub runs by (seg_min, seg_max, chain, contiguity) alone. Two independently "
            "inserted segments sharing an identical range that ended up chain-adjacent will be "
            "silently merged into one sentinel.",
            _SEG_UID_ANNOT,
        )

    segment_idxs = np.where(is_segment)[0]
    runs: list[tuple[int, int]] = []
    run_start = int(segment_idxs[0])
    run_prev = int(segment_idxs[0])
    for idx in map(int, segment_idxs[1:]):
        is_contiguous = idx == run_prev + 1
        same_range = (segmin_ann[idx] == segmin_ann[run_prev]) and (segmax_ann[idx] == segmax_ann[run_prev])
        same_chain = array.get_annotation(chain_key)[idx] == array.get_annotation(chain_key)[run_prev]
        # Two independently-inserted segments can share an identical range and
        # end up chain-adjacent after realization -- same_uid tells them apart.
        same_uid = seg_uid_ann is None or seg_uid_ann[idx] == seg_uid_ann[run_prev]
        if is_contiguous and same_range and same_chain and same_uid:
            run_prev = idx
            continue
        runs.append((run_start, run_prev))
        run_start = idx
        run_prev = idx
    runs.append((run_start, run_prev))

    result = array
    for run_start, run_end in reversed(runs):
        run_len = run_end - run_start + 1
        segmin = int(S_SEGMIN.annotation(result)[run_start])
        segmax = int(S_SEGMAX.annotation(result)[run_start])
        chain_value = result.get_annotation(chain_key)[run_start]

        # Keep first realized stub as sentinel.
        seg_name = f"{segmin}-{segmax}"
        _ensure_res_name_width(result, max(9, len(seg_name)))
        result.res_name[run_start] = seg_name

        if run_len > 1:
            keep = np.ones(result.array_length(), dtype=bool)
            keep[run_start + 1 : run_end + 1] = False
            result = result[keep]

            # Restore res_ids BEFORE fixing bonds so that get_residue_starts
            # sees the same residue grouping as the original sentinel array.
            shift = run_len - 1
            downstream_mask = np.arange(result.array_length()) > run_start
            same_chain_mask = result.get_annotation(chain_key) == chain_value
            shift_mask = downstream_mask & same_chain_mask
            result.res_id[shift_mask] = result.res_id[shift_mask] - shift

            # Fix bonds
            _install_sentinel_bonds(result, run_start, chain_key)

    _renumber_ids_and_termini(result)
    return result
