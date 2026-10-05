"""Tests for expandable segment sentinel functionality.

Covers: insert_expandable_segment, realize_expandable_segments, collapse_expandable_segments.
"""

import numpy as np
import pytest

from atomworks.constants import BIOTITE_COMMON_OPTIONAL_ANNOTATIONS, BIOTITE_DEFAULT_ANNOTATIONS, MASKED
from atomworks.enums import ChainType
from atomworks.io.config import ParseConfig
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, as_atom_array_plus
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.standard_annotations import S_SEGMAX, S_SEGMIN
from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from atomworks.ml.utils.expandable_segment import (
    _SEG_UID_ANNOT,
    collapse_expandable_segments,
    insert_expandable_segment,
    realize_expandable_segments,
)
from atomworks.ml.utils.testing import add_randomized_standard_annotations, cached_parse, save_and_load_atom_array

BIOTITE_RESERVED_ANNOTATIONS = [
    "ins_code",
    *BIOTITE_DEFAULT_ANNOTATIONS,
    *BIOTITE_COMMON_OPTIONAL_ANNOTATIONS,
]


def _compare_atom_arrays(
    arr1: AtomArrayPlus,
    arr2: AtomArrayPlus,
    annotations_to_compare: list[str],
    compare_bonds: bool = True,
) -> None:
    """Compares without bonds but enforcing order, then with bonds without enforcing order."""
    assert_same_atom_array_or_stack(
        arr1,
        arr2,
        annotations_to_compare=annotations_to_compare,
        enforce_order=True,
        compare_bonds=False,
        cast_to_common_dtype=True,
    )
    if compare_bonds:
        assert_same_atom_array_or_stack(
            arr1,
            arr2,
            annotations_to_compare=annotations_to_compare,
            enforce_order=False,
            compare_bonds=True,
            cast_to_common_dtype=True,
        )


def test_expandable_segment_sentinel_roundtrip():
    """Insert segment, realize, collapse, save/load: non-segment atoms are preserved throughout."""
    rng = np.random.default_rng(42)

    # Load a realistic structure with randomized standard annotations (including 2D)
    data = cached_parse("6lyz", **ParseConfig.from_preset("annotations_only").to_dict())
    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "is_res_start")
    atom_array = add_randomized_standard_annotations(atom_array)

    # Reset expandable segment annotations after randomizing
    S_SEGMIN.set_annotation(atom_array, S_SEGMIN.default_annotation(atom_array))
    S_SEGMAX.set_annotation(atom_array, S_SEGMAX.default_annotation(atom_array))

    # insert_expandable_segment changes some annotations expectedly, so we don't want to compare them
    annotations_to_compare = [
        annot
        for annot in atom_array.get_annotation_categories()
        if (
            annot
            not in [
                "res_id",
                "atom_id",
                "token_id",
                "within_poly_res_idx",
                "auth_seq_id",
                "condition_coordinate_1_atom",
                "transformation_id",
            ]
        )
        and (not (annot.startswith("label_") or annot.startswith("auth_")))
    ]

    # --- Insert an expandable segment sentinel in the middle but at a res_start ---
    res_start_inds = np.where(atom_array.is_res_start)[0]
    mid = res_start_inds[res_start_inds.size // 2]
    arr_with_segment = insert_expandable_segment(atom_array.copy(), seg_min=3, seg_max=8, atom_idx=mid)

    # Slicing out segment atoms must recover the original (bond graph legitimately
    # differs at the insertion point, so skip bond comparison for sliced arrays).
    non_segment = arr_with_segment[~S_SEGMIN.mask(arr_with_segment)]
    _compare_atom_arrays(atom_array, non_segment, annotations_to_compare, compare_bonds=False)

    # --- Realize the segment into concrete MASKED stubs ---
    realized = realize_expandable_segments(arr_with_segment.copy(), rng=rng)
    assert np.any(realized.res_name == MASKED), "Realized array should contain MASKED stubs"
    non_segment_realized = realized[~S_SEGMIN.mask(realized)]
    _compare_atom_arrays(atom_array, non_segment_realized, annotations_to_compare, compare_bonds=False)

    # --- Collapse stubs back to a single sentinel ---
    collapsed = collapse_expandable_segments(realized.copy())

    _compare_atom_arrays(arr_with_segment, collapsed, annotations_to_compare)

    # --- Round-trip the collapsed (sentinel) form through CIF ---
    extra_annotations_to_save = [
        annot
        for annot in annotations_to_compare
        if (annot not in BIOTITE_RESERVED_ANNOTATIONS)
        and (not (annot.startswith("label_") or annot.startswith("auth_")))
    ]
    loaded_collapsed = save_and_load_atom_array(
        collapsed, extra_annotations_to_save=extra_annotations_to_save, extra_fields_to_load=extra_annotations_to_save
    )
    _compare_atom_arrays(collapsed, loaded_collapsed, annotations_to_compare=annotations_to_compare)

    # --- Round-trip the realized (expanded) form, then collapse and verify ---
    loaded_realized = save_and_load_atom_array(
        realized, extra_annotations_to_save=extra_annotations_to_save, extra_fields_to_load=extra_annotations_to_save
    )
    _compare_atom_arrays(realized, loaded_realized, annotations_to_compare=annotations_to_compare)
    collapsed_again = collapse_expandable_segments(loaded_realized.copy())
    _compare_atom_arrays(arr_with_segment, collapsed_again, annotations_to_compare=annotations_to_compare)


def test_insert_expandable_segment_as_separate_chain_updates_chain_dependent_annotations():
    """Insert expandable segment as a new chain and verify chain-dependent annotations update consistently."""
    data = cached_parse("6lyz", **ParseConfig.from_preset("annotations_only").to_dict())
    atom_array = as_atom_array_plus(data["atom_array"])
    atom_array = add_randomized_standard_annotations(atom_array)

    # Reset expandable segment annotations after randomizing
    S_SEGMIN.set_annotation(atom_array, S_SEGMIN.default_annotation(atom_array))
    S_SEGMAX.set_annotation(atom_array, S_SEGMAX.default_annotation(atom_array))

    annots_before = set(atom_array.get_annotation_categories())
    new_chain_id = "ZZ"
    result = insert_expandable_segment(
        atom_array.copy(),
        seg_min=3,
        seg_max=8,
        new_chain=True,
        new_chain_id=new_chain_id,
        new_chain_type=ChainType.NON_POLYMER,
    )

    segment_mask = S_SEGMIN.mask(result)
    assert np.sum(segment_mask) == 1, "Expected exactly one inserted sentinel segment atom."
    segment_idx = int(np.where(segment_mask)[0][0])

    # New chain identity for sentinel
    assert result.chain_id[segment_idx] == new_chain_id
    assert result.res_name[segment_idx] == "3-8"
    assert int(result.res_id[segment_idx]) == 1

    # Chain-dependent annotations that should track new chain insertion
    if "chain_type" in annots_before:
        assert int(result.chain_type[segment_idx]) == int(ChainType.NON_POLYMER)
    if "is_polymer" in annots_before:
        assert bool(result.is_polymer[segment_idx]) is False
    if "pn_unit_id" in annots_before:
        assert result.pn_unit_id[segment_idx] == new_chain_id
    if "chain_iid" in annots_before:
        assert str(result.chain_iid[segment_idx]).startswith(f"{new_chain_id}_")
    if "pn_unit_iid" in annots_before:
        assert str(result.pn_unit_iid[segment_idx]).startswith(f"{new_chain_id}_")
    if "molecule_id" in annots_before:
        assert result.molecule_id[segment_idx] not in set(atom_array.molecule_id.tolist())
    if "molecule_iid" in annots_before:
        assert result.molecule_iid[segment_idx] not in set(atom_array.molecule_iid.tolist())
    if "chain_entity" in annots_before:
        assert result.chain_entity[segment_idx] not in set(atom_array.chain_entity.tolist())
    if "pn_unit_entity" in annots_before:
        assert result.pn_unit_entity[segment_idx] not in set(atom_array.pn_unit_entity.tolist())
    if "molecule_entity" in annots_before:
        assert result.molecule_entity[segment_idx] not in set(atom_array.molecule_entity.tolist())
    if "label_entity_id" in annots_before and "chain_entity" in annots_before:
        # New entity → fresh label_entity_id that doesn't clash with original ones.
        assert int(result.label_entity_id[segment_idx]) not in set(atom_array.label_entity_id.tolist())

    # All non-segment atoms should match original except expected globally rederived IDs/entities.
    non_segment = result[~segment_mask]
    annotations_to_compare = [
        annot
        for annot in atom_array.get_annotation_categories()
        if (
            annot
            not in [
                "res_id",
                "atom_id",
                "token_id",
                "within_chain_res_idx",
                "within_poly_res_idx",
                "chain_entity",
                "pn_unit_entity",
                "molecule_entity",
                "molecule_id",
                "molecule_iid",
                "pn_unit_id",
                "pn_unit_iid",
                "chain_iid",
                "label_entity_id",
                "auth_seq_id",
            ]
        )
        and (not (annot.startswith("label_") or annot.startswith("auth_")))
    ]
    _compare_atom_arrays(atom_array, non_segment, annotations_to_compare)


def test_insert_expandable_segment_new_chain_entity_id_avoids_all_collisions():
    """New-chain entity id must avoid every existing entity id, not just the last atom's."""
    data = cached_parse("6lyz", **ParseConfig.from_preset("annotations_only").to_dict())
    atom_array = as_atom_array_plus(data["atom_array"].copy())

    # Force a non-monotonic layout
    half = atom_array.array_length() // 2
    chain_id = atom_array.chain_id.copy()
    chain_id[:half] = "AAA"
    chain_id[half:] = "BBB"
    atom_array.set_annotation("chain_id", chain_id)

    entity_id = atom_array.label_entity_id.copy()
    entity_id[:half] = 3  # AAA (first in array order)
    entity_id[half:] = 2  # BBB (last in array order) -- old algo: 2 + 1 = 3, collides!
    atom_array.set_annotation("label_entity_id", entity_id.astype(np.int8))

    result = insert_expandable_segment(
        atom_array.copy(),
        seg_min=1,
        seg_max=1,
        new_chain=True,
        new_chain_id="ZZZ",
        new_chain_type=ChainType.NON_POLYMER,
    )
    segment_mask = S_SEGMIN.mask(result)
    new_entity_id = int(result.label_entity_id[segment_mask][0])
    assert new_entity_id not in {2, 3}, f"new sentinel entity id {new_entity_id} collides with an existing entity id"
    assert new_entity_id == 4, "expected max(existing entity ids) + 1"


def test_insert_expandable_segment_rejects_mid_residue_atom_idx():
    """``atom_idx`` must land on a residue boundary -- mid-residue must raise, not silently splice in."""
    data = cached_parse("6lyz", **ParseConfig.from_preset("annotations_only").to_dict())
    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "is_res_start")
    res_start_inds = np.where(atom_array.is_res_start)[0]

    mid_residue_start = None
    for s in map(int, res_start_inds):
        if s + 1 < atom_array.array_length() and not atom_array.is_res_start[s + 1]:
            mid_residue_start = s
            break
    assert mid_residue_start is not None, "test setup: no multi-atom residue found"
    mid_atom_idx = mid_residue_start + 1  # strictly inside the residue, not its first/last atom

    with pytest.raises(ValueError, match="not the first atom of its residue"):
        insert_expandable_segment(atom_array.copy(), seg_min=1, seg_max=1, atom_idx=mid_atom_idx, insert_before=True)

    with pytest.raises(ValueError, match="not the last atom of its residue"):
        insert_expandable_segment(atom_array.copy(), seg_min=1, seg_max=1, atom_idx=mid_atom_idx, insert_before=False)


def test_collapse_expandable_segments_keeps_independent_identical_range_segments_separate():
    """Two independently-inserted, identical-range segments that end up chain-adjacent
    after realization must collapse back into TWO sentinels, not merge into one.
    """
    data = cached_parse("6lyz", **ParseConfig.from_preset("annotations_only").to_dict())
    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "is_res_start")
    res_start_inds = np.where(atom_array.is_res_start)[0]
    mid = int(res_start_inds[res_start_inds.size // 2])

    # Insert two independent segments with an identical (seg_min, seg_max) at
    # the same position, so the second lands immediately adjacent to the first
    # -- both single-atom (seg_min=seg_max=1) so realization is deterministic.
    step1 = insert_expandable_segment(atom_array.copy(), seg_min=1, seg_max=1, atom_idx=mid, insert_before=True)
    step2 = insert_expandable_segment(step1.copy(), seg_min=1, seg_max=1, atom_idx=mid, insert_before=True)
    assert np.sum(S_SEGMIN.mask(step2)) == 2, "test setup: expected exactly two sentinels"

    realized = realize_expandable_segments(step2.copy(), rng=np.random.default_rng(0))
    collapsed = collapse_expandable_segments(realized.copy())

    assert np.sum(S_SEGMIN.mask(collapsed)) == 2, (
        "two independently-inserted segments sharing an identical range were merged "
        "into one sentinel instead of collapsing back into two"
    )


def test_realize_expandable_segments_backfills_missing_seg_uid():
    """Sentinels lacking ``_expseg_uid`` (e.g. loaded from CIF, which drops internal
    annotations, or created before this safeguard existed) still get disambiguated
    correctly -- ``realize_expandable_segments`` backfills a fresh id for each one
    before ``collapse_expandable_segments`` ever needs it.
    """
    data = cached_parse("6lyz", **ParseConfig.from_preset("annotations_only").to_dict())
    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "is_res_start")
    res_start_inds = np.where(atom_array.is_res_start)[0]
    mid = int(res_start_inds[res_start_inds.size // 2])

    step1 = insert_expandable_segment(atom_array.copy(), seg_min=1, seg_max=1, atom_idx=mid, insert_before=True)
    step2 = insert_expandable_segment(step1.copy(), seg_min=1, seg_max=1, atom_idx=mid, insert_before=True)
    assert np.sum(S_SEGMIN.mask(step2)) == 2, "test setup: expected exactly two sentinels"

    # Simulate the annotation never having existed at all.
    step2.del_annotation(_SEG_UID_ANNOT)
    assert _SEG_UID_ANNOT not in step2.get_annotation_categories()

    realized = realize_expandable_segments(step2.copy(), rng=np.random.default_rng(0))
    assert _SEG_UID_ANNOT in realized.get_annotation_categories(), "realize should backfill the annotation"

    collapsed = collapse_expandable_segments(realized.copy())
    assert np.sum(S_SEGMIN.mask(collapsed)) == 2, (
        "realize_expandable_segments failed to backfill distinct seg_uids for "
        "sentinels that were missing the annotation entirely"
    )


def _bond_partners(array: AtomArrayPlus, atom_idx: int) -> set[int]:
    """Return global indices of all atoms bonded to atom_idx."""
    bonds = array.bonds.as_array()
    partners: set[int] = set()
    for b in bonds:
        if int(b[0]) == atom_idx:
            partners.add(int(b[1]))
        elif int(b[1]) == atom_idx:
            partners.add(int(b[0]))
    return partners


def test_expandable_segment_bond_connectivity():
    """Each sentinel/stub has exactly 2 bond partners across insert, realize, and collapse."""
    data = cached_parse("6lyz", **ParseConfig.from_preset("annotations_only").to_dict())
    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "chain_type")

    ensure_annotations(atom_array, "is_res_start")
    res_start_inds = np.where(atom_array.is_res_start)[0]
    mid = res_start_inds[res_start_inds.size // 2]
    arr_with_segment = insert_expandable_segment(atom_array.copy(), seg_min=3, seg_max=3, atom_idx=mid)

    def check_segment_bond_counts(array, label):
        for idx in np.where(S_SEGMIN.mask(array))[0]:
            n = len(_bond_partners(array, int(idx)))
            assert n == 2, f"{label}: segment atom {idx} has {n} bond partners, expected 2"

    check_segment_bond_counts(arr_with_segment, "after insert")

    realized = realize_expandable_segments(arr_with_segment.copy(), rng=np.random.default_rng(0))
    check_segment_bond_counts(realized, "after realize")

    collapsed = collapse_expandable_segments(realized.copy())
    check_segment_bond_counts(collapsed, "after collapse")


def test_adjacent_expandable_segments_form_one_polymer_bond_graph():
    """Separately named adjacent segments remain connected through realization and collapse."""
    data = cached_parse("6lyz", **ParseConfig.from_preset("annotations_only").to_dict())
    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "chain_type")

    new_chain_id = "Z"
    with_segments = insert_expandable_segment(
        atom_array,
        seg_min=2,
        seg_max=2,
        new_chain=True,
        new_chain_id=new_chain_id,
        new_chain_type=ChainType.POLYPEPTIDE_L,
    )
    for length in (1, 3):
        chain_tail = int(np.where(with_segments.chain_id == new_chain_id)[0][-1])
        with_segments = insert_expandable_segment(
            with_segments,
            seg_min=length,
            seg_max=length,
            atom_idx=chain_tail,
            insert_before=False,
        )

    realized = realize_expandable_segments(with_segments.copy(), rng=np.random.default_rng(0))
    collapsed = collapse_expandable_segments(realized.copy())

    for label, array, expected_residues in (
        ("sentinels", with_segments, 3),
        ("realized", realized, 6),
        ("collapsed", collapsed, 3),
    ):
        chain = array[array.chain_id == new_chain_id]
        starts = get_residue_starts(chain)
        atom_to_residue = np.searchsorted(starts, np.arange(chain.array_length()), side="right") - 1
        bonded_residue_pairs = {
            tuple(sorted((int(atom_to_residue[a]), int(atom_to_residue[b]))))
            for a, b, _ in chain.bonds.as_array()
            if atom_to_residue[a] != atom_to_residue[b]
        }
        assert len(starts) == expected_residues
        assert bonded_residue_pairs == {(i, i + 1) for i in range(expected_residues - 1)}, label
