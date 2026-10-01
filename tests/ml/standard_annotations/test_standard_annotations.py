"""Tests for the StandardAnnotation framework"""

from pathlib import Path
from tempfile import TemporaryDirectory

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from atomworks.constants import MASKED
from atomworks.io.config import ParseConfig
from atomworks.io.parser import STANDARD_PARSER_ARGS, parse
from atomworks.io.utils.atom_array_plus import (
    as_atom_array_plus,
)
from atomworks.io.utils.io_utils import (
    load_any,
    to_cif_file,
)
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.io.utils.standard_annotations import STANDARD_ANNOTATIONS
from atomworks.io.utils.standard_annotations.serialization import _deserialize_standard_annotations
from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from atomworks.ml.conditions import C_CRD, C_IDX, C_NTR, C_SEQ
from atomworks.ml.utils.testing import (
    add_randomized_standard_annotations,
    cached_parse,
    get_pdb_mirror_path,
    save_and_load_atom_array,
)

# ---------------------------------------------------------------------------
# Save/load round-trips
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_atom_array():
    atom_array = AtomArray(3)
    atom_array.res_name = np.array(["ALA", "GLY", "SER"])
    atom_array.coord = np.arange(9, dtype=float).reshape(3, 3)
    return as_atom_array_plus(atom_array)


def test_masks_are_derived_from_annotations(minimal_atom_array):
    sequence = np.array(["ALA", MASKED, "SER"])
    coordinate = minimal_atom_array.coord.copy()
    coordinate[1] = np.nan
    index = np.array([True, False, True])

    C_SEQ.set_annotation(minimal_atom_array, sequence)
    C_CRD.set_annotation(minimal_atom_array, coordinate)
    C_IDX.set_annotation(minimal_atom_array, index)

    assert C_SEQ.mask(minimal_atom_array).tolist() == [True, False, True]
    assert C_CRD.mask(minimal_atom_array).tolist() == [True, False, True]
    assert np.array_equal(C_IDX.mask(minimal_atom_array), index)
    assert not any(name.startswith("mask_") for name in minimal_atom_array.get_annotation_categories())


def test_direct_aliases_are_non_destructive_and_conflicts_raise(minimal_atom_array):
    alias = C_NTR.aliases[0]
    minimal_atom_array.set_annotation(alias, np.array([True, False, True]))

    assert C_NTR.annotation(minimal_atom_array).tolist() == [True, False, True]
    assert C_NTR.full_name not in minimal_atom_array.get_annotation_categories()

    C_NTR.set_annotation(minimal_atom_array, np.array([False, True, False]))
    assert minimal_atom_array.get_annotation(alias).tolist() == [False, True, False]

    minimal_atom_array.set_annotation(alias, np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="Conflicting fields"):
        C_NTR.annotation(minimal_atom_array)


def test_legacy_mask_only_fields_are_migrated(minimal_atom_array):
    sequence_mask = np.array([True, False, True])
    coordinate_mask = np.array([False, True, True])
    minimal_atom_array.set_annotation("mask_sequence_1_residue", sequence_mask)
    minimal_atom_array.set_annotation("mask_coordinate_1_atom", coordinate_mask)

    loaded = _deserialize_standard_annotations(minimal_atom_array, pdbx.CIFBlock())

    assert C_SEQ.annotation(loaded).tolist() == ["ALA", MASKED, "SER"]
    assert np.array_equal(C_CRD.annotation(loaded)[coordinate_mask], minimal_atom_array.coord[coordinate_mask])
    assert np.isnan(C_CRD.annotation(loaded)[~coordinate_mask]).all()
    assert "mask_sequence_1_residue" not in loaded.get_annotation_categories()
    assert "mask_coordinate_1_atom" not in loaded.get_annotation_categories()


def test_legacy_coordinate_mask_is_restored_on_stack(minimal_atom_array):
    coordinate_mask = np.array([False, True, True])
    stack = struc.stack([minimal_atom_array, minimal_atom_array])
    stack.coord[1] += 10
    stack.set_annotation("mask_coordinate_1_atom", coordinate_mask)

    loaded = _deserialize_standard_annotations(stack, pdbx.CIFBlock())

    for model in range(2):
        np.testing.assert_array_equal(C_CRD.mask(loaded[model]), coordinate_mask)
        np.testing.assert_allclose(
            C_CRD.annotation(loaded[model])[coordinate_mask], stack.coord[model, coordinate_mask]
        )
    assert "mask_coordinate_1_atom" not in loaded.get_annotation_categories()


def test_legacy_index_prefers_boolean_alias(minimal_atom_array):
    minimal_atom_array.set_annotation(C_IDX.full_name, np.array([4, 5, 6]))
    minimal_atom_array.set_annotation("mask_index_1_residue", np.array([True, False, True]))

    loaded = _deserialize_standard_annotations(minimal_atom_array, pdbx.CIFBlock())

    assert C_IDX.annotation(loaded).dtype == bool
    assert C_IDX.annotation(loaded).tolist() == [True, False, True]
    assert "mask_index_1_residue" not in loaded.get_annotation_categories()


def test_save_and_load_with_standard_annotations():
    """Round-trip with non-default values for all StandardAnnotations, extra annotations included."""

    atom_array = load_any(get_pdb_mirror_path("6lyz"))[0]
    atom_array = as_atom_array_plus(atom_array)

    test_atom_array = add_randomized_standard_annotations(atom_array)

    # Extra non-SA scalar annotations
    extra_mask_original = np.zeros(len(test_atom_array), dtype=bool)
    extra_mask_original[np.random.choice(len(test_atom_array), 5, replace=False)] = True
    test_atom_array.set_annotation("extra_mask", extra_mask_original)

    extra_annotation_original = np.random.randint(0, 100, size=test_atom_array.array_length())
    test_atom_array.set_annotation("extra_annotation", extra_annotation_original)

    loaded = save_and_load_atom_array(
        test_atom_array,
        extra_fields_to_load=["extra_mask", "extra_annotation"],
        extra_annotations_to_save=["extra_mask", "extra_annotation"],
    )

    assert set(get_annotation_categories(loaded, n_body="all")).issuperset(
        set(STANDARD_ANNOTATIONS.get_field_names())
    ), "Loaded atom array must have all StandardAnnotation categories for this test to be valid."

    assert_same_atom_array_or_stack(test_atom_array, loaded, annotations_to_compare="arr1", enforce_order=False)


def test_loading_standard_annotations_requires_plus():
    with pytest.raises(ValueError, match="load_standard_annotations=True requires return_atom_array_plus=True"):
        ParseConfig.from_preset("minimal", load_standard_annotations=True, return_atom_array_plus=False)


def test_round_trip_parse():
    """Save with to_cif_file, load with parse + load_standard_annotations=True."""

    parser_args = STANDARD_PARSER_ARGS.copy()
    parser_args["fix_arginines"] = False
    parser_args["add_missing_atoms"] = False
    parser_args["convert_mse_to_met"] = False
    data = cached_parse(
        "6lyz",
        **parser_args,
    )
    atom_array = as_atom_array_plus(data["atom_array"])

    test_atom_array = add_randomized_standard_annotations(atom_array)

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.cif"
        to_cif_file(test_atom_array, path, save_standard_annotations=True)

        result = parse(
            path,
            config=ParseConfig(**{**parser_args, "load_standard_annotations": True, "return_atom_array_plus": True}),
        )
    assembly_ids = list(result["assemblies"].keys())
    loaded = result["assemblies"][assembly_ids[0]][0]

    assert_same_atom_array_or_stack(test_atom_array, loaded)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
