from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.io.utils.atom_array_plus import (
    AnnotationList2D,
    AtomArrayPlus,
    as_atom_array_plus,
)
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.ml.conditions import CONDITIONS
from atomworks.ml.utils.condition import (
    load_atom_array_with_conditions_from_cif,
    save_atom_array_with_conditions_to_cif,
)
from atomworks.ml.utils.testing import cached_parse


@pytest.fixture
def atom_array() -> AtomArrayPlus:
    cif_parser_args = {
        **STANDARD_PARSER_ARGS,
        **{
            "fix_arginines": False,
            "add_missing_atoms": False,  # this is crucial otherwise the annotations are deleted
        },
    }
    cif_parser_args.pop("add_bond_types_from_struct_conn")
    cif_parser_args.pop("remove_ccds")
    data = cached_parse("6lyz", **cif_parser_args)
    atom_array = data["atom_array"]
    atom_array = as_atom_array_plus(atom_array)
    return atom_array


def save_and_load_atom_array(atom_array: AtomArray) -> AtomArray:
    """
    Saves an atom array to a temporary file and loads it back in using standard functions.
    """
    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.cif"
        save_atom_array_with_conditions_to_cif(atom_array, temp_file)
        loaded_atom_array = load_atom_array_with_conditions_from_cif(temp_file, fill_missing_conditions=True)

    return loaded_atom_array


def test_save_and_load_atom_array_with_defaults(atom_array: AtomArray):
    """
    Test that saving and loading an atom array with all defaults preserves the atom array.
    """
    loaded_atom_array = save_and_load_atom_array(atom_array)

    possible = {c.full_name for c in CONDITIONS} | {c.mask_name for c in CONDITIONS}
    existing = set(
        get_annotation_categories(loaded_atom_array, n_body=1) + get_annotation_categories(loaded_atom_array, n_body=2)
    )
    assert possible.issubset(
        existing
    ), f"Some possible annotations are not found in loaded_atom_array: {possible - existing}"

    for condition_cls in CONDITIONS:
        default_mask = condition_cls.default_mask(atom_array)
        annotated_mask = condition_cls.mask(loaded_atom_array)
        if isinstance(default_mask, AnnotationList2D):
            default_mask = default_mask.as_dense_array()
        if isinstance(annotated_mask, AnnotationList2D):
            annotated_mask = annotated_mask.as_dense_array()
        assert np.array_equal(default_mask, annotated_mask, equal_nan=True)

        if not condition_cls.is_mask and condition_cls.name != "sequence":
            default_annotation = condition_cls.default_annotation(atom_array)
            annotated_annotation = condition_cls.annotation(loaded_atom_array)
            if isinstance(default_annotation, AnnotationList2D):
                default_annotation = default_annotation.as_dense_array()
            if isinstance(annotated_annotation, AnnotationList2D):
                annotated_annotation = annotated_annotation.as_dense_array()
            assert np.array_equal(default_annotation, annotated_annotation, equal_nan=True)


def test_save_and_load_atom_array_with_conditions(atom_array: AtomArray):
    """
    Same as test_save_and_load_atom_array_with_defaults but we randomly set some of each condition
    and make sure the annotations are preserved.
    """

    # Generate some non-zero values for all conditions
    frozen_condition_values = {}
    for condition_cls in CONDITIONS:
        if condition_cls.name == "coordinate":
            random_indices = np.random.choice(len(atom_array), size=5, replace=False)
            mask = np.zeros(len(atom_array), dtype=bool)
            mask[random_indices] = True
            condition_cls.set_mask(atom_array, mask)
            values = condition_cls.annotation(atom_array, default="generate")
            atom_array.set_annotation(condition_cls.full_name, values)
            frozen_condition_values[condition_cls.full_name] = {
                "mask": mask,
                "condition": values,
            }

        elif condition_cls.n_body == 1:
            random_indices = np.random.choice(len(atom_array), size=5, replace=False)
            random_values = np.random.random(size=5)
            if np.issubdtype(condition_cls.dtype, bool):
                random_values = random_values > 0.5
            elif np.issubdtype(condition_cls.dtype, np.integer):
                random_values = np.random.randint(0, 100, size=5)
            mask = np.zeros(len(atom_array), dtype=bool)
            mask[random_indices] = True
            frozen_condition_values[condition_cls.full_name] = {"mask": mask}

            if not condition_cls.is_mask and condition_cls.name != "sequence":
                condition_arr = np.zeros(len(atom_array), dtype=condition_cls.dtype)
                condition_arr[random_indices] = random_values
                frozen_condition_values[condition_cls.full_name] |= {"condition": condition_arr}
        elif condition_cls.n_body == 2:
            random_indices = np.random.choice(len(atom_array), size=(5, 2), replace=False)
            random_values = np.random.random(size=5)
            if np.issubdtype(condition_cls.dtype, bool):
                random_values = random_values > 0.5
            elif np.issubdtype(condition_cls.dtype, np.integer):
                random_values = random_values.astype(int)
            mask = AnnotationList2D(
                n_atoms=len(atom_array),
                pairs=random_indices,
                values=np.ones_like(random_values, dtype=bool),
            )
            frozen_condition_values[condition_cls.full_name] = {"mask": mask}

            if not condition_cls.is_mask:
                condition_arr = AnnotationList2D(
                    n_atoms=len(atom_array),
                    pairs=random_indices,
                    values=random_values,
                )
                frozen_condition_values[condition_cls.full_name] |= {"condition": condition_arr}
        else:
            raise ValueError(
                f"Condition {condition_cls.full_name} has {condition_cls.n_body} bodies, which is not supported."
            )

    # add annotations to atom_array
    for annotation_name, annotation_dict in frozen_condition_values.items():
        condition_cls = CONDITIONS.from_full_name(annotation_name)
        condition_cls.set_mask(atom_array, annotation_dict["mask"])
        if not condition_cls.is_mask and condition_cls.name != "sequence":
            if condition_cls.n_body == 1:
                condition_cls.set_annotation(atom_array, array=annotation_dict["condition"])
            elif condition_cls.n_body == 2:
                condition_cls.set_annotation(
                    atom_array,
                    pairs=annotation_dict["condition"].pairs,
                    values=annotation_dict["condition"].values,
                )
            else:
                raise ValueError(
                    f"Condition {condition_cls.full_name} has {condition_cls.n_body} bodies, which is not supported."
                )

    loaded_atom_array = save_and_load_atom_array(atom_array)

    # Check that all annotations are preserved after save/load
    for annotation_name in frozen_condition_values:
        condition_cls = CONDITIONS.from_full_name(annotation_name)
        target_annotation = condition_cls.annotation(atom_array)
        target_mask = condition_cls.mask(atom_array)

        loaded_annotation = condition_cls.annotation(loaded_atom_array)
        loaded_mask = condition_cls.mask(loaded_atom_array)

        if condition_cls.n_body > 1:
            target_mask = target_mask.as_dense_array()
            loaded_mask = loaded_mask.as_dense_array()
            target_annotation = target_annotation.as_dense_array()
            loaded_annotation = loaded_annotation.as_dense_array()

        assert np.array_equal(
            target_mask, loaded_mask, equal_nan=True
        ), f"Mismatch for {condition_cls.full_name}: {np.where(target_mask)} != {np.where(loaded_mask)}"
        assert np.array_equal(
            target_annotation[target_mask],
            loaded_annotation[loaded_mask],
            equal_nan=np.issubdtype(condition_cls.dtype, np.floating),
        ), f"Mismatch for {condition_cls.full_name}: {target_annotation[target_mask]} != {loaded_annotation[loaded_mask]}"


if __name__ == "__main__":
    # run pytest on this file with verbose output
    pytest.main(["-v", __file__])
