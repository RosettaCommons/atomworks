import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from biotite.structure import AtomArray
from scipy.spatial import cKDTree

from atomworks.common import immutable_lru_cache
from atomworks.constants import PDB_MIRROR_PATH
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.utils.atom_array_plus import (
    AnnotationList2D,
    AtomArrayPlus,
)
from atomworks.io.utils.io_utils import (
    to_cif_file,
)
from atomworks.io.utils.standard_annotations import STANDARD_ANNOTATIONS
from atomworks.ml.conditions import C_CTR, C_NTR, CONDITIONS


def get_pdb_mirror_path(pdbid: str, base_dir: str = PDB_MIRROR_PATH) -> str:
    """Convenience util to get the path to a CIF file on the DIGS"""
    # Assert that the base directory exists
    assert os.path.exists(base_dir), f"Base directory {base_dir} does not exist"

    # Build the path to the file
    pdbid = pdbid.lower()
    filename = f"{base_dir}/{pdbid[1:3]}/{pdbid}.cif.gz"
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename


@immutable_lru_cache(maxsize=1000, deepcopy=True)
def cached_parse(pdb_id: str, **kwargs) -> dict:
    """Wrapper around parse with caching to return an independent copy of the output dict."""
    data = parse(get_pdb_mirror_path(pdb_id), config=ParseConfig(**kwargs) if kwargs else None)
    if "atom_array" not in data:
        assembly_ids = list(data["assemblies"].keys())
        data["atom_array"] = data["assemblies"][assembly_ids[0]][0]
    data["pdb_id"] = pdb_id
    return data


def is_clash(atom_array_1: AtomArray, atom_array_2: AtomArray, clash_distance: float = 1.0) -> bool:
    """Check for clashes between two atom arrays."""
    tree = cKDTree(atom_array_2.coord)
    return len(tree.query_ball_point(atom_array_1.coord, r=clash_distance, return_length=True).nonzero()[0]) > 0


def add_randomized_standard_annotations(atom_array: AtomArrayPlus) -> AtomArrayPlus:
    """Add randomized standard annotations to an AtomArray"""
    original_categories = set(atom_array.get_annotation_categories())
    for sa_cls in STANDARD_ANNOTATIONS:
        if sa_cls in {C_CTR, C_NTR}:
            continue
        if sa_cls.n_body == 1:
            random_indices = np.random.choice(len(atom_array), size=5, replace=False)
            annotation_arr = sa_cls.default_annotation(atom_array).copy()
            modification_shape = (5, *annotation_arr.shape[1:])
            storage_dtype = sa_cls.storage_dtype()
            if np.issubdtype(storage_dtype, np.bool_):
                random_values = np.ones(modification_shape, dtype=bool)
            elif np.issubdtype(storage_dtype, np.integer):
                random_values = np.random.randint(1, 100, size=modification_shape)
            elif np.issubdtype(storage_dtype, np.floating):
                random_values = np.random.random(modification_shape)
            else:
                annotation_arr = annotation_arr.astype(storage_dtype)
                random_values = np.full(modification_shape, "TEST", dtype=storage_dtype)
            annotation_arr[random_indices, ...] = random_values
            sa_cls.set_annotation(atom_array, array=annotation_arr)

        elif sa_cls.n_body == 2:
            random_indices = np.random.choice(len(atom_array), size=(5, 2), replace=False)
            annotation_shape = sa_cls.default_annotation(atom_array).values.shape
            modification_shape = (5, *annotation_shape[1:])
            storage_dtype = sa_cls.storage_dtype()
            if np.issubdtype(storage_dtype, np.bool_):
                values = np.ones(modification_shape, dtype=bool)
            elif np.issubdtype(storage_dtype, np.integer):
                values = np.random.randint(1, 100, size=modification_shape).astype(storage_dtype)
            else:
                values = np.random.random(modification_shape)
            annotation_arr = AnnotationList2D(
                n_atoms=atom_array.array_length(),
                pairs=random_indices,
                values=values,
            )
            sa_cls.set_annotation(atom_array, pairs=annotation_arr.pairs, values=annotation_arr.values)

    # Remove ordinary annotations generated only as prerequisites of SA defaults.
    standard_annotation_names = set(STANDARD_ANNOTATIONS.get_field_names(n_body=1, include_aliases=True))
    for category in set(atom_array.get_annotation_categories()) - original_categories - standard_annotation_names:
        atom_array.del_annotation(category)

    return atom_array


def assert_tensor_or_array_equal(
    actual: np.ndarray | torch.Tensor | int | float | str | bool,
    expected: np.ndarray | torch.Tensor | int | float | str | bool,
    error_msg: str,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Assert tensors, arrays, or scalars are equal with type-aware tolerance.

    Uses exact equality for int/bool, tolerances for float, and treats NaN as equal.

    Args:
        atol: Absolute tolerance for floats. Defaults to ``1e-4``.
        rtol: Relative tolerance for floats. Defaults to ``1e-4``.

    Raises:
        AssertionError: If values don't match.
    """
    if torch.is_tensor(actual):
        if actual.dtype == torch.bool or actual.dtype in [torch.int32, torch.int64]:
            torch.testing.assert_close(
                actual, expected, atol=0, rtol=0, equal_nan=True, msg=lambda x: error_msg + ": " + x
            )
        else:
            torch.testing.assert_close(
                actual, expected, atol=atol, rtol=rtol, equal_nan=True, msg=lambda x: error_msg + ": " + x
            )
    elif isinstance(actual, np.ndarray):
        if actual.dtype.kind in ["U", "S"] or actual.dtype == bool or np.issubdtype(actual.dtype, np.integer):
            np.testing.assert_array_equal(actual, expected, err_msg=error_msg)
        else:
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True, err_msg=error_msg)
    else:
        # Handle numpy string conversion for scalars
        if isinstance(actual, np.str_):
            actual = str(actual)
        if isinstance(expected, np.str_):
            expected = str(expected)
        assert actual == expected, error_msg


def assert_equal(
    obtained: dict | list | np.ndarray | torch.Tensor | int | float | str | bool,
    expected: dict | list | np.ndarray | torch.Tensor | int | float | str | bool,
    path: str = "",
    allow_extra_keys: bool = False,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Recursively assert nested structures are equal with type-aware comparison.

    Handles dicts, lists, tensors, arrays, and scalars. Uses :py:func:`assert_tensor_or_array_equal`
    for leaf value comparisons.

    Args:
        path: Current path in structure for error messages. Defaults to ``""``.
        allow_extra_keys: Allow extra keys in ``obtained`` dicts. Defaults to ``False``.
        atol: Absolute tolerance for floats. Defaults to ``1e-4``.
        rtol: Relative tolerance for floats. Defaults to ``1e-4``.

    Raises:
        AssertionError: If values don't match.
    """
    error_msg = f"Mismatch at {path}" if path else "Values don't match"

    # Handle dictionaries recursively
    if isinstance(expected, dict):
        if not isinstance(obtained, dict):
            raise AssertionError(f"{error_msg}: expected dict, got {type(obtained)}")

        # Check keys
        missing = set(expected.keys()) - set(obtained.keys())
        if missing:
            raise AssertionError(f"{error_msg}: missing keys {missing}")
        if not allow_extra_keys:
            extra = set(obtained.keys()) - set(expected.keys())
            if extra:
                raise AssertionError(f"{error_msg}: extra keys {extra}")

        # Recurse on values
        for key in expected:
            new_path = f"{path}[{key!r}]" if path else repr(key)
            assert_equal(obtained[key], expected[key], new_path, allow_extra_keys, atol, rtol)
        return

    # Handle lists recursively
    if isinstance(expected, list):
        if not isinstance(obtained, list):
            raise AssertionError(f"{error_msg}: expected list, got {type(obtained)}")
        if len(obtained) != len(expected):
            raise AssertionError(f"{error_msg}: length mismatch {len(obtained)} != {len(expected)}")

        # Convert to numpy arrays for comparison if all elements are numeric
        try:
            obtained_arr = np.array(obtained)
            expected_arr = np.array(expected)
            # If conversion succeeded and types are comparable, use array comparison
            if obtained_arr.dtype.kind in ["i", "u", "f"] and expected_arr.dtype.kind in ["i", "u", "f"]:
                assert_tensor_or_array_equal(obtained_arr, expected_arr, error_msg, atol, rtol)
                return
        except (ValueError, TypeError):
            pass

        # Otherwise, compare element by element
        for i, (obtained_val, expected_val) in enumerate(zip(obtained, expected, strict=False)):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            assert_equal(obtained_val, expected_val, new_path, allow_extra_keys, atol, rtol)
        return

    # All leaf values (tensors, arrays, scalars) handled by assert_tensor_or_array_equal
    assert_tensor_or_array_equal(obtained, expected, error_msg, atol, rtol)


def save_and_load_atom_array(
    atom_array: AtomArrayPlus,
    extra_fields_to_load: list[str] | None = "all",
    extra_annotations_to_save: list[str] | None = "all",
) -> AtomArrayPlus:
    """Save an atom array to a temporary CIF and load it back."""
    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.cif"
        to_cif_file(
            atom_array,
            temp_file,
            extra_fields=extra_annotations_to_save or [],
            save_standard_annotations=True,
        )
        result = parse(
            temp_file,
            config=ParseConfig.from_preset(
                "minimal",
                load_standard_annotations=True,
                return_atom_array_plus=True,
                extra_fields=extra_fields_to_load or [],
            ),
        )
        assembly_ids = list(result["assemblies"].keys())
        loaded = result["assemblies"][assembly_ids[0]][0]
        CONDITIONS.fill_missing_conditions_with_defaults(loaded)
    return loaded
