"""Tests for extra fields utilities."""

import numpy as np

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.extra_fields import (
    get_default_array,
    infer_dtype_from_values,
    normalize_extra_fields,
    transfer_annotation_if_exists,
)
from atomworks.io.utils.io_utils import load_any
from atomworks.io.utils.testing import get_pdb_path
from atomworks.ml.utils.testing import cached_parse
from tests.io.conftest import TEST_DATA_IO


def test_normalize_extra_fields_formats():
    """Test normalization of various input formats."""
    # List format
    result = normalize_extra_fields(["field1", "field2"])
    assert list(result.keys()) == ["field1", "field2"]

    # Dict with None
    result = normalize_extra_fields({"field1": None})
    assert result["field1"] == {"default": None, "dtype": None}

    # Dict with default (dtype inferred)
    result = normalize_extra_fields({"field": {"default": 0}})
    assert result["field"]["dtype"] == np.int64

    # Dict with dtype only
    result = normalize_extra_fields({"field": {"dtype": np.int32}})
    assert result["field"]["dtype"] == np.int32
    assert result["field"]["default"] is None

    # None and "all" return empty
    assert normalize_extra_fields(None) == {}
    assert normalize_extra_fields("all") == {}


def test_normalize_require_defaults():
    """Test require_defaults validation."""
    # Fields without defaults are filtered out (with warning)
    result = normalize_extra_fields(["field1"], require_defaults=True)
    assert "field1" not in result  # Filtered out

    # Passes with default
    result = normalize_extra_fields({"f": {"default": 0}}, require_defaults=True)
    assert result["f"]["default"] == 0

    # Passes with dtype only
    result = normalize_extra_fields({"f": {"dtype": np.int32}}, require_defaults=True)
    assert result["f"]["dtype"] == np.int32

    # Mixed: keeps fields with defaults, filters out those without
    result = normalize_extra_fields(
        {"with_default": {"default": 0}, "without_default": None},
        require_defaults=True,
    )
    assert "with_default" in result
    assert "without_default" not in result


def test_infer_dtype_from_values():
    """Test dtype inference from string and numeric values."""
    # String to int
    dtype, converted = infer_dtype_from_values(np.array(["1", "2", "3"]))
    assert dtype == np.int64
    assert np.array_equal(converted, [1, 2, 3])

    # String to float
    dtype, _ = infer_dtype_from_values(np.array(["1.5", "2.0"]))
    assert dtype == np.float64

    # Non-numeric stays string
    dtype, _ = infer_dtype_from_values(np.array(["A", "B"]))
    assert dtype.kind == "U"

    # Already numeric unchanged
    dtype, _ = infer_dtype_from_values(np.array([1, 2, 3], dtype=np.int32))
    assert dtype == np.int32


def test_get_default_array():
    """Test default array generation."""
    # Explicit default
    arr = get_default_array({"default": -99, "dtype": np.int32}, 5)
    assert arr.shape == (5,)
    assert arr.dtype == np.int32
    assert np.all(arr == -99)

    # Dtype-only uses sensible defaults
    assert np.all(get_default_array({"default": None, "dtype": np.int32}, 3) == -1)
    assert np.all(np.isnan(get_default_array({"default": None, "dtype": np.float64}, 3)))


def test_load_any_backward_compatible(compressed_example):
    """Test old extra_fields formats still work."""
    path = compressed_example(TEST_DATA_IO / "1a8o_modified.cif")
    # List format
    result = load_any(path, extra_fields=["b_factor", "occupancy"])
    assert "b_factor" in result.get_annotation_categories()

    # Dict format
    result = load_any(path, extra_fields={"b_factor": None})
    assert "b_factor" in result.get_annotation_categories()
    assert np.issubdtype(result.b_factor.dtype, np.floating)


def test_parse_extra_fields_with_add_missing_atoms():
    """Test extra fields with add_missing_atoms."""
    # Fields without defaults are skipped (not loaded) with warning
    result = parse(get_pdb_path("1a8o"), config=ParseConfig(add_missing_atoms=True, extra_fields=["custom"]))
    asym_unit = result["asym_unit"][0]
    assert "custom" not in asym_unit.get_annotation_categories()

    # Works with defaults
    result = parse(
        get_pdb_path("1a8o"),
        config=ParseConfig(add_missing_atoms=True, extra_fields={"atom_id": {"default": -1, "dtype": np.int64}}),
    )
    asym_unit = result["asym_unit"][0]
    assert "atom_id" in asym_unit.get_annotation_categories()

    # Unresolved atoms have the default
    unresolved = asym_unit.occupancy == 0
    if np.any(unresolved):
        assert np.any(asym_unit.atom_id[unresolved] == -1)


def test_transfer_annotation_if_exists():
    """Test transferring annotation value when it exists, or doing nothing when it doesn't."""
    source = cached_parse("1a8o")["atom_array"]
    target = source.copy()

    # Test when annotation exists - broadcasts source[0] to all target atoms
    transfer_annotation_if_exists(source, target, "chain_id", 0)
    assert np.all(target.chain_id == source.chain_id[0])

    # Test when annotation doesn't exist - does nothing
    transfer_annotation_if_exists(source, target, "nonexistent", 0)
    assert "nonexistent" not in target.get_annotation_categories()


def test_af3_cond_design_cif_example(compressed_example):
    path = compressed_example(TEST_DATA_IO / "example_conditional_generation_output.cif")
    result = parse(
        path,
        config=ParseConfig(add_missing_atoms=False, extra_fields="all"),
    )
    # Check if processing runs through
    assert result is not None

    atom_array = result["assemblies"]["1"][0]
    assert atom_array.condition_motif_index is not None
    assert atom_array.condition_coords_gt_x is not None

    result = parse(
        path,
        config=ParseConfig(add_missing_atoms=False, extra_fields=["condition_coords_gt_x"]),
    )
    atom_array = result["assemblies"]["1"][0]
    assert atom_array.condition_coords_gt_x is not None
    assert "condition_coords_gt_y" not in atom_array._annot
