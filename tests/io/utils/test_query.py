import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.io.transforms.atom_array import is_any_coord_nan
from atomworks.io.utils.io_utils import load_any
from atomworks.io.utils.query import QueryExpression, idxs, mask, query
from atomworks.io.utils.standard_annotations import S_ATM
from atomworks.io.utils.testing import get_pdb_path
from atomworks.ml.conditions import C_IDX, C_NTR


@pytest.fixture(scope="module")
def atom_array():
    """Cache the loaded atom array for all tests in this module."""
    atom_array = load_any(get_pdb_path("6lyz"), model=1, include_bonds=True)
    return atom_array


@pytest.fixture(scope="module")
def atom_array_stack():
    """Cache the loaded atom array stack for all tests in this module."""
    atom_array_stack = load_any(get_pdb_path("6lyz"), include_bonds=True)
    return atom_array_stack


class TestBasicQueries:
    """Test basic comparison operations."""

    def test_chain_equality(self, atom_array):
        """Test chain equality queries."""
        result = query(atom_array, "chain_id == 'A'")
        expected = atom_array[atom_array.chain_id == "A"]
        assert len(result) == len(expected)
        assert np.array_equal(result.chain_id, expected.chain_id)

    def test_chain_inequality(self, atom_array):
        """Test chain inequality queries."""
        result = query(atom_array, "chain_id != 'A'")
        expected = atom_array[atom_array.chain_id != "A"]
        assert len(result) == len(expected)

    def test_residue_number_comparison(self, atom_array):
        """Test numeric comparisons on residue numbers."""
        result = query(atom_array, "res_id > 50")
        expected = atom_array[atom_array.res_id > 50]
        assert len(result) == len(expected)

        result = query(atom_array, "res_id <= 10")
        expected = atom_array[atom_array.res_id <= 10]
        assert len(result) == len(expected)

    def test_atom_name_queries(self, atom_array):
        """Test atom name queries."""
        result = query(atom_array, "atom_name == 'CA'")
        expected = atom_array[atom_array.atom_name == "CA"]
        assert len(result) == len(expected)
        assert np.all(result.atom_name == "CA")


class TestInOperations:
    """Test 'in' and 'not in' operations."""

    def test_residue_name_in_list(self, atom_array):
        """Test residue name in a list."""
        residues = ["ALA", "GLY", "VAL"]
        result = query(atom_array, "res_name in ['ALA', 'GLY', 'VAL']")
        expected = atom_array[np.isin(atom_array.res_name, residues)]
        assert len(result) == len(expected)
        assert np.all(np.isin(result.res_name, residues))

    def test_residue_name_not_in_list(self, atom_array):
        """Test residue name not in a list."""
        residues = ["ALA", "GLY", "VAL"]
        result = query(atom_array, "res_name not in ['ALA', 'GLY', 'VAL']")
        expected = atom_array[~np.isin(atom_array.res_name, residues)]
        assert len(result) == len(expected)
        assert np.all(~np.isin(result.res_name, residues))

    def test_atom_name_in_tuple(self, atom_array):
        """Test atom name in a tuple."""
        atoms = ("CA", "CB", "N")
        result = query(atom_array, "atom_name in ('CA', 'CB', 'N')")
        expected = atom_array[np.isin(atom_array.atom_name, atoms)]
        assert len(result) == len(expected)
        assert np.all(np.isin(result.atom_name, atoms))


class TestLogicalOperations:
    """Test logical combinations of queries."""

    def test_and_operation(self, atom_array):
        """Test AND operations with &."""
        result = query(atom_array, "(chain_id == 'A') & (atom_name == 'CA')")
        expected = atom_array[(atom_array.chain_id == "A") & (atom_array.atom_name == "CA")]
        assert len(result) == len(expected)
        assert np.all(result.chain_id == "A")
        assert np.all(result.atom_name == "CA")

    def test_or_operation(self, atom_array):
        """Test OR operations with |."""
        result = query(atom_array, "(atom_name == 'CA') | (atom_name == 'CB')")
        expected = atom_array[(atom_array.atom_name == "CA") | (atom_array.atom_name == "CB")]
        assert len(result) == len(expected)
        assert np.all(np.isin(result.atom_name, ["CA", "CB"]))

    def test_not_operation(self, atom_array):
        """Test NOT operations with ~."""
        result = query(atom_array, "~(atom_name == 'CA')")
        expected = atom_array[~(atom_array.atom_name == "CA")]
        assert len(result) == len(expected)
        assert np.all(result.atom_name != "CA")

    def test_complex_logical_combination(self, atom_array):
        """Test complex logical combinations."""
        result = query(atom_array, "(chain_id == 'A') & ((atom_name == 'CA') | (atom_name == 'CB')) & (res_id < 50)")

        # Build expected manually
        mask_chain = atom_array.chain_id == "A"
        mask_atoms = (atom_array.atom_name == "CA") | (atom_array.atom_name == "CB")
        mask_res = atom_array.res_id < 50
        expected = atom_array[mask_chain & mask_atoms & mask_res]

        assert len(result) == len(expected)
        assert np.all(result.chain_id == "A")
        assert np.all(np.isin(result.atom_name, ["CA", "CB"]))
        assert np.all(result.res_id < 50)


class TestCoordinateQueries:
    """Test coordinate-based queries."""

    def test_x_coordinate_queries(self, atom_array):
        """Test x-coordinate queries."""
        result = query(atom_array, "x > 0")
        expected = atom_array[atom_array.coord[:, 0] > 0]
        assert len(result) == len(expected)
        assert np.all(result.coord[:, 0] > 0)

    def test_y_coordinate_queries(self, atom_array):
        """Test y-coordinate queries."""
        result = query(atom_array, "y < 0")
        expected = atom_array[atom_array.coord[:, 1] < 0]
        assert len(result) == len(expected)
        assert np.all(result.coord[:, 1] < 0)

    def test_z_coordinate_queries(self, atom_array):
        """Test z-coordinate queries."""
        result = query(atom_array, "z >= 10")
        expected = atom_array[atom_array.coord[:, 2] >= 10]
        assert len(result) == len(expected)
        assert np.all(result.coord[:, 2] >= 10)

    def test_coordinate_combination(self, atom_array):
        """Test combination of coordinate queries."""
        result = query(atom_array, "(x > 0) & (y > 0) & (z > 0)")
        assert np.all(result.coord[:, 0] > 0)
        assert np.all(result.coord[:, 1] > 0)
        assert np.all(result.coord[:, 2] > 0)


class TestFunctionCalls:
    """Test function call queries."""

    @pytest.mark.parametrize("array_name", ["atom_array", "atom_array_stack"])
    def test_has_nan_coord_function(self, array_name, request):
        """Test has_nan_coord() function."""
        array = request.getfixturevalue(array_name)
        mask_result = mask(array, "~has_nan_coord()")
        # Should return atoms without NaN coordinates
        expected_mask = ~is_any_coord_nan(array)
        assert np.array_equal(mask_result, expected_mask)

    @pytest.mark.parametrize("array_name", ["atom_array", "atom_array_stack"])
    def test_has_bonds_function(self, array_name, request):
        """Test has_bonds() function."""
        array = request.getfixturevalue(array_name)
        mask_result = mask(array, "has_bonds()")
        # Should return atoms that are bonded
        bonded_idxs = np.unique(array.bonds.as_array()[:, :2])
        expected_mask = np.isin(np.arange(array.array_length()), bonded_idxs)
        assert np.array_equal(mask_result, expected_mask)

    @pytest.mark.parametrize("array_name", ["atom_array", "atom_array_stack"])
    def test_function_with_logical_operations(self, array_name, request):
        """Test functions combined with logical operations."""
        array = request.getfixturevalue(array_name)
        mask_result = mask(array, "~has_nan_coord() & (atom_name == 'CA')")
        assert np.all(array.atom_name[mask_result] == "CA")
        assert not np.any(is_any_coord_nan(array) & mask_result)


class TestReturnTypes:
    """Test different return types (query, mask, idxs)."""

    def test_mask_return_type(self, atom_array):
        """Test that mask returns boolean array."""
        result_mask = mask(atom_array, "chain_id == 'A'")
        expected_mask = atom_array.chain_id == "A"

        assert isinstance(result_mask, np.ndarray)
        assert result_mask.dtype == bool
        assert len(result_mask) == atom_array.array_length()
        assert np.array_equal(result_mask, expected_mask)

    def test_idxs_return_type(self, atom_array):
        """Test that idxs returns integer array."""
        result_idxs = idxs(atom_array, "chain_id == 'A'")
        expected_idxs = np.where(atom_array.chain_id == "A")[0]

        assert isinstance(result_idxs, np.ndarray)
        assert result_idxs.dtype == np.int64 or result_idxs.dtype == np.int32
        assert np.array_equal(result_idxs, expected_idxs)

    def test_query_return_type(self, atom_array):
        """Test that query returns AtomArray."""
        result = query(atom_array, "chain_id == 'A'")
        expected = atom_array[atom_array.chain_id == "A"]

        assert type(result) == type(atom_array)
        assert result.array_length() == expected.array_length()


class TestMonkeyPatching:
    """Test monkey-patched methods on AtomArray."""

    def test_query_method(self, atom_array):
        """Test atom_array.query() method."""
        result = atom_array.query("chain_id == 'A'")
        expected = query(atom_array, "chain_id == 'A'")
        assert len(result) == len(expected)

    def test_mask_method(self, atom_array):
        """Test atom_array.mask() method."""
        result = atom_array.mask("chain_id == 'A'")
        expected = mask(atom_array, "chain_id == 'A'")
        assert np.array_equal(result, expected)

    def test_idxs_method(self, atom_array):
        """Test atom_array.idxs() method."""
        result = atom_array.idxs("chain_id == 'A'")
        expected = idxs(atom_array, "chain_id == 'A'")
        assert np.array_equal(result, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_result(self, atom_array):
        """Test query that returns empty result."""
        result = query(atom_array, "res_id > 10000")  # Should be empty
        assert len(result) == 0

    def test_all_atoms_result(self, atom_array):
        """Test query that returns all atoms."""
        result = query(atom_array, "res_id >= 0")  # Should return all
        assert len(result) == len(atom_array[atom_array.res_id >= 0])

    def test_invalid_attribute_error(self, atom_array):
        """Test error when querying non-existent attribute."""
        with pytest.raises(NameError, match="Name 'nonexistent_attr' is not defined"):
            query(atom_array, "nonexistent_attr == 'A'")

    def test_invalid_function_error(self, atom_array):
        """Test error when calling non-existent function."""
        with pytest.raises(NameError, match="Function 'nonexistent_func' is not defined"):
            query(atom_array, "nonexistent_func()")

    def test_function_with_arguments_error(self, atom_array):
        """Test error when calling function with arguments."""
        with pytest.raises(ValueError, match="does not accept arguments"):
            query(atom_array, "has_nan_coord('arg')")


class TestChainedComparisons:
    """Test chained comparison operations."""

    def test_chained_numeric_comparison(self, atom_array):
        """Test chained numeric comparisons."""
        result = query(atom_array, "10 <= res_id <= 50")
        expected = atom_array[(atom_array.res_id >= 10) & (atom_array.res_id <= 50)]
        assert len(result) == len(expected)
        assert np.all((result.res_id >= 10) & (result.res_id <= 50))

    def test_chained_coordinate_comparison(self, atom_array):
        """Test chained coordinate comparisons."""
        result = query(atom_array, "-10 < x < 10")
        expected = atom_array[(atom_array.coord[:, 0] > -10) & (atom_array.coord[:, 0] < 10)]
        assert len(result) == len(expected)
        assert np.all((result.coord[:, 0] > -10) & (result.coord[:, 0] < 10))


class TestSpecialCases:
    """Test special cases and data types."""

    def test_string_comparison(self, atom_array):
        """Test string comparisons with different operators."""
        # Test that string inequality works
        result = query(atom_array, "atom_name != 'CA'")
        assert np.all(result.atom_name != "CA")

    def test_boolean_annotation_if_exists(self, atom_array):
        """Test boolean annotation queries if they exist."""
        # Check if there are any boolean annotations
        for attr in atom_array.get_annotation_categories():
            values = getattr(atom_array, attr)
            if hasattr(values, "dtype") and values.dtype == bool:
                # Test boolean query
                result = query(atom_array, f"{attr}")
                expected = atom_array[values]
                assert len(result) == len(expected)
                break

    def test_numeric_vs_string_types(self, atom_array):
        """Test that numeric and string types are handled correctly."""
        # This should work - comparing numbers
        result1 = query(atom_array, "res_id == 1")
        # This should also work - comparing strings
        result2 = query(atom_array, "atom_name == 'CA'")

        assert isinstance(result1, type(atom_array))
        assert isinstance(result2, type(atom_array))


@pytest.mark.parametrize(
    "expr, expected",
    [
        # bare path selection via sel()
        ("sel('A/*/*/CA')", lambda a: (a.chain_id == "A") & (a.atom_name == "CA")),
        # path selection intersected with a query predicate
        ("sel('A/*/*/CA') & (res_id < 50)", lambda a: (a.chain_id == "A") & (a.atom_name == "CA") & (a.res_id < 50)),
        # bracket list inside sel()
        ("sel('A/*/*/[CA,CB]')", lambda a: (a.chain_id == "A") & np.isin(a.atom_name, ["CA", "CB"])),
        # top-level comma unions tokens inside sel()
        ("sel('A/*/*/CA, A/*/*/CB')", lambda a: (a.chain_id == "A") & np.isin(a.atom_name, ["CA", "CB"])),
        # empty sel() is non-raising: contributes an all-False mask and composes with |
        ("sel('ZZ/*/*/CA') | (atom_name == 'CA')", lambda a: a.atom_name == "CA"),
    ],
)
def test_sel_in_query(atom_array, expr, expected):
    """``sel('...')`` bridges path-selection syntax into query expressions."""
    assert np.array_equal(mask(atom_array, expr), expected(atom_array))


def test_query_expression_reuse(atom_array):
    """Test that QueryExpression can be reused across different arrays."""

    # Create a query expression
    expr = QueryExpression("atom_name == 'CA'")

    # Use it multiple times
    result1 = expr.query(atom_array)
    result2 = expr.query(atom_array)

    assert result1.array_length() == result2.array_length()
    assert np.array_equal(result1.atom_name, result2.atom_name)

    # Test that mask and idxs work too
    mask1 = expr.mask(atom_array)
    mask2 = expr.mask(atom_array)
    assert np.array_equal(mask1, mask2)

    idxs1 = expr.idxs(atom_array)
    idxs2 = expr.idxs(atom_array)
    assert np.array_equal(idxs1, idxs2)


def test_standard_annotation_mask_query(atom_array):
    array = atom_array.copy()
    expected = np.arange(array.array_length()) % 2 == 0
    C_IDX.set_annotation(array, expected)

    assert np.array_equal(mask(array, "sa_mask('index')"), expected)
    assert np.array_equal(mask(array, f"sa_mask('{C_IDX.full_name}')"), expected)


def test_standard_annotation_mask_query_resolves_shortcuts_and_storage_aliases(atom_array):
    array = atom_array.copy()
    expected = np.arange(array.array_length()) % 2 == 0
    array.set_annotation("mask_n-terminus_1_residue", expected)
    S_ATM.set_annotation(array, expected)

    assert np.array_equal(mask(array, "sa_mask('C_NTR')"), expected)
    assert np.array_equal(mask(array, f"sa_mask('{C_NTR.full_name}')"), expected)
    assert np.array_equal(mask(array, "sa_mask('mask_n-terminus_1_residue')"), expected)
    assert np.array_equal(mask(array, "sa_mask('S_ATM')"), expected)


def test_apply_and_spread_query():
    array = AtomArray(4)
    array.chain_id = np.array(["A", "A", "A", "A"])
    array.res_id = np.array([1, 1, 2, 2])
    array.atom_name = np.array(["N", "CA", "N", "C"])
    array.set_annotation("b_factor", np.array([1.0, 3.0, 4.0, 6.0]))

    assert mask(array, "apply_and_spread('residue', atom_name == 'CA', 'any')").tolist() == [True, True, False, False]
    assert mask(array, "apply_and_spread('residue', b_factor, 'mean') > 4").tolist() == [False, False, True, True]
