from copy import deepcopy

import biotite.structure as struc
import numpy as np
import pytest

from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import (
    AnnotationList2D,
    AtomArrayPlus,
    AtomArrayPlusStack,
    concatenate_any,
    concatenate_atom_array_plus,
    insert_atoms,
    stack_any,
    stack_atom_array_plus,
)
from atomworks.io.utils.io_utils import CIFWriteConfig, to_cif_buffer
from atomworks.io.utils.standard_annotations import S_ATM, S_SEGMAX, S_SEGMIN
from atomworks.io.utils.standard_annotations.base import Level, StandardAnnotationBase, StandardAnnotationMeta
from atomworks.ml.conditions import C_DIS, C_SEQ
from tests.conftest import TEST_DATA_DIR


# --- Fixtures ---
@pytest.fixture
def simple_array():
    """Create a simple AtomArrayPlus with a single 2D annotation."""
    arr = AtomArrayPlus(4)
    arr.set_annotation_2d("distances", [(0, 1), (1, 2), (2, 3)], [1.1, 2.2, 3.3])
    return arr


@pytest.fixture
def complex_array():
    """Create a complex AtomArrayPlus with multiple 2D annotations."""
    arr = AtomArrayPlus(7)
    arr.set_annotation_2d(
        "dist", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    )
    arr.set_annotation_2d("contact_strength", [(0, 2), (2, 4), (4, 6)], [111.0, 112.0, 113.0])
    return arr


@pytest.fixture
def multi_array_pair():
    """Create a pair of AtomArrayPlus objects for concatenation tests."""
    arr1 = AtomArrayPlus(2)
    arr1.set_annotation_2d("dist", [(0, 1)], [1.0])

    arr2 = AtomArrayPlus(3)
    arr2.set_annotation_2d("dist", [(1, 2)], [2.0])

    return arr1, arr2


@pytest.fixture
def complete_atom_array_plus():
    """A canonical AtomArrayPlus with 4 atoms, bonds, and a 2D annotation"""
    arr = AtomArrayPlus(4)

    # 1D annotations
    arr.coord = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    arr.element = np.array(["C", "N", "O", "S"])

    # 2D annotation
    arr.set_annotation_2d("dist", [(0, 1), (1, 2), (2, 3)], [1.0, 2.0, 3.0])

    # Bonds
    bonds = struc.BondList(4)
    bonds.add_bond(0, 1)
    bonds.add_bond(1, 2)
    bonds.add_bond(2, 3)
    bonds.add_bond(1, 3)
    arr.bonds = bonds

    return arr


@pytest.fixture
def simple_annotation_list_2d():
    """A simple AnnotationList2D with 4 atoms and 3 pairs."""
    return AnnotationList2D(4, np.array([[0, 1], [1, 2], [2, 3]]), np.array([1.1, 2.2, 3.3]))


# --- Test Classes ---
class TestAnnotationList2D:
    """Tests for the AnnotationList2D class (sparse 2D annotation logic).

    AnnotationList2D should function analagously to a BondList.
    """

    def test_set_and_retrieve(self, simple_annotation_list_2d):
        """Test setting and retrieving a 2D annotation."""
        arr = simple_annotation_list_2d
        # Check type and length
        assert isinstance(arr, AnnotationList2D)
        assert len(arr) == 3

        # Check values and pairs
        assert np.allclose(arr.values, [1.1, 2.2, 3.3])
        assert np.array_equal(arr.pairs, np.array([[0, 1], [1, 2], [2, 3]]))

    def test_sparse_representation(self, simple_annotation_list_2d):
        """Test sparse representation of 2D annotations."""
        arr = deepcopy(simple_annotation_list_2d)
        sparse = arr.as_array()

        assert np.array_equal(sparse[:, 0:2], np.array([[0, 1], [1, 2], [2, 3]]))
        assert np.array_equal(sparse[:, 2], np.array([1.1, 2.2, 3.3]))

    def test_dense_representation(self, simple_annotation_list_2d):
        """Test dense representation of 2D annotations."""
        arr = deepcopy(simple_annotation_list_2d)
        dense = arr.as_dense_array()

        assert np.allclose(dense[0, 1], 1.1)
        assert np.allclose(dense[1, 2], 2.2)
        assert np.allclose(dense[2, 3], 3.3)
        assert np.isnan(dense[0, 2])
        assert dense.shape == (4, 4)

    def test_add_row(self):
        """Test adding a row to an AnnotationList2D."""
        arr = AnnotationList2D(3, np.array([[0, 1]]), np.array([5]))
        arr.add_row(1, 2, 7)

        assert len(arr) == 2
        assert (1, 2, 7) in [arr[i] for i in range(len(arr))]

    def test_from_dense(self):
        """Test creating an AnnotationList2D from a dense array."""
        dense = np.array([[np.nan, 1, np.nan], [1, np.nan, np.nan], [np.nan, np.nan, 1]], dtype=float)
        ann = AnnotationList2D.from_dense_array(dense)

        assert ann.n_atoms == 3
        assert np.array_equal(ann.as_array(), np.array([[0, 1, 1], [1, 0, 1], [2, 2, 1]]))
        assert np.allclose(ann.as_dense_array(), dense, equal_nan=True)

    def test_concatenate(self):
        """Test concatenation of AnnotationList2D objects."""
        a1 = AnnotationList2D(2, np.array([[0, 1]]), np.array([1.0]))
        a2 = AnnotationList2D(3, np.array([[1, 2]]), np.array([2.0]))
        a_cat = AnnotationList2D.concatenate([a1, a2], [2, 3])

        assert a_cat.n_atoms == 5
        assert set(map(tuple, a_cat.pairs)) == {(0, 1), (3, 4)}
        assert np.allclose(sorted(a_cat.values), [1.0, 2.0])

        dense = a_cat.as_dense_array()

        assert np.allclose(dense[0, 1], 1.0)
        assert np.allclose(dense[3, 4], 2.0)
        assert np.isnan(dense[1, 2])
        assert dense.shape == (5, 5)

    def test_indexing(self):
        """Test various indexing types on AnnotationList2D."""
        arr = AnnotationList2D(5, np.array([[0, 1], [1, 2], [2, 3], [3, 4]]), np.array([10.0, 20.0, 30.0, 40.0]))

        # Slice: keep atoms 1, 2, 3
        sliced = arr[[1, 2, 3]]
        assert sliced.n_atoms == 3
        assert set(map(tuple, sliced.pairs)) == {(0, 1), (1, 2)}
        assert np.allclose(sorted(sliced.values), [20.0, 30.0])

        # Negative indexing
        arr2 = AnnotationList2D(4, np.array([[0, 1], [1, 2], [2, 3]]), np.array([1.0, 2.0, 3.0]))
        sliced2 = arr2[[0, 1, -1]]
        assert sliced2.n_atoms == 3
        assert set(map(tuple, sliced2.pairs)) == {(0, 1)}
        assert np.allclose(sliced2.values, [1.0])

    def test_empty_annotation(self):
        """Test setting and retrieving an empty 2D annotation."""
        arr = AnnotationList2D(3, [], [])
        assert arr.pairs.shape == (0, 2)
        assert arr.values.shape == (0,)
        assert arr.n_atoms == 3

    def test_symmetrization(self):
        """Test ensure_symmetric adds missing pairs and detects value mismatches."""
        # Adds missing reverse pairs
        annot = AnnotationList2D(3, np.array([[0, 0], [0, 1], [1, 2]]), np.array([0.5, 1.5, 2.5]))
        result = annot.symmetrized()
        assert len(result.pairs) == 5
        assert np.allclose(result.pairs, np.array([[0, 0], [0, 1], [1, 2], [1, 0], [2, 1]]))
        assert np.allclose(result.values, [0.5, 1.5, 2.5, 1.5, 2.5])

        # Errors on value mismatch
        with pytest.raises(ValueError, match="^Asymmetric input values"):
            AnnotationList2D(3, np.array([[0, 1], [1, 0]]), np.array([1.0, 2.0])).symmetrized()

        # Handles NaN correctly (NaN == NaN)
        assert len(AnnotationList2D(3, np.array([[0, 1], [1, 0]]), np.array([np.nan, np.nan])).symmetrized().pairs) == 2

        # Some more edge cases
        annot = AnnotationList2D(
            4,
            np.array([[0, 1], [0, 2], [2, 1], [1, 3]]),  # upper and lower triangle pairs
            np.array([1.0, 2.0, 3.0, 4.0]),
        )
        result = annot.symmetrized()
        assert len(result.pairs) == 8
        assert np.allclose(
            result.as_dense_array(),
            np.array(
                [
                    [np.nan, 1.0, 2.0, np.nan],
                    [1.0, np.nan, 3.0, 4.0],
                    [2.0, 3.0, np.nan, np.nan],
                    [np.nan, 4.0, np.nan, np.nan],
                ]
            ),
            equal_nan=True,
        )


def _make_single_atom(element: str, coord: list[float], annotations_2d: list[str] = None) -> AtomArrayPlus:
    """Helper function to create a single-atom AtomArrayPlus with the given element, coordinates, and optional 2D annotation names."""
    atom = AtomArrayPlus(1)
    atom.coord = np.array([coord], dtype=float)
    atom.element = np.array([element])
    if annotations_2d:
        for name in annotations_2d:
            atom.set_annotation_2d(name, [], [])
    return atom


class TestAtomArrayPlus:
    """Tests for AtomArrayPlus, including 2D annotation logic, slicing, copying, and equality."""

    @pytest.mark.parametrize("as_bcif", [False, True])
    def test_system_annotations_survive_structure_operations(self, monkeypatch, as_bcif):
        """System scalars survive structure operations and CIF model selection; conflicting models raise."""
        monkeypatch.setattr(StandardAnnotationMeta, "_registry", StandardAnnotationMeta._registry.copy())

        class PointGroup(StandardAnnotationBase):
            name = "testpointgroup"
            n_body, level, dtype = 0, Level.SYSTEM, str
            default_value = ""

            @classmethod
            def mask_from_annotation(cls, annotation):
                return annotation != ""

        config = ParseConfig.from_preset("minimal", load_standard_annotations=True, return_atom_array_plus=True)
        atoms = parse(TEST_DATA_DIR / "io" / "6lyz.bcif", config=config)["asym_unit"][0]
        value = np.array("C3")
        PointGroup.set_annotation(atoms, value)
        value[...] = "D2"
        assert PointGroup.full_name not in atoms.get_annotation_categories()
        for result in (
            atoms.copy(),
            atoms[:2],
            concatenate_atom_array_plus([atoms, atoms]),
            stack_atom_array_plus([atoms, atoms]),
        ):
            annotation = PointGroup.annotation(result)
            assert annotation.shape == () and annotation.item() == "C3"
            annotation[...] = "D2"
            assert PointGroup.annotation(atoms).item() == "C3"
        other = atoms.copy()
        PointGroup.set_annotation(other, "D2")
        with pytest.raises(ValueError, match="conflicting system annotation"):
            concatenate_atom_array_plus([atoms, other])

        config = config.replace(file_type="bcif" if as_bcif else "cif")
        for second in (atoms, other):
            buffer = to_cif_buffer(
                [atoms, second], model_ids=[7, 3], as_bcif=as_bcif, config=CIFWriteConfig(include_chem_comp=False)
            )
            if second is other:
                with pytest.raises(ValueError, match="Conflicting system values"):
                    parse(buffer, config=config)
            else:
                assert PointGroup.annotation(parse(buffer, config=config)["asym_unit"]).item() == "C3"
            for model, expected in ((1, "C3"), (-1, PointGroup.annotation(second).item())):
                buffer.seek(0)
                loaded = parse(buffer, config=config.replace(model=model))["asym_unit"]
                assert PointGroup.annotation(loaded).item() == expected
                assert PointGroup.mask(loaded).item()

    def test_set_and_get_annotation_2d(self, simple_array):
        """Test setting and retrieving a 2D annotation on AtomArrayPlus."""
        arr = simple_array.copy()

        # Set annotation
        arr.set_annotation_2d("foo", [(0, 1)], [42])
        ann = arr.get_annotation_2d("foo")
        assert np.array_equal(ann.pairs, np.array([[0, 1]]))
        assert np.array_equal(ann.values, np.array([42]))

        # Overwrite
        arr.set_annotation_2d("foo", [(1, 0)], [24])
        ann2 = arr.get_annotation_2d("foo")
        assert np.array_equal(ann2.pairs, np.array([[1, 0]]))
        assert np.array_equal(ann2.values, np.array([24]))

        source = AnnotationList2D(len(arr), [(0, 1)], [42])
        arr.set_annotation("foo", source, n_body=2)
        source.add_row(1, 2, 24)
        assert len(arr.get_annotation("foo", n_body=2)) == 1
        source.pairs[0, 0] = len(arr)
        with pytest.raises(AssertionError, match="valid atom indices"):
            arr.set_annotation("foo", source, n_body=2)

    def test_get_annotation_2d_categories(self, simple_array):
        """Test retrieving all 2D annotation names from AtomArrayPlus."""
        arr = deepcopy(simple_array)
        arr.set_annotation_2d("foo", [(0, 1)], [42])
        arr.set_annotation_2d("bar", [(1, 0)], [24])
        names = arr.get_annotation_2d_categories()
        assert set(names) == {"distances", "foo", "bar"}

    def test_del_annotation_2d(self, simple_array):
        """Test deleting 2D annotations from AtomArrayPlus."""
        arr = deepcopy(simple_array)
        arr.set_annotation_2d("foo", [(0, 1)], [42])
        arr.set_annotation_2d("bar", [(1, 0)], [24])

        # Verify initial state
        assert set(arr.get_annotation_2d_categories()) == {"distances", "foo", "bar"}

        # Delete one annotation
        arr.del_annotation_2d("foo")
        assert set(arr.get_annotation_2d_categories()) == {"distances", "bar"}

        # Delete another
        arr.del_annotation_2d("distances")
        assert set(arr.get_annotation_2d_categories()) == {"bar"}

        # Delete nonexistent annotation (should not raise error, like AtomArray.del_annotation)
        arr.del_annotation_2d("nonexistent")  # Should do nothing
        assert set(arr.get_annotation_2d_categories()) == {"bar"}

    def test_slice_and_filter_2d_annotations(self, complex_array):
        """Test filtering of 2D annotations when slicing AtomArrayPlus."""
        arr = complex_array.copy()

        # ... filter with non-contiguous indices
        sliced = arr[[1, 2, 4, 6]]

        dist = sliced.get_annotation_2d("dist")
        assert dist.n_atoms == 4
        assert set(map(tuple, dist.pairs)) == {(0, 1)}
        dense = dist.as_dense_array()
        assert np.all(np.isnan(dense[1:, :1])) or np.all(np.isnan(dense[2:, :2]))  # keep original check
        assert dense.shape == (4, 4)

    def test_concatenate_and_slice(self, multi_array_pair):
        """Test concatenation followed by slicing on AtomArrayPlus."""
        arr1, arr2 = deepcopy(multi_array_pair)
        arr1.box = np.eye(3)
        originals = [array.copy() for array in (arr1, arr2)]
        # ... concatenate
        arr_cat = concatenate_any([arr1, arr2])

        # ... slice
        sliced = arr_cat[[1, 2, 3]]

        dist = sliced.get_annotation_2d("dist")
        assert dist.n_atoms == 3
        assert set(map(tuple, dist.pairs)) == set()
        assert len(dist.values) == 0

        dense = dist.as_dense_array()
        assert np.all(np.isnan(dense))
        assert dense.shape == (3, 3)

        arr_cat.coord[:] = -1
        arr_cat.box[:] = 0
        arr_cat.get_annotation_2d("dist").values[:] = -1
        for array, original in zip((arr1, arr2), originals, strict=True):
            np.testing.assert_array_equal(array.coord, original.coord)
            np.testing.assert_array_equal(array.box, original.box)
            np.testing.assert_array_equal(
                array.get_annotation_2d("dist").values, original.get_annotation_2d("dist").values
            )

    def test_concatenate_fills_only_missing_standard_annotations(self):
        """StandardAnnotations use union-with-defaults; ordinary annotations still use intersection."""
        left = AtomArrayPlus(2)
        right = AtomArrayPlus(2)
        for chain_id, array in zip(("A", "B"), (left, right), strict=True):
            array.chain_id[:] = chain_id
            array.res_id[:] = [1, 2]
            array.res_name[:] = "ALA"
            array.atom_name[:] = "CA"
            array.element[:] = "C"

        S_ATM.set_annotation(left, np.array([False, True]))
        S_ATM.set_annotation(right, np.array([True, False]))

        S_SEGMIN.set_annotation(left, np.array([-1, 8]))
        S_SEGMAX.set_annotation(left, np.array([-1, 12]))

        C_SEQ.set_annotation(right, np.array(["ALA", "<M>"]))
        ensure_annotations(left, "is_protein_backbone")

        original_categories = [set(array.get_annotation_categories()) for array in (left, right)]
        with pytest.raises(ValueError, match="Annotation mismatch across inputs"):
            concatenate_atom_array_plus([left, right])
        with pytest.warns(UserWarning, match="dropping them"):
            warned = concatenate_atom_array_plus([left, right], on_annotation_mismatch_policy="warn")
        dropped = concatenate_atom_array_plus([left, right], on_annotation_mismatch_policy="drop")
        with pytest.raises(ValueError, match="is_protein_backbone"):
            concatenate_atom_array_plus([left, right], fill_missing_standard_annotations=True)
        with pytest.warns(UserWarning, match="is_protein_backbone"):
            warned_and_filled = concatenate_atom_array_plus(
                [left, right],
                on_annotation_mismatch_policy="warn",
                fill_missing_standard_annotations=True,
            )
        filled = concatenate_atom_array_plus(
            [left, right],
            on_annotation_mismatch_policy="drop",
            fill_missing_standard_annotations=True,
        )

        shared_standard_fields = {S_ATM.full_name}
        one_sided_standard_fields = {
            S_SEGMIN.full_name,
            S_SEGMAX.full_name,
            C_SEQ.full_name,
        }
        assert shared_standard_fields | one_sided_standard_fields <= set(filled.get_annotation_categories())
        assert shared_standard_fields <= set(dropped.get_annotation_categories())
        assert one_sided_standard_fields.isdisjoint(dropped.get_annotation_categories())
        assert set(warned.get_annotation_categories()) == set(dropped.get_annotation_categories())
        assert set(warned_and_filled.get_annotation_categories()) == set(filled.get_annotation_categories())
        assert "is_protein_backbone" not in filled.get_annotation_categories()
        assert "is_protein_backbone" not in dropped.get_annotation_categories()

        assert np.array_equal(S_SEGMIN.annotation(filled), [-1, 8, -1, -1])
        assert np.array_equal(S_SEGMAX.annotation(filled), [-1, 12, -1, -1])
        assert np.array_equal(C_SEQ.mask(filled), [False, False, True, False])
        assert [set(array.get_annotation_categories()) for array in (left, right)] == original_categories

    def test_equality_and_nan_handling(self, simple_array):
        """Test equality and NaN handling for AtomArrayPlus 2D annotations."""
        arr1 = simple_array.copy()
        arr2 = simple_array.copy()

        arr1.set_annotation_2d("foo", [(1, 2)], [float("nan")])
        arr2.set_annotation_2d("foo", [(1, 2)], [float("nan")])

        assert arr1.equal_annotations(arr2, equal_nan=True)
        assert not arr1.equal_annotations(arr2, equal_nan=False)

        # ... change one value and ensure they are not equal
        arr2.set_annotation_2d("distances", [(0, 1)], [2.0])
        assert not arr1.equal_annotations(arr2, equal_nan=True)

    def test_copy_preserves_and_is_deep(self, simple_array):
        """Test that copying AtomArrayPlus preserves 2D annotations and is deep."""
        arr = simple_array.copy()
        # (Basic copy)
        arr.set_annotation_2d("foo", [(0, 1)], [42])
        arr2 = arr.copy()
        assert arr.equal_annotations(arr2)

        # ... change one value and ensure they are not equal (checks for deep copy)
        arr2.set_annotation_2d("foo", [(1, 0)], [24])
        assert not arr.equal_annotations(arr2)

    def test_set_empty_2d_annotation(self, simple_array):
        """Test setting and retrieving an empty 2D annotation on AtomArrayPlus."""
        arr = simple_array
        arr.set_annotation_2d("empty", [], [])
        ann = arr.get_annotation_2d("empty")

        assert ann.pairs.shape == (0, 2)
        assert ann.values.shape == (0,)
        assert ann.n_atoms == 4

    def test_atomarray_roundtrip(self, complete_atom_array_plus):
        """Test AtomArrayPlus -> AtomArray -> AtomArrayPlus roundtrip preserves bonds and 1D annotations, but loses 2D annotations."""
        arr_plus = complete_atom_array_plus.copy()
        arr = arr_plus.as_atom_array()
        arr_plus2 = AtomArrayPlus.from_atom_array(arr)

        # Bonds are preserved
        assert arr_plus2.bonds == arr_plus.bonds

        # 2D annotations are lost
        assert arr_plus2.get_annotation_2d_categories() == []

        # 1D annotations and coordinates are preserved
        np.testing.assert_allclose(arr_plus2.coord, arr_plus.coord)
        assert np.array_equal(arr_plus2.element, arr_plus.element)

    def test_insert_atoms_with_2d_annotations(self, complete_atom_array_plus):
        """Test inserting atoms with 2D annotations and bond remapping."""
        arr = complete_atom_array_plus.copy()

        annotations_2d = arr.get_annotation_2d_categories()
        new_atoms = [
            _make_single_atom("H", [0.5, 0, 0], annotations_2d),
            _make_single_atom("F", [2.5, 0, 0], annotations_2d),
        ]

        # Insert
        insert_positions = [1, 3]
        arr_inserted = insert_atoms(arr, new_atoms, insert_positions)

        # Check coordinates
        assert list(arr_inserted.element) == ["C", "H", "N", "O", "F", "S"]
        np.testing.assert_allclose(
            arr_inserted.coord, np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0], [2.5, 0, 0], [3, 0, 0]])
        )

        # Check 2D annotations
        dist = arr_inserted.get_annotation_2d("dist")
        assert np.array_equal(dist.pairs, np.array([[0, 2], [2, 3], [3, 5]]))
        assert np.allclose(dist.values, [1.0, 2.0, 3.0])

        # Check bonds
        bond_array = arr_inserted.bonds.as_array()
        expected_bonds = np.array([[0, 2, 0], [2, 3, 0], [3, 5, 0], [2, 5, 0]])
        sorted_bonds = np.sort(bond_array[:, :2], axis=1)
        sorted_expected = np.sort(expected_bonds[:, :2], axis=1)
        assert sorted_bonds.shape == sorted_expected.shape
        assert np.all([any(np.array_equal(sb, se) for se in sorted_expected) for sb in sorted_bonds])

    def test_insert_atoms_multiple_at_same_position(self, complete_atom_array_plus):
        """Test inserting multiple atoms at the same position and bond remapping using fixture for the base array."""
        arr = complete_atom_array_plus.copy()
        annotations_2d = arr.get_annotation_2d_categories()

        new_atoms = [
            _make_single_atom("H", [0.5, 0, 0], annotations_2d),
            _make_single_atom("F", [0.6, 0, 0], annotations_2d),
            _make_single_atom("Cl", [0.7, 0, 0], annotations_2d),
        ]

        insert_positions = [1, 1, 1]
        arr_inserted = insert_atoms(arr, new_atoms, insert_positions)

        assert list(arr_inserted.element) == ["C", "H", "F", "Cl", "N", "O", "S"]
        np.testing.assert_allclose(
            arr_inserted.coord,
            np.array([[0, 0, 0], [0.5, 0, 0], [0.6, 0, 0], [0.7, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
        )

        dist = arr_inserted.get_annotation_2d("dist")
        assert np.array_equal(dist.pairs, np.array([[0, 4], [4, 5], [5, 6]]))
        assert np.allclose(dist.values, [1.0, 2.0, 3.0])

        bond_array = arr_inserted.bonds.as_array()
        expected_bonds = np.array([[0, 4, 0], [4, 5, 0], [5, 6, 0], [4, 6, 0]])
        sorted_bonds = np.sort(bond_array[:, :2], axis=1)
        sorted_expected = np.sort(expected_bonds[:, :2], axis=1)

        assert sorted_bonds.shape == sorted_expected.shape
        assert np.all([any(np.array_equal(sb, se) for se in sorted_expected) for sb in sorted_bonds])

    def test_insert_atoms_annotation_mismatch_policy(self):
        """Annotation mismatch handling is forwarded through insertion."""
        arr = AtomArrayPlus(1)
        new_atom = AtomArrayPlus(1)
        S_SEGMIN.set_annotation(new_atom, np.array([8]))

        with pytest.raises(ValueError, match="Annotation mismatch across inputs"):
            insert_atoms(arr, [new_atom], [1])

        inserted = insert_atoms(
            arr,
            [new_atom],
            [1],
            fill_missing_standard_annotations=True,
        )
        assert np.array_equal(S_SEGMIN.mask(inserted), [False, True])
        assert np.array_equal(S_SEGMIN.annotation(inserted), [-1, 8])

    def test_equal_annotations_with_per_stack(self, complete_atom_array_plus):
        """Test equality checking with per-stack annotations."""
        # Create two identical stacks
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        # Make stacks
        stack1 = stack_atom_array_plus([arr1, arr2])
        stack2 = stack_atom_array_plus([arr1, arr2])

        # Test 1: Identical per-stack annotations should be equal
        b_factors = np.zeros((2, 4), dtype=float)
        b_factors[0] = [10.0, 20.0, 30.0, 40.0]
        b_factors[1] = [15.0, 25.0, 35.0, 45.0]

        stack1.set_per_stack_annotation("b_factor", b_factors.copy())
        stack2.set_per_stack_annotation("b_factor", b_factors.copy())

        assert stack1.equal_annotations(stack2)

        # Test 2: Different per-stack values should not be equal
        stack2._annot_per_stack["b_factor"][1, 2] = 99.0
        assert not stack1.equal_annotations(stack2)

        # Reset for next test
        stack2._annot_per_stack["b_factor"][1, 2] = 35.0
        assert stack1.equal_annotations(stack2)

        # Test 3: Equal NaN values should be considered equal when equal_nan=True
        occupancy = np.zeros((2, 4), dtype=float)
        occupancy[0, 0] = np.nan
        occupancy[1, 0] = np.nan

        stack1.set_per_stack_annotation("occupancy", occupancy.copy())
        stack2.set_per_stack_annotation("occupancy", occupancy.copy())

        assert stack1.equal_annotations(stack2, equal_nan=True)
        assert not stack1.equal_annotations(stack2, equal_nan=False)


class TestAtomArrayPlusStack:
    """Tests for AtomArrayPlusStack, including stacking and 2D annotation preservation."""

    def test_stack_plus_preserves_2d_annotations(self):
        """Test that stacking AtomArrayPlus objects preserves 2D annotations."""
        arr1 = AtomArrayPlus(3)
        arr1.set_annotation_2d("dist", [(0, 1), (1, 2)], [1.0, 2.0])
        # (Annotations must be the same for stack_any to work, including 2D annotations)
        arr2 = AtomArrayPlus(3)
        arr2.set_annotation_2d("dist", [(0, 1), (1, 2)], [1.0, 2.0])

        stack = stack_any([arr1, arr2])
        assert isinstance(stack, AtomArrayPlusStack)
        assert "dist" in stack._annot_2d

        ann = stack._annot_2d["dist"]
        assert ann.n_atoms == 3
        assert np.allclose(ann.values, [1.0, 2.0])
        assert np.array_equal(ann.pairs, np.array([[0, 1], [1, 2]]))

    def test_per_stack_annotation_add_and_get(self, complete_atom_array_plus):
        """Test adding and retrieving per-stack annotations."""
        # Create a stack with two identical arrays
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        stack = stack_atom_array_plus([arr1, arr2])

        # Add a per-stack annotation where all models have identical values
        values = np.zeros((2, 4), dtype=float)
        values[0, :] = [10.0, 20.0, 30.0, 40.0]
        values[1, :] = [10.0, 20.0, 30.0, 40.0]  # same as first model
        stack.set_per_stack_annotation("b_factor", values)

        # Check if the annotation exists
        assert "b_factor" in stack._annot_per_stack

        # Get the annotation
        b_factors = stack.get_per_stack_annotation("b_factor")
        assert b_factors.shape == (2, 4)
        assert np.allclose(b_factors[0], [10.0, 20.0, 30.0, 40.0])
        assert np.allclose(b_factors[1], [10.0, 20.0, 30.0, 40.0])

        # Get categories
        categories = stack.get_per_stack_annotation_categories()
        assert "b_factor" in categories

        # Verify accessing individual arrays has correct annotations
        arr0 = stack.get_array(0)
        assert np.allclose(arr0.b_factor, [10.0, 20.0, 30.0, 40.0])

        arr1 = stack.get_array(1)
        assert np.allclose(arr1.b_factor, [10.0, 20.0, 30.0, 40.0])

    def test_per_stack_annotation_set(self, complete_atom_array_plus):
        """Test setting per-stack annotations."""
        # Create a stack with two identical arrays
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        stack = stack_atom_array_plus([arr1, arr2])

        # Create a per-stack annotation with identical values for all models
        b_factors = np.zeros((2, 4), dtype=float)
        b_factors[0] = [10.0, 20.0, 30.0, 40.0]
        b_factors[1] = [10.0, 20.0, 30.0, 40.0]  # same as first model

        # Set the annotation
        stack.set_per_stack_annotation("b_factor", b_factors)

        # Check if setting worked
        retrieved = stack.get_per_stack_annotation("b_factor")
        assert np.allclose(retrieved, b_factors)

        # Test that it raises ValueError with wrong shape
        wrong_shape = np.zeros((3, 4), dtype=float)
        with pytest.raises(ValueError, match=r"Expected array shape"):
            stack.set_per_stack_annotation("wrong_shape", wrong_shape)

        # Test multidimensional per-stack annotations
        annot = np.zeros((2, 4, 3), dtype=float)
        stack.set_per_stack_annotation("annot", annot)
        assert stack.get_per_stack_annotation("annot").shape == (2, 4, 3)

        wrong_multidim = np.zeros((3, 4, 3), dtype=float)
        with pytest.raises(ValueError, match=r"Expected array shape"):
            stack.set_per_stack_annotation("wrong_multidim", wrong_multidim)

    def test_to_per_stack_annotation(self, complete_atom_array_plus):
        """Test converting a regular annotation to a per-stack annotation."""
        # Create a stack with two identical arrays
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        stack = stack_atom_array_plus([arr1, arr2])

        # Add a regular annotation that's the same for all models
        stack.set_annotation("charge", np.array([-1.0, 0.0, 1.0, 0.0]))

        # Convert to per-stack
        stack.to_per_stack_annotation("charge")

        # Verify it's now a per-stack annotation
        assert "charge" not in stack._annot
        assert "charge" in stack._annot_per_stack

        # Check the values are correct for both models
        charges = stack.get_per_stack_annotation("charge")
        assert np.allclose(charges[0], [-1.0, 0.0, 1.0, 0.0])
        assert np.allclose(charges[1], [-1.0, 0.0, 1.0, 0.0])

    def test_from_per_stack_annotation(self, complete_atom_array_plus):
        """Test converting a per-stack annotation to a regular annotation."""
        # Create a stack with two identical arrays
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        stack = stack_atom_array_plus([arr1, arr2])

        # Add a per-stack annotation with identical values
        charges = np.zeros((2, 4), dtype=float)
        charges[0] = [-1.0, 0.0, 1.0, 0.0]
        charges[1] = [-1.0, 0.0, 1.0, 0.0]
        stack.set_per_stack_annotation("charge", charges)

        # Convert to regular annotation
        stack.from_per_stack_annotation("charge")

        # Verify conversion worked
        assert "charge" in stack._annot
        assert "charge" not in stack._annot_per_stack
        assert np.allclose(stack.charge, [-1.0, 0.0, 1.0, 0.0])

    def test_indexing_with_per_stack_annotations_case1(self, complete_atom_array_plus):
        """Test Case 1: Integer indexing of AtomArrayPlusStack with per-stack annotations."""
        # Create a stack with two identical arrays
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        stack = stack_atom_array_plus([arr1, arr2])

        # Add per-stack annotation with identical values
        b_factors = np.zeros((2, 4), dtype=float)
        b_factors[0] = [10.0, 20.0, 30.0, 40.0]
        b_factors[1] = [10.0, 20.0, 30.0, 40.0]
        stack.set_per_stack_annotation("b_factor", b_factors)

        # Case 1: Integer indexing - get a single model
        model0 = stack[0]
        assert isinstance(model0, AtomArrayPlus)
        assert np.allclose(model0.b_factor, [10.0, 20.0, 30.0, 40.0])

        model1 = stack[1]
        assert isinstance(model1, AtomArrayPlus)
        assert np.allclose(model1.b_factor, [10.0, 20.0, 30.0, 40.0])

    def test_indexing_with_per_stack_annotations_case2(self, complete_atom_array_plus):
        """Test Case 2: Tuple with integer first index on AtomArrayPlusStack with per-stack annotations."""
        # Create a stack with two identical arrays
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        stack = stack_atom_array_plus([arr1, arr2])

        # Add per-stack annotation with identical values
        b_factors = np.zeros((2, 4), dtype=float)
        b_factors[0] = [10.0, 20.0, 30.0, 40.0]
        b_factors[1] = [10.0, 20.0, 30.0, 40.0]
        stack.set_per_stack_annotation("b_factor", b_factors)

        # Case 2a: (int, int) indexing - gets a single atom
        atom = stack[0, 1]
        assert isinstance(atom, struc.Atom)  # just an atom, no annotations

        # Case 2b: (int, slice) indexing - gets atoms from a single model
        atoms_slice = stack[0, 1:3]
        assert isinstance(atoms_slice, AtomArrayPlus)
        assert np.allclose(atoms_slice.b_factor, [20.0, 30.0])

        atoms_slice = stack[1, :2]
        assert isinstance(atoms_slice, AtomArrayPlus)
        assert np.allclose(atoms_slice.b_factor, [10.0, 20.0])

    def test_indexing_with_per_stack_annotations_case3a(self, complete_atom_array_plus):
        """Test Case 3a: 2D indexing of AtomArrayPlusStack with per-stack annotations."""
        # Create a stack with identical arrays
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        arr3 = complete_atom_array_plus.copy()
        stack = stack_atom_array_plus([arr1, arr2, arr3])

        # Add per-stack annotation with identical values
        b_factors = np.zeros((3, 4), dtype=float)
        b_factors[0] = [10.0, 20.0, 30.0, 40.0]
        b_factors[1] = [10.0, 20.0, 30.0, 40.0]
        b_factors[2] = [10.0, 20.0, 30.0, 40.0]
        stack.set_per_stack_annotation("b_factor", b_factors)

        # Case 3a: Two-dimensional indexing - slice both stack and atom dimensions
        sub_stack = stack[0:2, 1:3]
        assert isinstance(sub_stack, AtomArrayPlusStack)
        assert sub_stack.stack_depth() == 2
        assert sub_stack.array_length() == 2

        # Check per-stack annotation was sliced correctly
        per_stack_b = sub_stack.get_per_stack_annotation("b_factor")
        assert per_stack_b.shape == (2, 2)
        assert np.allclose(per_stack_b[0], [20.0, 30.0])
        assert np.allclose(per_stack_b[1], [20.0, 30.0])

        # Extract models and check annotations
        model0 = sub_stack[0]
        assert np.allclose(model0.b_factor, [20.0, 30.0])

        model1 = sub_stack[1]
        assert np.allclose(model1.b_factor, [20.0, 30.0])

    def test_indexing_with_per_stack_annotations_case3b(self, complete_atom_array_plus):
        """Test Case 3b: 1D indexing of AtomArrayPlusStack with per-stack annotations."""
        # Create a stack with identical arrays
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        arr3 = complete_atom_array_plus.copy()
        stack = stack_atom_array_plus([arr1, arr2, arr3])

        # Add per-stack annotation with identical values
        b_factors = np.zeros((3, 4), dtype=float)
        b_factors[0] = [10.0, 20.0, 30.0, 40.0]
        b_factors[1] = [10.0, 20.0, 30.0, 40.0]
        b_factors[2] = [10.0, 20.0, 30.0, 40.0]
        stack.set_per_stack_annotation("b_factor", b_factors)

        # Case 3b: One-dimensional indexing - slice only stack dimension
        sub_stack = stack[1:3]
        assert isinstance(sub_stack, AtomArrayPlusStack)
        assert sub_stack.stack_depth() == 2
        assert sub_stack.array_length() == 4  # all atoms kept

        # Check per-stack annotation was sliced correctly
        per_stack_b = sub_stack.get_per_stack_annotation("b_factor")
        assert per_stack_b.shape == (2, 4)
        assert np.allclose(per_stack_b[0], [10.0, 20.0, 30.0, 40.0])
        assert np.allclose(per_stack_b[1], [10.0, 20.0, 30.0, 40.0])

        # Extract models and check annotations
        model0 = sub_stack[0]  # this is actually model1 from original stack
        assert np.allclose(model0.b_factor, [10.0, 20.0, 30.0, 40.0])

        # Test with boolean mask
        mask = np.array([False, True, True])
        mask_stack = stack[mask]
        assert mask_stack.stack_depth() == 2

        mask_b = mask_stack.get_per_stack_annotation("b_factor")
        assert mask_b.shape == (2, 4)
        assert np.allclose(mask_b[0], [10.0, 20.0, 30.0, 40.0])
        assert np.allclose(mask_b[1], [10.0, 20.0, 30.0, 40.0])

        # Test fancy indexing
        fancy_stack = stack[[0, 2]]
        assert fancy_stack.stack_depth() == 2

        fancy_b = fancy_stack.get_per_stack_annotation("b_factor")
        assert np.allclose(fancy_b[0], [10.0, 20.0, 30.0, 40.0])
        assert np.allclose(fancy_b[1], [10.0, 20.0, 30.0, 40.0])

    def test_stack_with_2d_annotations(self):
        """Test stacking arrays with 2D annotations."""
        # Create arrays with 2D annotations
        arr1 = AtomArrayPlus(4)
        arr1.chain_id = np.array(["A", "A", "B", "B"])

        arr2 = AtomArrayPlus(4)
        arr2.chain_id = np.array(["A", "A", "B", "B"])  # Same as arr1

        # Add 2D annotations (should be preserved in stack)
        arr1.set_annotation_2d("dist", [(0, 1), (1, 2)], [3.8, 4.2])
        arr2.set_annotation_2d("dist", [(0, 1), (1, 2)], [3.8, 4.2])

        # Stack the arrays (should succeed because annotations are equal)
        stack = stack_any([arr1, arr2])

        # Verify 2D annotations are preserved
        assert "dist" in stack.get_annotation_2d_categories()
        dist_ann = stack.get_annotation_2d("dist")
        assert np.array_equal(dist_ann.pairs, np.array([[0, 1], [1, 2]]))
        assert np.allclose(dist_ann.values, [3.8, 4.2])

    def test_stack_requires_equal_annotations(self):
        """Test that stacking requires equal annotations."""
        # Create arrays with different annotations
        arr1 = AtomArrayPlus(3)
        arr1.set_annotation("b_factor", np.array([10.0, 20.0, 30.0]))

        arr2 = AtomArrayPlus(3)
        arr2.set_annotation("b_factor", np.array([15.0, 25.0, 35.0]))  # different from arr1

        # Verify it's in the _annot dictionary
        assert "b_factor" in arr1._annot
        assert "b_factor" in arr2._annot
        assert not np.array_equal(arr1.b_factor, arr2.b_factor)

        # Attempt to stack should raise an error
        with pytest.raises(ValueError, match="are not equal to the annotations"):
            stack_atom_array_plus([arr1, arr2])

        # With identical annotations, it should work fine
        arr2.set_annotation("b_factor", np.array([10.0, 20.0, 30.0]))  # now same as arr1
        _ = stack_any([arr1, arr2])

    def test_with_identical_annotations(self):
        """Test stacking arrays with identical annotations."""
        # Create arrays with identical annotations
        arr1 = AtomArrayPlus(3)
        arr1.set_annotation("b_factor", np.array([10.0, 20.0, 30.0]))

        arr2 = AtomArrayPlus(3)
        arr2.set_annotation("b_factor", np.array([10.0, 20.0, 30.0]))  # same as arr1

        # Stack should work fine
        stack = stack_atom_array_plus([arr1, arr2])

        # Both models should have the same b_factor values
        model0 = stack.get_array(0)
        assert np.allclose(model0.b_factor, [10.0, 20.0, 30.0])

        model1 = stack.get_array(1)
        assert np.allclose(model1.b_factor, [10.0, 20.0, 30.0])

    def test_stack_roundtrip_to_atom_array_stack_and_back(self, complete_atom_array_plus):
        """
        Test round-trip: AtomArrayPlusStack -> AtomArrayStack -> AtomArrayPlusStack.

        Ensures 1D annotations, bonds, and coordinates are preserved, but 2D annotations are lost and can be restored.
        """
        arr1 = complete_atom_array_plus.copy()
        arr2 = complete_atom_array_plus.copy()
        stack = stack_any([arr1, arr2])

        # Convert to AtomArrayStack (loses 2D annotations)...
        arr_stack = stack.as_atom_array_stack()
        # ... and back to AtomArrayPlusStack
        stack2 = AtomArrayPlusStack.from_atom_array_stack(arr_stack)

        # 2D annotations are lost
        assert stack2.get_annotation_2d_categories() == []

        # 1D annotations and coordinates are preserved
        np.testing.assert_allclose(stack2.coord, stack.coord)
        assert np.array_equal(stack2.element, stack.element)

        # Bonds are preserved
        assert stack2.bonds == stack.bonds


@pytest.mark.parametrize("use_alias", [False, True])
def test_concatenation_fills_missing_scalar_and_pair_conditions(use_alias):
    left, right = AtomArrayPlus(2), AtomArrayPlus(1)
    sequence_name = C_SEQ.aliases[0] if use_alias else C_SEQ.full_name
    distance_name = C_DIS.aliases[0] if use_alias else C_DIS.full_name
    left.set_annotation(sequence_name, np.array(["A1AAG", "<M>"]))
    left.set_annotation_2d(distance_name, pairs=np.array([[0, 1]]), values=np.array([3.5]))

    combined = concatenate_atom_array_plus([left, right], fill_missing_standard_annotations=True)

    np.testing.assert_array_equal(C_SEQ.annotation(combined), ["A1AAG", "<M>", "<M>"])
    np.testing.assert_array_equal(C_SEQ.mask(combined), [True, False, False])
    distance = C_DIS.annotation(combined)
    assert distance.n_atoms == 3
    np.testing.assert_array_equal(distance.pairs, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(distance.values, [3.5, 3.5])
    assert not C_SEQ.has_annotation(right)
    assert not C_DIS.has_annotation(right)


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=INFO", __file__])
