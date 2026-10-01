"""
Tests for the annotator module functionality.

This module tests the annotation registry system, ensure_annotations function,
and the ability to add and remove annotations from AtomArray objects.
"""

import numpy as np
import pytest

from atomworks.io.utils.annotator import (
    ANNOTATOR_REGISTRY,
    clear_generated_annotations,
    ensure_annotations,
    remove_annotations,
)
from atomworks.io.utils.standard_annotations.definitions import Atomize
from atomworks.ml.conditions.definitions import Sequence
from atomworks.ml.utils.testing import cached_parse


@pytest.fixture
def atom_array():
    """Fixture providing a test atom array."""
    data = cached_parse("6lyz")
    return data["atom_array"]


def test_ensure_annotations_basic(atom_array):
    """Test basic functionality of ensure_annotations."""
    # Get a few annotations from the registry
    available_annotations = list(ANNOTATOR_REGISTRY.keys())[:3]  # Take first 3

    # Ensure these annotations don't exist initially (clean slate)
    initial_categories = atom_array.get_annotation_categories()
    for annot in available_annotations:
        if annot in initial_categories:
            atom_array.del_annotation(annot)

    # Apply annotations
    ensure_annotations(atom_array, *available_annotations)

    # Check that annotations are now present
    final_categories = atom_array.get_annotation_categories()
    for annot in available_annotations:
        assert annot in final_categories, f"Annotation '{annot}' should be present after ensure_annotations"

        # Check that annotation has correct shape
        annotation_data = atom_array.get_annotation(annot)
        assert (
            len(annotation_data) == atom_array.array_length()
        ), f"Annotation '{annot}' should have same length as atom array"
        assert isinstance(annotation_data, np.ndarray), f"Annotation '{annot}' should be a numpy array"


def test_ensure_annotations_all_registry(atom_array):
    """Test ensure_annotations with all registered annotations."""
    # Get all available annotations
    desired_annotations = list(ANNOTATOR_REGISTRY.keys())

    # Apply all annotations
    ensure_annotations(atom_array, *desired_annotations)

    # Check that all annotations are present
    final_categories = atom_array.get_annotation_categories()
    missing_annotations = [annot for annot in desired_annotations if annot not in final_categories]

    assert len(missing_annotations) == 0, f"Missing annotations: {missing_annotations}"

    # Verify each annotation has correct properties
    for annot in desired_annotations:
        annotation_data = atom_array.get_annotation(annot)
        assert len(annotation_data) == atom_array.array_length(), f"Annotation '{annot}' has wrong length"
        assert isinstance(annotation_data, np.ndarray), f"Annotation '{annot}' is not a numpy array"


def test_clear_generated_annotations(atom_array):
    """Test clearing all generated annotations."""
    # Get some annotations to work with
    desired_annotations = list(ANNOTATOR_REGISTRY.keys())[:5]  # Take first 5

    # Apply annotations
    ensure_annotations(atom_array, *desired_annotations)

    # Verify they exist
    categories_before = atom_array.get_annotation_categories()
    for annot in desired_annotations:
        assert annot in categories_before, f"Annotation '{annot}' should exist before clearing"

    # Clear generated annotations
    clear_generated_annotations(atom_array)

    # Check that registered annotations are removed
    categories_after = atom_array.get_annotation_categories()
    remaining_registered = [annot for annot in desired_annotations if annot in categories_after]

    assert len(remaining_registered) == 0, f"These registered annotations should be removed: {remaining_registered}"


def test_remove_specific_annotations(atom_array):
    """Test removing specific annotations."""
    # Get some annotations to work with
    all_annotations = list(ANNOTATOR_REGISTRY.keys())
    annotations_to_add = all_annotations[:4]  # Take first 4
    annotations_to_remove = annotations_to_add[:2]  # Remove first 2
    annotations_to_keep = annotations_to_add[2:]  # Keep last 2

    # Apply annotations
    ensure_annotations(atom_array, *annotations_to_add)

    # Remove specific annotations
    remove_annotations(atom_array, *annotations_to_remove)

    # Check results
    final_categories = atom_array.get_annotation_categories()

    # Removed annotations should be gone
    for annot in annotations_to_remove:
        assert annot not in final_categories, f"Annotation '{annot}' should be removed"

    # Kept annotations should still be there
    for annot in annotations_to_keep:
        assert annot in final_categories, f"Annotation '{annot}' should still be present"


@pytest.mark.parametrize(
    "annotation_cls",
    [
        Atomize,
        Sequence,
    ],
)
def test_ensure_annotations_standard_annotation(atom_array, annotation_cls):
    """Test that ensure_annotations generates StandardAnnotation defaults by registry and full name."""
    annotation_name = annotation_cls.full_name
    # Ensure the annotation is absent to start
    assert (
        annotation_name not in atom_array.get_annotation_categories()
    ), f"Annotation '{annotation_name}' should not be present before test"

    ensure_annotations(atom_array, annotation_name)

    assert (
        annotation_name in atom_array.get_annotation_categories()
    ), f"Annotation '{annotation_name}' should be present after ensure_annotations"
    annotation_data = atom_array.get_annotation(annotation_name)
    assert len(annotation_data) == atom_array.array_length()
    assert isinstance(annotation_data, np.ndarray)


def test_ensure_annotations_standard_annotation_registry_name(atom_array):
    ensure_annotations(atom_array, Sequence.name)

    assert Sequence.full_name in atom_array.get_annotation_categories()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
