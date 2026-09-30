"""
Convenience utils for common validation checks in transforms.

All checks take a `data` dictionary as input and raise an error if the check fails.
"""

from typing import Any

from atomworks.io.utils.selection import get_annotation_categories


def check_contains_keys(data: dict[str, Any], keys: list[str]) -> None:
    """Check if a key is in a dictionary."""
    for key in keys:
        if key not in data:
            raise KeyError(f"Key `{key}` not in data. Available keys: {list(data.keys())}")


def check_does_not_contain_keys(data: dict[str, Any], keys: list[str]) -> None:
    """Check if a key is not in a dictionary."""
    for key in keys:
        if key in data:
            raise KeyError(f"Key `{key}` already exists in data")


def check_is_instance(data: dict[str, Any], key: str, expected_type: type) -> None:
    """Check if the value of a key in a dictionary is of a certain type."""
    if not isinstance(data[key], expected_type):
        raise ValueError(f"Key `{key}` in data is not of type `{expected_type}`, got {type(data[key])}")


def check_is_shape(data: dict[str, Any], key: str, expected_shape: tuple[int, ...]) -> None:
    """Check if the value of a key in a dictionary has a certain shape."""
    if data[key].shape != expected_shape:
        raise ValueError(f"Key `{key}` in data has shape {data[key].shape} but expected shape {expected_shape}")


def check_nonzero_length(data: dict[str, Any], key: str) -> None:
    """Check if the length of the value of a key in a dictionary is nonzero."""
    if len(data[key]) == 0:
        raise ValueError(f"Key {key} in data has length 0")


def check_atom_array_annotation(
    data: dict[str, Any], required: list[str], forbidden: list[str] = [], n_body: int = 1
) -> None:
    """Check if `atom_array` key has the annotations specified in `required`."""
    annotations = set(get_annotation_categories(data["atom_array"], n_body=n_body))

    if not set(required).issubset(annotations):
        missing = set(required) - annotations
        raise ValueError(f"Key `atom_array` is missing the following annotations: {missing}")
    if len(forbidden) > 0 and set(forbidden).issubset(annotations):
        raise ValueError(f"Key `atom_array` has the following forbidden annotations: {forbidden}")


def check_atom_array_has_bonds(data: dict[str, Any]) -> None:
    """Check if `atom_array` key has bonds."""
    if data["atom_array"].bonds is None:
        raise ValueError("Key `atom_array` in data has no `bonds` defined.")
