"""Common utility functions used throughout the project."""

import copy
import hashlib
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any

import numpy as np
from toolz.curried import compose, reduce


def exists(obj: Any) -> bool:
    """Check that obj is not None.

    Args:
        obj: The object to check.

    Returns:
        True if obj is not None, False otherwise.
    """
    return obj is not None


def default(obj: Any, default: Any) -> Any:
    """Return obj if not None, otherwise return default.

    Args:
        obj: The primary object to return.
        default: The fallback value if obj is None.

    Returns:
        obj if it is not None, otherwise default.
    """
    return obj if exists(obj) else default


def to_hashable(element: Any) -> Any:
    """Convert an element to a hashable type.

    Args:
        element: The element to convert.

    Returns:
        The element if already hashable, otherwise converted to a tuple.
    """
    return element if isinstance(element, int | str | np.integer | np.str_) else tuple(element)


def string_to_md5_hash(s: str, truncate: int = 32) -> str:
    """Generate an MD5 hash of a string and return the first truncate characters.

    Args:
        s: The string to hash.
        truncate: Number of characters to return from the hash.

    Returns:
        The truncated MD5 hash as a string.
    """
    full_hash = hashlib.md5(s.encode("utf-8")).hexdigest()
    return full_hash[:truncate]


def sum_string_arrays(*objs: np.ndarray | str) -> np.ndarray:
    """Sum a list of string arrays or strings into a single string array.

    Concatenates the arrays and determines the shortest string length to set as dtype.

    Args:
        *objs: Variable number of string arrays or strings to sum.

    Returns:
        A single concatenated string array.
    """
    return reduce(np.char.add, objs).astype(object).astype(str)


def not_isin(element: np.ndarray, array: np.ndarray, **isin_kwargs) -> np.ndarray:
    """Like ~np.isin, but more efficient.

    Args:
        element: The array to test.
        array: The array of values to test against.
        **isin_kwargs: Additional keyword arguments for np.isin.

    Returns:
        Boolean array indicating which elements are not in the array.
    """
    return np.isin(element, array, invert=True, **isin_kwargs)


def listmap(func: Callable, *iterables) -> list:
    """Like map, but returns a list instead of an iterator.

    Args:
        func: The function to apply.
        *iterables: Variable number of iterables to map over.

    Returns:
        A list containing the results of applying func to the iterables.
    """
    return compose(list, map)(func, *iterables)


def as_list(value: Any) -> list:
    """Convert a value to a list.

    Handles various types using duck typing:
        - Iterable objects (lists, tuples, strings, etc.): converted to list
        - Single values: wrapped in a list

    Args:
        value: The value to convert to a list.

    Returns:
        A list containing the value(s).
    """
    try:
        # Try to iterate over the value (duck typing approach)
        # Exclude strings since they're iterable but we want to treat them as single values
        if isinstance(value, str):
            return [value]
        return list(value)
    except TypeError:
        # If it's not iterable, wrap it in a list
        return [value]


def immutable_lru_cache(maxsize: int = 128, typed: bool = False, deepcopy: bool = True) -> Callable:
    """An immutable version of lru_cache for caching functions that return mutable objects.

    Args:
        maxsize: Maximum number of items to cache.
        typed: Whether to treat different types as separate cache entries.
        deepcopy: Whether to use deep copy for immutable caching.

    Returns:
        A decorator that provides immutable caching functionality.
    """
    copy_func = copy.deepcopy if deepcopy else copy.copy

    def decorator(func: Callable) -> Callable:
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return copy_func(cached_func(*args, **kwargs))

        return wrapper

    return decorator


class KeyToIntMapper:
    """Maps keys to unique integers based on the order of the first appearance of the key.

    This is useful for mapping id's such as chain_id, chain_entity, molecule_iid, etc.
    to integers.

    Example:
        >>> chain_id_to_int = KeyToIntMapper()
        >>> chain_id_to_int("A")  # 0
        >>> chain_id_to_int("C")  # 1
        >>> chain_id_to_int("A")  # 0
        >>> chain_id_to_int("B")  # 2
    """

    def __init__(self):
        """Initialize KeyToIntMapper with empty mapping."""
        self.key_to_id = {}
        self.next_id = 0

    def __call__(self, value: Any) -> int:
        """Map a key to a unique integer.

        Args:
            value: The key to map.

        Returns:
            The unique integer assigned to the key.
        """
        if value not in self.key_to_id:
            self.key_to_id[value] = self.next_id
            self.next_id += 1
        return self.key_to_id[value]
