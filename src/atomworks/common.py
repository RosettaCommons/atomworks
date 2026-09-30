"""
Common utility functions used throughout the project.

Timeout utilities are adapted from https://github.com/pnpnpn/timeout-decorator/blob/master/timeout_decorator/timeout_decorator.py (MIT License)
and from https://github.com/chaidiscovery/chai-lab/blob/main/chai_lab/utils/timeout.py
"""

import copy
import hashlib
import multiprocessing
import queue as _queue
import signal
import time
from collections.abc import Callable
from enum import Enum
from functools import lru_cache, wraps
from multiprocessing import Queue
from typing import Any, Literal, Never

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


def immutable_lru_cache(
    maxsize: int = 128,
    typed: bool = False,
    deepcopy: bool = True,
    copy_func: Callable | None = None,
) -> Callable:
    """An immutable version of lru_cache for caching functions that return mutable objects.

    Args:
        maxsize: Maximum number of items to cache.
        typed: Whether to treat different types as separate cache entries.
        deepcopy: Whether to use deep copy for immutable caching.
        copy_func: Custom copy function to use. If provided, overrides deepcopy parameter.
            Should be a callable that takes the cached object and returns a copy.

    Returns:
        A decorator that provides immutable caching functionality.

    Example:
        >>> # Use biotite's fast copy for AtomArrays
        >>> @immutable_lru_cache(maxsize=200, copy_func=lambda x: x.copy())
        >>> def get_template(code):
        ...     return atom_array_from_ccd_code(code)
    """
    if copy_func is None:
        copy_func = copy.deepcopy if deepcopy else copy.copy

    def decorator(func: Callable) -> Callable:
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return copy_func(cached_func(*args, **kwargs))

        # Expose cache methods from the underlying lru_cache
        wrapper.cache_clear = cached_func.cache_clear
        wrapper.cache_info = cached_func.cache_info

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


def timeout(timeout: float | int | None = None, strategy: Literal["signal", "subprocess"] = "subprocess") -> Callable:
    """
    Decorator to apply a timeout to a function.

    The `signal` strategy is more efficient and slightly faster, but does not work in all contexts
    (e.g. with some C dependencies like RDKit, on certain operating systems).
    The `subprocess` strategy is always available, but slightly slower and with a higher overhead.
    """
    if timeout is None:
        return do_nothing()
    match strategy:
        case "signal":
            # timeout based on signal module
            return timeout_using_signal(timeout)
        case "subprocess":
            # timeout based on subprocess module
            return timeout_using_subprocess(timeout)
        case _:
            raise ValueError(f"Invalid strategy: {strategy}. Must be 'signal' or 'subprocess'.")


def do_nothing(*args, **kwargs) -> Callable:
    """A decorator that does nothing and simply returns the original function.

    This decorator can be used as a placeholder or for testing purposes when you want
    to conditionally apply decorators without changing the code structure.

    Returns:
        A decorator function that returns the original function unchanged.

    Example:
        .. code-block:: python

            @do_nothing_decorator()
            def my_function():
                return "Hello, World!"


            # or:
            do_nothing(bla=123, blub=456)(my_function)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapped_func

    return decorator


def timeout_using_signal(timeout: float | int | None) -> Callable:
    """
    Build a decorator that applies a timeout to a function using the signal module.

    This decorator sets up a signal handler to raise a TimeoutError if the decorated function
    exceeds the specified timeout duration. It uses the SIGALRM signal to implement the timeout.

    Use for example as:
    ```python
    result = timeout_using_signal(timeout=10.0)(my_function)(*args, **kwargs)
    ```

    Args:
        timeout (float | int | None): The timeout duration in seconds.

    Returns:
        Callable: A decorator function that can be applied to other functions to add timeout functionality.
    """

    def decorate(func: Callable) -> Callable:
        @wraps(func)
        def wrapped_func(*args, **kwargs):  # noqa: ANN202
            _start_time = time.time()

            def _timeout_handler(*_) -> Never:
                # ... raise TimeoutError if called
                _elapsed_time = time.time() - _start_time
                raise TimeoutError(f"Function timed out after {_elapsed_time:.3f} seconds")

            # ... set the timeout handler and record the prior handler to restore later
            _prior_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            # ... start the timer
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                return func(*args, **kwargs)
            finally:
                # ... reset the timer
                signal.setitimer(signal.ITIMER_REAL, 0)
                # ... restore the prior handler
                signal.signal(signal.SIGALRM, _prior_handler)

        return wrapped_func

    return decorate


def _timeout_handler(queue: Queue, func: Callable, args: Any, kwargs: Any) -> None:
    """
    Util function to be used only in `timeout_using_subprocess`.
    This util function is in the outer scope to allow pickling during ddp multiprocessing.
    """
    try:
        result = func(*args, **kwargs)
        queue.put((_TimeoutHandlerStatus.SUCCESS, result))
    except Exception as e:
        queue.put((_TimeoutHandlerStatus.EXCEPTION, e))


def timeout_using_subprocess(timeout: float | int | None) -> Callable:
    """Force function to timeout after specified time.

    The returned decorator uses a subprocess to execute the function, allowing for timeout
    functionality even for CPU-bound operations that cannot be interrupted by signals.

    Args:
        timeout (float | int | None): The maximum time in seconds allowed for the function to execute.

    Returns:
        Callable: A decorator that can be applied to a function.

    Raises:
        TimeoutError: If the function does not return before the timeout.
        ChildProcessException: If the child process dies unexpectedly.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapped_func(*args, **kwargs):  # noqa: ANN202
            # NOTE: 'fork' context is useful to speed up the timeout handling,
            #  as using 'spawn' instead will re-trigger imports that are needed to run the function
            #  and understand the context in which it is used in, which can be slow.
            ctx = multiprocessing.get_context("fork")
            queue = ctx.Queue()

            # ... create subprocess to run the function
            proc = ctx.Process(target=_timeout_handler, args=(queue, func, args, kwargs), daemon=True)
            # ... start the subprocess (ensure it is not a daemon to allow doing this in multiprocessing)
            with _AllowSubprocessForDeamonicProcess():
                proc.start()

            # ... wait for the subprocess to finish and check if it timed out
            proc.join(timeout=float(timeout))

            # ... if the subprocess is still running, terminate it and raise a TimeoutError
            if proc.is_alive():
                proc.terminate()
                proc.join()
                raise TimeoutError(f"Function {func} timed out after {timeout} seconds")

            # ... try retrieving the result, if available
            try:
                status, value = queue.get(timeout=0.1)  # short timeout to prevent hang
                # NOTE: Hang can happen when the child process dies unexpectedly
                #       and the main process is waiting for the result in the queue.
                # See Issue(https://bugs.python.org/issue43805)
            except _queue.Empty:
                raise ChildProcessError("Child process died unexpectedly")  # noqa: B904

            match status:
                case _TimeoutHandlerStatus.SUCCESS:
                    # ... return the result of the function
                    return value
                case _TimeoutHandlerStatus.EXCEPTION:
                    # ... raise the exception caught in the child process
                    raise value
                case _:
                    # ... this code should be unreachable, if reached raise an error
                    raise ValueError(f"Invalid status: {status}. Must be 'SUCCESS' or 'EXCEPTION'.")

        return wrapped_func

    return decorator


# TODO: This is dangerous: revert once the underlying problem in rdkit is fixed
# RDKit Issue(https://github.com/rdkit/rdkit/discussions/7289)
class _AllowSubprocessForDeamonicProcess:
    """Context Manager to resolve AssertionError: daemonic processes are not allowed to have children
    See https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic"""

    def __init__(self):
        self.conf: dict = multiprocessing.process.current_process()._config  # type: ignore
        if "daemon" in self.conf:
            self.daemon_status_set = True
        else:
            self.daemon_status_set = False
        self.daemon_status_value = self.conf.get("daemon")

    def __enter__(self):
        if self.daemon_status_set:
            del self.conf["daemon"]

    def __exit__(self, *args, **kwargs):
        if self.daemon_status_set:
            self.conf["daemon"] = self.daemon_status_value


class _TimeoutHandlerStatus(Enum):
    """Status of the timeout handler."""

    SUCCESS = 0
    EXCEPTION = 1


class ChildProcessError(Exception):
    """Exception raised when a child process dies unexpectedly."""

    pass
