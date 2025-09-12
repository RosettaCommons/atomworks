"""
Timeout utilities for applying time limits to functions and blocks of code.

Adapted from https://github.com/pnpnpn/timeout-decorator/blob/master/timeout_decorator/timeout_decorator.py (MIT License)
and from https://github.com/chaidiscovery/chai-lab/blob/main/chai_lab/utils/timeout.py
"""

import multiprocessing
import queue as _queue
import signal
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from multiprocessing import Queue
from typing import Any, Literal, Never


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
