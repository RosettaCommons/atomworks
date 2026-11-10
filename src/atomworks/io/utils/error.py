import contextlib
import logging
import sys
import traceback
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def context(
    msg: str,
    cleanup: Callable[[], None] = lambda: None,
    raise_error: bool = True,
    log_level: int = logging.ERROR,
    exc_types: tuple = (Exception,),
) -> Any:  # Using Any since yield can be used with any type
    """
    Production-ready context manager for handling exceptions with configurable error handling and logging.

    Args:
        - msg (str): Message to prepend to the error description
        - cleanup (callable): Optional cleanup function to call when an exception occurs. Defaults to no-op
        - raise_error (bool): If True, logs and re-raises the exception. If False, only logs the exception
        - log_level (int): Logging level to use (from logging module constants). Defaults to logging.ERROR
        - exc_types (tuple): Tuple of exception types to catch. Defaults to (Exception,)

    Yields:
        Any: The yielded value from the context block

    Raises:
        Exception: Re-raises the caught exception if raise_error is True
    """
    try:
        yield
    except exc_types as ex:
        # Format the error message with more robust handling
        error_msg = f"{msg}: {ex!s}" if str(ex) else msg

        # Get full traceback
        exc_info = sys.exc_info()
        full_tb = "".join(traceback.format_exception(*exc_info)) if exc_info[0] else ""

        try:
            # Attempt cleanup before potentially raising
            cleanup()
        except Exception as cleanup_ex:
            logger.error(
                "Cleanup failed after error '%s': %s\n%s",
                error_msg,
                str(cleanup_ex),
                full_tb,
            )

        # Log the original error
        logger.log(
            log_level,
            "Encountered error in context: \n\t%s\n\n%s",
            error_msg,
            full_tb,
        )

        if raise_error:
            # Update exception args to include context
            ex.args = (error_msg,) + ex.args[1:]
            raise

    except BaseException as ex:  # Catches system exits, keyboard interrupts etc.
        # Format the error message with more robust handling
        error_msg = f"Unexpected error in context: \n\t{msg}\n\n{ex!s}" if str(ex) else msg

        # Get full traceback similar to main exception handling
        exc_info = sys.exc_info()
        full_tb = "".join(traceback.format_exception(*exc_info)) if exc_info[0] else ""

        try:
            # Attempt cleanup before raising
            cleanup()
        except Exception as cleanup_ex:
            logger.critical(
                "Cleanup failed after unexpected error '%s': %s\n%s",
                error_msg,
                str(cleanup_ex),
                full_tb,
            )

        # Update exception args to include context, similar to main exception handling
        ex.args = (error_msg,) + ex.args[1:]
        raise  # Re-raise the original exception (preserving its type)
