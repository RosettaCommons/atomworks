import logging

import pytest

from atomworks.ml.utils.error import context


def test_context_successful_execution():
    """Test that context manager passes through when no exception occurs."""
    with context("Test operation"):
        result = 1 + 1
    assert result == 2


def test_context_with_handled_exception():
    """Test that context manager properly handles expected exceptions."""
    cleanup_called = False

    def cleanup():
        nonlocal cleanup_called
        cleanup_called = True

    with pytest.raises(ValueError) as exc_info, context("Test operation", cleanup=cleanup):
        raise ValueError("test error")

    assert cleanup_called
    assert "Test operation" in str(exc_info.value)
    assert "test error" in str(exc_info.value)


def test_context_without_error_raising():
    """Test that context manager can suppress exceptions when raise_error=False."""
    with context("Test operation", raise_error=False):
        raise ValueError("test error")
    # If we get here, the test passed because the exception was suppressed


def test_context_with_custom_exception_types():
    """Test that context manager only catches specified exception types."""
    with pytest.raises(KeyError), context("Test operation", exc_types=(ValueError,)):  # Should not be caught
        raise KeyError("test error")


def test_context_with_custom_logger(caplog):
    """Test that context manager uses the specified logger and log level."""
    test_logger = logging.getLogger("test_logger")

    with (
        caplog.at_level(logging.INFO),
        context("Test operation", raise_error=False, log_level=logging.INFO, logger=test_logger),
    ):
        raise ValueError("test error")

    assert "Test operation" in caplog.text
    assert "test error" in caplog.text


def test_context_cleanup_error(caplog):
    """Test that context manager handles cleanup failures properly."""

    def failing_cleanup():
        raise RuntimeError("cleanup failed")

    with caplog.at_level(logging.ERROR), pytest.raises(ValueError), context("Test operation", cleanup=failing_cleanup):
        raise ValueError("test error")

    assert "Cleanup failed" in caplog.text
    assert "cleanup failed" in caplog.text


def test_context_with_system_exit():
    """Test that context manager properly handles SystemExit."""
    with pytest.raises(SystemExit) as exc_info, context("Test operation"):
        raise SystemExit(1)

    assert "Unexpected error in context" in str(exc_info.value)


def test_context_with_keyboard_interrupt():
    """Test that context manager properly handles KeyboardInterrupt."""
    with pytest.raises(KeyboardInterrupt) as exc_info, context("Test operation"):
        raise KeyboardInterrupt()

    assert "Test operation" in str(exc_info.value)


def test_context_skip_allowed_exceptions():
    """Test that context manager only skips specified exception types when raise_error=False."""
    # Should be suppressed
    with context("Test operation", raise_error=False, exc_types=(ValueError,)):
        raise ValueError("test error")

    # Should still raise
    with pytest.raises(KeyError), context("Test operation", raise_error=False, exc_types=(ValueError,)):
        raise KeyError("test error")
