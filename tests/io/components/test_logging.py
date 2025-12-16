import importlib
import logging
import os

import atomworks.io


def test_atomworks_logging_level():
    """
    Test if the logging level for atomworks.io is correctly configured.

    This test checks if the atomworks.io logger's level is set according to the ATOMWORKS_LOG_LEVEL
    environment variable, or defaults to WARNING if not set.
    """
    # Get the expected level from the environment variable, default to WARNING
    expected_level_str = os.environ.get("ATOMWORKS_LOG_LEVEL", "WARNING").upper()
    expected_level = getattr(logging, expected_level_str)

    # Get the atomworks.io logger
    atomworks_logger = logging.getLogger("atomworks")

    # Assert that the current logging level matches the expected level
    assert (
        atomworks_logger.level == expected_level
    ), f"Expected atomworks.logging level to be {logging.getLevelName(expected_level)}, but it was {logging.getLevelName(atomworks_logger.level)}"


def test_atomworks_logging_level_env_var():
    """
    Test if the logging level for atomworks.logging responds to changes in the ATOMWORKS_LOG_LEVEL environment variable.
    """
    # Store the original environment variable value
    original_level = os.environ.get("ATOMWORKS_LOG_LEVEL")

    try:
        # Set the environment variable
        os.environ["ATOMWORKS_LOG_LEVEL"] = "DEBUG"

        # Re-import atomworks to trigger logger configuration
        importlib.reload(atomworks)

        # Get the atomworks logger
        atomworks_logger = logging.getLogger("atomworks")

        # Assert that the current logging level matches the set environment variable
        assert (
            atomworks_logger.level == logging.DEBUG
        ), f"Expected atomworks.logging level to be DEBUG, but it was {logging.getLevelName(atomworks_logger.level)}"

    finally:
        # Clean up: restore the original environment variable
        if original_level is None:
            del os.environ["ATOMWORKS_LOG_LEVEL"]
        else:
            os.environ["ATOMWORKS_LOG_LEVEL"] = original_level

        # Reset the atomworks logger to its original state
        importlib.reload(atomworks)
