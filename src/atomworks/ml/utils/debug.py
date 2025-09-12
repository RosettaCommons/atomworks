"""Debug utilities for ML components.

Provides functions for saving failed examples and debugging ML pipelines.
"""

import logging
import os
import pickle
import re
from datetime import datetime

from atomworks.common import default

logger = logging.getLogger("atomworks.ml")
_USER = default(os.getenv("USER"), "")

try:
    import wandb
except ImportError:
    wandb = None


def _remove_special_characters(s: str) -> str:
    """Remove special characters from a string.

    Args:
        s: The string to clean.

    Returns:
        The cleaned string with only alphanumeric characters and underscores.
    """
    assert isinstance(s, str)
    # Remove unwanted characters using regex
    clean_s = re.sub(r"[^a-zA-Z0-9_]", "", s)
    return f"{clean_s}"


def save_failed_example_to_disk(
    example_id: str,
    fail_dir: str,
    *,
    data: dict = {},
    rng_state_dict: dict = {},
    error_msg: str = "",
) -> None:
    """Attempts to save a failed example to disk as a pickle file.

    Args:
        example_id: The ID of the example.
        fail_dir: The directory where the failed example should be saved.
        data: Optional data dictionary to save.
        rng_state_dict: The random number generator state dictionary.
        error_msg: The error message associated with the failure.
    """
    try:
        # Get wandb run ID if currently in a wandb run
        run_id = ""
        if wandb is not None and hasattr(wandb, "run") and wandb.run is not None:
            run_id = wandb.run.id
        file_path = os.path.join(fail_dir, run_id, _remove_special_characters(example_id) + ".pkl")

        # Ensure the fail directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True, mode=0o777)  # Allow everyone to read/write

        with open(file_path, "wb") as f:
            data = {
                "example_id": example_id,
                "rng_state_dict": rng_state_dict,
                "error_msg": error_msg,
                "wandb_run_id": run_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": _USER,
            } | data
            pickle.dump(data, f)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        logger.warning(
            f"Failed to save failed example to disk: {e}. Are you sure the directory exists? Do you have write permissions?"
        )
