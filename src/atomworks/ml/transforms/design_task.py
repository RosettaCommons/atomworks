"""
Defines a framework for creating and applying protein design tasks to AtomArrays.

This module provides a framework for transforming a generic `AtomArray` (normally
sampled from the RCSB PDB or a distillation dataset at training time)
into a well-defined design task for protein design. A design task annotates an `AtomArray`
with a set of conditions that specify the design problem, such as coordinate-conditioned
motif scaffolding or unconditional generation.

The main components of this module are:
- `DesignTaskABC`: An abstract base class that defines the interface for all design
  tasks. A concrete implementation must define:
    - `can_apply(data)`: A method to check if the task is applicable to the given
      `AtomArray` (e.g., checking for the presence of specific residues or features).
    - `forward(data)`: The core logic that applies the task by adding conditional
      annotations to the `AtomArray`. This will normally sample some motif and
      subsequently annotate all conditions for the task.
    - `annotate_for_task_selection(data)`: An optional method to add any annotations
      required by `can_apply()`. It is recommended to use the `ensure_annotations`
      utility from `annotator.py` to lazily generate and reuse annotations.

- `SampleDesignTask`: A transform that acts as a router for different design tasks.
  During training, it takes a dictionary of design tasks and their sampling frequencies,
  evaluates which tasks `can_apply`, and samples one to apply to the data.

This system allows for flexible and data-dependent task generation within a training
pipeline. The typical workflow is:
1. A raw structure is parsed into a cleaned `AtomArray`.
2. `SampleDesignTask` selects and applies a `DesignTask` to the `AtomArray`.
3. The resulting `AtomArray`, now annotated with a specific design task, is passed
   to a model-specific featurization pipeline which is shared by the inference pipeline.

At inference time, an `AtomArray` is provided that already specifies a design
task through its annotations, bypassing the sampling step. It is then passed through the same
featurization pipeline as the training data.
"""

import abc
import logging
from typing import Any, ClassVar

import numpy as np

from atomworks.ml.transforms.base import Transform

logger = logging.getLogger(__name__)


class SampleDesignTask(Transform):
    """Samples and applies a design task from a set of available tasks based on their frequencies.

    This transform evaluates all available design tasks to determine which ones can be applied
    to the current data, then samples one task based on the relative frequencies specified
    in the design_tasks dictionary. The sampled task name is added to the data dictionary
    under the "task" key.

    Args:
        design_tasks: Dictionary mapping task names to task configurations. Each task config
            must contain:
            - "transform": A DesignTaskABC instance that implements the task logic
            - "frequency": Positive float indicating the relative sampling frequency
        rng: Optional random number generator. If None, uses numpy.random.
    """

    def __init__(
        self,
        design_tasks: dict[str, dict[str, Any]],
        rng: np.random.Generator | None = None,
    ):
        # Check the format of the design tasks dictionary and remove tasks with zero frequency
        design_tasks_to_use = {}
        removed_design_task_names = []
        for name, task in design_tasks.items():
            assert "transform" in task, f"Design task {name} must have a 'transform' key"
            assert "frequency" in task, f"Design task {name} must have a 'frequency' key"
            if task["frequency"] > 0:
                design_tasks_to_use[name] = task
            else:
                removed_design_task_names.append(name)

        if removed_design_task_names:
            logger.info(
                f"Removed {len(removed_design_task_names)} design tasks with zero frequency: {', '.join(removed_design_task_names)}"
            )

        if not design_tasks_to_use:
            logger.warning(
                "No design tasks with non-zero frequency found. SampleDesignTask will act as an identity transform."
            )

        self.design_tasks = design_tasks_to_use
        self.rng = rng

    def forward(self, data: dict) -> dict:
        """Sample and apply a design task to the input data.

        This method:
        1. Evaluates all design tasks to find eligible ones that can be applied
        2. Samples one task based on the relative frequencies
        3. Adds the sampled task name to the data dictionary

        If no design tasks are configured, acts as an identity transform.

        Args:
            data: Input data dictionary containing atom array and other fields.

        Returns:
            Data dictionary with the sampled task name added under the "task" key,
            or unchanged if no design tasks are configured.

        Raises:
            AssertionError: If no eligible tasks are found for the current data.
        """
        # Early return if no design tasks configured (identity transform)
        if not self.design_tasks:
            return data

        # 1. Build list of eligible tasks
        eligible_tasks, frequencies = [], []

        for name, task in self.design_tasks.items():
            # ... annotate the data for the task selection
            data = task["transform"].annotate_for_task_selection(data)

            # ... evaluate whether the task can be applied to the data
            if task["transform"].can_apply(data):
                eligible_tasks.append(name)
                frequencies.append(task["frequency"])

        # 2. Sample a task
        assert len(eligible_tasks) > 0, "No eligible tasks found"
        probabilities = np.array(frequencies) / np.sum(frequencies)
        choice = self.rng.choice if self.rng else np.random.choice
        task_name = choice(eligible_tasks, p=probabilities)

        # 3. Annotate the given data with the sampled task
        data["task"] = {"name": task_name}

        return data


class DesignTaskABC(Transform, abc.ABC):
    """Abstract base class for design task transforms.

    This abstract base class defines the interface for design task transforms that can be
    sampled and applied during training. Design task transforms represent different
    protein design objectives that can be conditionally applied based on the input data.

    The pattern works as follows:
    1. Each concrete design task implements `can_apply()` to determine if it's applicable
    2. Each task implements `annotate_for_task_selection()` to add required annotations that
        are needed in `can_apply()` to determine if the task can be applied.
    3. The central `SampleDesignTask` transform evaluates all tasks and samples one based on
        frequencies provided in its `design_tasks` dictionary among the eligible tasks.
    4. The sampled `DesignTaskTransform` is then applied to the data to generate a design task.

    This enables flexible, data-dependent task sampling during training while maintaining
    a clean separation between task selection logic and task execution.
    """

    requires_previous_transforms: ClassVar[list[str]] = [SampleDesignTask]

    def annotate_for_task_selection(self, data: dict) -> dict:
        """Add any annotations to the data that are required for the task selection.

        This method allows design tasks to add annotations to the data that they need
        to evaluate whether they can be applied. Usually the `atom_array` object of the
        data dictionary is annotated with any extra annotations.

        Preferrably you should use the `annotator` module to add annotations to the data.
        Those can easily & lazily be generated & removed.

        Args:
            data: The input data dictionary containing the atom array and other fields.

        Returns:
            The data dictionary with additional annotations added.
        """
        return data

    @abc.abstractmethod
    def can_apply(self, data: dict) -> bool:
        """Evaluate whether the design task can be applied to the data.

        This method should check if the current data meets the requirements for
        applying this design task.

        Args:
            data: The input data dictionary containing the atom array and annotations.

        Returns:
            True if the task can be applied, False otherwise.
        """
        raise NotImplementedError

    def _check_forward(self, data: dict) -> None:
        assert "task" in data, "Design task must be applied to the data"
        assert "mask" in data["task"], "Design task must have a mask"
        assert "class" in data["task"], "Design task must have a class"
        assert "name" in data["task"], "Design task must have a name"

    def __call__(self, data: dict) -> dict:
        """Call `forward` and ensure that the data is annotated with the task and has a mask."""
        super().__call__(data)  # <-- calls `forward` via the `Transform` base class
        self._check_forward(data)
        return data
