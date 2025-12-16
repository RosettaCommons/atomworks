"""Base classes for transformations."""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import pprint
import re
import time
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import numpy as np
import torch
from toolz import valmap

from atomworks.ml.transforms._checks import check_contains_keys, check_does_not_contain_keys
from atomworks.ml.utils.rng import capture_rng_states, rng_state, serialize_rng_state_dict

logger = logging.getLogger("transforms")
DEBUG = os.getenv("DEBUG", True)
if DEBUG:
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug mode is on")
    import traceback
else:
    logger.setLevel(logging.INFO)


class TransformPipelineError(Exception):
    """A custom error class for Transform pipelines (via :class:`Compose`).

    Attributes:
        rng_state_dict: Optional RNG state dictionary for debugging purposes.
    """

    def __init__(self, message: str, rng_state_dict: dict[str, Any] | None = None):
        """Initialize TransformPipelineError.

        Args:
            message: The error message.
            rng_state_dict: Optional RNG state dictionary for debugging purposes.
        """
        super().__init__(message)
        # expose RNG state dict for debugging
        self.rng_state_dict = rng_state_dict


class TransformedDict(dict):
    """A thin wrapper around a dictionary that can be used to track the transform history.

    Behaves just like a regular dictionary but includes a ``__transform_history__`` attribute
    that tracks the sequence of transforms applied to the data.
    """

    def __new__(cls, __existing_dict_to_wrap: dict[str, Any] | None = None, **kwargs):
        """Create a new instance or return the existing TransformedDict instance.

        Note:
            To get a pure dictionary, simply use ``dict(transformed_dict)`` on a TransformedDict instance.
            TransformedDict's behave just like dicts for all intents and purposes.

        Args:
            __existing_dict_to_wrap: This is useful for wrapping an existing dictionary.
                The odd name is used as an unlikely name to avoid conflicts with the dict class.
            **kwargs: Additional keyword arguments to pass to the dictionary constructor.
        """
        # if the argument is already a TransformedDict, return it
        if isinstance(__existing_dict_to_wrap, TransformedDict):
            return __existing_dict_to_wrap

        # otherwise, instantiate a new built-in `dict`
        instance = super().__new__(cls)
        # ... and update it with the provided dictionary if given
        if __existing_dict_to_wrap is not None:
            assert len(kwargs) == 0, "Either `__existing_dict_to_wrap` or `kwargs` must be provided, but not both."
            # ... if the argument is a dict, update the instance with the dict
            instance.update(__existing_dict_to_wrap)
        # ... or update it with the keyword arguments if given
        else:
            # ... if the argument is a dict, update the instance with the dict
            instance.update(kwargs)

        # set the transform history tracker
        instance.__transform_history__ = []
        return instance


class Transform(ABC):
    """Abstract base class for transformations on dictionary objects.

    To write a subclass, you need to implement the :meth:`forward` method.
    Optionally, you can override :meth:`check_input` for input validation.

    Attributes:
        validate_input: Whether to validate the input.
        raise_if_invalid_input: Whether to raise an error if the input is invalid.
        requires_previous_transforms: Transforms that must have been applied before this transform.
        incompatible_previous_transforms: Transforms that cannot have preceded this transform.
        previous_transforms_order_matters: Whether the order of the transforms is important.
        _track_transform_history: Whether to track the transform history.
    """

    validate_input: bool = True
    raise_if_invalid_input: bool = True
    requires_previous_transforms: ClassVar[list[str]] = []
    incompatible_previous_transforms: ClassVar[list[str]] = []
    previous_transforms_order_matters: bool = False
    _track_transform_history: bool = True

    # To be implemented by subclasses (optional)
    def check_input(self, data: dict[str, Any]) -> None:  # noqa: B027
        """Check if the input dictionary is valid for the transform.

        Args:
            data: The input dictionary to validate.

        Raises:
            Exception: If the input is invalid.
        """
        pass

    @abstractmethod
    def forward(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Apply a transformation to the input dictionary and return the transformed dictionary.

        Args:
            data: The input dictionary to transform.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The transformed dictionary.
        """
        pass

    # Internal logic for formatting error messages, debugging, logging and transform history tracking
    def _format_error_msg(self, e: Exception) -> str:
        """Format the error message with optional traceback when in DEBUG mode.

        Args:
            e: The exception that occurred.

        Returns:
            Formatted error message.
        """
        msg = f"Invalid input for {self.__class__.__name__}: {e}"
        if DEBUG:
            msg += f"\n\n{traceback.format_exc()}\n" + "=" * 80
        return msg

    def _transform_to_str(self, t: str | Transform | ABCMeta) -> str:
        """Convert a transform to a string.

        Args:
            t: The transform to convert (string, Transform instance, or Transform class).

        Returns:
            String representation of the transform.
        """
        if isinstance(t, str):
            # case: transform was provided as string, e.g. as `"RemoveKeys"`
            return t
        elif isinstance(t, ABCMeta):
            # case: transform was provided as class, e.g. as `RemoveKeys`
            return t.__name__
        elif isinstance(t, Transform):
            # case: transform was provided as instance, e.g. as `RemoveKeys()`
            return t.__class__.__name__
        else:
            raise ValueError(f"Transform `{t}` cannot be converted to a string form for comparison of history.")

    def _ensure_has_transform_history(self, data: dict[str, Any] | TransformedDict) -> TransformedDict:
        """Ensure that the data dictionary has a transform history by wrapping it in a TransformedDict.

        Args:
            data: The data dictionary to wrap.

        Returns:
            TransformedDict instance with transform history.
        """
        data = TransformedDict(data)
        return data

    def _get_transform_history(self, data: TransformedDict) -> list[str]:
        """Get the transform history from the data.

        Args:
            data: The TransformedDict containing the history.

        Returns:
            List of transform names in the history.
        """
        return data.__transform_history__

    def _maybe_update_transform_history(self, data: TransformedDict) -> dict[str, Any]:
        """Update the transform history by appending the current transform to the transform history.

        Args:
            data: The TransformedDict to update.

        Returns:
            The updated data dictionary.
        """
        if self._track_transform_history:
            this_transform_record = {
                "name": self.__class__.__name__,
                "instance": hex(id(self)),
                "start_time": time.time(),
                "end_time": None,
                "processing_time": None,
            }
            # record the current transform in the transform history
            data.__transform_history__ = [*data.__transform_history__, this_transform_record]

        return data

    def _maybe_restore_transform_history(self, data: TransformedDict, transform_history: list[str]) -> dict[str, Any]:
        """Restore the transform history, in case the data was copied.

        Args:
            data: The TransformedDict to restore history for.
            transform_history: The history to restore.

        Returns:
            The data with restored history.
        """
        if not hasattr(data, "__transform_history__") or len(data.__transform_history__) == 0:
            # restore previous transform history if it is not present (e.g. if the data was copied)
            data.__transform_history__ = transform_history
        return data

    def _maybe_record_processing_time(self, data: TransformedDict) -> dict[str, Any]:
        """Record the processing time for the transform.

        Args:
            data: The TransformedDict to record timing for.

        Returns:
            The data with updated timing information.
        """
        if self._track_transform_history and len(data.__transform_history__) > 0:
            for reverse_idx in range(len(data.__transform_history__) - 1, -1, -1):
                # record the processing time for the current transform
                record = data.__transform_history__[reverse_idx]
                if record["instance"] == hex(id(self)):
                    start_time = record["start_time"]
                    end_time = time.time()
                    data.__transform_history__[reverse_idx]["end_time"] = end_time
                    data.__transform_history__[reverse_idx]["processing_time"] = end_time - start_time
        return data

    def _check_transform_history(self, data: TransformedDict) -> None:
        """Check if the previous transforms are valid for the transform.

        Args:
            data: The TransformedDict to check.

        Raises:
            TransformPipelineError: If the transform history is invalid.
        """
        # extract the transform history
        history = [record["name"] for record in data.__transform_history__]

        # ensure that `incompatible_previous_transforms` did not get applied
        for t in self.incompatible_previous_transforms:
            t = self._transform_to_str(t)
            if t in history:
                raise ValueError(
                    f"Transform `{self.__class__.__name__}` cannot be applied if any of the transforms {self.incompatible_previous_transforms} "
                    f"have been applied before it. Current transform history: {history}"
                )

        # get indices of `requires_previous_transforms` in the transform history
        indices = []
        for t in self.requires_previous_transforms:
            t = self._transform_to_str(t)
            pattern = re.compile(t)
            matches = [index for index, t in enumerate(history) if pattern.search(t)]
            if len(matches) == 0:
                raise ValueError(f"Transform `{t}` is missing from the transform history, which is {history}.")
            elif len(matches) > 1:
                raise ValueError(
                    f"Transform `{t}` appears multiple times in the transform history, which is {history}."
                )
            assert len(matches) == 1
            indices.append(matches[0])

        # check if the indices are in the correct order
        if self.previous_transforms_order_matters and (indices != sorted(indices)):
            current_order = ">".join([history[i] for i in sorted(indices)])
            required_order = ">".join(self.requires_previous_transforms)
            raise ValueError(
                f"Transform `{self.__class__.__name__}` requires the transforms {required_order} "
                f"to have been applied before it in this order, but the current order is {current_order}."
            )

    def __call__(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Validate and apply the transformation to the given dictionary.

        Args:
            data: The input dictionary to transform.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The transformed dictionary.

        Raises:
            TransformPipelineError: If the input is invalid and raise_if_invalid_input is True.
        """
        # enable history tracking if it is not already enabled
        data = self._ensure_has_transform_history(data)

        # validate input
        if self.validate_input:
            try:
                # check if the input is valid
                self._check_transform_history(data)
                self.check_input(data)
            except Exception as e:
                # if it is not valid, log or raise an error
                formatted_msg = self._format_error_msg(e)
                if self.raise_if_invalid_input:
                    logger.error(formatted_msg)
                    raise TransformPipelineError(formatted_msg) from e
                else:
                    logger.warning(formatted_msg)
                    return data

        # update transform history if it is being tracked
        data = self._maybe_update_transform_history(data)

        # get previous transform history (needed for `_maybe_restore_transform_history` later)
        # (NOTE: It is neccessary to carry the transform history outside the `forward` method
        #   and the `data` object to allow users to seamlessly copy the dict and work with the
        #   dict without losing the transform history.)
        transform_history = self._get_transform_history(data)

        # apply the transformation
        data = self.forward(data, *args, **kwargs)
        assert isinstance(
            data, dict
        ), f"`forward` method of {self.__class__.__name__} must return a dictionary, not {type(data)}."

        # restore the transform history if `data` was copied (which loses the transform history)
        data = self._ensure_has_transform_history(data)
        data = self._maybe_restore_transform_history(data, transform_history)
        data = self._maybe_record_processing_time(data)

        return data

    def __repr__(self) -> str:
        """String representation of the transform for debugging, notebooks and logging.

        Returns:
            String representation of the transform.
        """
        # Get all the attributes of the class
        repr_str = f"{self.__class__.__name__} at {hex(id(self))}"

        if len(self.__dict__) > 0:
            attributes = [
                f"{k}={pprint.pformat(v, indent=2, depth=1, compact=True, sort_dicts=False)}"
                for k, v in self.__dict__.items()
            ]
            repr_str += "(\n " + ",\n  ".join(attributes) + "\n)"
        return repr_str

    def __add__(self, other: Transform) -> Compose:
        """Add two transforms together to create a Compose instance.

        Args:
            other: Another Transform or Compose instance.

        Returns:
            A new Compose instance containing both transforms.

        Raises:
            ValueError: If other is not a Transform or Compose instance.
        """
        # Case 1: self & other are `Compose` instances
        #  ... overridden in `Compose` class
        # Case 2: self is a `Compose` instance and other is a `Transform` instance
        #  ... overridden in `Compose` class

        # Case 3: self is a `Transform` instance and other is a `Compose` instance
        if isinstance(self, Transform) and isinstance(other, Compose):
            return Compose([self, *other.transforms], track_rng_state=other.track_rng_state)
        # Case 4: self & other are simple `Transform` instances
        elif isinstance(self, Transform) and isinstance(other, Transform):
            return Compose([self, other])
        # Case 5: other is not a `Transform` instance
        else:
            raise ValueError(f"Expected a Transform or Compose, but got a {type(other)}")


class Compose(Transform):
    """Compose multiple transformations together.

    This class allows you to chain multiple transformations and apply them sequentially to a data dictionary.
    It is particularly useful for preprocessing pipelines where multiple steps need to be applied in a specific order.

    Attributes:
        transforms: A list of transformations to be applied.
        track_rng_state: Whether to track and serialize the random number generator (RNG) state. This is
            useful for debugging when dealing with probabilistic transformations. The RNG state is returned with
            the error message if the transform pipeline fails, allowing you to instantiate the same RNG state
            with ``eval`` for debugging.
    """

    _track_transform_history: bool = False  # Compose does not show up in the transform history

    def __init__(self, transforms: list[Transform], track_rng_state: bool = True, print_rng_state: bool = False):
        """Initialize the Compose transformation pipeline.

        Args:
            transforms: A list of transformations to be applied sequentially.
            track_rng_state: Whether to track and serialize the random number generator (RNG) state.
                This is useful for debugging when dealing with probabilistic transformations. The RNG state
                is returned with the error message if the transform pipeline fails, allowing you to instantiate
                the same RNG state with ``eval`` for debugging.
            print_rng_state: Whether to print the RNG state upon failure. This can be useful
                for debugging and reproducing specific states for transforms with stochasticity.

        Raises:
            ValueError: If transforms is not a list or tuple, if it is empty, or if it contains elements that
                are not instances of Transform.
        """
        if not isinstance(transforms, list | tuple):
            raise ValueError(f"Expected a list or tuple of Transforms, but got a {type(transforms)}")

        if not len(transforms) > 0:
            raise ValueError("Got an empty list of transforms.")

        if not all(isinstance(t, Transform) for t in transforms):
            invalid_type = next(t for t in transforms if not isinstance(t, Transform))
            raise ValueError(f"Expected a list or tuple of Transforms, but got a {type(invalid_type)}")

        self.transforms = transforms
        self.track_rng_state = track_rng_state
        self.latest_rng_state_dict = None
        self.print_rng_state = print_rng_state

    def __add__(self, other: Transform | list[Transform] | Compose) -> Compose:
        """Add another transform or compose to this compose.

        Args:
            other: Another Transform, list of Transforms, or Compose instance.

        Returns:
            A new Compose instance containing all transforms.

        Raises:
            ValueError: If other is not a valid type.
        """
        if isinstance(other, Compose):
            return Compose(
                self.transforms + other.transforms, track_rng_state=self.track_rng_state or other.track_rng_state
            )
        elif isinstance(other, Transform):
            return Compose([*self.transforms, other], track_rng_state=self.track_rng_state)
        elif isinstance(other, list):
            return Compose(self.transforms + other, track_rng_state=self.track_rng_state)
        else:
            raise ValueError(f"Expected a Transform or list of Transforms or Compose, but got a {type(other)}")

    def check_input(self, data: dict) -> None:
        """Check if the input is valid for the compose.

        Compose is always valid, so this method does nothing.

        Args:
            data: The input data to check.
        """
        # Compose is always valid
        pass

    def _stop_transforms(
        self,
        next_transform: Transform,
        next_transform_idx: int,
        stop_before: Transform | int | str | None = None,
    ) -> bool:
        """Check if transforms should stop before the next transform.

        Args:
            next_transform: The next transform to apply.
            next_transform_idx: The index of the next transform.
            stop_before: The transform, name, or index to stop before.

        Returns:
            True if transforms should stop before the next transform.

        Raises:
            ValueError: If stop_before is not a valid type.
        """
        if stop_before is None:
            return False
        elif isinstance(stop_before, int):
            return next_transform_idx == stop_before
        elif isinstance(stop_before, str):
            return next_transform.__class__.__name__ == stop_before
        elif isinstance(stop_before, Transform):
            return next_transform.__class__.__name__ == stop_before.__class__.__name__
        else:
            raise ValueError(f"Expected a Transform or str or int, but got a {type(stop_before)}")

    def forward(
        self,
        data: dict,
        rng_state_dict: dict[str, Any] | None = None,
        _stop_before: Transform | str | int | None = None,
    ) -> dict:
        """Apply a series of transformations to the input data.

        Args:
            data: The input data to be transformed.
            rng_state_dict: Random number generator state dictionary.
                If provided, sets the RNG state before applying transforms.
            _stop_before: Specifies a point to stop the transformation
                process. Can be a Transform instance, a string (transform class name), or an integer (index).

        Returns:
            The transformed data.

        Raises:
            Exception: If any transform in the pipeline fails, with details about the failure point and RNG state.
        """

        # set the RNG state context if given
        with (
            rng_state(rng_state_dict, include_cuda=False) if rng_state_dict else contextlib.nullcontext()
        ) as rng_state_dict:
            if self.track_rng_state and rng_state_dict is None:
                # collect RNG states at the start of the pipeline and execute the transforms
                rng_state_dict = capture_rng_states()
                self.latest_rng_state_dict = rng_state_dict

            try:
                # execute the transforms
                for idx, transform in enumerate(self.transforms):
                    if self._stop_transforms(transform, idx, _stop_before):
                        # ... capability to stop before a specific transform for debugging
                        break

                    # ... otherwise apply the transform
                    data = transform(
                        data
                    )  # BREAKPOINT: Set debug breakpoint here to step through the transforms one-by-one
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # construct error message including the RNG states
                failed_transform_name = transform.__class__.__name__

                # Give more informative messages for failures within a ConditionalRoute or RandomRoute
                if (
                    failed_transform_name in ["ConditionalRoute", "RandomRoute"]
                    and hasattr(data, "__transform_history__")
                    and len(data.__transform_history__) > 0
                ):
                    last_transform = data.__transform_history__[-1]["name"]
                    if last_transform != failed_transform_name:
                        failed_transform_name += f" -> {last_transform}"

                msg = f"Transforms failed at stage `{failed_transform_name}`: " + str(e)
                if "example_id" in data:
                    msg += f"\nFailure occurred for example ID: {data['example_id']}."
                if self.track_rng_state and self.print_rng_state:
                    msg += "\nRandom number generator states at the start of the pipeline (you can instantiate the string below with `eval` for debugging):\n"
                    msg += repr(serialize_rng_state_dict(rng_state_dict))

                # Update error message of original exception
                e.args = (msg,)

                # Raise the new custom exception with the original traceback
                raise e.with_traceback(e.__traceback__)  # noqa: B904

        return data

    def __repr__(self) -> str:
        return "Compose(\n  " + ",\n  ".join([str(t.__class__.__name__) for t in self.transforms]) + "\n)"

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: int | slice | Iterable[int]) -> Transform:
        if isinstance(idx, slice):
            return Compose(self.transforms[idx], track_rng_state=self.track_rng_state)
        elif hasattr(idx, "__iter__"):
            return Compose([self.transforms[i] for i in idx], track_rng_state=self.track_rng_state)
        else:
            return self.transforms[idx]


class ListBuilder:
    """A convenience class to build lists element by element with the '+=' operator"""

    def __init__(self):
        self.list = []

    def __add__(self, other: Any) -> ListBuilder:
        if isinstance(other, list):
            self.list.extend(other)
        elif isinstance(other, ListBuilder):
            self.list.extend(other.list)
        else:
            self.list.append(other)
        return self

    def tolist(self) -> list[Any]:
        return self.list


class RemoveKeys(Transform):
    """
    Remove keys from the data dictionary.
    """

    def __init__(self, keys: list[str], require_keys_exist: bool = True):
        self.keys = keys
        self.validate_input = require_keys_exist

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, self.keys)

    def forward(self, data: dict) -> dict:
        for key in self.keys:
            if key in data:
                del data[key]
        return data


class SubsetToKeys(Transform):
    """
    Keep only the keys in the data dictionary.
    """

    def __init__(self, keys: list[str], require_keys_exist: bool = True):
        self.keys = keys
        self.validate_input = require_keys_exist

    def check_input(self, data: dict) -> None:
        pass

    def forward(self, data: dict) -> dict:
        return {key: data[key] for key in self.keys if key in data}


class AddData(Transform):
    """
    Add data to the data dictionary.
    """

    def __init__(self, data: dict, allow_overwrite: bool = False):
        self.data = data
        self.validate_input = not allow_overwrite

    def check_input(self, data: dict) -> None:
        check_does_not_contain_keys(data, self.data.keys())

    def forward(self, data: dict) -> dict:
        data.update(self.data)
        return data


class LogData(Transform):
    """
    Log the data dictionary. Meant for debugging.
    """

    _track_transform_history: bool = False  # LogData does not show up in the transform history

    def __init__(self, log_level: int = logging.INFO, depth: int | None = 1, **pprint_kwargs):
        assert depth is None or depth > 0, "Depth must be a positive integer or None"
        self.log_level = log_level
        self.depth = depth
        self.pprint_kwargs = pprint_kwargs

    def check_input(self, data: dict) -> None:
        pass

    def forward(self, data: dict) -> dict:
        # Construct log message
        msg = "=" * 80 + "\n"
        msg += f"Data: \n{pprint.pformat(data, indent=2, depth=self.depth, sort_dicts=False, **self.pprint_kwargs)}\n"
        msg += "=" * 80

        # Log the message
        logger.log(
            level=self.log_level,
            msg=msg,
        )

        return data


class PickleToDisk(Transform):
    """
    Save the data dictionary to a pickle file.
    """

    def __init__(
        self,
        dir_path: str,
        file_name_func: Callable[[dict], str] | None = None,
        save_transform_history: bool = False,
        overwrite: bool = False,
    ):
        self.dir_path = dir_path
        self.file_name_func = file_name_func
        self.overwrite = overwrite
        self.save_transform_history = save_transform_history

        if not file_name_func:
            file_name_func = lambda data: f"{data['id']}.pkl"  # noqa

        # Ensure the directory exists
        os.makedirs(self.dir_path, exist_ok=True)

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["id"])

    def forward(self, data: dict) -> dict:
        file_name = self.file_name_func(data)
        file_path = os.path.join(self.dir_path, file_name)
        if os.path.exists(file_path) and not self.overwrite:
            raise ValueError(f"File {file_path} already exists. Set overwrite=True to overwrite it.")

        with open(file_path, "wb") as f:
            # NOTE: We cast the data to a dict to ensure that the data is serializable
            #  and that deserialization does not fail due to the presence of custom classes.
            #  (in particular the `TransformedDict` class)
            pickle.dump(dict(data), f)

        return data


class RaiseError(Transform):
    """
    Raises an error for testing and debugging purposes.
    """

    def __init__(self, error_type: Exception = ValueError, error_message: str = "User requested raising an error."):
        self.error_type = error_type
        self.error_message = error_message

    def check_input(self, data: dict[str, Any]) -> None:
        pass

    def forward(self, data: dict) -> dict:
        raise self.error_type(self.error_message)


class Identity(Transform):
    """
    Identity transform. Does nothing and just passes the data through.
    """

    validate_input = False
    raise_if_invalid_input = False
    _track_transform_history = False

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


class RandomRoute(Transform):
    """
    Route probabilistically between various transforms.

    This transform is useful for routing between different transforms probabilistically, e.g. for
    sampling different cropping strategies.
    """

    validate_input = False
    raise_if_invalid_input = False
    _track_transform_history: bool = True  # RandomRoute records history because it changes the RNG state

    def __init__(self, transforms: list[Transform], probs: list[float]):
        """
        Initializes the RandomRoute transform.

        Args:
            transforms (list[Transform]): A list of transformations to route between.
            probs (list[float]): A list of probabilities corresponding to each transform. The probabilities
                must be non-negative and sum to 1. There must be as many probabilities as there are transforms.

        Raises:
            AssertionError: If inputs are invalid (e.g. probabilities don't add up, are negative, etc.)
        """
        # Validate inputs
        assert len(transforms) == len(probs), (
            f"Number of transforms must match number of probabilities. "
            f"Got {len(transforms)} transforms and {len(probs)} probabilities."
        )
        assert np.isclose(np.sum(probs), 1) or np.isclose(
            np.sum(probs), 0
        ), f"Probabilities must sum to 1 or 0. Got {np.sum(probs)}"
        assert all(isinstance(t, Transform) for t in transforms), (
            f"All elements in transforms must be Transform instances. "
            f"Got {type(next(t for t in transforms if not isinstance(t, Transform)))}"
        )

        self.transforms = transforms
        self.probs = probs

    @classmethod
    def from_dict(cls, transform_dict: dict[Transform, float]) -> RandomRoute:
        probs = list(transform_dict.values())
        transforms = list(transform_dict.keys())
        return cls(transforms, probs)

    @classmethod
    def from_list(cls, transform_list: list[tuple[float, Transform]]) -> RandomRoute:
        probs, transforms = zip(*transform_list, strict=False)
        return cls(transforms, probs)

    def check_input(self, data: dict[str, Any]) -> None:
        pass

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        # Choose a transform probabilistically

        # EDGE CASE: If the probabilities sum to 0, skip the transform
        if np.isclose(np.sum(self.probs), 0):
            # skip
            return data

        idx = np.random.choice(len(self.transforms), p=self.probs)

        # Apply the transform
        return self.transforms[idx](data)


class ConditionalRoute(Transform):
    """
    Route conditionally between various transforms.

    This Transform is useful for routing between different transforms based on a condition, e.g. skipping transforms during inference.
    """

    def __init__(self, condition_func: Callable[[dict[str, Any]], Any], transform_map: dict[Any, Transform]):
        """
        Initialize the ConditionalRoute transformation.

        Args:
            condition_func (Callable[[dict[str, Any]], Any]): A function that takes the data dictionary and returns a condition value.
            transform_map (dict[Any, Transform]): A dictionary mapping condition values to their corresponding transforms.

        Example:
            ```python
            ConditionalRoute(
                condition_func=lambda data: data.get("mode", "inference"),
                transform_map={
                    "train": TrainingTransform(),
                    "inference": Identity(),
                    # Defaults to Identity if no match; "inference" included for clarity
                },
            )
            ```
        """
        self.condition_func = condition_func
        self.transform_map = transform_map

    def check_input(self, data: dict[str, Any]) -> None:
        # No specific input validation required for routing
        pass

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply the appropriate transformation based on the condition value.

        Args:
            data (dict[str, Any]): The input data dictionary.

        Returns:
            dict[str, Any]: The transformed data dictionary.
        """
        condition_value = self.condition_func(data)
        transform = self.transform_map.get(condition_value, Identity())
        return transform(data)


def convert_to_torch(data: dict[str, Any], keys: list[str], device: str = "cpu") -> dict[str, Any]:
    """Convert the contents of specified `data` keys to torch tensors and move them to the specified device.

    For each given top-level `data` key, all nested numpy arrays are converted to torch tensors.

    Args:
        data (dict[str, Any]): The input data dictionary.
        keys (list[str]): List of `data` keys within which to search for numpy arrays to convert to torch tensors.
        device (str): The device to which the tensors should be moved (e.g., 'cpu', 'cuda'). Default is 'cpu'.

    Returns:
        dict[str, Any]: The data dictionary with numpy arrays converted to torch tensors.
    """
    # Set of supported numpy data types
    supported_dtypes = (
        np.float64,
        np.float32,
        np.float16,
        np.complex64,
        np.complex128,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
        np.bool_,
    )

    def _convert_to_tensor(value: Any) -> Any:
        """Convert a value to a torch tensor if it is a numpy array or recursively handle nested dictionaries."""
        if isinstance(value, np.ndarray) and value.dtype in supported_dtypes:
            return torch.tensor(value, device=device)
        elif isinstance(value, dict):
            return valmap(_convert_to_tensor, value)
        elif isinstance(value, list):
            return [_convert_to_tensor(v) for v in value]
        else:
            return value

    for key in keys:
        if key in data:
            data[key] = _convert_to_tensor(data[key])
        else:
            raise KeyError(f"Key '{key}' not found in the data dictionary.")

    return data


class ConvertToTorch(Transform):
    """
    Converts the contents of specified `data` keys to torch tensors and moves them to the specified device.
    """

    def __init__(self, keys: list[str], device: str = "cpu"):
        self.keys = keys
        self.device = device

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, self.keys)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        return convert_to_torch(data, self.keys, self.device)


class RaiseOnCondition(Transform):
    """
    Raises a user-specified exception if a given condition is met.
    """

    def __init__(self, condition: callable, error_message: str, exception_to_raise: type[Exception] = ValueError):
        self.condition = condition
        self.error_message = error_message
        self.exception_class = exception_to_raise

    def check_input(self, data: dict[str, Any]) -> None:
        pass

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.condition(data):
            raise self.exception_class(self.error_message)
        return data


class ApplyFunction(Transform):
    """
    Applies a function to the data dictionary.
    """

    def __init__(self, func: callable):
        self.func = func

    def check_input(self, data: dict[str, Any]) -> None:
        pass

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.func(data)
