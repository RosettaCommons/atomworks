"""
StandardAnnotation system with:
 - a ``StandardAnnotationBase`` abstract base class for typed, defaulted, registry-driven
   annotations on ``AtomArray`` objects
 - metaclasses ``StandardAnnotationMeta`` that handles auto-registration and attribute pre-computation
 - accessor singleton ``STANDARD_ANNOTATIONS``
"""

from __future__ import annotations

from abc import ABC, ABCMeta
from collections.abc import Callable
from enum import Enum, StrEnum
from typing import Any, ClassVar, Literal

import numpy as np
from biotite.structure import AtomArray

from atomworks.io.utils import scatter
from atomworks.io.utils.atom_array_plus import AnnotationList2D
from atomworks.io.utils.selection import (
    _validate_n_body_and_type,
    get_annotation,
    get_annotation_categories,
    get_residue_starts,
)

__all__ = [
    "NO_DEFAULT",
    "STANDARD_ANNOTATIONS",
    "AnnotationRegistryAccessor",
    "StandardAnnotationBase",
]

MAX_TEXT_ANNOTATION_WIDTH = 64


class _DefaultValue(Enum):
    MISSING = "no uniform default"


NO_DEFAULT = _DefaultValue.MISSING
"""Sentinel for annotations without a structure-independent scalar fill value."""


class Level(StrEnum):
    """
    Level-based hierarchy for describing information in a molecular structure.

    Used for example to specify the level at which a `StandardAnnotation` applies.
    """

    ATOM = "atom"
    TOKEN = "token"
    RESIDUE = "residue"
    CHAIN = "chain"
    MOLECULE = "molecule"
    SYSTEM = "system"

    def _get_segment_or_group(self, atom_array: AtomArray) -> tuple[str, np.ndarray]:
        if self == Level.ATOM:
            raise ValueError("Apply and spread does not make sense for atom level.")
        if self == Level.TOKEN:
            try:
                from atomworks.ml.utils.token import get_token_starts
            except ImportError:
                raise ImportError(
                    "`atomworks.ml` module not found. All token-level functionality requires `atomworks.ml`, "
                    "so `Level.TOKEN` is not supported when it is unavailable."
                ) from None
            return "segment", get_token_starts(atom_array, add_exclusive_stop=True)
        elif self == Level.RESIDUE:
            return "segment", get_residue_starts(atom_array, add_exclusive_stop=True)
        elif self == Level.CHAIN:
            chain_key = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
            return "group", atom_array.get_annotation(chain_key)
        elif self == Level.MOLECULE:
            return "group", atom_array.get_annotation("molecule_id")
        elif self == Level.SYSTEM:
            return "group", np.ones(atom_array.array_length(), dtype=np.int32)
        else:
            raise ValueError(f"Invalid level: {self}")

    def apply(self, atom_array: AtomArray, data: np.ndarray, func: Callable[[AtomArray], Any]) -> Any:
        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.apply_segment_wise(grouping, data, func)
        elif strategy == "group":
            return scatter.apply_group_wise(grouping, data, func)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def spread(self, atom_array: AtomArray, data: np.ndarray) -> np.ndarray:
        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.spread_segment_wise(grouping, data)
        elif strategy == "group":
            return scatter.spread_group_wise(grouping, data)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def apply_and_spread(
        self,
        atom_array: AtomArray,
        data: np.ndarray,
        func: Callable[[np.ndarray], Any],
    ) -> np.ndarray:
        """
        Apply a function to the data and spread the result back to the original positions.
        """

        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.apply_and_spread_segment_wise(grouping, data, func)
        elif strategy == "group":
            return scatter.apply_and_spread_group_wise(grouping, data, func)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")


class StandardAnnotationMeta(ABCMeta):
    """Metaclass for all standard annotations.

    Registers concrete subclasses into a shared registry and pre-computes
    the derived canonical storage name (``full_name``).

    The registry ``StandardAnnotationMeta._registry`` is shared across all
    subclasses. This enforces name uniqueness across ALL standard annotations.
    """

    _registry: ClassVar[dict[str, type[StandardAnnotationBase]]] = {}

    @classmethod
    def _registered_names(cls) -> dict[str, type[StandardAnnotationBase]]:
        """Return every public name already claimed by a registered annotation."""
        return {
            candidate: registered_cls
            for registered_name, registered_cls in cls._registry.items()
            for candidate in (
                registered_name,
                registered_cls.full_name,
                *registered_cls.aliases,
            )
        }

    def __new__(meta, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs):  # noqa: N804
        annotation_name = namespace.get("name")
        if annotation_name:
            # Validate required attributes
            required_attrs = ["name", "n_body", "level", "dtype"]
            for attr in required_attrs:
                assert attr in namespace, f"StandardAnnotations must define the {attr} attribute."

            assert (
                "_" not in annotation_name
            ), "StandardAnnotations must not contain underscores in their names (reserved for internal use)."

            declared_aliases = namespace.get("aliases", ())
            if isinstance(declared_aliases, str) or not isinstance(declared_aliases, list | tuple):
                raise TypeError("StandardAnnotation aliases must be a list or tuple of strings.")
            aliases = tuple(declared_aliases)
            if any(not isinstance(alias, str) or not alias for alias in aliases):
                raise TypeError("StandardAnnotation aliases must be non-empty strings.")
            if len(set(aliases)) != len(aliases):
                raise ValueError(f"StandardAnnotation '{annotation_name}' declares duplicate aliases.")

            if "alias" in namespace:
                raise TypeError(
                    "StandardAnnotation alias= is no longer supported; override get_full_name() for a custom "
                    "canonical name or use aliases= for accepted legacy names."
                )

        cls = super().__new__(meta, name, bases, namespace, **kwargs)

        if annotation_name:
            cls.aliases = aliases
            cls.enum = cls.dtype if isinstance(cls.dtype, type) and issubclass(cls.dtype, Enum) else None
            if isinstance(cls.default_value, Enum) and cls.default_value is not NO_DEFAULT:
                cls.default_value = cls.default_value.value

            if "is_symmetric" not in namespace:
                assert cls.n_body <= 1, "Multi-body StandardAnnotations must define the is_symmetric attribute."
                cls.is_symmetric = True

            cls.level = Level(cls.level)
            assert (
                cls.n_body != 0 or cls.level == Level.SYSTEM
            ), "0-body StandardAnnotations must use level=Level.SYSTEM."

            # Eagerly compute and cache the canonical storage name.
            cls.full_name = cls.get_full_name()

            if cls.full_name in cls.aliases:
                raise ValueError(
                    f"StandardAnnotation '{annotation_name}' aliases its canonical name '{cls.full_name}'."
                )

            claimed = StandardAnnotationMeta._registered_names()
            for candidate in (annotation_name, cls.full_name, *cls.aliases):
                if existing := claimed.get(candidate):
                    raise ValueError(
                        f"StandardAnnotation name or alias '{candidate}' is already registered to "
                        f"{existing.__name__}."
                    )

            # Register in shared registry
            StandardAnnotationMeta._registry[annotation_name] = cls

        return cls

    def __repr__(self) -> str:  # noqa: N804
        try:
            if self.n_body > 1:
                return (
                    f"{self.__name__}(name={self.name}, n_body={self.n_body}, "
                    f"level={self.level}, is_symmetric={self.is_symmetric})"
                )
            else:
                return f"{self.__name__}(name={self.name}, n_body={self.n_body}, level={self.level})"
        except AttributeError:
            # NOTE: This is a fallback for the abstract base class to ensure that methods like `__mro__` continue to work
            return super().__repr__()


class StandardAnnotationBase(ABC, metaclass=StandardAnnotationMeta):
    """Abstract base class for all standard annotations.

    A standard annotation is a typed, defaulted annotation with mask semantics
    that can be stored on an :py:class:`~biotite.structure.AtomArray` and
    serialized/deserialized from CIF files.

    Attributes:
        (required on concrete subclasses)
        name: Registry name. No underscores allowed.
        n_body: Number of bodies (0 for a single system-wide value, 1 for per-atom/residue,
            2 for pairwise).
        level: Level at which the annotation applies.
        dtype: Semantic scalar type for the annotation values. Enum subclasses
            describe closed categorical domains.

        (optional)
        aliases: Alternative field names accepted when reading legacy data.
        is_symmetric: Whether pairwise annotation is symmetric. Required if ``n_body > 1``.
        value_shape: Per-item shape, excluding the atom or pair axis. Scalars use ``()``.
        default_value: Scalar fill value for a uniform default, or :py:data:`NO_DEFAULT`.
            Enum members are normalized to their stored scalar values at class creation.
            Structure-dependent and sparse defaults override :py:meth:`default_annotation`.

        (computed by metaclass)
        full_name: Canonical persisted field name.
        enum: Enum class from ``dtype``, or ``None`` for open-ended scalar types.
    """

    _prefix: ClassVar[str] = "annotation"

    # --- Attributes to be defined by concrete subclasses ---
    name: ClassVar[str]
    n_body: ClassVar[int]
    level: ClassVar[Level]
    dtype: ClassVar[np.dtype | type[np.generic] | type[float] | type[int] | type[str] | type[bool] | type[Enum]]
    aliases: ClassVar[tuple[str, ...]] = ()
    is_symmetric: ClassVar[bool]  # only needs to be set if n_body > 1
    value_shape: ClassVar[tuple[int, ...]] = ()
    default_value: ClassVar[Any] = NO_DEFAULT

    # --- Pre-computed attributes (set by metaclass) ---
    full_name: ClassVar[str]
    enum: ClassVar[type[Enum] | None]

    def __init__(self):
        """DO NOT INSTANTIATE THIS CLASS. Use its class methods directly."""
        raise RuntimeError(
            f"{self.__class__.__name__} is a definitional class and should not be instantiated. "
            "Use its class methods directly."
        )

    @classmethod
    def _resolve_level(cls, level: Level | str | None) -> Level:
        """Resolves the level to a Level enum."""
        return Level(level if level is not None else cls.level)

    @classmethod
    def _resolve_n_body(cls, n_body: int | None) -> int:
        """Resolves the number of bodies to an integer."""
        return n_body if n_body is not None else cls.n_body

    # --- Name generation methods ---
    @classmethod
    def get_full_name(cls) -> str:
        """Returns the full name of the StandardAnnotation at a given body order and level."""
        return f"{cls._prefix}_{cls.name}_{cls.n_body}_{cls.level}"

    @classmethod
    def storage_dtype(cls) -> np.dtype:
        """Return the NumPy dtype used to store annotation values."""
        if isinstance(cls.dtype, type) and issubclass(cls.dtype, Enum):
            return np.asarray([member.value for member in cls.dtype]).dtype
        dtype = np.dtype(cls.dtype)
        if cls.n_body != 0 and dtype.kind in ("U", "S") and dtype.itemsize == 0:
            return np.dtype((dtype.type, MAX_TEXT_ANNOTATION_WIDTH))
        return dtype

    @classmethod
    def get_feature_name(
        cls,
        n_body: int | None = None,
        level: Level | str | None = None,
        suffix: str = "",
    ) -> str:
        """Returns the feature name at a given body order and level.

        Args:
            n_body: The body order to get the feature name for.
            level: The level to get the feature name for.
            suffix: An optional suffix to add to the feature name to allow
             flexibility for adding multiple different features for a single StandardAnnotation.

        Returns:
            The feature name of the StandardAnnotation. This will be of the form:
            `feature-<suffix>_<standard_annotation_name>_<n_body>_<level>`
        """
        level, n_body = cls._resolve_level(level), cls._resolve_n_body(n_body)
        suffix_str = f"-{suffix}" if suffix and not suffix.startswith("-") else suffix
        return f"feature{suffix_str}_{cls.name}_{n_body}_{level}"

    @classmethod
    def has_annotation(cls, atom_array: AtomArray) -> bool:
        """Check if the value annotation for this StandardAnnotation is present on the AtomArray."""
        categories = get_annotation_categories(atom_array, n_body=cls.n_body)
        return any(name in categories for name in (cls.full_name, *cls.aliases))

    @staticmethod
    def _values_equal(first: np.ndarray, second: np.ndarray) -> bool:
        first, second = np.asarray(first), np.asarray(second)
        equal_nan = np.issubdtype(first.dtype, np.inexact) and np.issubdtype(second.dtype, np.inexact)
        return np.array_equal(first, second, equal_nan=equal_nan)

    @classmethod
    def _annotations_equal(
        cls,
        first: np.ndarray | AnnotationList2D,
        second: np.ndarray | AnnotationList2D,
    ) -> bool:
        if isinstance(first, AnnotationList2D) and isinstance(second, AnnotationList2D):
            return np.array_equal(first.pairs, second.pairs) and cls._values_equal(first.values, second.values)
        if isinstance(first, AnnotationList2D) or isinstance(second, AnnotationList2D):
            return False
        return cls._values_equal(first, second)

    @classmethod
    def _read_existing(cls, atom_array: AtomArray) -> np.ndarray | AnnotationList2D | None:
        """Read the canonical field or one alias, rejecting conflicting duplicates."""
        existing = [
            (name, value)
            for name in (cls.full_name, *cls.aliases)
            if (value := get_annotation(atom_array, name, n_body=cls.n_body)) is not None
        ]
        if not existing:
            return None
        first_name, first = existing[0]
        for other_name, other in existing[1:]:
            if not cls._annotations_equal(first, other):
                raise ValueError(
                    f"Conflicting fields '{first_name}' and '{other_name}' both resolve to "
                    f"StandardAnnotation '{cls.name}'."
                )
        return first

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray | AnnotationList2D:
        """Fill a system or one-body annotation, or override for structure-dependent/sparse defaults."""
        if cls.default_value is NO_DEFAULT or cls.n_body not in (0, 1):
            raise NotImplementedError(f"StandardAnnotation `{cls.name}` must implement default_annotation().")
        shape = () if cls.n_body == 0 else (atom_array.array_length(), *cls.value_shape)
        dtype = cls.storage_dtype()
        if dtype.kind in ("U", "S") and dtype.itemsize == 0:
            dtype = np.asarray(cls.default_value, dtype=dtype).dtype
        return np.full(shape, cls.default_value, dtype=dtype)

    @classmethod
    def mask_from_annotation(cls, annotation: np.ndarray | AnnotationList2D) -> np.ndarray | AnnotationList2D:
        """Derive this StandardAnnotation's mask from its value annotation."""
        if np.issubdtype(cls.storage_dtype(), np.bool_):
            return annotation
        raise NotImplementedError(
            f"StandardAnnotation `{cls.name}` (class `{cls.__name__}`) does not define mask_from_annotation()."
        )

    # --- Core Functionality ---
    @classmethod
    def mask(
        cls,
        atom_array: AtomArray,
        default: Any | Literal["generate", "raise"] = "generate",
    ) -> np.ndarray | AnnotationList2D:
        """Derive a mask from the persisted or generated value annotation.

        Args:
            atom_array: The AtomArray to get the mask from.
            default: Behavior when the value annotation is absent. If ``"generate"``,
                derive from :meth:`default_annotation`; if ``"raise"``, raise.
                If any other value, that value is returned.

        Returns:
            The mask for this StandardAnnotation.

        Raises:
            ValueError: If the annotation is absent and ``default`` is ``"raise"``.
        """
        annotation = cls._read_existing(atom_array)
        if annotation is None:
            if isinstance(default, str) and default == "generate":
                annotation = cls.default_annotation(atom_array)
            elif isinstance(default, str) and default == "raise":
                raise ValueError(f"AtomArray is missing {cls.n_body}-body annotation `{cls.full_name}`.")
            else:
                return default
        return cls.mask_from_annotation(annotation)

    @classmethod
    def annotation(
        cls,
        atom_array: AtomArray,
        default: Any | Literal["generate", "raise"] = "generate",
    ) -> np.ndarray | AnnotationList2D:
        """Gets an annotation from an AtomArray, falling back to a generated default.

        Args:
            atom_array: The AtomArray to get the annotation from.
            default: The default value to return if the annotation is not found.
                If ``"generate"``, the default annotation is generated and returned.
                If ``"raise"``, a ValueError is raised if the annotation is not found.
                If any other value, that value is returned.

        Returns:
            The annotation for this StandardAnnotation.

        Raises:
            ValueError: If the annotation is not found and ``default`` is ``"raise"``.
        """
        annotation = cls._read_existing(atom_array)

        if annotation is None:
            if isinstance(default, str) and default == "generate":
                annotation = cls.default_annotation(atom_array)
            elif isinstance(default, str) and default == "raise":
                raise ValueError(f"AtomArray is missing {cls.n_body}-body annotation `{cls.full_name}`.")
            else:
                annotation = default

        return annotation

    @classmethod
    def annotation_and_mask_at_declared_level(cls, atom_array: AtomArray) -> tuple[np.ndarray, np.ndarray]:
        """Return values and active masks with one entry per segment at the StandardAnnotation's Level.

        System annotations return one entry; 2d annotations return all entries in the AnnotationList2D.
        """
        annotation = cls.annotation(atom_array, default="generate")
        mask = cls.mask(atom_array, default="generate")
        if cls.n_body == 0:
            return np.asarray(annotation).reshape(1), np.asarray(mask, dtype=bool).reshape(1)
        if isinstance(annotation, AnnotationList2D):
            if not isinstance(mask, AnnotationList2D):
                raise TypeError(f"Two-body annotation `{cls.name}` returned a one-body mask.")
            return np.asarray(annotation.values), np.asarray(mask.values, dtype=bool)
        if isinstance(mask, AnnotationList2D):
            raise TypeError(f"One-body annotation `{cls.name}` returned a two-body mask.")

        n_atoms = atom_array.array_length()
        if cls.level == Level.ATOM or n_atoms == 0:
            indices = np.arange(n_atoms)
        else:
            indices = cls.level.apply(atom_array, np.arange(n_atoms), lambda values: values[0])

        return np.asarray(annotation)[indices], np.asarray(mask, dtype=bool)[indices]

    @classmethod
    def set_annotation(cls, atom_array: AtomArray, *args, **annotation_kwargs) -> None:
        """Set the annotation, synchronizing existing aliases and enforcing declared pairwise symmetry."""
        if cls.n_body in (0, 1):
            if len(args) == 0:
                array = annotation_kwargs.pop("array")
            elif len(args) == 1:
                array = args[0]
            else:
                raise ValueError(f"Only one argument is allowed for {cls.n_body}-body annotations. Got {len(args)}.")
            assert len(annotation_kwargs) == 0, "Unexpected annotation keyword arguments."

            if cls.n_body == 0:
                _validate_n_body_and_type(atom_array, cls.n_body, f"set {cls.full_name}")
                array = np.asarray(array, dtype=cls.storage_dtype())
                atom_array.set_annotation(cls.full_name, array, n_body=0)
            else:
                atom_array.set_annotation(cls.full_name, array)

            categories = get_annotation_categories(atom_array, n_body=cls.n_body)
            for alias in cls.aliases:
                if alias in categories:
                    if cls.n_body == 0:
                        atom_array.set_annotation(alias, array, n_body=0)
                    else:
                        atom_array.set_annotation(alias, array)

        elif cls.n_body == 2:
            _validate_n_body_and_type(atom_array, cls.n_body, f"set {cls.full_name}")

            if len(args) == 1:
                assert isinstance(args[0], AnnotationList2D), "Only AnnotationList2D is allowed for 2-body annotations."
                annot = args[0]
            elif len(args) == 0:
                annot = AnnotationList2D(atom_array.array_length(), **annotation_kwargs)
            else:
                raise ValueError(f"Only one argument is allowed for 2-body annotations. Got {len(args)} arguments.")

            if cls.is_symmetric:
                annot = annot.symmetrized()
            atom_array.set_annotation(cls.full_name, annot, n_body=2)
            categories = get_annotation_categories(atom_array, n_body=2)
            for alias in cls.aliases:
                if alias in categories:
                    atom_array.set_annotation(alias, annot, n_body=2)
        else:
            raise NotImplementedError("Currently only 0-body, 1-body and 2-body annotations are supported.")

    @classmethod
    def is_valid(cls, atom_array: AtomArray) -> bool:
        """
        Check if the annotation is consistent at the relevant level.
        (i.e. do all atoms/tokens/residues/chains/molecules/systems have the same value for the StandardAnnotation?)

        Returns:
            bool: True if valid, False otherwise.
        """
        if cls.n_body == 0 or cls.level == Level.ATOM:
            return True

        if cls.n_body == 1:
            is_same = lambda x: np.all(x == x[0]) if len(x) > 0 else True  # noqa: E731
            is_annotation_valid = cls.level.apply(atom_array, cls.annotation(atom_array), is_same)
            return np.all(is_annotation_valid)

        # TODO: Implement n-body aggregation
        return True


class AnnotationRegistryAccessor:
    """Base class for annotation registry accessor singletons.

    Provides dynamic attribute-based and subscript access to a registry dict,
    plus utilities for querying field names by body order.

    Subclasses must override :py:attr:`_registry` to return the appropriate
    :py:class:`StandardAnnotationMeta` registry dict.
    """

    _entry_kind: ClassVar[str] = "standard annotation"

    @property
    def _registry(self) -> dict[str, type[StandardAnnotationBase]]:
        raise NotImplementedError

    def __getattr__(self, name: str) -> type[StandardAnnotationBase]:
        """Dynamically retrieves an entry from the registry."""
        name = name.replace("_", "-")
        return self.__getitem__(name)

    def __getitem__(self, name: str) -> type[StandardAnnotationBase]:
        """Retrieves an entry by its registry name."""
        try:
            return self._registry[name]
        except KeyError:
            raise AttributeError(
                f"No {self._entry_kind} named '{name}' is registered. " f"Available {self._entry_kind}s: {self.list()}"
            ) from None

    def list(self) -> list[str]:
        """Returns a list of registry names of all registered entries."""
        return list(self._registry.keys())

    def __iter__(self):
        """Iterate over all registered classes."""
        for name in self.list():
            yield self[name]

    def get_valid_full_names(self) -> frozenset[str]:
        """Returns the set of full names of all registered entries."""
        return frozenset(cls.full_name for cls in self)

    def get(self, name: str) -> type[StandardAnnotationBase]:
        """Retrieve an entry by registry name, canonical field name, or alias."""
        if name in self._registry:
            return self._registry[name]
        return self.from_full_name(name)

    def from_full_name(self, full_name: str) -> type[StandardAnnotationBase]:
        """Resolve a canonical full name or accepted alias to a registered class."""
        for cls in self:
            if full_name in (cls.full_name, *cls.aliases):
                return cls
        raise KeyError(f"No {self._entry_kind} with full_name '{full_name}' found.")

    def get_field_names(self, n_body: int | Literal["all"] = "all", *, include_aliases: bool = False) -> list[str]:
        """Return persisted StandardAnnotation field names for the given body order.

        Args:
            n_body: Body order to filter by, or ``"all"`` for no filtering.
                Defaults to ``"all"``.
            include_aliases: Include accepted legacy read names.

        Returns:
            Deduplicated list of field names.
        """
        names: list[str] = []
        for cls in self:
            if n_body != "all" and cls.n_body != n_body:
                continue
            names.append(cls.full_name)
            if include_aliases:
                names.extend(cls.aliases)
        return list(dict.fromkeys(names))


class StandardAnnotationAccessor(AnnotationRegistryAccessor):
    """Provides dynamic, attribute-based access to all registered standard annotations.

    Iterates ``StandardAnnotationMeta._registry``, which also includes subclasses of
    :py:class:`StandardAnnotationBase`.
    """

    @property
    def _registry(self) -> dict[str, type[StandardAnnotationBase]]:
        return StandardAnnotationMeta._registry

    def __repr__(self) -> str:
        return f"StandardAnnotations({self.list()})"


STANDARD_ANNOTATIONS = StandardAnnotationAccessor()
