import ast
import operator
import re
from collections.abc import Callable
from functools import reduce
from itertools import product
from types import MappingProxyType
from typing import Any

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.common import not_isin
from atomworks.io.transforms.atom_array import is_any_coord_nan


class QueryExpression:
    """Query evaluator for biotite AtomArrays using pandas-like syntax.

    Examples:
        Select all CA atoms in chain A:
            >>> expr = QueryExpression("(chain_id == 'A') & (atom_name == 'CA')")
            >>> ca_atoms = expr.query(atom_array)

        Select atoms without NaN coordinates:
            >>> expr = QueryExpression("~has_nan_coord()")
            >>> valid_atoms = expr.query(atom_array)

        Select bonded atoms in specific residues:
            >>> expr = QueryExpression("has_bonds() & (res_name in ['ALA', 'GLY', 'VAL'])")

        Combine path-selection syntax with predicates via ``sel('...')``:
            >>> expr = QueryExpression("sel('[A,B]/ALA') & (x > 0)")

        Combine derived StandardAnnotation masks using their exported shortcuts:
            >>> expr = QueryExpression('sa_mask("C_CRD") & sa_mask("C_IDX")')

        Expand a residue-wise reduction back to atoms:
            >>> expr = QueryExpression("apply_and_spread('residue', atom_name == 'CA', 'any')")
    """

    # Functions that take one string-like argument, as opposed to the
    # zero-argument predicates below (``has_bonds()`` etc.).
    STRING_ARG_FUNCTIONS = frozenset({"sa_mask", "sel"})

    # Map string operators to functions
    OPS = MappingProxyType(
        {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            # Special handling for In/NotIn will be done in _eval_node
            ast.In: None,
            ast.NotIn: None,
            # Logical operators
            ast.And: np.logical_and,
            ast.Or: np.logical_or,
            ast.Not: np.logical_not,
            # Bitwise operators (which act as logical for boolean arrays)
            ast.BitAnd: np.bitwise_and,
            ast.BitOr: np.bitwise_or,
            ast.Invert: np.invert,
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }
    )
    REDUCTIONS = MappingProxyType(
        {
            "all": np.all,
            "any": np.any,
            "max": np.max,
            "mean": np.mean,
            "min": np.min,
            "sum": np.sum,
        }
    )

    def __init__(self, expr: str) -> None:
        """Initialize QueryExpression with a query string.

        Args:
            expr: The query expression string to parse and evaluate.
        """
        self.expr = expr
        # Parse once during initialization for efficiency
        self.tree = ast.parse(expr, mode="eval")

    def mask(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        """Apply the query expression to an AtomArray and return a boolean mask.

        Args:
            atom_array: The atom array to query.

        Returns:
            Boolean numpy array indicating which atoms match the query.
        """
        namespace = self._build_namespace(atom_array)
        functions = self._build_functions(atom_array)
        mask = self._eval_node(self.tree.body, namespace, functions, atom_array)

        # Ensure result is boolean array of correct length
        mask = self._ensure_bool_array(mask, atom_array.array_length())

        return mask

    def query(self, atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Apply the query expression to an AtomArray and return a filtered AtomArray.

        Args:
            atom_array: The atom array to query.

        Returns:
            Filtered atom array containing only atoms that match the query expression.
        """
        mask = self.mask(atom_array)
        return atom_array[mask]

    def idxs(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        """Apply the query expression to an AtomArray and return the indices of the matching atoms.

        Args:
            atom_array: The atom array to query.

        Returns:
            Numpy array of indices for atoms that match the query expression.
        """
        mask = self.mask(atom_array)
        return np.where(mask)[0]

    @staticmethod
    def _build_namespace(atom_array: AtomArray) -> dict[str, Any]:
        """Build namespace of queryable attributes.

        Args:
            atom_array: The atom array to build namespace from.

        Returns:
            Dictionary mapping attribute names to their values.
        """
        namespace = {}

        # Add all annotation arrays as queryable attributes
        for attr in atom_array.get_annotation_categories():
            namespace[attr] = getattr(atom_array, attr)

        # Add coordinate attributes
        if isinstance(atom_array, AtomArray):
            namespace["x"] = atom_array.coord[:, 0]
            namespace["y"] = atom_array.coord[:, 1]
            namespace["z"] = atom_array.coord[:, 2]

        return namespace

    @staticmethod
    def _build_functions(atom_array: AtomArray) -> dict[str, Callable]:
        """Build available functions that can be called in queries.

        Args:
            atom_array: The atom array to build functions for.

        Returns:
            Dictionary mapping function names to callable functions.
        """
        functions = {
            "apply_and_spread": lambda level, data, reduction: QueryExpression._apply_and_spread(
                atom_array, level, data, reduction
            ),
            "has_nan_coord": lambda: QueryExpression._has_nan_coord(atom_array),
            "has_bonds": lambda: QueryExpression._has_bonds(atom_array),
            "sa_mask": lambda name: QueryExpression._standard_annotation_mask(atom_array, name),
            # Bridge into the path-selection DSL: ``sel('A/ALA/1/[CA,CB]')`` returns a mask.
            # Non-raising on empty match so it composes inside boolean expressions.
            "sel": lambda selection_str: AtomSelectionStack.from_query(selection_str).get_mask(
                atom_array, raise_on_empty=False
            ),
        }
        return functions

    @staticmethod
    def _apply_and_spread(atom_array: AtomArray, level: str, data: Any, reduction: str) -> np.ndarray:
        """Reduce query data at a structural level and spread it back to atoms."""
        from atomworks.io.utils.standard_annotations.base import Level

        if not isinstance(level, str):
            raise TypeError("apply_and_spread() level must be a string")
        if not isinstance(reduction, str):
            raise TypeError("apply_and_spread() reduction must be a string")
        if reduction not in QueryExpression.REDUCTIONS:
            raise ValueError(
                f"Unknown apply_and_spread() reduction '{reduction}'. "
                f"Expected one of {sorted(QueryExpression.REDUCTIONS)}."
            )
        data = np.asarray(data)
        if data.ndim == 0 or len(data) != atom_array.array_length():
            raise ValueError(
                "apply_and_spread() data must have one value per atom; "
                f"got shape {data.shape} for {atom_array.array_length()} atoms."
            )
        return Level(level).apply_and_spread(atom_array, data, QueryExpression.REDUCTIONS[reduction])

    @staticmethod
    def _standard_annotation_mask(atom_array: AtomArray, name: str) -> np.ndarray:
        """Derive a registered StandardAnnotation mask without storing it."""
        # Lazy import avoids a cycle through StandardAnnotation defaults that use query utilities.
        from atomworks.io.utils.standard_annotations import STANDARD_ANNOTATIONS

        sa_cls = STANDARD_ANNOTATIONS.get(name)
        if sa_cls.n_body != 1:
            raise ValueError(
                f"sa_mask('{name}') requires a one-body standard annotation, but "
                f"`{sa_cls.full_name}` is {sa_cls.n_body}-body and has no per-atom mask."
            )
        return sa_cls.mask(atom_array)

    @staticmethod
    def _has_nan_coord(atom_array: AtomArray) -> np.ndarray:
        """Check if atom has NaN coordinates.

        Args:
            atom_array: The atom array to check.

        Returns:
            Boolean numpy array indicating which atoms have NaN coordinates.
        """
        return is_any_coord_nan(atom_array)

    @staticmethod
    def _has_bonds(atom_array: AtomArray) -> np.ndarray:
        """Check if atom is involved in a bond.

        Args:
            atom_array: The atom array to check.

        Returns:
            Boolean numpy array indicating which atoms are involved in bonds.
        """
        if atom_array.bonds is None:
            return np.zeros(atom_array.array_length(), dtype=bool)
        _bonded_idxs = np.unique(atom_array.bonds.as_array()[:, :2])
        return np.isin(np.arange(atom_array.array_length()), _bonded_idxs)

    @staticmethod
    def _ensure_bool_array(mask: Any, expected_length: int) -> np.ndarray:
        """Ensure mask is a boolean numpy array of the correct length.

        Args:
            mask: The mask to ensure is a boolean array.
            expected_length: The expected length of the array.

        Returns:
            Boolean numpy array of the correct length.

        Raises:
            ValueError: If the mask length doesn't match the expected length.
        """
        # Convert to numpy array if needed
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, dtype=bool)

        # Handle scalar boolean result
        if mask.shape == () or mask.ndim == 0:
            mask = np.full(expected_length, bool(mask), dtype=bool)

        # Ensure boolean dtype
        if mask.dtype != bool:
            mask = mask.astype(bool)

        # Check length
        if len(mask) != expected_length:
            raise ValueError(
                f"Query resulted in mask of length {len(mask)}, " f"but AtomArray has length {expected_length}"
            )

        return mask

    def _handle_in_operator(self, left: Any, right: Any, invert: bool = False) -> np.ndarray:
        """Handle 'in' and 'not in' operators with numpy arrays.

        Args:
            left: Left operand of the in/not in operation.
            right: Right operand of the in/not in operation.
            invert: Whether to invert the result (for 'not in').

        Returns:
            Boolean numpy array result of the in/not in operation.

        Raises:
            TypeError: If the right operand is not iterable.
        """
        # Convert right to list/array if needed
        if isinstance(right, (list | tuple | np.ndarray)):
            # Use numpy's isin for array operations
            if isinstance(left, np.ndarray):
                return not_isin(left, right) if invert else np.isin(left, right)
            else:
                # Single value
                return (left not in right) if invert else (left in right)
        else:
            raise TypeError(f"Argument of type '{type(right)}' is not iterable")

    def _eval_node(
        self, node: ast.AST, namespace: dict[str, Any], functions: dict[str, Callable], atom_array: AtomArray
    ) -> Any:
        """Recursively evaluate an AST node.

        Args:
            node: The AST node to evaluate.
            namespace: Dictionary of available variables.
            functions: Dictionary of available functions.
            atom_array: The atom array being queried.

        Returns:
            The result of evaluating the AST node.

        Raises:
            ValueError: If an unsupported operation or node type is encountered.
            NameError: If a name or function is not defined.
        """
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, namespace, functions, atom_array)
            results = []

            for op, comparator in zip(node.ops, node.comparators, strict=False):
                right = self._eval_node(comparator, namespace, functions, atom_array)

                # Special handling for In/NotIn operators
                if isinstance(op, ast.In):
                    results.append(self._handle_in_operator(left, right, invert=False))
                elif isinstance(op, ast.NotIn):
                    results.append(self._handle_in_operator(left, right, invert=True))
                else:
                    op_func = self.OPS[type(op)]
                    results.append(op_func(left, right))

                left = right

            # Chain multiple comparisons with AND
            if len(results) > 1:
                result = results[0]
                for r in results[1:]:
                    result = np.logical_and(result, r)
                return result
            else:
                return results[0]

        elif isinstance(node, ast.BoolOp):
            op_func = self.OPS[type(node.op)]
            values = [self._eval_node(value, namespace, functions, atom_array) for value in node.values]

            # Ensure all values are boolean arrays of correct length
            values = [self._ensure_bool_array(v, atom_array.array_length()) for v in values]

            # Use numpy operations for boolean arrays
            result = values[0]
            for val in values[1:]:
                result = op_func(result, val)
            return result

        elif isinstance(node, ast.BinOp):
            # Handle bitwise operations (& and |)
            if type(node.op) in [ast.BitAnd, ast.BitOr]:
                left = self._eval_node(node.left, namespace, functions, atom_array)
                right = self._eval_node(node.right, namespace, functions, atom_array)

                # Ensure boolean arrays
                left = self._ensure_bool_array(left, atom_array.array_length())
                right = self._ensure_bool_array(right, atom_array.array_length())

                op_func = self.OPS[type(node.op)]
                return op_func(left, right)
            else:
                raise ValueError(f"Unsupported binary operation: {type(node.op)}")

        elif isinstance(node, ast.UnaryOp):
            op_func = self.OPS[type(node.op)]
            operand = self._eval_node(node.operand, namespace, functions, atom_array)

            # Ensure boolean array for logical operations
            if type(node.op) in [ast.Not, ast.Invert]:
                operand = self._ensure_bool_array(operand, atom_array.array_length())

            return op_func(operand)

        elif isinstance(node, ast.Call):
            # Handle function calls
            if not isinstance(node.func, ast.Name):
                raise ValueError("Complex function calls not supported")
            func_name = node.func.id
            if func_name not in functions:
                raise NameError(f"Function '{func_name}' is not defined")

            if func_name == "apply_and_spread":
                if len(node.args) != 3 or node.keywords:
                    raise ValueError("Function 'apply_and_spread' expects level, data, and reduction arguments")
                level = self._eval_node(node.args[0], namespace, functions, atom_array)
                data = self._eval_node(node.args[1], namespace, functions, atom_array)
                reduction = self._eval_node(node.args[2], namespace, functions, atom_array)
                return functions[func_name](level, data, reduction)
            elif func_name in self.STRING_ARG_FUNCTIONS:
                if len(node.args) != 1 or node.keywords:
                    raise ValueError(f"Function '{func_name}' expects a single argument")
                argument = self._eval_node(node.args[0], namespace, functions, atom_array)
                if not isinstance(argument, str):
                    raise TypeError(f"Function '{func_name}' argument must evaluate to a string")
                result = functions[func_name](argument)
            else:
                # Zero-argument predicates (has_bonds(), has_nan_coord()).
                if node.args or node.keywords:
                    raise ValueError(f"Function '{func_name}' does not accept arguments")
                result = functions[func_name]()
            # Ensure it returns a boolean array of correct length
            return self._ensure_bool_array(result, atom_array.array_length())

        elif isinstance(node, ast.Name):
            if node.id in namespace:
                return namespace[node.id]
            raise NameError(f"Name '{node.id}' is not defined")

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.List):
            return [self._eval_node(elt, namespace, functions, atom_array) for elt in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt, namespace, functions, atom_array) for elt in node.elts)

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def __str__(self):
        return self.expr

    def __repr__(self):
        return f"QueryExpression('{self.expr}')"


def query(atom_array: AtomArray | AtomArrayStack, expr: str) -> AtomArray | AtomArrayStack:
    """
    Query the AtomArray using pandas-like syntax.
    Args:
        atom_array: The atom array to query.
        expr: Query expression in pandas-like syntax.

    Returns:
        Filtered atom array containing only atoms that match the query expression.

    Examples
    --------
    >>> # Select all CA atoms in chain A
    >>> ca_atoms = query(atom_array, "(chain_id == 'A') & (atom_name == 'CA')")

    >>> # Select atoms without NaN coordinates
    >>> valid_atoms = query(atom_array, "~has_nan_coord()")

    >>> # Select bonded atoms in specific residues
    >>> bonded = query(atom_array, "has_bonds() & (res_name in ['ALA', 'GLY', 'VAL'])")
    """
    querier = QueryExpression(expr)
    return querier.query(atom_array)


def mask(atom_array: AtomArray | AtomArrayStack, expr: str) -> np.ndarray:
    """
    Query the AtomArray using pandas-like syntax and return a boolean mask.
    """
    querier = QueryExpression(expr)
    return querier.mask(atom_array)


def idxs(atom_array: AtomArray | AtomArrayStack, expr: str) -> np.ndarray:
    """
    Query the AtomArray using pandas-like syntax and return the indices of the matching atoms.
    """
    querier = QueryExpression(expr)
    return querier.idxs(atom_array)


# ---------------------------------------------------------------------------
# Path-selection DSL
#
# A terse, path-like selection language over five ordered fields:
#     CHAIN_ID / RES_NAME / RES_ID / ATOM_NAME / TRANSFORMATION_ID
#
# It complements the :py:class:`QueryExpression` language above and is bridged
# into it via the ``sel('...')`` function, e.g. ``arr.query("sel('[A,B]/ALA') & (x > 0)")``.
# ---------------------------------------------------------------------------


class AtomSelection:
    """A single-valued selection of atoms in a molecular structure.

    A selection is specified by ``chain_id``, ``res_name``, ``res_id``, ``atom_name``,
    and (optionally) ``transformation_id``. Each field is either an exact value or the
    wildcard ``"*"`` (match anything). Fields are combined with logical AND.

    For example:
        - specifying only ``chain_id`` selects all atoms in that chain
        - specifying ``chain_id`` and ``res_name`` selects all atoms of that residue type in that chain
        - specifying only ``atom_name`` selects all atoms with that name, in any chain/residue

    For multi-valued selections (lists, ranges, unions), use :py:class:`AtomSelectionStack`.
    """

    def __init__(
        self,
        chain_id: str = "*",
        res_name: str = "*",
        res_id: int | str = "*",
        atom_name: str = "*",
        transformation_id: int | str = "*",
    ):
        self.chain_id = chain_id
        self.res_name = res_name
        self.atom_name = atom_name
        self.res_id = int(res_id) if res_id != "*" else res_id
        self.transformation_id = str(transformation_id)

    def __str__(self) -> str:
        parts = [self.chain_id, self.res_name, str(self.res_id), self.atom_name, str(self.transformation_id)]

        # Remove trailing '*' values
        while parts and parts[-1] == "*":
            parts.pop()

        return "/".join(parts)

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            # Convert the string to an AtomSelection for comparison
            other = self.from_selection_str(other)

        if not isinstance(other, AtomSelection):
            return False

        return (
            self.chain_id == other.chain_id
            and self.res_name == other.res_name
            and self.res_id == other.res_id
            and self.atom_name == other.atom_name
            and self.transformation_id == other.transformation_id
        )

    @classmethod
    def from_selection_str(cls, selection_string: str) -> "AtomSelection":
        """Create a selection from ``CHAIN/RES_NAME/RES_ID/ATOM/TRANSFORM`` syntax.

        ``"*"`` acts as a wildcard for any field; trailing fields may be omitted and default to ``"*"``.
        Bracket-list syntax (``[...]``) is not accepted here - use :py:meth:`AtomSelectionStack.from_query`
        or the query ``sel('...')`` function for multi-valued selections.

        Examples:
            >>> AtomSelection.from_selection_str("A/ALA/1/CA")
            A/ALA/1/CA
            >>> AtomSelection.from_selection_str("*/ALA/*/CB")
            */ALA/*/CB
        """
        selection = parse_selection_string(selection_string)
        return cls(
            chain_id=selection.chain_id,
            res_name=selection.res_name,
            res_id=selection.res_id,
            atom_name=selection.atom_name,
            transformation_id=selection.transformation_id,
        )

    @classmethod
    def from_pymol_str(cls, pymol_string: str) -> "AtomSelection":
        """Create a selection from a PyMOL atom label ``CHAIN/RES_NAME`RES_ID/ATOM``.

        Such strings are produced by clicking an atom/residue in PyMOL, e.g. ``"A/ASP`37/OD2"``.
        ``"*"`` may be used as a wildcard. ``transformation_id`` is not supported by PyMOL strings.
        """
        selection = parse_pymol_string(pymol_string)
        return cls(
            chain_id=selection.chain_id,
            res_name=selection.res_name,
            res_id=selection.res_id,
            atom_name=selection.atom_name,
        )

    def get_mask(self, atom_array: AtomArray, raise_on_empty: bool = True) -> np.ndarray:
        """Create a boolean mask for this selection on ``atom_array``.

        Args:
          raise_on_empty: If ``True`` (default), raise :py:class:`ValueError` when the selection
            matches no atoms. Set ``False`` for union/composition contexts (used by ``sel('...')``).
        """
        return get_mask_from_atom_selection(atom_array, self, raise_on_empty=raise_on_empty)

    def get_idxs(self, atom_array: AtomArray) -> np.ndarray:
        """Get the indices of atoms selected by this AtomSelection."""
        return np.where(self.get_mask(atom_array))[0]


def parse_selection_string(selection_string: str) -> AtomSelection:
    """Parse a ``CHAIN/RES_NAME/RES_ID/ATOM/TRANSFORM`` string into an :py:class:`AtomSelection`.

    ``"*"`` acts as a wildcard for any field. Trailing fields may be omitted and default to ``"*"``.

    Raises:
      ValueError: If bracket-list syntax ``[...]`` is used (multi-valued selections must go through
        :py:meth:`AtomSelectionStack.from_query`).
    """
    if "[" in selection_string or "]" in selection_string:
        raise ValueError(
            f"Bracket-list syntax '[...]' is not supported for single selections: {selection_string!r}. "
            "Use AtomSelectionStack.from_query(...) or the query sel('...') function instead."
        )
    granularity_tiers = ["chain_id", "res_name", "res_id", "atom_name", "transformation_id"]
    values = selection_string.split("/")

    # Create a dictionary with available tiers and values
    selection_dict = {tier: value for tier, value in zip(granularity_tiers, values, strict=False) if value != "*"}

    return AtomSelection(**selection_dict)


def parse_pymol_string(pymol_string: str) -> AtomSelection:
    """Parse a PyMOL ``CHAIN/RES_NAME`RES_ID/ATOM`` string into an :py:class:`AtomSelection`.

    Wildcards (``"*"``) are supported; ``transformation_id`` is not.
    """
    # Replace backtick with slash to standardize the format
    standardized_string = pymol_string.replace("`", "/")
    return parse_selection_string(standardized_string)


def get_mask_from_selection_string(atom_array: AtomArray, selection_string: str) -> np.ndarray:
    """Create a boolean mask from a ``CHAIN/RES_NAME/RES_ID/ATOM/TRANSFORM`` selection string.

    ``"*"`` acts as a wildcard for any field.
    """
    return get_mask_from_atom_selection(atom_array, parse_selection_string(selection_string))


def get_mask_from_atom_selection(
    atom_array: AtomArray, atom_selection: AtomSelection, raise_on_empty: bool = True
) -> np.ndarray:
    """Create a boolean mask from an :py:class:`AtomSelection`.

    Args:
      raise_on_empty: If ``True`` (default), raise :py:class:`ValueError` when no atoms match.
    """
    mask = np.ones(atom_array.array_length(), dtype=bool)

    # ``"*"`` is the wildcard; an empty/omitted string field is treated as wildcard too (this also
    # avoids touching a possibly-absent ``transformation_id`` annotation on a trailing-slash select).
    # ``res_id`` is the exception: it is an int and ``0`` is a real residue id, so it must compare
    # against the sentinel directly rather than via truthiness (``0`` is falsy).
    if atom_selection.chain_id and atom_selection.chain_id != "*":
        mask &= atom_array.chain_id == atom_selection.chain_id

    if atom_selection.res_name and atom_selection.res_name != "*":
        mask &= atom_array.res_name == atom_selection.res_name

    if atom_selection.res_id != "*":
        mask &= atom_array.res_id == atom_selection.res_id

    if atom_selection.atom_name and atom_selection.atom_name != "*":
        mask &= atom_array.atom_name == atom_selection.atom_name

    if atom_selection.transformation_id and atom_selection.transformation_id != "*":
        mask &= atom_array.transformation_id == atom_selection.transformation_id

    if raise_on_empty and not np.any(mask):
        raise ValueError(f"No atoms found for selection: {atom_selection}")

    return mask


class AtomSelectionStack:
    """A union (logical OR) of :py:class:`AtomSelection` objects.

    Enables a single string to select multiple atoms/segments via
    :py:meth:`from_query` (extended syntax with ``[...]`` lists and ranges) or
    :py:meth:`from_contig` (contiguous residue ranges).
    """

    def __init__(self, selections: list[AtomSelection]):
        self.selections = selections

    @classmethod
    def from_contig(cls, contig: str) -> "AtomSelectionStack":
        """Create a stack from contiguous residue ranges like ``"A1-2, B3-10"``."""
        # First define a regex that matches the elements of the contig string
        CONTIG_REGEX = re.compile(r"([A-Za-z]+)(\d+)-(\d+)")  # noqa
        selections = []
        for selection in contig.replace(" ", "").split(","):
            match = CONTIG_REGEX.match(selection)
            if not match:
                raise ValueError(f"Invalid contig string: {selection}")
            chain_id, start, stop = match.groups()
            # Create a new AtomSelection for each match
            for i in range(int(start), int(stop) + 1):
                # Create a new AtomSelection for each residue in the range
                atom_selection = AtomSelection(chain_id=chain_id, res_id=i)
                selections.append(atom_selection)
        return cls(selections)

    @classmethod
    def from_query(cls, query: str | list[str]) -> "AtomSelectionStack":
        """Create a stack from the extended path-selection syntax.

        Grammar (fields in order ``CHAIN/RES_NAME/RES_ID/ATOM/TRANSFORM``, trailing fields
        default to ``"*"``):

        - Each field is a scalar (``A``), the wildcard ``*``, or a bracket list ``[a, b, ...]``.
        - Inside a bracket list, each item is a scalar or - for ``res_id`` only - an inclusive
          range ``lo-hi`` (e.g. ``[1-5, 9, 12-14]``). Ranges must be bracketed; a bare ``5-10``
          is not a range.
        - Multiple whole tokens, separated by top-level commas (or given as a ``list[str]``),
          are unioned. Commas inside ``[...]`` are part of the list, not token separators.

        Examples:
            >>> AtomSelectionStack.from_query("[A,B]/ALA/1/[CA,CB]")  # 4 selections, unioned
            >>> AtomSelectionStack.from_query("A/*/[5-10]")  # residues 5..10 in chain A
            >>> AtomSelectionStack.from_query("A/*/[5-10], B/*/[3-8]")  # different range per chain
        """
        tokens = cls._parse_query_tokens(query)
        selections: list[AtomSelection] = []
        for token in tokens:
            field_values = cls._parse_token_fields(token)
            selections.extend(cls._build_selections_from_fields(field_values))
        return cls(selections)

    @staticmethod
    def _split_top_level(string: str, sep: str) -> list[str]:
        """Split ``string`` on ``sep`` only at bracket depth 0 (commas inside ``[...]`` are kept)."""
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        for ch in string:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            if ch == sep and depth == 0:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        parts.append("".join(buf))
        return parts

    @classmethod
    def _parse_query_tokens(cls, query: str | list[str]) -> list[str]:
        """Split query input into individual (bracket-aware) union tokens."""
        items = [query] if isinstance(query, str) else list(query)
        raw = [tok for item in items for tok in cls._split_top_level(item, ",")]
        return [tok.strip() for tok in raw if tok.strip()]

    @classmethod
    def _parse_token_fields(cls, token: str) -> dict[str, list[Any]]:
        """Parse a single token into per-field option lists."""
        parts = token.split("/")
        while len(parts) < 5:
            parts.append("*")
        chain_val, res_name_val, res_id_val, atom_name_val, trans_id_val = parts[:5]
        return {
            "chain_id": cls._parse_field_value(chain_val, is_res_id=False),
            "res_name": cls._parse_field_value(res_name_val, is_res_id=False),
            "res_id": cls._parse_field_value(res_id_val, is_res_id=True),
            "atom_name": cls._parse_field_value(atom_name_val, is_res_id=False),
            "transformation_id": cls._parse_field_value(trans_id_val, is_res_id=False),
        }

    @classmethod
    def _parse_field_value(cls, value: str, *, is_res_id: bool = False) -> list[Any]:
        """Parse a single field into a list of options (expanding ``[...]`` lists and ranges)."""
        v = value.strip()
        if v in ("*", ""):
            return ["*"]
        if "[" in v or "]" in v:
            if not (v.startswith("[") and v.endswith("]")):
                raise ValueError(f"Malformed bracket list in selection field: {v!r}")
            items = [item.strip() for item in v[1:-1].split(",") if item.strip()]
            if not items:
                raise ValueError(f"Empty bracket list in selection field: {v!r}")
            options: list[Any] = []
            for item in items:
                options.extend(cls._expand_item(item, is_res_id=is_res_id))
            return options
        # Bare value: scalar only (ranges must be wrapped in brackets)
        return cls._scalar_option(v, is_res_id=is_res_id)

    @classmethod
    def _expand_item(cls, item: str, *, is_res_id: bool = False) -> list[Any]:
        """Expand a single bracket-list item (scalar, wildcard, or res_id range) into options."""
        if item == "*":
            return ["*"]
        if is_res_id:
            match = re.fullmatch(r"(-?\d+)-(-?\d+)", item)
            if match:
                start_i, stop_i = int(match.group(1)), int(match.group(2))
                step = 1 if start_i <= stop_i else -1
                return list(range(start_i, stop_i + step, step))
        return cls._scalar_option(item, is_res_id=is_res_id)

    @staticmethod
    def _scalar_option(value: str, *, is_res_id: bool = False) -> list[Any]:
        """Coerce a single scalar field value into a one-element option list."""
        if is_res_id:
            try:
                return [int(value)]
            except ValueError:
                raise ValueError(
                    f"Invalid res_id {value!r}: ranges must be wrapped in brackets, e.g. '[5-10]'."
                ) from None
        return [value]

    @classmethod
    def _build_selections_from_fields(cls, field_values: dict[str, list[Any]]) -> list[AtomSelection]:
        """Build the Cartesian product of per-field options into individual selections."""
        return [
            AtomSelection(chain_id=c, res_name=r, res_id=i, atom_name=a, transformation_id=t)
            for c, r, i, a, t in product(
                field_values["chain_id"],
                field_values["res_name"],
                field_values["res_id"],
                field_values["atom_name"],
                field_values["transformation_id"],
            )
        ]

    def get_mask(self, atom_array: AtomArray | AtomArrayStack, raise_on_empty: bool = True) -> np.ndarray:
        """Create a boolean mask by unioning (logical OR) all member selections.

        Args:
          raise_on_empty: Passed through to each member selection. If ``True`` (default),
            a member matching no atoms raises :py:class:`ValueError`. ``sel('...')`` sets this
            to ``False`` so unions degrade gracefully.
        """
        if not self.selections:
            return np.zeros(atom_array.array_length(), dtype=bool)
        masks = [selection.get_mask(atom_array, raise_on_empty=raise_on_empty) for selection in self.selections]
        return reduce(np.logical_or, masks)

    def get_center_of_mass(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        """Return the center of mass of the selected atoms."""
        mask = self.get_mask(atom_array)
        if not np.any(mask):
            raise ValueError("No atoms selected by the AtomSelectionStack.")

        if isinstance(atom_array, AtomArray):
            return atom_array.coord[mask].mean(axis=0)
        elif isinstance(atom_array, AtomArrayStack):
            return atom_array.coord[:, mask].mean(axis=1)
        else:
            raise ValueError(f"Cannot get center of mass for {type(atom_array)}!")

    def get_principal_components(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        """Return principal axes (eigenvectors) of the selected atoms via SVD.

        Returns:
          ``(3, 3)`` array for :py:class:`~biotite.structure.AtomArray`.
          ``(n_models, 3, 3)`` array for :py:class:`~biotite.structure.AtomArrayStack`.
        """
        mask = self.get_mask(atom_array)
        if not np.any(mask):
            raise ValueError("No atoms selected by the AtomSelectionStack.")

        if isinstance(atom_array, AtomArray):
            coords = atom_array.coord[mask]  # (N_atoms, 3)
            coords_centered = coords - coords.mean(axis=0)
            # SVD for principal axes
            _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
            return vh.T  # (3, 3), columns are principal axes
        elif isinstance(atom_array, AtomArrayStack):
            coords = atom_array.coord[:, mask, :]  # (n_models, N_atoms, 3)
            pcs = []
            for model_coords in coords:
                model_centered = model_coords - model_coords.mean(axis=0)
                _, _, vh = np.linalg.svd(model_centered, full_matrices=False)
                pcs.append(vh.T)  # (3, 3)
            return np.stack(pcs, axis=0)  # (n_models, 3, 3)
        else:
            raise ValueError(f"Cannot get principal components for {type(atom_array)}!")
