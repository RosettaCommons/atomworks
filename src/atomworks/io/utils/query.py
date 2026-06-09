import ast
import operator
from collections.abc import Callable
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
    """

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
            "has_nan_coord": lambda: QueryExpression._has_nan_coord(atom_array),
            "has_bonds": lambda: QueryExpression._has_bonds(atom_array),
        }
        return functions

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
        if not hasattr(atom_array, "bonds"):
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
                f"Query resulted in mask of length {len(mask)}, but AtomArray has length {expected_length}"
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
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in functions:
                    # Call the function (no arguments supported for now)
                    if node.args or node.keywords:
                        raise ValueError(f"Function '{func_name}' does not accept arguments")
                    result = functions[func_name]()
                    # Ensure it returns a boolean array of correct length
                    return self._ensure_bool_array(result, atom_array.array_length())
                else:
                    raise NameError(f"Function '{func_name}' is not defined")
            else:
                raise ValueError("Complex function calls not supported")

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
