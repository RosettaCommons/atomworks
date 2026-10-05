"""Internal coordinates in the Rosetta convention, and TMol's names for hydrogens no dictionary describes.

An atom is placed against three placed ancestors (parent, grandparent,
great-grandparent) by ``d``, the distance to its parent, ``theta``, ``180 -
angle(atom, parent, grandparent)``, and ``phi``, the IUPAC dihedral
``atom-parent-grandparent-great_grandparent`` (:func:`biotite.structure.dihedral`). TMol uses the same
convention, so a structure moves between the two libraries without its atoms shifting.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = ["build_coordinates", "names_from_parent"]


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 1e-8, v / norm, np.nan)


def _perpendicular(v: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each row of *v*: its cross product with the axis it has least of."""
    axis = np.zeros(v.shape)
    axis[np.arange(len(v)), np.argmin(np.abs(np.nan_to_num(v, nan=1.0)), axis=-1)] = 1.0
    return _unit(np.cross(v, axis))


def _frame_axes(
    parent: np.ndarray, grandparent: np.ndarray, great_grandparent: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The axes an atom is placed on: x from grandparent to parent, y towards the great-grandparent."""
    parent, grandparent, great_grandparent = np.broadcast_arrays(parent, grandparent, great_grandparent)
    x_axis = _unit(parent - grandparent)
    second = great_grandparent - grandparent
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    # Normalised first, so near-collinearity is judged by an angle rather than by a distance.
    with np.errstate(invalid="ignore", divide="ignore"):
        y_axis = np.where(norm > 1e-8, second / norm, 0.0)
    z_axis = np.cross(x_axis, y_axis)
    collinear = np.linalg.norm(z_axis, axis=-1) < 1e-8
    if collinear.any():
        z_axis[collinear] = _perpendicular(x_axis[collinear])
    z_axis = _unit(z_axis)
    return x_axis, np.cross(z_axis, x_axis), z_axis


def build_coordinates(
    parent: np.ndarray, grandparent: np.ndarray, great_grandparent: np.ndarray, geometry: np.ndarray
) -> np.ndarray:
    """Place atoms at ``geometry = (d, theta, phi)`` against their ancestors, over any leading dims.

    Collinear ancestors leave the dihedral free; a parent on its grandparent gives NaN.

    Args:
        parent: ``[..., 3]`` parent positions.
        grandparent: ``[..., 3]``.
        great_grandparent: ``[..., 3]``.
        geometry: ``[..., 3]`` ``(d, theta, phi)``, radians.

    Returns:
        ``[..., 3]`` positions.
    """
    x_axis, y_axis, z_axis = _frame_axes(parent, grandparent, great_grandparent)
    d, theta, phi = np.moveaxis(np.asarray(geometry, dtype=np.float64), -1, 0)
    offset = np.stack([d * np.cos(theta), d * np.sin(theta) * np.cos(phi), d * np.sin(theta) * np.sin(phi)], -1)
    return parent + x_axis * offset[..., :1] + y_axis * offset[..., 1:2] + z_axis * offset[..., 2:]


def _measure_icoors(
    atom: np.ndarray, parent: np.ndarray, grandparent: np.ndarray, great_grandparent: np.ndarray
) -> np.ndarray:
    """``(d, theta, phi)`` of each atom on the axes :func:`build_coordinates` places it on, its inverse.

    A great-grandparent without a position leaves ``phi`` NaN rather than measured on a stand-in axis.
    """
    offset, axes = atom - parent, _frame_axes(parent, grandparent, great_grandparent)
    x, y, z = (np.einsum("...i,...i->...", offset, axis) for axis in axes)
    phi = np.where(np.isfinite(great_grandparent).all(axis=-1), np.arctan2(z, y), np.nan)
    return np.stack([np.linalg.norm(offset, axis=-1), np.arctan2(np.hypot(y, z), x), phi], axis=-1)


def _numbered(prefix: str, count: int, taken: set[str]) -> list[str]:
    """The first *count* names ``<prefix><n>``, ``n`` from 1, that are not *taken*."""
    candidates = (f"{prefix}{n}" for n in itertools.count(1))
    return list(itertools.islice((name for name in candidates if name not in taken), count))


def names_from_parent(heavy_name: str, element: str, count: int, taken: set[str]) -> list[str]:
    """TMol's names for hydrogens of a component no dictionary describes.

    One H on X is ``HX``, several ``HX1``, ``HX2``. Past four characters the parent's
    element is dropped (``H101`` on C10); still too long or taken, ``H<element><n>``.
    """
    stems = [heavy_name]
    if heavy_name.upper().startswith(element.upper()) and len(heavy_name) > len(element):
        stems.append(heavy_name[len(element) :])
    for stem in stems:
        names = [f"H{stem}"] if count == 1 else [f"H{stem}{i}" for i in range(1, count + 1)]
        if all(len(name) <= 4 and name not in taken for name in names):
            return names
    return _numbered(f"H{element}", count, taken)
