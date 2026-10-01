import logging
from itertools import pairwise
from typing import Literal

import biotite.structure as struc
import numpy as np

from atomworks.io.utils.standard_annotations import S_ATM, S_SEGMAX, S_SEGMIN
from atomworks.io.utils.standard_annotations.base import Level
from atomworks.io.utils.visualize import get_pymol_session, view_pymol
from atomworks.ml.conditions import C_DIS, Condition

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# Conditions opted out of automatic mask-display selection
# generation in :py:func:`get_atom_array_style_cmd`. Extensions (e.g.
# proteinfoundation) register entries via :py:func:`hide_condition_display`
_HIDE_MASKS_FOR_CONDITIONS: set[type] = set()


def hide_condition_display(cond) -> None:  # noqa: ANN001 - cond is a condition class
    """Opt *cond* out of automatic mask-display selection generation.

    Args:
        cond: A condition class (e.g. ``C_IDX``) whose display selection should
            not be auto-created by :py:func:`get_atom_array_style_cmd`.
    """
    _HIDE_MASKS_FOR_CONDITIONS.add(cond)


def show_condition_display(cond) -> None:  # noqa: ANN001 - cond is a condition class
    """Re-enable automatic mask-display for *cond* (undo :py:func:`hide_condition_display`)."""
    _HIDE_MASKS_FOR_CONDITIONS.discard(cond)


# Marks atoms whose true coordinates are NaN (unresolved / filled-missing atoms).
_DISPLAY_PLACEHOLDER = "_display_placeholder"


def display_atom_ids(array: struc.AtomArray) -> np.ndarray:
    """Return the PyMOL atom ids for *array* — the ``atom_id`` annotation if present, else 1-indexed."""
    if "atom_id" in array.get_annotation_categories():
        return array.get_annotation("atom_id")
    return np.arange(1, array.array_length() + 1)


def _fill_nan_with_residue_centroid(array: struc.AtomArray, nan_mask: np.ndarray) -> None:
    """Move each NaN-coord atom to its residue's resolved-atom centroid, in place.

    Fully-unresolved residues (no resolved atom to average) fall back to the origin.

    Args:
        array: Array to modify (``array.coord`` is mutated).
        nan_mask: Boolean mask of the atoms with NaN coordinates.
    """
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    for s, e in pairwise(starts):
        res_nan = nan_mask[s:e]
        if not res_nan.any():
            continue
        centroid = array.coord[s:e][~res_nan].mean(axis=0) if not res_nan.all() else np.zeros(3)
        array.coord[np.arange(s, e)[res_nan]] = centroid


def prepare_for_display(
    array: struc.AtomArray,
    nan_coord_policy: Literal["remove", "zero", "centroid"] | None = None,
) -> struc.AtomArray:
    """Return a display copy of the array with expandable segment sentinel coordinates filled in.

    Expandable segment sentinels carry NaN coordinates internally (correct for
    atomworks and file I/O), but PyMOL cannot render NaN positions.  This function
    copies the array and assigns each sentinel a display position:

    - **In-chain sentinel**: midpoint of the nearest flanking CA atoms on either side.
      Falls back to whichever flanking atom is available.
    - **New-chain sentinel** (no flanking atoms): placed at the origin.

    Args:
        array: AtomArrayPlus to prepare.  Not modified.
        nan_coord_policy: How to handle atoms whose coords are still NaN after
            sentinel interpolation. ``"remove"`` drops them (PyMOL never sees
            them). ``"zero"`` sets their coords to the origin. ``"centroid"``
            keeps them (so they stay selectable/editable), moves each to its
            residue centroid, and marks them via :py:data:`_DISPLAY_PLACEHOLDER`
            so the viewer can hide them. ``None`` leaves them as NaN (caller is
            responsible for downstream handling).

    Returns:
        Copy of the array with sentinel coordinates set for display.
    """
    array = array.copy()  # never change the main array

    # always disambiguate chain_id using chain_iid
    if "chain_iid" in array.get_annotation_categories():
        array.chain_id = array.get_annotation("chain_iid").astype(str)
    else:
        raise ValueError("Expected chain_iid annotation for display disambiguation, but not found.")

    segment_mask = S_SEGMIN.mask(array, default="generate")
    for segment_idx in np.where(segment_mask)[0]:
        prev_coord = next_coord = None
        for j in range(int(segment_idx) - 1, -1, -1):
            if not segment_mask[j] and array.atom_name[j] == "CA" and not np.any(np.isnan(array.coord[j])):
                prev_coord = array.coord[j]
                break
        for j in range(int(segment_idx) + 1, len(array)):
            if not segment_mask[j] and array.atom_name[j] == "CA" and not np.any(np.isnan(array.coord[j])):
                next_coord = array.coord[j]
                break

        if prev_coord is not None and next_coord is not None:
            array.coord[segment_idx] = (prev_coord + next_coord) / 2.0
        elif prev_coord is not None:
            array.coord[segment_idx] = prev_coord
        elif next_coord is not None:
            array.coord[segment_idx] = next_coord
        else:
            array.coord[segment_idx] = np.zeros(3)

    if nan_coord_policy is not None:
        nan_mask = np.any(np.isnan(array.coord), axis=1)
        if nan_coord_policy == "centroid":
            if nan_mask.any():
                _fill_nan_with_residue_centroid(array, nan_mask)
            prior = (
                array.get_annotation(_DISPLAY_PLACEHOLDER).astype(bool)
                if _DISPLAY_PLACEHOLDER in array.get_annotation_categories()
                else np.zeros(array.array_length(), dtype=bool)
            )
            array.set_annotation(_DISPLAY_PLACEHOLDER, prior | nan_mask)
        elif nan_mask.any():
            if nan_coord_policy == "remove":
                array = array[~nan_mask]
            elif nan_coord_policy == "zero":
                array.coord[nan_mask] = 0.0

    return array


def count_hidden_nan_atoms(array: struc.AtomArray) -> int:
    """Return the number of atoms :py:func:`prepare_for_display` would drop.

    Sentinels (``S_SEGMIN``-masked atoms) are excluded because they receive
    interpolated display coordinates rather than being hidden.
    """
    nan_mask = np.any(np.isnan(array.coord), axis=1)
    if S_SEGMIN.has_annotation(array):
        nan_mask = nan_mask & ~S_SEGMIN.mask(array)
    return int(nan_mask.sum())


def get_atom_array_style_cmd(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
    obj: str,
    label: bool = False,
    max_distances: int = 100,
    grid_slot: int | None = None,
    nan_coord_policy: Literal["remove", "zero", "centroid"] = "remove",
) -> str:
    """Generate PyMOL commands to style an atom array visualization.

    Creates a series of PyMOL commands that style different parts of a molecular structure, including:
    - Applying a color spectrum to polymer chains
    - Showing backbone atoms as sticks and CA atoms as spheres
    - Styling non-polymer atoms with different colors and representation
    - Highlighting specially annotated atoms with different colors and visualizations

    Args:
        atom_array: The biotite AtomArray or AtomArrayStack to be styled
        obj: PyMOL object name to apply the styling to
        label: Whether to label all atoms with their 0-indexed atom_id
        max_distances: Maximum number of distance lines to show (pymol hangs when there's too many distance objects)

    Returns:
        str: A PyMOL command string that styles the atom array
    """
    # Work on a display copy: expandable segment sentinels get interpolated coordinates so
    # PyMOL can position them.  The original array is never modified.
    # NaN coords must match whatever policy was used when loading into PyMOL,
    # otherwise atom IDs between the commands below and the PyMOL object diverge.
    atom_array = prepare_for_display(atom_array, nan_coord_policy=nan_coord_policy)

    grid_slot = grid_slot or np.random.randint(0, 10_000)
    commands = [f"hide everything, {obj}"]
    annotations = atom_array.get_annotation_categories()

    # pymol 1-indexes atom ids unless an explicit atom_id annotation is present
    offset = 0 if "atom_id" in annotations else 1
    atom_ids = display_atom_ids(atom_array)

    # Style the backbone for each polymer chain with a color spectrum
    for chain_id in struc.get_chains(atom_array):
        if (~atom_array.hetero[atom_array.chain_id == chain_id]).any():
            commands.append(
                f"spectrum resi, RFd_darkblue RFd_blue RFd_lightblue RFd_purple RFd_pink RFd_melon RFd_navaho, "
                f"{obj} and chain {chain_id} and elem C"
            )

    # Add basic styling commands for protein backbone and non-polymer components
    commands.extend(
        [
            f"show sticks, model {obj} and name n+c+ca+cb",
            f"show spheres, model {obj} and name ca",
            f"set sphere_scale, 0.23, model {obj} and name ca",
            f"set sphere_transparency, 0, model {obj} and name ca",
            f"color grey60, model {obj} and not polymer and elem C",
            f"show nb_spheres, model {obj} and not polymer",
            f"show sticks, model {obj} and not polymer",
        ]
    )
    if label:
        # label 0-indexed for correspondence with biotite
        commands.append("label all, ID") if offset == 0 else commands.append(f"label all, ID-{offset}")

    # Style atoms marked for "atomize" if present.
    # Delete any prior selection first so it never lingers when atomize is absent.
    commands.append(f"delete {obj}_atomize")
    if S_ATM.has_annotation(atom_array) and S_ATM.mask(atom_array).any():
        atomize_ids = np.where(S_ATM.mask(atom_array))[0]
        atomize_ids = atom_ids[atomize_ids]
        commands.extend(
            [
                f"select {obj}_atomize, model {obj} and id {'+'.join(str(id) for id in atomize_ids)}",
                f"show sticks, {obj}_atomize and byres {obj}_atomize",
            ]
        )

    # Style constraints, if present:
    # 2-body:
    # ... add distance lines between atoms if specified in annotations
    if hasattr(atom_array, "_annot_2d"):
        distance_commands = []
        if C_DIS.full_name in atom_array._annot_2d:
            constraint_data = C_DIS.annotation(atom_array).as_array()
            if len(constraint_data) > 0:
                _atom_idxs = np.unique(constraint_data[:, :2].flatten()).astype(int)
                _atom_ids = atom_ids[_atom_idxs]
                _selection = f'{obj} and id {"+".join(str(id) for id in _atom_ids)}'
                commands.extend(
                    [
                        f"delete m2d_{obj}",
                        f"select m2d_{obj}, {_selection}",
                        f"show spheres, m2d_{obj}",
                        f"set sphere_color, lime, m2d_{obj}",
                        f"set sphere_scale, 0.25, m2d_{obj}",
                        f"set sphere_transparency, 0.5, m2d_{obj}",
                        f"show sticks, byres m2d_{obj}",
                    ]
                )

                if len(constraint_data) > max_distances:
                    logger.warning(f"Too many distance conditions ({len(constraint_data)}), sampling {max_distances}.")
                    constraint_idxs = np.arange(len(constraint_data))
                    constraint_idxs = np.random.choice(constraint_idxs, max_distances, replace=False)
                    constraint_data = constraint_data[constraint_idxs]

                for row in constraint_data:
                    idx_i, idx_j, value = row
                    if (idx_i > idx_j) or (value == 0):
                        continue

                    i, j = atom_ids[idx_i], atom_ids[idx_j]
                    # ... if we have a stack, we grab the last frame for the distance computation
                    if isinstance(atom_array, struc.AtomArrayStack):
                        distance = struc.distance(atom_array[0, idx_i], atom_array[0, idx_j])
                    else:
                        distance = struc.distance(atom_array[idx_i], atom_array[idx_j])

                    distance_name = f"d{idx_i}-{idx_j}_{value:.2f}_{distance:.2f}"
                    distance_commands.extend(
                        [
                            f"distance {distance_name}, model {obj} and id {i}, model {obj} and id {j}",
                            f"set grid_slot, {grid_slot}, {distance_name}",
                        ]
                    )

        commands.extend(distance_commands)

    # Expandable segment sentinel visualization — coordinates were filled in by prepare_for_display above.
    # Delete any prior selection first so it never lingers when no sentinels remain.
    commands.append(f"delete {obj}_segments")
    if S_SEGMIN.has_annotation(atom_array):
        segment_mask = S_SEGMIN.mask(atom_array)
        if segment_mask.any():
            segment_indices = np.where(segment_mask)[0]
            segment_atom_ids = atom_ids[segment_indices]
            segment_sel = f'model {obj} and id {"+".join(str(i) for i in segment_atom_ids)}'
            commands.extend(
                [
                    f"select {obj}_segments, {segment_sel}",
                    f"show spheres, {obj}_segments",
                    f"color yellow, {obj}_segments",
                    f"set sphere_scale, 0.6, {obj}_segments",
                    f"set sphere_transparency, 0.3, {obj}_segments",
                ]
            )
            # Label each sentinel using annotation values so realized stubs
            # (res_name == MASKED) also get the correct range shown.
            segmin_annot = S_SEGMIN.annotation(atom_array)
            segmax_annot = S_SEGMAX.annotation(atom_array)
            for segment_idx, atom_id in zip(segment_indices, segment_atom_ids, strict=False):
                segmin = int(segmin_annot[segment_idx])
                segmax = int(segmax_annot[segment_idx])
                commands.append(f'label model {obj} and id {atom_id}, "{segmin}-{segmax}"')

    # Handle 1-D conditions: surface a named selection for any condition whose
    # (possibly generated) default mask is non-empty. Conditions registered via
    # hide_condition_display() are skipped
    for cond in Condition:
        if cond.n_body != 1:
            continue
        sel_name = f"mask_{cond.name}_{cond.n_body}_{cond.level}_{obj}"
        # Always delete the prior selection first so nothing stale ever lingers —
        # every recompute reflects the current array exactly.
        commands.append(f"delete {sel_name}")
        if cond in _HIDE_MASKS_FOR_CONDITIONS:
            continue
        mask = cond.mask(atom_array, default="generate")
        if not mask.any():
            continue

        _atom_ids = atom_ids[np.where(mask)[0]]
        # enable=0: create the selection without activating it, so a condition
        # mask never masquerades as the live ``sele`` selection.
        if cond.level == Level.ATOM:
            _selection = f'model {obj} and id {"+".join(str(id) for id in _atom_ids)}'
            commands.append(f"select {sel_name}, {_selection}, enable=0")
        elif cond.level == Level.RESIDUE or cond.level == Level.TOKEN:
            _selection = f'model {obj} and byres (id {"+".join(str(id) for id in _atom_ids)})'
            commands.append(f"select {sel_name}, {_selection}, enable=0")

    return "\n".join(commands)


def viz(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
    id: str = "obj",
    clear: bool = False,
    label: bool = False,
    max_distances: int = 100,
    view_ori_token: bool = False,
    stylize: bool = False,
) -> None:
    """Quickly visualize a molecular structure in PyMOL with predefined styling.

    This function creates a PyMOL session, loads the atom array structure, and applies
    a set of styling commands to make the visualization informative and aesthetically pleasing.
    The styling highlights different structural components and annotated features.

    Args:
        atom_array: The biotite AtomArray or AtomArrayStack to visualize
        id: PyMOL object identifier (default: "obj")
        clear: Whether to clear existing PyMOL objects before visualization (default: True)
        label: Whether to label all atoms with their 0-indexed atom_id (default: True)
        view_ori_token: Whether to view the ori token (default: False)
        stylize: Whether to apply styling as defined in `get_atom_array_style_cmd` (default: False)

    Example:
        >>> from biotite.structure import AtomArray
        >>> # Create or load an atom array
        >>> structure = AtomArray(...)
        >>> # Visualize the structure in PyMOL
        >>> viz(structure)
    """
    atom_array = atom_array.copy()
    # pymol only considers chain_id annotation, which can make weird looking artifacts if we have two different chains with the same chain_id
    # We always disambiguate by using the chain_iid annotation, so we need to have pymol use that to do the same
    if "chain_iid" in atom_array.get_annotation_categories():
        atom_array.chain_id = atom_array.get_annotation("chain_iid")
        atom_array.chain_id = atom_array.chain_id.astype(str)

    pymol = get_pymol_session()
    pymol.do("set valence, 1; set connect_mode, 2;")
    if clear:
        pymol.do("delete d*")
        pymol.delete("all")
    slot = np.random.randint(0, 10_000)
    obj_name = view_pymol(atom_array, id=id, grid_slot=slot)
    if stylize:
        cmd = get_atom_array_style_cmd(atom_array, obj_name, label=label, grid_slot=slot, max_distances=max_distances)
        pymol.do(cmd)

    if view_ori_token:
        pymol.do(f"pseudoatom ori_{obj_name}, pos=[0,0,0]")
        pymol.do(
            [
                f"set grid_slot, {slot}, ori_{obj_name}",
                f"show spheres, ori_{obj_name}",
                f"set sphere_color, white, ori_{obj_name}",
                f"set sphere_scale, 0.5, ori_{obj_name}",
                f"set sphere_transparency, 0.5, ori_{obj_name}",
            ]
        )
