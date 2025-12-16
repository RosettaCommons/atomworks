"""
Flexible masking utilities for sampling masks over atom arrays.

This module provides utilities for creating masks over atom arrays using a flexible
seed-grow-merge pattern. This is useful for tasks like creating island masks
or bond-graph based masks.
"""

import abc
import copy
import logging
from collections.abc import Callable

import biotite.structure as struc
import networkx as nx
import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import KDTree

from atomworks.common import exists
from atomworks.constants import STANDARD_AA_TIP_ATOM_NAMES
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array import apply_and_spread
from atomworks.io.utils.bonds import _atom_array_to_networkx_graph
from atomworks.io.utils.query import QueryExpression
from atomworks.io.utils.selection import annot_start_stop_idxs
from atomworks.ml.utils.token import get_token_starts

logger = logging.getLogger(__name__)


# ==== Seed sampling functions ====
class SampleSeed(abc.ABC):
    def __init__(
        self,
        is_eligible: str | np.ndarray | None = None,
        avoid_same: tuple[str, ...] | None = None,
        rng: np.random.Generator | None = None,
    ):
        if not exists(is_eligible):
            self.is_eligible = None
        elif isinstance(is_eligible, str):
            self.is_eligible = QueryExpression(is_eligible)
        else:
            self.is_eligible = is_eligible
        self.avoid_same = (avoid_same,) if isinstance(avoid_same, str) else avoid_same
        self.rng = rng

    def _get_is_eligible(self, atom_array: AtomArray) -> np.ndarray:
        if not exists(self.is_eligible):
            is_eligible = np.ones(atom_array.array_length(), dtype=bool)
        elif isinstance(self.is_eligible, QueryExpression):
            is_eligible = self.is_eligible.mask(atom_array)
        else:
            is_eligible = self.is_eligible

        if not is_eligible.any():
            raise ValueError(f"No eligible indices found with filters: {self.is_eligible}")

        return is_eligible

    def _get_already_sampled(self, atom_array: AtomArray, total_mask: np.ndarray) -> np.ndarray:
        if exists(self.avoid_same):
            segments = annot_start_stop_idxs(atom_array, annots=self.avoid_same, add_exclusive_stop=True)
            already_sampled = apply_and_spread(segments, total_mask, np.any)
        else:
            already_sampled = total_mask
        return already_sampled

    @abc.abstractmethod
    def __call__(
        self,
        atom_array: AtomArray,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> int:
        raise NotImplementedError("Subclasses must implement this method")


class SampleUniformly(SampleSeed):
    def __call__(
        self,
        atom_array: AtomArray,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> int:
        # ... get the eligible indices (all if no filters)
        is_eligible = self._get_is_eligible(atom_array)

        # ... get the already sampled indices
        already_sampled = self._get_already_sampled(atom_array, total_mask)

        # ... get the available indices for sampling
        available_indices = np.where(is_eligible & ~already_sampled)[0]
        if len(available_indices) == 0:
            raise ValueError("No available indices to sample from.")

        choice = self.rng.choice if self.rng else np.random.choice
        return choice(available_indices)


class SampleWithPotential(SampleSeed):
    """
    Samples an atom index with a probability based on a potential function
    to already sampled atoms.

    This class uses a KDTree for efficient nearest neighbor searches to compute
    distances between available (unsampled) atoms and already sampled atoms.
    The distances are then converted into weights using a provided potential
    function, and an atom is sampled based on these weights.

    Args:
        potential: A callable that takes an array of distances and returns
            an array of weights/attraction values.  Higher values indicate
            greater attraction.
        rng: A NumPy random number generator.  Defaults to the default RNG.

    Example:
        def inverse_distance_potential(distances: np.ndarray) -> np.ndarray:
            # Simple potential that is inversely proportional to distance.
            # Adding a small constant to avoid division by zero.
            return 1.0 / (distances + 0.1)

        sampler = SampleWithPotentialFromUnsampled(potential=inverse_distance_potential)
    """

    def __init__(
        self,
        potential: Callable[[np.ndarray], np.ndarray],
        is_eligible: str | None = None,
        avoid_same: tuple[str, ...] | None = ("chain_iid", "res_name", "res_id"),
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            is_eligible=is_eligible,
            avoid_same=avoid_same,
            rng=rng,
        )
        self.potential = potential

    def __call__(
        self,
        atom_array: AtomArray,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> int:
        """
        Samples an atom index based on attraction potential.

        Args:
            atom_array: The AtomArray to sample from.
            total_mask: Mask for all selections so far.
            all_masks: List of all masks.

        Returns:
            The index of the sampled atom.

        Raises:
            ValueError: If there are no available indices to sample from.
        """
        is_nan_coords = np.isnan(atom_array.coord).any(axis=1)
        is_eligible = self._get_is_eligible(atom_array)
        already_sampled = self._get_already_sampled(atom_array, total_mask)
        is_available = is_eligible & ~already_sampled & ~is_nan_coords
        if not is_available.any():
            raise ValueError("No available indices to sample from.")

        sampled_idxs = np.where(already_sampled & ~is_nan_coords)[0]
        available_idxs = np.where(is_available)[0]  # (N_available,)

        if len(sampled_idxs) == 0:
            choice = self.rng.choice if self.rng else np.random.choice
            return choice(available_idxs)

        # ... get the distances to the sampled atoms
        sampled_coords = atom_array.coord[sampled_idxs]  # (N_sampled, 3)
        available_coords = atom_array.coord[available_idxs]  # (N_available, 3)

        # Build KDTree from sampled coordinates & query for distances
        tree = KDTree(sampled_coords)
        distances, _ = tree.query(available_coords)  # (N_available,)

        # Convert distances to probabilities
        potentials = self.potential(distances)  # (N_available,)
        probabilities = potentials / np.sum(potentials)  # (N_available,)

        # Sample an index based on the probabilities
        choice = self.rng.choice if self.rng else np.random.choice
        sampled_index = choice(available_idxs, p=probabilities)

        return sampled_index


def _seed_to_tip_atom_seed(seed: int, atom_array: AtomArray, rng: np.random.Generator) -> int:
    res_id = atom_array.res_id[seed]
    res_name = atom_array.res_name[seed]
    chain_id = atom_array.chain_id[seed]
    res_mask = (atom_array.res_id == res_id) & (atom_array.res_name == res_name) & (atom_array.chain_id == chain_id)
    tip_atom_names = STANDARD_AA_TIP_ATOM_NAMES[res_name]
    tip_atom_idxs = np.where(np.isin(atom_array.atom_name[res_mask], tip_atom_names))[0]
    # ... convert to global
    tip_atom_idxs = np.where(res_mask)[0][tip_atom_idxs]
    if len(tip_atom_idxs) == 0:
        raise ValueError(f"No tip atoms found for residue {res_name}")
    choice = rng.choice if rng else np.random.choice
    selected_tip_idx = choice(tip_atom_idxs)
    return selected_tip_idx


class SampleTipAtomUniformly(SampleUniformly):
    def __init__(self, **kwargs):
        if "is_eligible" not in kwargs:
            protein_chain_types = [c.value for c in ChainType.get_proteins()]
            kwargs["is_eligible"] = f"(chain_type in {protein_chain_types}) & (occupancy > 0) & (~has_nan_coord())"
        super().__init__(**kwargs)

    def __call__(
        self,
        atom_array: AtomArray,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> int:
        seed = super().__call__(atom_array, total_mask, all_masks)
        return _seed_to_tip_atom_seed(seed, atom_array, self.rng)


class SampleTipAtomWithPotential(SampleWithPotential):
    def __init__(
        self,
        potential: Callable[[np.ndarray], np.ndarray],
        is_eligible: str = "is_tip_atom & (res_min_occupancy > 0)",
        avoid_same: tuple[str, ...] | None = ("chain_id", "res_name", "res_id"),
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            potential=potential,
            is_eligible=is_eligible,
            avoid_same=avoid_same,
            rng=rng,
        )

    def __call__(
        self,
        atom_array: AtomArray,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> int:
        seed = super().__call__(atom_array, total_mask, all_masks)
        return _seed_to_tip_atom_seed(seed, atom_array, self.rng)


# ==== Grow functions ====
class GrowMask(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        atom_array: AtomArray,
        seed_idx: int,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> np.ndarray:
        pass


class GrowToSegment(GrowMask):
    def __init__(
        self,
        require_same_annotation: tuple[str, ...] = (
            "chain_id",
            "res_name",
            "res_id",
            "transformation_id",
        ),
    ):
        self.require_same_annotation = require_same_annotation

    def __call__(
        self,
        atom_array: AtomArray,
        seed_idx: int,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> np.ndarray:
        segment_start_stop_idxs = annot_start_stop_idxs(
            atom_array, annots=self.require_same_annotation, add_exclusive_stop=True
        )
        atom_mask = np.zeros(atom_array.array_length(), dtype=bool)
        atom_mask[seed_idx] = True
        atom_mask = apply_and_spread(segment_start_stop_idxs, atom_mask, np.any)
        return atom_mask


class GrowToSameAnnotation(GrowMask):
    def __init__(
        self,
        require_same_annotation: tuple[str, ...] = (
            "chain_id",
            "res_name",
            "res_id",
            "transformation_id",
        ),
    ):
        self.require_same_annotation = require_same_annotation

    def __call__(
        self,
        atom_array: AtomArray,
        seed_idx: int,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> np.ndarray:
        atom_mask = np.zeros(atom_array.array_length(), dtype=bool)
        atom_mask[seed_idx] = True
        masks = []
        for annot in self.require_same_annotation:
            annots = atom_array.get_annotation(annot)
            masks.append(annots == annots[seed_idx])
        atom_mask |= np.logical_and.reduce(masks)  # (N,)
        return atom_mask


class GrowToToken(GrowToSegment):
    def __init__(self, **kwargs):
        super().__init__(require_same_annotation=("token_id",), **kwargs)


class GrowToResidue(GrowToSameAnnotation):
    def __init__(self, **kwargs):
        super().__init__(require_same_annotation=("chain_iid", "res_id", "res_name"), **kwargs)


class GrowToChain(GrowToSameAnnotation):
    def __init__(self, **kwargs):
        super().__init__(require_same_annotation=("chain_iid",), **kwargs)


class GrowToPNUnit(GrowToSameAnnotation):
    def __init__(self, **kwargs):
        super().__init__(require_same_annotation=("pn_unit_iid",), **kwargs)


class GrowToMolecule(GrowToSameAnnotation):
    def __init__(self, **kwargs):
        super().__init__(require_same_annotation=("molecule_iid",), **kwargs)


class GrowToSegmentIsland(GrowMask):
    r"""Grows an initial seed to a contiguous region of segments, an "island".

    This class defines an "island" as a contiguous block of "segments". The growth process is constrained
    to a "region".

    - **Region**: A contiguous block of atoms where all atoms share the same values for annotations specified
      in `require_same_annotation`. The island cannot grow beyond the region containing the seed atom.
      For example, if `require_same_annotation` is `("chain_iid",)`, islands cannot span multiple chains.

    - **Segment**: A contiguous block of atoms within a region where all atoms share the same values for
      annotations specified in `segment_has_same_annotation`. For example, if this is set to
      `("chain_iid", "res_name", "res_id")`, each segment is a residue.

    The island is created by randomly selecting a number of segments (`island_size`) between `island_min_size`
    and `island_max_size`. Then, a starting segment is chosen randomly such that the resulting island is
    guaranteed to contain the original seed atom's segment.

    Here's a pictorial example:
    ```
      . = atom
      [ ... ] = segment
      < ... > = region
      * = seed atom
    ```

    AtomArray:
    ```
    < [ .. | .. | .. | .* | .. | .. ] > < [ .. | .. | .. ] >
      \_____________________________/     \______________/
                 Region 1                      Region 2
    ```

    If we seed in the 4th segment of Region 1 and sample an island of size 3,
    a possible resulting island could be:

    ```
    < [ .. | .. | .. | .* | .. | .. ] > < [ .. | .. | .. ] >
            \____________/
          (Island of size 3)
    ```
    """

    def __init__(
        self,
        segment_has_same_annotation: tuple[str, ...] = (
            "chain_iid",
            "res_name",
            "res_id",
        ),
        require_same_annotation: tuple[str, ...] = ("chain_iid",),
        island_min_size: int = 2,
        island_max_size: int = 10,
        rng: np.random.Generator | None = None,
    ):
        r"""
        Grows an initial seed to a contiguous region of segments, an "island".

        Args:
            segment_has_same_annotation: A tuple of annotation names that define a segment. A contiguous
                set of atoms that agree in the `segment_has_same_annotation` annotations constitutes *ONE*
                segment.
            require_same_annotation: A tuple of annotation names that define the region within which
                the island can grow. The island cannot grow beyond the region containing the seed atom.
                For example, if `require_same_annotation` is `("chain_iid",)`, islands cannot span multiple chains.
            island_min_size: The minimum number of segments in the island.
            island_max_size: The maximum number of segments in the island.
            rng: An optional numpy random number generator for reproducible sampling. If None (recommended),
                the default numpy random number generator will be used.

        Raises:
            ValueError: If `island_min_size` is greater than `island_max_size`.

        Here's a pictorial example of the definitions:
            ```
            . = atom
            [ ... ] = segment (defined by `segment_has_same_annotation`)
            < ... > = region (defined by `require_same_annotation`)
            * = seed atom
            ```

            AtomArray:
            ```
            < [ .. | .. | .. | .* | .. | .. ] > < [ .. | .. | .. ] >
            \_____________________________/     \______________/
                        Region 1                      Region 2
            ```

            If we seed in the 4th segment of Region 1 and sample an island of size 3,
            a possible resulting island could be:

            ```
            < [ .. | .. | .. | .* | .. | .. ] > < [ .. | .. | .. ] >
                    \____________/
                (Island of size 3)
            ```
        """
        if island_min_size > island_max_size:
            raise ValueError(f"{island_min_size=} must be less than or equal to {island_max_size=}")
        if island_min_size == island_max_size == 1:
            raise ValueError("Use `GrowToSegment` instead of `GrowToSegmentIsland` for deterministic island size 1")
        self.require_same_annotation = require_same_annotation
        self.segment_has_same_annotation = segment_has_same_annotation
        self.island_min_size = island_min_size
        self.island_max_size = island_max_size
        self.rng = rng

    def __call__(
        self,
        atom_array: AtomArray,
        seed_idx: int,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> np.ndarray:
        # Determine the allowable, contiguous region within which the seed is allowed to grow by combining segments
        regions = annot_start_stop_idxs(atom_array, annots=self.require_same_annotation, add_exclusive_stop=True)
        region_starts = regions[:-1]
        region_stops = regions[1:]
        seed_region_idx = np.searchsorted(region_starts, seed_idx, side="right") - 1
        assert (
            region_starts[seed_region_idx] <= seed_idx < region_stops[seed_region_idx]
        ), f"{seed_idx=} is not within the allowed region ({region_starts[seed_region_idx]}-{region_stops[seed_region_idx]})"
        region_start = region_starts[seed_region_idx]
        region_stop = region_stops[seed_region_idx]

        # Split the allowed region into segments from which we build up an island
        segments = annot_start_stop_idxs(atom_array, annots=self.segment_has_same_annotation, add_exclusive_stop=True)
        segment_starts = segments[:-1]
        segment_stops = segments[1:]
        # ... get the segment idxs that are within the allowed region
        _is_candidate_segment = np.where((region_start <= segment_starts) & (region_stop >= segment_stops))[0]
        segment_starts = segment_starts[_is_candidate_segment]
        segment_stops = segment_stops[_is_candidate_segment]

        # ... check that the region contains segments and that the seed is within the region
        assert len(segment_starts) > 0, "No candidate segments found"
        seed_segment_idx = np.searchsorted(segment_starts, seed_idx, side="right") - 1
        assert (
            segment_starts[seed_segment_idx] <= seed_idx < segment_stops[seed_segment_idx]
        ), f"{seed_idx=} is not within the allowed region ({segment_starts[seed_segment_idx]}-{segment_stops[seed_segment_idx]})"

        # Sample island size and start position (to include the seed)
        n_candidate_segments = len(segment_starts)
        min_island_size = min(self.island_min_size, n_candidate_segments)
        max_island_size = min(self.island_max_size, n_candidate_segments)
        random_integers = self.rng.integers if self.rng else np.random.randint
        if min_island_size < max_island_size:
            island_size = random_integers(min_island_size, max_island_size + 1)
        else:
            island_size = min_island_size

        max_start_pos = min(seed_segment_idx, n_candidate_segments - island_size)
        min_start_pos = max(0, seed_segment_idx - island_size + 1)

        if min_start_pos < max_start_pos:
            island_start_pos = random_integers(min_start_pos, max_start_pos + 1)
        else:
            island_start_pos = max_start_pos

        # Build island mask
        island_mask = np.zeros(atom_array.array_length(), dtype=bool)
        island_mask[segment_starts[island_start_pos] : segment_stops[island_start_pos + island_size - 1]] = True
        assert island_mask[seed_idx], f"seed_idx {seed_idx} is not in the island mask "

        return island_mask


class GrowToResidueIsland(GrowToSegmentIsland):
    """Uses a `Residue` as the segment and grows it to an island within a `Chain`."""

    def __init__(self, **kwargs):
        super().__init__(
            segment_has_same_annotation=(
                "chain_iid",
                "res_name",
                "res_id",
            ),  # (Set segment unit to be a residue)
            require_same_annotation=("chain_iid",),  # (Disallow growing across chain boundaries)
            **kwargs,
        )


class GrowToTokenIsland(GrowToSegmentIsland):
    """Uses a `Token` as the segment and grows it to an island within a `Chain`."""

    def __init__(self, **kwargs):
        super().__init__(
            segment_has_same_annotation=("token_id",),  # (Set segment unit to be a token)
            require_same_annotation=("chain_iid",),  # (Disallow growing across chain boundaries)
            **kwargs,
        )


class GrowByHoppingAlongBondGraph(GrowMask):
    """
    Grows a mask by hopping along the bond graph from a seed atom.

    This class implements a `GrowMask` strategy that expands a mask from a seed atom by traversing the bond graph
    of the atom array. The growth is controlled by the expected number of hops (`n_hops_expected`), and can be
    constrained to stay within the same token or chain.
    """

    def __init__(
        self,
        n_hops_expected: int = 1,
        require_same_annotation: tuple[str, ...] = (
            "res_name",
            "res_id",
            "chain_id",
            "transformation_id",
        ),
        rng: np.random.Generator | None = None,
        atom_array: AtomArray | None = None,
    ):
        """
        Args:
            n_hops_expected: Expected number of hops to make along the bond graph.
            allow_other_tokens: Whether to allow the mask to grow into other tokens.
            allow_other_chains: Whether to allow the mask to grow into other chains.
            rng: Random number generator.
            atom_array: AtomArray to grow the mask on. Optional, but if provided will precompute the bond graph and save a lot of time
        """
        self.n_hops_expected = n_hops_expected
        self.require_same_annotation = require_same_annotation
        self.rng = rng
        self.graph = None
        if atom_array is not None:
            self.graph = _atom_array_to_networkx_graph(
                atom_array,
                bond_order=False,
                cast_aromatic_bonds_to_same_type=True,
            )

    def __call__(
        self,
        atom_array: AtomArray,
        seed_idx: int,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> np.ndarray:
        """
        Grows the mask by hopping along the bond graph.

        Args:
            atom_array: The atom array to grow the mask on.
            seed_idx: The index of the seed atom.
            total_mask: The total mask.
            all_masks: A list of all masks.

        Returns:
            The grown mask.
        """
        # Determine allowed atoms based on token and chain constraints
        is_allowed = np.ones(atom_array.array_length(), dtype=bool)  # (N,)
        for annotation in self.require_same_annotation:
            _annotation = atom_array.get_annotation(annotation)
            _target_annotation = _annotation[seed_idx]
            is_allowed = is_allowed & (_annotation == _target_annotation)

        # Get the relevant graph around the seed
        graph = self.graph
        if graph is None:
            graph = _atom_array_to_networkx_graph(atom_array, bond_order=False, cast_aromatic_bonds_to_same_type=True)

        # Sample number of hops from geometric distribution
        geometric = self.rng.geometric if self.rng else np.random.geometric
        n_hops = geometric(p=1 / (1 + self.n_hops_expected)) - 1

        # Get atom indices within n hops from the seed
        src_node = seed_idx
        paths = nx.single_source_shortest_path_length(graph, source=src_node, cutoff=n_hops)
        atom_indices = list(paths.keys())

        # Apply the mask to the selected atoms, considering 'is_allowed'
        this_mask = np.zeros(atom_array.array_length(), dtype=bool)  # (N,)
        for atom_idx in atom_indices:
            if is_allowed[atom_idx]:
                this_mask[atom_idx] = True

        return this_mask


# ==== Merge functions ====
class MergeMask(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        atom_array: AtomArray,
        seed_idx: int,
        total_mask: np.ndarray,
        all_masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        pass


# ==== Check budget functions ====
class CheckBudget(abc.ABC):
    @abc.abstractmethod
    def __call__(self, atom_array: AtomArray, total_mask: np.ndarray, all_masks: list[np.ndarray]) -> bool:
        pass


class CheckAtomBudget(CheckBudget):
    def __init__(self, n_min_atoms: int, n_max_atoms: int):
        self.n_min_atoms = n_min_atoms
        self.n_max_atoms = n_max_atoms
        assert (
            self.n_min_atoms <= self.n_max_atoms
        ), f"Can never satisfy budget with {self.n_min_atoms=}<= {self.n_max_atoms=}"

    def __call__(self, atom_array: AtomArray, total_mask: np.ndarray, all_masks: list[np.ndarray]) -> bool:
        n_atoms = len(total_mask)
        assert self.n_min_atoms <= n_atoms, f"Can never satisfy budget with {self.n_min_atoms=}<= {n_atoms=}"
        n_selected_atoms = total_mask.sum()
        if n_selected_atoms < self.n_min_atoms:
            # ... continue sampling
            return False
        if n_selected_atoms > self.n_max_atoms:
            # ... remove the last mask and try again
            all_masks.pop()
            return False
        return True


class CheckResidueBudget(CheckBudget):
    def __init__(
        self,
        n_min_residues: int,
        n_max_residues: int,
        reduce: Callable[[AtomArray], np.ndarray] | None = np.all,
    ):
        self.n_min_residues = n_min_residues
        self.n_max_residues = n_max_residues
        assert (
            self.n_min_residues <= self.n_max_residues
        ), f"Can never satisfy budget with {self.n_min_residues=}<= {self.n_max_residues=}"
        self.reduce = reduce

    def __call__(self, atom_array: AtomArray, total_mask: np.ndarray, all_masks: list[np.ndarray]) -> bool:
        res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        n_res = len(res_starts) - 1
        assert self.n_min_residues <= n_res, f"Can never satisfy budget with {self.n_min_residues=}<= {n_res=}"

        res_mask_atom_lvl = apply_and_spread(res_starts, total_mask, self.reduce)
        res_mask_res_lvl = res_mask_atom_lvl[res_starts[:-1]]
        n_selected_res = res_mask_res_lvl.sum()
        if n_selected_res < self.n_min_residues:
            # ... continue sampling
            return False
        if n_selected_res > self.n_max_residues:
            # ... remove the last mask and try again
            all_masks.pop()
            return False
        return True


class CheckTokenBudget(CheckBudget):
    def __init__(self, n_min_tokens: int, n_max_tokens: int):
        self.n_min_tokens = n_min_tokens
        self.n_max_tokens = n_max_tokens
        assert (
            self.n_min_tokens <= self.n_max_tokens
        ), f"Can never satisfy budget with {self.n_min_tokens=}<= {self.n_max_tokens=}"

    def __call__(self, atom_array: AtomArray, total_mask: np.ndarray, all_masks: list[np.ndarray]) -> bool:
        token_starts = get_token_starts(atom_array)
        n_tokens = len(token_starts)
        assert self.n_min_tokens <= n_tokens, f"Can never satisfy budget with {self.n_min_tokens=}<= {n_tokens=}"

        token_mask_atom_lvl = apply_and_spread(token_starts, total_mask, np.any)
        token_mask_token_lvl = token_mask_atom_lvl[token_starts[:-1]]
        n_selected_tokens = token_mask_token_lvl.sum()
        if n_selected_tokens < self.n_min_tokens:
            return False
        if n_selected_tokens > self.n_max_tokens:
            all_masks.pop()
            return False
        return True


class CheckNumMasksBudget(CheckBudget):
    def __init__(self, n_masks: int):
        assert n_masks > 0, f"Can never satisfy budget with {n_masks=}"
        self.n_masks = n_masks

    def __call__(self, atom_array: AtomArray, total_mask: np.ndarray, all_masks: list[np.ndarray]) -> bool:
        return len(all_masks) >= self.n_masks


def sample_mask_via_seed_grow_merge(
    atom_array: AtomArray,
    all_masks: list[np.ndarray] | None = None,
    fn_sample_seed: SampleSeed = SampleUniformly(),  # noqa: B008
    fn_grow_seed: GrowMask | None = None,
    fn_merge_masks: MergeMask | None = None,
    fn_check_budget: CheckBudget | None = None,
    max_iterations: int = 100,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Sample a mask over an atom array using a flexible `seed-grow-merge` pattern.

    This function is a general purpose utility for sampling masks over atom arrays. It follows this pseudo-code:
    ```text
        until budget is full or max_iterations:
            - Sample `seed_idx = fn_sample_seed(atom_array, total_mask, all_masks)`
                (default: sample uniformly from unsampled atoms)
            - Grow `this_mask = fn_grow_seed(atom_array, seed_idx, total_mask, all_masks)`
                (default: grow nothing, just set `this_mask[seed_idx] = True`)
            - Merge `all_masks = fn_merge_masks(atom_array, seed_idx, this_mask, total_mask, all_masks)`
                (default: append)
            - Get `total_mask = np.logical_or.reduce(all_masks)`
            - Check budget `is_finished = fn_check_budget(total_mask, all_masks)`
            - If `is_finished`, break
    ```

    Args:
        atom_array: The atom array to sample masks over.
        fn_sample_seed: Function that samples a seed index.
            Takes (atom_array, total_mask, all_masks) and returns an integer index.
        fn_grow_seed: Optional function to grow a seed mask.
            Takes (atom_array, seed_idx, total_mask, all_masks) and returns a boolean mask of shape
            (n_atoms,).
        fn_merge_masks: Optional function to merge masks.
            Takes (atom_array, seed_idx, this_mask, total_mask, all_masks) and returns a list of boolean masks,
            each of shape (n_atoms,).
        fn_check_budget: Optional function to check if the budget is full.
            Takes (atom_array, total_mask, all_masks) and returns a boolean. If None, the loop runs for
            `max_iterations`.
        max_iterations: Maximum number of iterations to run.

    Returns:
        total_mask: Boolean array of shape (n_atoms,) with all sampled masks combined
        all_masks: List of individual boolean masks that were sampled
    """
    n_atoms = atom_array.array_length()
    if all_masks is None or len(all_masks) == 0:
        all_masks = []
        total_mask = np.zeros(n_atoms, dtype=bool)
    else:
        all_masks = copy.deepcopy(list(all_masks))
        total_mask = np.logical_or.reduce(all_masks)

    for _ in range(max_iterations):
        # Sample seed if function provided
        if fn_sample_seed is not None:
            try:
                seed_idx = fn_sample_seed(atom_array, total_mask, all_masks)
            except ValueError:
                all_masks.append(total_mask)
                return total_mask, all_masks
        else:
            # ... do not sample anything
            all_masks.append(total_mask)
            return total_mask, all_masks
        if not isinstance(seed_idx, int | np.integer):
            raise TypeError(f"seed_idx must be an integer or numpy integer. Got {type(seed_idx)}")
        assert 0 <= seed_idx < n_atoms, f"seed_idx must be between 0 and {n_atoms}. Got {seed_idx}"

        # Grow seed if function provided
        if fn_grow_seed is not None:
            this_mask = fn_grow_seed(atom_array, seed_idx, total_mask, all_masks)
        else:
            # ... grow nothing, just set `this_mask[seed_idx] = True`
            this_mask = np.zeros(n_atoms, dtype=bool)
            this_mask[seed_idx] = True
        assert this_mask.shape == (
            n_atoms,
        ), f"this_mask must be a boolean array of shape ({n_atoms=},). Got {this_mask.shape}"
        assert this_mask.dtype == bool, f"this_mask must be a boolean array. Got {this_mask.dtype}"

        # Merge masks if function provided
        if fn_merge_masks is not None:
            all_masks = fn_merge_masks(atom_array, seed_idx, this_mask, total_mask, all_masks)
        else:
            # ... merge by simply appending the mask
            all_masks.append(this_mask)

        # Update total mask
        total_mask = np.logical_or.reduce(all_masks)

        # Check if budget is full
        if fn_check_budget is None:
            # ... always return if no budget check function provided
            return total_mask, all_masks
        elif fn_check_budget(atom_array, total_mask, all_masks):
            # ... return if budget is full
            return total_mask, all_masks

    logger.warning(
        f"Max iterations reached. Returning current mask. Total mask: {total_mask.sum()}, all masks: {len(all_masks)}"
    )
    return total_mask, all_masks
