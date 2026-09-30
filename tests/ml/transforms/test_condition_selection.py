import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.io.transforms.atom_array import (
    is_any_coord_nan,
    remove_hydrogens,
    remove_waters,
)
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.transforms import mask_generator as mg
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse


@pytest.fixture
def atom_array() -> AtomArray:
    data = cached_parse("6lyz", hydrogen_policy="remove")
    atom_array = data["atom_array"]
    atom_array = remove_hydrogens(atom_array)
    atom_array = remove_waters(atom_array)
    return atom_array


def test_sample_residue(atom_array: AtomArray):
    with rng_state(create_rng_state_from_seeds(np_seed=1, py_seed=1, torch_seed=1)):
        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=mg.SampleUniformly(
                is_eligible="(occupancy > 0) & (~has_nan_coord())",
                avoid_same=("chain_id", "res_name", "res_id"),
            ),
            fn_grow_seed=mg.GrowToResidue(),
            fn_check_budget=mg.CheckResidueBudget(n_min_residues=3, n_max_residues=3),
        )

        # ... check the total mask
        assert np.unique(atom_array.res_id[total_mask]).size == 3
        assert np.all(atom_array.occupancy[total_mask] > 0)
        assert np.all(~is_any_coord_nan(atom_array[total_mask]))

        # ... check individual masks that were sampled
        for mask in all_masks:
            assert np.unique(atom_array.res_id[mask]).size == 1
            assert np.all(atom_array.occupancy[mask] > 0)
            assert np.all(~is_any_coord_nan(atom_array[mask]))


def test_sample_deterministic(atom_array: AtomArray):
    total_masks = []
    is_eligible_mask = (atom_array.occupancy > 0) & ~np.isnan(atom_array.coord).any(axis=1)

    for i in range(2):
        with rng_state(create_rng_state_from_seeds(np_seed=1, py_seed=1, torch_seed=1)):
            # Run once with the mask as a string, once with the mask as a numpy array
            is_eligible = "(occupancy > 0) & (~has_nan_coord())" if i == 0 else is_eligible_mask

            total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
                atom_array,
                fn_sample_seed=mg.SampleUniformly(
                    is_eligible=is_eligible,
                    avoid_same=("chain_id", "res_name", "res_id"),
                ),
                fn_grow_seed=mg.GrowToResidue(),
            )
            total_masks.append(total_mask)

    assert np.all(total_masks[0] == total_masks[1])


def test_sample_with_attractive_potential(atom_array: AtomArray):
    with rng_state(create_rng_state_from_seeds(np_seed=1, py_seed=1, torch_seed=1)):
        local_sampler = mg.SampleWithPotential(
            is_eligible="(occupancy > 0) & (~has_nan_coord())",
            avoid_same=("chain_id", "res_name", "res_id"),
            potential=lambda x: np.exp(-x * 20),  # <-- extremely local potential
        )

        total_mask = np.zeros(len(atom_array), dtype=bool)
        all_masks = []
        seeds = []
        for _ in range(3):
            seed = local_sampler(atom_array, total_mask, all_masks)
            total_mask[seed] = True
            all_masks.append(total_mask.copy())
            seeds.append(seed)
        assert np.unique(atom_array[seeds].res_id).size == 3
        assert struc.distance(atom_array[seeds[0]], atom_array[seeds[1]]) < 8.0
        assert struc.distance(atom_array[seeds[0]], atom_array[seeds[2]]) < 8.0
        assert struc.distance(atom_array[seeds[1]], atom_array[seeds[2]]) < 8.0
        assert np.all(atom_array.occupancy[total_mask] > 0)
        assert np.all(~is_any_coord_nan(atom_array[total_mask]))
        # Check expectation:
        assert np.array_equal(atom_array[seeds].res_id, np.array([5, 6, 7]))


def test_sample_with_repulsive_potential(atom_array: AtomArray):
    with rng_state(create_rng_state_from_seeds(np_seed=1, py_seed=1, torch_seed=1)):
        non_local_sampler = mg.SampleWithPotential(
            is_eligible="(occupancy > 0) & (~has_nan_coord())",
            avoid_same=("chain_id", "res_name", "res_id"),
            potential=lambda x: np.exp(x * 10),  # <-- extremely non-local potential
        )

        total_mask = np.zeros(len(atom_array), dtype=bool)
        all_masks = []
        seeds = []
        for _ in range(3):
            seed = non_local_sampler(atom_array, total_mask, all_masks)
            total_mask[seed] = True
            all_masks.append(total_mask.copy())
            seeds.append(seed)
        assert np.unique(atom_array[seeds].res_id).size == 3
        assert struc.distance(atom_array[seeds[0]], atom_array[seeds[1]]) > 15.0
        assert struc.distance(atom_array[seeds[0]], atom_array[seeds[2]]) > 15.0
        assert struc.distance(atom_array[seeds[1]], atom_array[seeds[2]]) > 15.0
        assert np.all(atom_array.occupancy[total_mask] > 0)
        assert np.all(~is_any_coord_nan(atom_array[total_mask]))


def test_sample_repulsive_islands(atom_array: AtomArray):
    with rng_state(create_rng_state_from_seeds(np_seed=1, py_seed=1, torch_seed=1)):
        repulsive_sampler = mg.SampleWithPotential(
            is_eligible="(occupancy > 0) & (~has_nan_coord())",
            avoid_same=("chain_id", "res_name", "res_id"),
            potential=lambda x: np.exp(x * 10),  # <-- extremely non-local potential
        )
        island_grower = mg.GrowToResidueIsland(
            island_min_size=3,
            island_max_size=5,
        )
        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=repulsive_sampler,
            fn_grow_seed=island_grower,
            fn_check_budget=mg.CheckNumMasksBudget(n_masks=3),
        )
        assert len(all_masks) == 3

        for mask in all_masks:
            assert np.all(atom_array.occupancy[mask] > 0)
            assert np.all(~is_any_coord_nan(atom_array[mask]))
            assert np.unique(atom_array.res_id[mask]).size >= 3
            assert np.unique(atom_array.res_name[mask]).size <= 5
            # ... check that the mask is contiguous
            assert np.all(np.diff(np.where(mask)[0]) == 1)

        # For development, plot the masks
        # plt.plot(total_mask)
        # plt.savefig("total_mask.png")


def test_sample_subgraph_atoms(atom_array: AtomArray):
    ensure_annotations(atom_array, "is_tip_atom", "res_min_occupancy")
    with rng_state(create_rng_state_from_seeds(np_seed=1, py_seed=1, torch_seed=1)):
        sample_fn = mg.SampleTipAtomWithPotential(
            avoid_same=("chain_id", "res_name", "res_id"),
            potential=lambda x: np.exp(-x / 5.0),  # <-- fairly local potential
        )
        grow_fn = mg.GrowByHoppingAlongBondGraph(
            n_hops_expected=3,
            require_same_annotation=("chain_id", "res_name", "res_id"),
        )
        check_fn = mg.CheckResidueBudget(n_min_residues=3, n_max_residues=10, reduce=np.any)
        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=sample_fn,
            fn_grow_seed=grow_fn,
            fn_check_budget=check_fn,
        )
        assert 3 <= len(all_masks) <= 10
        for mask in all_masks:
            assert np.all(atom_array.occupancy[mask] > 0)
            assert np.all(~is_any_coord_nan(atom_array[mask]))
            assert np.unique(atom_array.res_id[mask]).size == 1


if __name__ == "__main__":
    # run pytest on this file with verbose output
    pytest.main(["-v", __file__])
