import numpy as np
from biotite.structure import AtomArray

from atomworks.ml.transforms.symmetry import (
    _create_instance_to_entity_map,
    _n_possible_isomorphisms,
    get_isomorphisms_from_symmetry_groups,
    identify_isomorphic_chains_based_on_chain_entity,
    identify_isomorphic_chains_based_on_molecule_entity,
)


def test_create_instance_to_entity_map():
    iids = np.array([0, 1, 2, 3, 4, 5])
    entities = np.array([1, 1, 1, 2, 3, 3])
    expected_output = {1: np.array([0, 1, 2]), 2: np.array([3]), 3: np.array([4, 5])}
    output = _create_instance_to_entity_map(iids, entities)
    for key in expected_output:
        assert np.array_equal(output[key], expected_output[key])


def test_n_possible_isomorphisms():
    group_to_instance_map = {1: [0, 1, 2], 2: [3], 3: [4, 5]}
    expected_output = 12
    output = _n_possible_isomorphisms(group_to_instance_map)
    assert output == expected_output


def test_get_isomorphisms_from_symmetry_groups():
    group_to_instance_map = {1: [0, 1, 2], 2: [3], 3: [4, 5]}
    expected_output = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 2, 1, 3, 4, 5],
            [1, 0, 2, 3, 4, 5],
            [1, 2, 0, 3, 4, 5],
            [2, 0, 1, 3, 4, 5],
            [2, 1, 0, 3, 4, 5],
            [0, 1, 2, 3, 5, 4],
            [0, 2, 1, 3, 5, 4],
            [1, 0, 2, 3, 5, 4],
            [1, 2, 0, 3, 5, 4],
            [2, 0, 1, 3, 5, 4],
            [2, 1, 0, 3, 5, 4],
        ],
        dtype=np.uint32,
    )
    output = get_isomorphisms_from_symmetry_groups(group_to_instance_map)
    assert np.array_equal(output, expected_output)


def test_get_isomorphisms_from_symmetry_groups_with_max_set():
    group_to_instance_map = {1: [0, 1, 2], 2: [3], 3: [4, 5]}
    expected_output = np.array(
        [[0, 1, 2, 3, 4, 5], [0, 2, 1, 3, 4, 5], [1, 0, 2, 3, 4, 5], [0, 1, 2, 3, 5, 4], [0, 2, 1, 3, 5, 4]],
        dtype=np.uint32,
    )
    output = get_isomorphisms_from_symmetry_groups(group_to_instance_map, max_isomorphisms=5)
    assert np.array_equal(output, expected_output)


def test_identify_isomorphic_chains_based_on_molecule_entity():
    atom_array = AtomArray(6)
    atom_array.set_annotation("molecule_iid", np.array([0, 1, 2, 3, 4, 5]))
    atom_array.set_annotation("molecule_entity", np.array([1, 1, 1, 2, 3, 3]))
    expected_output = {1: np.array([0, 1, 2]), 2: np.array([3]), 3: np.array([4, 5])}
    output = identify_isomorphic_chains_based_on_molecule_entity(atom_array)
    for key in expected_output:
        assert np.array_equal(output[key], expected_output[key])


def test_identify_isomorphic_chains_based_on_chain_entity():
    atom_array = AtomArray(6)
    atom_array.set_annotation("chain_iid", np.array([0, 1, 2, 3, 4, 5]))
    atom_array.set_annotation("chain_entity", np.array([1, 1, 1, 2, 3, 3]))
    expected_output = {1: np.array([0, 1, 2]), 2: np.array([3]), 3: np.array([4, 5])}
    output = identify_isomorphic_chains_based_on_chain_entity(atom_array)
    for key in expected_output:
        assert np.array_equal(output[key], expected_output[key])


if __name__ == "__main__":
    test_get_isomorphisms_from_symmetry_groups_with_max_set()
