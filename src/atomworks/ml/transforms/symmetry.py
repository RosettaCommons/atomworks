import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any, ClassVar

import biotite.structure as struc
import einops
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import torch
from biotite.structure import AtomArray
from rdkit.Chem import BondType as RDKitBondType

from atomworks.constants import HYDROGEN_LIKE_SYMBOLS
from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.io.utils.bonds import hash_atom_array
from atomworks.io.utils.ccd import get_standard_ccd_codes
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.io import cache_based_on_subset_of_args

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# (The order given to a bond whose real order is arbitrary because the group is delocalized)
RESONANT_BOND_ORDER = 1.5


def apply_automorphs(data: torch.Tensor, automorphs: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Create data permutations of the input data for each of the automorphs.

    This function generates permutations of the input tensor data based on the provided automorphisms.
    Each permutation corresponds to a different automorphism, effectively reordering the data according
    to the automorphisms.

    Args:
        data: The input tensor to be permuted. The first dimension has to correspond to
            the number of atoms.
        automorphs: A tensor or numpy array of shape [n_automorphs, n_atoms, 2]
            representing the automorphisms. Each automorphism is a list of paired atom indices
            (from_idx, to_idx). The from_idx column is essentially just a repetition of np.arange(n_atoms).

    Returns:
        A tensor of shape [n_automorphs, ``*data.shape``] containing the permuted
            data for each automorphism.

    Example:
        .. code-block:: python

            data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            # Example automorphisms (2 automorphisms for 3 atoms)
            automorphs = np.array([
                [[0, 0],
                 [1, 1],
                 [2, 2]],
                [[0, 2],
                 [1, 0],
                 [2, 1]]
            ... ])
            permuted_data = create_automorph_permutations(data, automorphs)
            print(permuted_data)
            # tensor([[[1.0, 2.0],
            #          [3.0, 4.0],
            #          [5.0, 6.0]],
            #
            #         [[5.0, 6.0],
            #          [1.0, 2.0],
            #          [3.0, 4.0]]])
    """
    automorphs = torch.as_tensor(automorphs)
    n_automorphs, n_atoms, _ = automorphs.shape
    assert data.shape[0] == n_atoms, "Tensor must have the same number of atoms (dim=0) as the automorphisms."

    _data_extra_dims = data.shape[1:]

    # einindex pattern: 'gather(tensor, automorphs[..., -1], 'natom ..., nauto [natom] -> nauto natom ...)
    # expand the tensor to have an automorphism dimension (creates a view)
    data_automorphs = data.expand(n_automorphs, *data.shape)

    # get the src atom indices for each automorphism, i.e. the atom indices from which to grab the information
    #  based on the original tensor ordering to create the given automorph
    #  ... convention of `from` and `to` is based on original RF2AA convention. Note that either convention
    #   would be valid since the inverse of an (auto/iso)morphism is also an (auto/iso)morphism.
    to_atom_idx, from_atom_idx = automorphs.unbind(dim=-1)  # [n_automorphs, n_atoms]
    # ... and expand to match the dimensions of the expanded tensor
    to_atom_idx = einops.rearrange(
        to_atom_idx, "n_automorphs n_atoms -> n_automorphs n_atoms " + " ".join(["1"] * len(_data_extra_dims))
    ).expand(*data_automorphs.shape)
    from_atom_idx = einops.rearrange(
        from_atom_idx, "n_automorphs n_atoms -> n_automorphs n_atoms " + " ".join(["1"] * len(_data_extra_dims))
    ).expand(*data_automorphs.shape)

    # gather the information from the src atom indices to create all the automorphs
    data_automorphs = torch.gather(data_automorphs, dim=1, index=from_atom_idx)  # [n_automorphs, *data.shape]

    # NOTE: the `scatter` operation is technically only needed if the first column of autmorphs (automorphs[..., 0])
    #  is not sorted. (since it will resolve to the identity in the other cases).
    #  This is very rare, but for example for automorphisms of the structure `R2R`
    #  openbabel returns automorphisms where the first column is not sorted. To resolve this one can either (1)
    #  sort the last dimension of automorphs such that the first column is sorted or (2) use the `scatter` operation
    #  to permute the atoms in the data to the desired positions. We chose the latter here, as it is more
    #  general and can also handle cases where the identity automorphism is not given as sorted indices
    data_automorphs = torch.scatter(
        data_automorphs, dim=1, index=to_atom_idx, src=data_automorphs
    )  # [n_automorphs, *data.shape]

    return data_automorphs


def _create_instance_to_entity_map(iids: np.ndarray, entities: np.ndarray) -> dict[int | str, np.ndarray]:
    """Create a mapping from entity to the instance iids that are instances of that entity."""
    assert iids.shape == entities.shape, "IIDs and entities must have the same shape."
    iids, first_idx_in_instance = np.unique(iids, return_index=True)  # [atom/residue -> instance level]
    entities = entities[first_idx_in_instance]  # [atom/residue -> instance level]
    entity_to_iid = {entity: iids[entities == entity] for entity in np.unique(entities)}
    return entity_to_iid


def _n_possible_isomorphisms(group_to_instance_map: dict[int | str, Sequence[int]]) -> int:
    """Compute the number of possible isomorphisms for a given entity->instances mapping."""
    try:
        product = np.prod([math.factorial(len(v)) for v in group_to_instance_map.values()])
        result = int(product)
    except OverflowError:
        # Handle overflow by returning the maximum integer value
        return (1 << 31) - 1

    # Check if the result is less than zero (paranoia)
    if result < 0:
        return (1 << 31) - 1  # Return the maximum 32-bit integer value

    return result


def get_isomorphisms_from_symmetry_groups(
    group_to_instance_map: dict[int | str, Sequence[int]], max_isomorphisms: int = 1000
) -> np.ndarray:
    """Create an array of all possible isomorphisms for a given entity to instances mapping.

    Args:
        group_to_instance_map (dict[int | str, Sequence[int]]): A dictionary mapping entities to their instances.
            For example, `{1: [0, 1, 2], 2: [3], 3: [4, 5]}`.
        max_isomorphisms (int): The maximum number of isomorphisms to return. Defaults to 1000.

    Returns:
        isomorphisms (np.ndarray): A 2D array of shape `(n_isomorphisms, n_instances)` containing all possible
            isomorphisms. `n_isomorphisms` is the number of possible isomorphisms and `n_instances` is the total
            number of instances. The values are the `id`s (i.e. the values in the `group_to_instance_map`) of the
            instances in the isomorphism. Each group of instances in the `isomorphisms` array appears
            consecutively in the array (column-wise) and the order of the group is the order of the instances
            in the `group_to_instance_map`.

    Example:
        >>> get_isomorphisms_from_symmetry_groups({1: [0, 1, 2], 2: [3], 3: [4, 5]})
        #       1, 1, 1, 2, 3, 3  <-(symmetry group)
        #     --------------------
        array([[0, 1, 2, 3, 4, 5],
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
               [2, 1, 0, 3, 5, 4]], dtype=uint32)
    """
    n_isomorphisms = _n_possible_isomorphisms(group_to_instance_map)
    n_instances = np.sum([len(v) for v in group_to_instance_map.values()])

    # Define helper function to get the number of permutations
    get_perm_count = lambda x: np.prod([len(g) for g in x])  # noqa

    if n_isomorphisms > max_isomorphisms:
        logger.warning(
            f"Number of isomorphisms ({n_isomorphisms}) is greater than the maximum allowed ({max_isomorphisms})."
            f"Symmetries will be truncated to the first {max_isomorphisms} isomorphisms."
        )
        n_isomorphisms = max_isomorphisms

        # Handle the case where the number of isomorphisms is too large by generating a subset of the isomorphisms
        # We need to ensure we have at least one isomorphism for each group of instances
        permutations = [[] for _ in group_to_instance_map]
        _perm_iters = itertools.zip_longest(
            *[itertools.permutations(v) for v in group_to_instance_map.values()], fillvalue=None
        )
        while get_perm_count(permutations) < n_isomorphisms:
            next_set_of_perms = next(_perm_iters)
            for group_idx, permutation in enumerate(next_set_of_perms):
                if permutation is None:
                    continue
                permutations[group_idx].append(permutation)

                if get_perm_count(permutations) >= n_isomorphisms:
                    break
        permutations = [np.array(v) for v in permutations]
    else:
        permutations = [np.array(list(itertools.permutations(v))) for v in group_to_instance_map.values()]

    # Initialize the isomorphisms array
    isomorphisms = np.empty((n_isomorphisms, n_instances), dtype=permutations[0].dtype)

    # Fill the isomorphisms array by taking the cartesian product of the permutations
    start_idx = 0
    n_reps = 1
    for perm in permutations:
        # tile: repeat entire array N times (e.g. [1, 2, 3] -> [1, 2, 3, 1, 2, 3])
        # repeat: repeat each element N times (e.g. [1, 2, 3] -> [1, 1, 2, 2, 3, 3])
        n_perm, n_instances = perm.shape

        # Determine tile size and number of tiles needed to fill the array
        tile_size = n_perm * n_reps  # ... each tile has the permutation repeated n_reps times
        n_tile = math.ceil(n_isomorphisms / tile_size)  # ... number of tiles needed to fill the array
        assert (n_isomorphisms <= get_perm_count(permutations)) or (
            n_tile * tile_size == n_isomorphisms
        ), f"{n_tile} * {tile_size} != {n_isomorphisms} (={n_tile * tile_size})"

        # Fill the isomorphisms array with `n_tile` copies of `perm`, each of which is repeated `n_reps` times
        isomorphisms[:, start_idx : start_idx + n_instances] = np.tile(np.repeat(perm, n_reps, axis=0), (n_tile, 1))[
            :n_isomorphisms
        ]

        start_idx += n_instances  # ... move to start of next group (i.e. move along column)
        n_reps *= n_perm  # ... update the number of repetitions for the next group

    return isomorphisms


def instance_to_token_lvl_isomorphisms(
    instance_isomorphisms: np.ndarray, instance_token_idxs: list[np.ndarray]
) -> np.ndarray:
    """Convert instance-level isomorphisms to token-level isomorphisms.

    This function takes a set of instance-level isomorphisms and their corresponding token indices,
    and maps the instance isomorphisms to token-level indices.

    Args:
        instance_isomorphisms (np.ndarray): A 2D array of shape (n_permutations, n_instances) where each row
            represents a permutation of instance indices.
        instance_token_idxs (list of np.ndarray): A list where each element is an array of token indices
            corresponding to each instance.

    Returns:
        token_lvl_isomorphisms (np.ndarray): A 2D array of shape (n_permutations, total_tokens) containing the
            token-level isomorphisms.

    Example:
    >>> instance_isomorphisms = np.array([[0, 1], [1, 0]])  # Example instance-level isomorphisms
    >>> instance_token_idxs = [
    ...     np.array([0, 1]),
    ...     np.array([2, 3]),
    ... ]  # Example token indices for each instance
    >>> token_lvl_isomorphisms = instance_to_token_lvl_isomorphisms(instance_isomorphisms, instance_token_idxs)
    >>>  [[0 1 2 3]
    >>>   [2 3 0 1]]
    """
    n_perm, n_instances = instance_isomorphisms.shape
    total_tokens = sum(len(idx) for idx in instance_token_idxs)

    # Turn `instance_isomorphs` [n_perm, n_instances], which can be an array of id's into
    #  an array of instance indices, so that we can use it to directly index into the
    #  `instance_token_idxs` array.
    _instance_idx_isomorphisms = np.empty_like(instance_isomorphisms, dtype=np.int32)
    for instance_idx, instance_id in enumerate(instance_isomorphisms[0]):
        _instance_idx_isomorphisms = np.where(
            instance_isomorphisms == instance_id, instance_idx, _instance_idx_isomorphisms
        )

    # Now we can use the instance indices to index into the `instance_token_idxs` array to create
    #  the token-level isomorphisms.
    token_lvl_isomorphisms = np.empty((n_perm, total_tokens), dtype=np.int32)
    for row, permutation in enumerate(_instance_idx_isomorphisms):
        col_start = 0
        for p in permutation:
            token_lvl_isomorphisms[row, col_start : col_start + len(instance_token_idxs[p])] = instance_token_idxs[p]
            col_start += len(instance_token_idxs[p])

    return token_lvl_isomorphisms


def identify_isomorphic_chains_based_on_molecule_entity(atom_array: AtomArray) -> dict[int | str, list[int | str]]:
    """
    Identifies isomorphic molecules based on the molecule entity annotation.

    This function creates a dictionary mapping molecule entities to their corresponding molecule IDs.
    Molecules with the same entity are considered isomorphic.

    Args:
        - atom_array (AtomArray): The atom array containing molecule entity and molecule ID annotations.

    Returns:
        - dict[int | str, list[int | str]]: A dictionary where keys are molecule entities and values
            are lists of molecule IDs belonging to that entity.

    Example:
        >>> atom_array = AtomArray(...)  # AtomArray with molecule_entity and molecule_iid annotations
        >>> isomorphic_molecules = identify_isomorphic_chains_based_on_molecule_entity(atom_array)
        >>> print(isomorphic_molecules)
        {"A,B": [1, 2, 3], "C": [4, 5]}
    """
    # Create a dictionary of molecule entities to molecule iids
    # e.g. {"A,B": [1, 2, 3], "C": [4, 5]}
    isomorphic_molecules = _create_instance_to_entity_map(atom_array.molecule_iid, atom_array.molecule_entity)
    return isomorphic_molecules


def identify_isomorphic_chains_based_on_chain_entity(atom_array: AtomArray) -> dict[int | str, list[int | str]]:
    """
    Identifies isomorphic chains based on the chain entity annotation.

    This function creates a dictionary mapping chain entities to their corresponding chain IDs.
    Chains with the same entity are considered isomorphic.

    Args:
        - atom_array (AtomArray): The atom array containing chain entity and chain ID annotations.

    Returns:
        - dict[int | str, list[int | str]]: A dictionary where keys are chain entities and values
            are lists of chain IDs belonging to that entity.

    Example:
        >>> atom_array = AtomArray(...)  # AtomArray with chain_entity and chain_iid annotations
        >>> isomorphic_chains = identify_isomorphic_chains_based_on_chain_entity(atom_array)
        >>> print(isomorphic_chains)
        {1: ['A', 'B', 'C'], 2: ['D', 'E', 'F'], 3: ['G']}
    """
    # Create a dictionary of chain entities to chain iids
    # e.g. {1: [A, B, C], 2: [D, E, F], 3: [G]}
    isomorphic_chains = _create_instance_to_entity_map(atom_array.chain_iid, atom_array.chain_entity)
    return isomorphic_chains


class AddPostCropMoleculeEntityToFreeFloatingLigands(Transform):
    """
    Relabels the molecule entities of free-floating (i.e. not bonded to a polymer), cropped ligands.
    This is relevant for identifying identical, swappable ligands, which are treated as swappable
    in the RF2AA loss.

    The relabelled molecule entity labels are stored in the `post_crop_molecule_entity` annotation
    of the AtomArray. This ensures that any downstream processes can accurately reference the modified
    entities without confusion.
    """

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data, required=["molecule_entity", "molecule_iid", "is_polymer", "atom_id", "pn_unit_iid", "atomize"]
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        if "crop_info" not in data:
            # Cropping did not occur; return the identity transform (e.g., we're in the validation loop)
            return data

        atom_array = data["atom_array"]
        atom_array.set_annotation("post_crop_molecule_entity", atom_array.molecule_entity.copy())

        # Initialize lookup table for re-labelling identical free-floating ligand fragments
        max_molecule_entity_in_use = np.max(atom_array.molecule_entity)
        hash_to_entity = {}

        # Iterate over all molecules that atomized (non-polys, covalent modifications, non-canonicals)
        _is_ligand = atom_array.atomize
        for molecule_iid in np.unique(atom_array.molecule_iid[_is_ligand]):
            is_in_molecule = atom_array.molecule_iid == molecule_iid
            is_free_floating_ligand = not np.any(atom_array.is_polymer[is_in_molecule])

            if not is_free_floating_ligand:
                # ... ignore ligands bonded to polymers
                # TODO: DOUBLE CHECK - set ligands bonded to polymers to their own, new entity
                mol_hash = molecule_iid
            else:
                # Hash the current free-floating ligand molecule
                # ... get a molecular graph with edge attributes = `bond_type`
                # ... c.f. https://github.com/biotite-dev/biotite/blob/v0.41.0/src/biotite/structure/bonds.pyx#L427-L434
                mol_hash = hash_atom_array(atom_array[is_in_molecule], annotations=["element"], bond_order=True)

            # ... add molecule hash to lookup table for assigning
            if mol_hash not in hash_to_entity:
                # ... if the hash is not yet in the lookup table, assign the next available entity
                hash_to_entity[mol_hash] = max_molecule_entity_in_use + 1
                max_molecule_entity_in_use += 1

            atom_array.post_crop_molecule_entity[is_in_molecule] = hash_to_entity[mol_hash]

        data["atom_array"] = atom_array
        return data


# ... `BondType` is an enum, not an order: AROMATIC_SINGLE is 5, not a quintuple bond. Every
# aromatic flavour is delocalized by definition, so biotite's own perception is honoured here and
# does not depend on RDKit succeeding -- `atom_array_to_rdkit` swallows its own sanitization
# failure, so a silent miss would otherwise match an aromatic ring as a literal kekule structure.
# Anything unrecognised (ANY, COORDINATION) keeps a label of its own, so it neither matches a real
# order nor looks like a triple bond.
_RESONANT_BOND_TYPES = frozenset(
    {
        struc.BondType.AROMATIC,
        struc.BondType.AROMATIC_SINGLE,
        struc.BondType.AROMATIC_DOUBLE,
        struc.BondType.AROMATIC_TRIPLE,
    }
)
_ORDER_BY_BOND_TYPE = {
    struc.BondType.SINGLE: 1.0,
    struc.BondType.DOUBLE: 2.0,
    struc.BondType.TRIPLE: 3.0,
    struc.BondType.QUADRUPLE: 4.0,
}


def _bond_order(bond_type: int) -> float:
    """Map a biotite `BondType` to a comparable order, with every aromatic flavour collapsed."""
    if bond_type in _RESONANT_BOND_TYPES:
        return RESONANT_BOND_ORDER
    return _ORDER_BY_BOND_TYPE.get(bond_type, -1.0 - float(bond_type))


def _resonance_normalized_bond_orders(
    atom_array: AtomArray, n_heavy_neighbors: np.ndarray
) -> dict[frozenset[int], float]:
    """Label each bond with an order under which resonance-equivalent bonds compare equal.

    RDKit equalizes conjugated bonds; a second pass equalizes bonds from an atom to its terminal
    same-element neighbours (e.g. phosphate/sulfonate ``P=O``/``S=O``). Terminality is judged on
    the whole structure, so a bond severed by unit splitting cannot make an atom look terminal.

    Returns:
        Bond order per bond, keyed by ``frozenset({i, j})``, ``RESONANT_BOND_ORDER`` where arbitrary.
    """
    # Seed every bond with its literal order and build the heavy-atom adjacency.
    orders, raw_orders, neighbours = {}, {}, defaultdict(list)
    for i, j, order in atom_array.bonds.as_array():
        bond = frozenset((int(i), int(j)))
        orders[bond] = _bond_order(int(order))
        raw_orders[bond] = orders[bond]
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))

    try:
        # Sanitize to perceive aromaticity/conjugation, then map RDKit atoms back via `rdkit_atom_id`.
        mol = atom_array_to_rdkit(atom_array, hydrogen_policy="keep", sanitize=True)
        from_mol_index = {
            atom.GetIdx(): atom.GetIntProp("rdkit_atom_id") for atom in mol.GetAtoms() if atom.HasProp("rdkit_atom_id")
        }
        for bond in mol.GetBonds():
            begin, end = from_mol_index.get(bond.GetBeginAtomIdx()), from_mol_index.get(bond.GetEndAtomIdx())
            if begin is None or end is None:
                continue  # ... a bond to an atom RDKit added; it has no counterpart to relabel
            if bond.GetBondType() == RDKitBondType.AROMATIC or (
                bond.GetIsConjugated() and bond.GetBondType() in (RDKitBondType.SINGLE, RDKitBondType.DOUBLE)
            ):
                orders[frozenset((begin, end))] = RESONANT_BOND_ORDER
    except (ValueError, RuntimeError, AssertionError) as exc:
        # atom_array_to_rdkit raises MolSanitizeException (a ValueError) or an AssertionError when it
        # cannot perceive aromaticity. Equalizing every bond is permissive (it can over-merge), but
        # keeps a hard fragment usable rather than aborting the whole search.
        logger.warning("RDKit could not perceive conjugation (%s); ignoring bond order for this fragment", exc)
        return dict.fromkeys(orders, RESONANT_BOND_ORDER)

    # Equalize delocalized terminal groups RDKit does not flag as conjugated: an atom's terminal
    # same-element neighbours (e.g. phosphate/sulfonate oxygens), which are resonance-equivalent.
    for centre, adjacent in neighbours.items():
        terminal_by_element = defaultdict(list)
        for neighbour in adjacent:
            if n_heavy_neighbors[neighbour] == 1:
                terminal_by_element[atom_array.element[neighbour]].append(neighbour)
        for element, group in terminal_by_element.items():
            bonds = [frozenset((centre, neighbour)) for neighbour in group]
            # ... carbon has no lone pair to delocalize with, so differing orders on terminal
            # carbons are real (an isopropenyl =CH2 is not its -CH3); and a triple bond is never
            # a resonance form of a single one (cyanamide's nitrile N is not its amine N)
            if len(group) < 2 or element == "C" or any(raw_orders[bond] == 3.0 for bond in bonds):
                continue
            for bond in bonds:
                orders[bond] = RESONANT_BOND_ORDER
    return orders


@cache_based_on_subset_of_args(cache_keys=["hash_key", "normalize_bond_orders", "max_automorphs"])
def generate_automorphisms_from_atom_array_with_networkx(
    atom_array: AtomArray | Callable[[], AtomArray],
    max_automorphs: int = 1000,
    node_features: str | list = "element",
    normalize_bond_orders: bool = True,
    hash_key: Hashable = None,
) -> np.ndarray:
    """Generate automorphisms of a molecular graph using NetworkX.

    Matches like RDKit (bonds must agree) but on resonance-normalized orders, so a bond order the
    CCD assigned arbitrarily within a delocalized group does not count as a real difference.

    Args:
        atom_array: The molecular structure, or a factory evaluated only on a cache miss.
        max_automorphs (int): The maximum number of automorphisms to generate. Default is 1000.
        node_features (str or list of str): The node-level features to use for coloring nodes.
            Can be a single feature (e.g., 'element') or a list of features (e.g., ['element', 'charge']).
            Default is 'element'.
        normalize_bond_orders (bool): Match on :py:func:`_resonance_normalized_bond_orders` (the
            default; uses the `n_heavy_neighbors` annotation if present, else computes it).
            ``False`` ignores bond order entirely and is deprecated, as it admits chemically
            wrong automorphisms.
        hash_key (Hashable): A hashable key to use for caching automorphisms. If None, no caching is used.
            Used by the decorator `cache_based_on_subset_of_args`, so cannot be deleted (even if unused in
            this function).

    Returns:
        np.ndarray: An array where the first row is the identity permutation [0, 1, 2, ..., n],
            and subsequent rows are the permutations representing automorphisms.

    Example:
        >>> atom_array = struc.info.residue("H2O")  # Water molecule; two identical hydrogen atoms
        >>> automorphisms = generate_automorphisms_from_atom_array_with_networkx(atom_array)
        >>> print(automorphisms)
        [[0, 1, 2],
         [0, 2, 1]]  # Example output for a simple molecule like H2O
    """
    if callable(atom_array):
        atom_array = atom_array()
    # ... convert the AtomArray to a NetworkX graph
    graph = atom_array.bonds.as_graph()

    edge_match = None
    if normalize_bond_orders:
        # ... compute heavy-atom degrees on the fly when a caller has not annotated them
        n_heavy_neighbors = (
            atom_array.n_heavy_neighbors
            if "n_heavy_neighbors" in atom_array.get_annotation_categories()
            else count_heavy_neighbors(atom_array)
        )
        orders = _resonance_normalized_bond_orders(atom_array, n_heavy_neighbors)
        nx.set_edge_attributes(graph, {(i, j): orders[frozenset((i, j))] for i, j in graph.edges}, "bond_type")
        edge_match = iso.categorical_edge_match("bond_type", None)
    else:
        warnings.warn(
            "normalize_bond_orders=False (ignoring bond order) is deprecated; it admits chemically "
            "wrong automorphisms. Pass normalize_bond_orders=True.",
            DeprecationWarning,
            stacklevel=2,
        )
        # None preserves edge existence while making every bond compare equal
        nx.set_edge_attributes(graph, None, "bond_type")

    # ... check if we're missing any atoms (e.g., disconnected ions within heme groups)
    if len(graph.nodes) != len(atom_array):
        for idx in range(len(atom_array)):
            if idx not in graph.nodes:
                graph.add_node(idx)

    # node_features must be a list; convert to list if it is a string
    if isinstance(node_features, str):
        node_features = [node_features]

    # ... set node attributes based on the specified features
    # NOTE: Features must be present in the AtomArray annotations
    for feature in node_features:
        nx.set_node_attributes(
            graph, {idx: atom_array.get_annotation(feature)[idx] for idx in range(len(atom_array))}, feature
        )

    # ... build the automorphism generator
    matcher = iso.GraphMatcher(
        graph,
        graph,
        node_match=iso.categorical_node_match(node_features, [""] * len(node_features)),
        edge_match=edge_match,
    )
    automorphism_generator = matcher.isomorphisms_iter()

    # List to store permutations; the first row is the identity permutation
    identity_permutation = list(range(len(atom_array)))
    permutations = [identity_permutation]

    for i, mapping in enumerate(automorphism_generator):
        # Early stopping if the number of automorphisms exceeds the maximum
        if i >= max_automorphs:
            break

        # ... convert the mapping dictionary to a permutation list
        permutation = [mapping[i] for i in identity_permutation]

        if permutation != identity_permutation:  # Skip the identity permutation
            permutations.append(permutation)

    return np.array(permutations)


def count_heavy_neighbors(atom_array: AtomArray) -> np.ndarray:
    """Count the non-hydrogen bond partners of each atom, across the whole structure.

    The count is a per-atom node colour for the automorphism search: it tells a terminal atom
    from an internal one and so keeps distinct residues apart. It is computed on the whole array
    (not per unit) so a bond severed by unit splitting cannot lower a degree. An array without a
    ``BondList`` has no neighbours, so every count is zero.
    """
    if atom_array.bonds is None:
        return np.zeros(atom_array.array_length(), dtype=int)
    bonds = atom_array.bonds.as_array()[:, :2]
    is_heavy = ~np.isin(atom_array.element, HYDROGEN_LIKE_SYMBOLS)
    return np.bincount(bonds[is_heavy[bonds].all(axis=1)].ravel(), minlength=len(atom_array))


def build_automorphism_units(atom_array: AtomArray, residue_index: np.ndarray | None = None) -> list[np.ndarray]:
    """Split an AtomArray into the fragments searched for automorphisms independently.

    One unit per `pn_unit`, except an all-polymer `pn_unit` is split per residue - so a ligand (or
    a residue reassigned onto one by covalent-modification handling) is searched whole, while a
    plain chain is searched residue-wise. Automorphisms therefore never span a `pn_unit`.

    Returns:
        Sorted global atom indices per unit.
    """
    n_atoms = atom_array.array_length()
    if n_atoms == 0:
        return []

    is_polymer = np.asarray(atom_array.is_polymer, dtype=bool)
    if residue_index is None:
        residue_starts = struc.get_residue_starts(atom_array)
        residue_index = np.searchsorted(residue_starts, np.arange(n_atoms), side="right")
    _, pn_unit_index = np.unique(atom_array.pn_unit_iid, return_inverse=True)

    # ... a pn_unit is split per residue only if every one of its atoms is polymer
    n_non_polymer = np.bincount(pn_unit_index, weights=~is_polymer, minlength=pn_unit_index.max() + 1)
    split_per_residue = (n_non_polymer == 0)[pn_unit_index]
    unit_of_atom = np.where(split_per_residue, residue_index, n_atoms + 1 + pn_unit_index)

    atoms_by_unit = np.argsort(unit_of_atom, kind="stable")
    return np.split(atoms_by_unit, np.flatnonzero(np.diff(unit_of_atom[atoms_by_unit])) + 1)


def find_automorphisms_with_networkx(atom_array: AtomArray, max_automorphs: int = 1000) -> list[np.ndarray]:
    """
    Finds automorphisms in an AtomArray using NetworkX, returning indices of atoms that can be permuted.

    The array is split by :py:func:`build_automorphism_units`, and each unit is matched on element
    and whole-structure heavy-atom degree, with bond orders normalized by
    :py:func:`_resonance_normalized_bond_orders` so that resonance is honoured but a localized
    double bond still distinguishes the atoms it joins.

    Args:
        atom_array (AtomArray): The input AtomArray object. Must have the following annotations:
            `pn_unit_iid`, `is_polymer`, `res_id`, `res_name`, `atom_name`, and `element`.
        max_automorphs (int, optional): The maximum number of automorphisms to generate. Default is 1000.

    Returns:
        list[np.ndarray]: One array per unit, each of shape `(n_automorphisms, n_unit_atoms)` holding
            global atom indices, with the identity permutation first.

    Example:
        >>> automorphisms = find_automorphisms_with_networkx(atom_array)
        # [
        #     array([[0, 1, 2, 3, 4, 5],   # identity; indices are global
        #            [0, 1, 2, 3, 5, 4]]), # atoms 4 and 5 are swappable
        #     array([[6, 7, 8, 9, 10, 11]]),
        # ]
    """
    standard_ccd_codes = get_standard_ccd_codes()
    n_heavy_neighbors = count_heavy_neighbors(atom_array)
    residue_starts = struc.get_residue_starts(atom_array)
    residue_index = np.searchsorted(residue_starts, np.arange(atom_array.array_length()), side="right")
    is_polymer = np.asarray(atom_array.is_polymer, dtype=bool)
    has_charge = "charge" in atom_array.get_annotation_categories()

    all_automorphs = []
    for indices in build_automorphism_units(atom_array, residue_index=residue_index):
        unit = None
        unit_residues = residue_index[indices]
        # Stable unit grouping keeps indices sorted, so equal endpoints imply one residue.
        ranks = (
            np.zeros(len(indices), dtype=unit_residues.dtype)
            if unit_residues[0] == unit_residues[-1]
            else np.unique(unit_residues, return_inverse=True)[1]
        )
        residue_rank = np.where(is_polymer[indices], ranks, -1)

        res_names = atom_array.res_name[indices].tolist()
        if all(res_name in standard_ccd_codes for res_name in set(res_names)):
            bonds = tuple(
                sorted((min(i, j), max(i, j), order) for i, j, order in atom_array.bonds[indices].as_array().tolist())
            )
            hash_key = (tuple(res_names), tuple(atom_array.atom_name[indices].tolist()), bonds)
        else:
            unit = atom_array[indices]
            hash_key = hash_atom_array(unit, annotations=["element"], bond_order=True)

        charge_key = tuple(map(int, atom_array.charge[indices].tolist())) if has_charge else ()

        def materialize_unit(
            indices: np.ndarray = indices, unit: AtomArray | None = unit, residue_rank: np.ndarray = residue_rank
        ) -> AtomArray:
            annotated = atom_array[indices] if unit is None else unit
            annotated.set_annotation("n_heavy_neighbors", n_heavy_neighbors[indices])
            annotated.set_annotation("polymer_residue_rank", residue_rank)
            return annotated

        automorphs = generate_automorphisms_from_atom_array_with_networkx(
            materialize_unit,
            max_automorphs=max_automorphs,
            node_features=["element", "n_heavy_neighbors", "polymer_residue_rank"],
            normalize_bond_orders=True,
            hash_key=(
                hash_key,
                tuple(n_heavy_neighbors[indices].tolist()),
                tuple(residue_rank.tolist()),
                charge_key,
                max_automorphs,
            ),
        )
        all_automorphs.append(indices[automorphs])

    # We do not concatenate automorphisms to avoid building a large, sparse tensor
    return all_automorphs


class FindAutomorphismsWithNetworkX(Transform):
    """
    Generates a list of automorphisms (including both polymer and non-polymer residues) for a given atom array.
    Used in AF-3/AF-Multimer-style symmetry resolution
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data, required=["pn_unit_iid", "atomize", "is_polymer", "res_id", "res_name", "atom_name", "element"]
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        automorphisms = find_automorphisms_with_networkx(atom_array=data["atom_array"], max_automorphs=1000)
        data["automorphisms"] = automorphisms
        return data
