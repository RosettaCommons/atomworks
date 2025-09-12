import itertools
import logging
import math
from collections.abc import Hashable, Sequence
from typing import Any, ClassVar

import biotite.structure as struc
import einops
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import torch
from biotite.structure import AtomArray

from atomworks.io.utils.bonds import hash_atom_array
from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING, TokenEncoding
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.encoding import atom_array_to_encoding
from atomworks.ml.utils.io import cache_based_on_subset_of_args
from atomworks.ml.utils.token import get_token_count, get_token_starts

try:
    from atomworks.ml.transforms.openbabel_utils import find_automorphisms
except ImportError:

    def find_automorphisms(atom_array: AtomArray) -> np.ndarray:
        raise ImportError("OpenBabel is not installed. Please install it to use this function.")


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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


class CreateSymmetryCopyAxisLikeRF2AA(Transform):
    """
    Create the `symmetry` axis for the `xyz` and `mask` features that go into RF2AA. These are required to
    resolve the equivalence between equivalent polymers and non-polymer configurations in the RF2AA loss.

    This transform generates what we loosely refer to as 'symmetry copies' of the coordinates and mask values
    (True indicates an atom exists) by computing and applying isomorphisms between molecules (for polymers) and
    automorphisms within each molecule (for non-polymers). This transform is very bespoke to the RF2AA loss
    and implementation and is not intended to be used outside of the RF2AA codebase.

    The transform roughly follows these steps:

    1. **Input Validation**:
        - Ensures the input data contains the necessary keys and types, including atom arrays, encoded data, and crop
          information. It also checks that the data satisfies the assumptions in RF2AA, namely that all polymer tokens
          occur before non-polymer (or atomized) tokens in the atom array and encoding. Atomized bits of polymers
          are treated as non-polymers in this transform.

    2. **Polymer Symmetries**:
        - Identifies which polymers are equivalent (isomorphic) based on molecule entities.
        - Generates all possible combinations of in-group permutations (isomorphisms) for these polymers.
        - Maps these isomorphisms from the instance level to the token level.
        - Encodes the pre-cropped atom array to get the xyz coordinates and masks.
        - Applies the isomorphisms to the pre-cropped xyz and mask, then subsets to the crop tokens for the permuted
          post-cropped xyz and mask.
        - Ensures the first column corresponds to the unpermuted, original polymer coordinates and mask.

    3. **Non-Polymer Symmetries**:
        - Checks if there are any non-polymers in the crop.
        - Identifies the full molecules (pre-crop) for each non-polymer that has tokens in the crop.
        - Computes the automorphisms for each of these full molecules.
        - Applies the automorphisms to the coordinates and masks of the encoded full molecules, then subsets to the
          crop tokens.
        - Concatenates all the automorphs together, padded to the maximum number of automorphs for any molecule.
        - Ensures all non-polymers are entirely atomized.

    4. **Combining Results**:
        - Combines the polymer and non-polymer xyz and masks by concatenating them along the `token` axis and
          padding the newly created symmetry axis.
        - Updates the encoded data with the combined xyz and mask, which will be used as input for RF2AA.

    The effect of this function is to:
        1. Update the 'encoded' key of the data dict with the symmetry copies of the xyz and mask.
        2. Add the 'symmetry_info' key to the data dict, which contains metadata on the symmetry.

    Example:
        >>> transform = CreateSymmetryCopyAxisLikeRF2AA()
        >>> data = {
        ...     "atom_array": AtomArray(...),
        ...     "encoded": {"xyz": torch.tensor(...), "mask": torch.tensor(...)},
        ...     "openbabel": {...},
        ...     "crop_info": {"atom_array": AtomArray(...), "crop_token_idxs": np.array(...)},
        ... }
        >>> transformed_data = transform.forward(data)
        >>> # transformed_data["encoded"]["xyz"] and transformed_data["encoded"]["mask"] now contain the symmetry copies,
        >>> # i.e. they are of shape
        >>> #  - [n_permutations, n_crop_tokens, n_atoms_per_token, 3]
        >>> #  - [n_permutations, n_crop_tokens, n_atoms_per_token]
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "SortLikeRF2AA",  # this transform assumes that all polymer tokens occur before non-polymer tokens
        "AddOpenBabelMoleculesForAtomizedMolecules",  # this transform needs openbabel molecules to identify automorphs in non-polys
        "EncodeAtomArray",  # this transform needs the encoded data as a sanity check
    ]
    previous_transforms_order_matters = True

    def __init__(
        self,
        encoding: TokenEncoding = RF2AA_ATOM36_ENCODING,
        max_automorphs: int = 1000,
        max_isomorphisms: int = 1000,
    ):
        """
        Initializes the CreateSymmetryCopyAxisLikeRF2AA transform. See the class docstring for more details
        on this transform.

        Args:
            - encoding (TokenEncoding): The encoding scheme to use for the tokens. Default is RF2AA_ATOM36_ENCODING.
            - max_automorphs (int): The maximum number of automorphisms to consider. Default is 1000.
            - max_isomorphisms (int): The maximum number of isomorphisms to consider. Default is 1000.
        """
        self.encoding = encoding
        self.max_automorphs = max_automorphs
        self.max_isomorphisms = max_isomorphisms

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "encoded", "openbabel"])

        # check cropped atom_array:
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data, required=["molecule_entity", "molecule_iid", "is_polymer", "atom_id", "pn_unit_iid", "atomize"]
        )

        # check encoded data:
        check_contains_keys(data["encoded"], ["xyz", "mask"])
        check_is_instance(data["encoded"], "xyz", (torch.Tensor, np.ndarray))
        check_is_instance(data["encoded"], "mask", (torch.Tensor, np.ndarray))

        # check crop data (pre-cropped atom array), if provided (e.g. during training, but not during validation):
        if "crop_info" in data:
            check_contains_keys(data["crop_info"], ["atom_array"])
            check_is_instance(data["crop_info"], "atom_array", AtomArray)
            check_atom_array_annotation(
                data["crop_info"],
                required=["molecule_entity", "molecule_iid", "is_polymer", "atom_id", "pn_unit_iid", "atomize"],
            )

    def assert_nonpoly_come_after_polys(self, atom_array: AtomArray) -> None:
        is_poly = lambda x: x.is_polymer & ~x.atomize  # noqa
        has_poly_and_nonpoly = lambda x: np.all(np.isin([True, False], is_poly(x)))  # noqa

        if has_poly_and_nonpoly(atom_array):
            assert np.max(np.where(is_poly(atom_array))[0]) < np.min(np.where(~is_poly(atom_array))[0])

    def handle_polymer_isomorphisms(
        self,
        pre_poly_array: AtomArray,
        post_poly_array: AtomArray,
        crop_tmask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handles polymer symmetries by computing all swaps between isomorphic (i.e., equivalent) polymers
        that are at least partially in the crop.

        NOTE: This function only swaps full chains. Swaps within atoms of a polymer (e.g., residue
        naming ambiguities) are not considered and are handled elsewhere.

        The process involves the following steps:

        1. Subset the crop mask and pre-cropped atom array to only include polymers that are at least
           partially in the crop.
        2. Among these, identify polymers that are equal to each other (i.e., symmetry groups).
        3. Generate all possible combinations of in-group permutations (isomorphisms).
        4. Apply these isomorphisms to the coordinates and masks of the pre-cropped, encoded polymers.
        5. Crop to the relevant bits that appear in the crop.
        6. De-duplicate the isomorphisms to remove any redundancies.

        Args:
            pre_poly_array: The atom array representing the state before cropping,
              containing polymer tokens.
            post_poly_array: The atom array representing the state after cropping,
              containing polymer tokens.
            crop_tmask: A boolean mask indicating which tokens are included in the crop.

        Returns:
            poly_xyz: The xyz coordinates of the polymers after applying the isomorphisms.
                It has shape [n_permutations, n_crop_tokens, n_atoms_per_token, 3].
            poly_mask: The mask of the polymers after applying the isomorphisms.
                It has shape [n_permutations, n_crop_tokens, n_atoms_per_token].
        """
        # NOTATION: a = atom-level, t = token-level, tidx = token-level index, tmask = token-level mask

        # Subset the crop mask only to the poly tokens
        poly_crop_tmask = crop_tmask[: get_token_count(pre_poly_array)]

        # Subset pre-crop poly array to only the molecule instances that are at least
        #  partially in the crop
        #  ... to do so, first find molecule iids in crop
        molecule_iids_in_crop = np.unique(post_poly_array.molecule_iid)
        is_molecule_in_crop_amask = np.isin(pre_poly_array.molecule_iid, molecule_iids_in_crop)
        is_molecule_in_crop_tmask = is_molecule_in_crop_amask[get_token_starts(pre_poly_array)]
        #  ... then subset to these molecule instances
        pre_poly_array = pre_poly_array[is_molecule_in_crop_amask]
        poly_crop_tmask = poly_crop_tmask[is_molecule_in_crop_tmask]

        # Among the molecules that are at least partially in the crop,
        # identify which polymers are `equal` to each other (i.e. symmetry groups)
        isomorphic_polys = identify_isomorphic_chains_based_on_molecule_entity(pre_poly_array)

        # ... and generate all possible combinations of in-group permutations (=ismorphisms)
        poly_isomorphisms = get_isomorphisms_from_symmetry_groups(
            isomorphic_polys, self.max_isomorphisms
        )  # [n_permutations, n_instances]

        # ... and map these isomorphisms from the instance level to the token-level
        _pre_poly_array_tidxs = get_token_starts(pre_poly_array)
        _identity_isomorphism = poly_isomorphisms[0]
        tidxs_per_instance = [
            np.where(pre_poly_array.molecule_iid[_pre_poly_array_tidxs] == mol_iid)[0]
            for mol_iid in _identity_isomorphism
        ]  # [ [n_tokens_in_instance1], [n_tokens_in_instance2], ... ]
        poly_isomorphisms = instance_to_token_lvl_isomorphisms(
            poly_isomorphisms, tidxs_per_instance
        )  # [n_permutations, n_tokens]

        # Encode pre-cropped atom array to get the xyz coordinates
        pre_poly_encoded = atom_array_to_encoding(pre_poly_array, self.encoding)
        pre_poly_xyz = torch.tensor(pre_poly_encoded["xyz"])  # [n_tokens, n_atoms_per_token, 3]
        pre_poly_mask = torch.tensor(pre_poly_encoded["mask"])  # [n_tokens, n_atoms_per_token]

        # ... generate [to_idx, from_idx] pairs for `apply_automorphs`
        from_tidx = torch.tensor(np.hstack(tidxs_per_instance), dtype=torch.long).view(1, -1)  # [1, n_tokens]
        to_tidx = torch.tensor(poly_isomorphisms, dtype=torch.long)  # [n_permutations, n_tokens]
        poly_isomorphisms = torch.stack(
            [to_tidx, from_tidx.expand_as(to_tidx)], dim=-1
        )  # [n_permutations, n_tokens, 2]

        # Act with isomorphisms on the pre-cropped xyz and mask and then subset to the crop tokens for
        # the permuted post-cropped xyz and mask.
        poly_xyz = apply_automorphs(pre_poly_xyz, poly_isomorphisms)[
            :, poly_crop_tmask
        ]  # [n_permutations, n_crop_tokens, n_atoms_per_token, 3]
        poly_mask = apply_automorphs(pre_poly_mask, poly_isomorphisms)[
            :, poly_crop_tmask
        ]  # [n_permutations, n_crop_tokens, n_atoms_per_token]

        # ... de-duplicate the isomorphisms (duplications can happen if the cropped tokens do not differ between
        #  two isomorphisms, but the atom swaps are in a part that is not within the crop)
        _, is_first_unique = np.unique(poly_isomorphisms[:, poly_crop_tmask], axis=0, return_index=True)
        is_first_unique = np.sort(is_first_unique)
        poly_xyz = poly_xyz[is_first_unique]  # [n_unique_permutations, n_crop_tokens, n_atoms_per_token, 3]
        poly_mask = poly_mask[is_first_unique]  # [n_unique_permutations, n_crop_tokens, n_atoms_per_token]

        # ... remove any permutations that would lead to fully unresolved chains -- this
        #  can happen in extremely rare cases such as e.g. (6G5F), where the in a homodimer
        #  only the first half of chain A is resolved but only the second half of chain B.
        is_fully_unresolved = ~(poly_mask.any(dim=(1, 2)))
        poly_xyz = poly_xyz[~is_fully_unresolved]
        poly_mask = poly_mask[~is_fully_unresolved]

        return poly_xyz, poly_mask

    def handle_nonpoly_automorphisms(
        self,
        pre_nonpoly_array: AtomArray,
        post_nonpoly_array: AtomArray,
        crop_tmask: np.ndarray,
        openbabel_data: dict[int, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handles non-polymer symmetries by computing automorphs within each non-polymer that
        is at least partially in the crop.

        This function calculates the swapped coordinate and mask values for each molecule and
        concatenates all automorphs together, padding the n_permutations dimension to the
        maximum number of automorphs for any molecule.

        WARNING: Unlike polymer symmetries, inter-molecule symmetries are not considered here, as they
        are managed by the RF2AA loss function through a greedy search.

        For non-polymers, the following steps are performed:

        1. Subset the pre-cropped non-poly array to only include the non-poly molecules that are
           at least partially in the crop.
        2. Compute the automorphs for each identified full molecule (i.e. BEFORE cropping).
        3. Apply the computed automorphs to the coordinates and masks of the encoded full molecules.
        4. Crop to the relevant sections of the molecules that appear in the crop.
        5. Concatenate all automorphs together, padding to the maximum number of automorphs
           for any molecule.
        6. De-duplicate the automorphs (duplications can happen if the cropped tokens do not differ between
           two automorphs, but the atom swaps are in a part that is not within the crop)

        Args:
            pre_nonpoly_array: The pre-cropped non-polymer array to process.
            post_nonpoly_array: The post-cropped non-polymer array to process.
            crop_tmask: A boolean mask indicating which tokens are in the crop.
            openbabel_data: A dictionary containing Open Babel data for molecules.

        Returns:
            nonpoly_xyzs: A tensor containing the coordinates of the non-polymer automorphs.
            nonpoly_masks: A tensor containing the masks of the non-polymer automorphs.
            symmetry_info: A dictionary containing the symmetry information.
        """
        n_nonpoly_token_in_crop = get_token_count(post_nonpoly_array)

        # All non-polymers must be entirely atomized. Check explicitly.
        assert np.all(pre_nonpoly_array.atomize)
        assert np.all(post_nonpoly_array.atomize)

        # ... subset to the (full! = pre-cropped) molecules that are still in the crop
        pn_unit_iids_in_crop = np.unique(post_nonpoly_array.pn_unit_iid)  # [n_nonpoly_instances_in_crop]
        atom_ids_in_crop = post_nonpoly_array.atom_id  # [n_nonpoly_tokens_in_crop]
        is_molecule_in_crop_amask = np.isin(pre_nonpoly_array.pn_unit_iid, pn_unit_iids_in_crop)  # [n_tokens]
        pre_nonpoly_array = pre_nonpoly_array[is_molecule_in_crop_amask]  # [n_nonpoly_tokens_in_crop]

        # ... encode the full non-poly molecules
        pre_nonpoly_encoded = atom_array_to_encoding(pre_nonpoly_array, self.encoding)
        pre_nonpoly_xyz = torch.tensor(pre_nonpoly_encoded["xyz"])  # [n_tokens, n_atoms_per_token, 3]
        pre_nonpoly_mask = torch.tensor(pre_nonpoly_encoded["mask"])  # [n_tokens, n_atoms_per_token]

        # ... generate the automorphs for each non-poly molecule
        nonpoly_xyzs = []
        nonpoly_masks = []
        symmetry_info = {}
        n_max_automorphs = 1
        # NOTE: We need to iterate over the `covalently connected, but atomized` bits of each
        #  `pn_unit` here. This is not the same as iterating over the `pn_unit` bits themselves,
        #  as in rare cases they can be disconnected (e.g. with the pn_unit was created by
        #  atomizing e.g. non-adjacent non-canonicals on a polymer chain, they would still
        #  have the `pn_unit_iid` of the original polymer chain but be disconnected from each other)
        # See `7tmj` and residue `AIB` for an example of this.
        n_max_automorphs_in_crop = 0
        for is_in_nonpoly_molecule in struc.get_molecule_masks(pre_nonpoly_array):
            start_atom_id = pre_nonpoly_array.atom_id[is_in_nonpoly_molecule][0]
            token_is_in_crop = np.isin(pre_nonpoly_array.atom_id[is_in_nonpoly_molecule], atom_ids_in_crop)

            # ... compute the automorphisms for each non-polymer molecule of which bits appear in the post-crop
            #  (molecules themselves are pre-cropped though)
            automorphs = find_automorphisms(
                openbabel_data[start_atom_id], max_automorphs=self.max_automorphs
            )  # [n_permutations, n_atoms]
            n_max_automorphs = max(n_max_automorphs, len(automorphs))

            # ... apply automorphisms to coordinates and masks of *this* molecule and then subset to the crop tokens
            _xyz = apply_automorphs(pre_nonpoly_xyz[is_in_nonpoly_molecule], automorphs)[
                :, token_is_in_crop
            ]  # [n_automorphs, n_crop_tokens, n_atoms_per_token, 3]
            _mask = apply_automorphs(pre_nonpoly_mask[is_in_nonpoly_molecule], automorphs)[
                :, token_is_in_crop
            ]  # [n_automorphs, n_crop_tokens, n_atoms_per_token]

            # ... de-duplicate the automorophs (duplications can happen if the cropped tokens do not differ between
            #  two automorphs, but the atom swaps are in a part that is not within the crop)
            _, is_first_unique = np.unique(automorphs[:, token_is_in_crop], axis=0, return_index=True)
            logger.debug(
                f"Found {len(automorphs)} automorphs, but only {len(is_first_unique)} are unique within the crop."
            )

            is_first_unique = np.sort(is_first_unique)
            _xyz = _xyz[is_first_unique]
            _mask = _mask[is_first_unique]

            # ... append to the list of xyz and masks
            nonpoly_xyzs.append(_xyz)
            nonpoly_masks.append(_mask)
            # ... log metadata (mostly for debugging & inspection)
            _res_names = "-".join(np.unique(pre_nonpoly_array.res_name[is_in_nonpoly_molecule]))
            pn_unit_iid = pre_nonpoly_array.pn_unit_iid[is_in_nonpoly_molecule][0]
            symmetry_info[(pn_unit_iid, _res_names, start_atom_id)] = {
                "full_automorphs": len(automorphs),
                "crop_automorphs": len(is_first_unique),
            }
            n_max_automorphs_in_crop = max(n_max_automorphs_in_crop, len(is_first_unique))

        # ... initialize a padded full xyz & mask and then insert the non-poly xyz & masks
        nonpoly_xyz = torch.full(
            (n_max_automorphs_in_crop, n_nonpoly_token_in_crop, self.encoding.n_atoms_per_token, 3), np.nan
        )
        nonpoly_mask = torch.zeros(
            (n_max_automorphs_in_crop, n_nonpoly_token_in_crop, self.encoding.n_atoms_per_token), dtype=torch.bool
        )

        # ... fill
        start_idx = 0
        for xyz, mask in zip(nonpoly_xyzs, nonpoly_masks, strict=False):
            n_permutations = xyz.shape[0]
            n_atoms = xyz.shape[1]
            nonpoly_xyz[:n_permutations, start_idx : start_idx + n_atoms] = xyz
            nonpoly_mask[:n_permutations, start_idx : start_idx + n_atoms] = mask
            start_idx += n_atoms

        return nonpoly_xyz, nonpoly_mask, symmetry_info

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        # NOTATION: a = atom-level, t = token-level, tidx = token-level index, tmask = token-level mask

        encoded = data["encoded"]
        pre_array = (
            data["crop_info"]["atom_array"] if "crop_info" in data else data["atom_array"]
        )  # AtomArray before cropping
        post_array = data["atom_array"]  # AtomArray after cropping

        # Get crop information at token level
        crop_tidxs = (
            data["crop_info"]["crop_token_idxs"] if "crop_info" in data else np.arange(get_token_count(pre_array))
        )
        crop_tmask = np.zeros(get_token_count(pre_array), dtype=bool)
        crop_tmask[crop_tidxs] = True

        if "symmetry_info" not in data:
            # ... create the symmetry info dict if it doesn't exist, to store convenient symmetry metadata
            data["symmetry_info"] = {}

        # NOTE: Atomized bits are treated as `non-polymer`s even though they can be part of a polymer instance
        #  (is_polymer=True)
        # NOTE: This transform assumes that all `polymer` entries occur *BEFORE* all `non-polymer` entries in the
        #  atom array and the encoding. This is required due to way the original RF2AA code handles symmetries.
        self.assert_nonpoly_come_after_polys(pre_array)
        self.assert_nonpoly_come_after_polys(post_array)

        # Handle polymer symmetries
        is_poly = lambda x: x.is_polymer & ~x.atomize  # noqa

        post_poly_array = post_array[is_poly(post_array)]
        if len(post_poly_array) > 0:
            poly_xyz, poly_mask = self.handle_polymer_isomorphisms(
                pre_poly_array=pre_array[is_poly(pre_array)],
                post_poly_array=post_poly_array,
                crop_tmask=crop_tmask,
            )

            # Validate: first column should correspond to unpermuted, original poly coordinates and mask (=identity isomorphism)
            n_poly_token_in_crop = get_token_count(post_poly_array)
            assert torch.allclose(poly_xyz[0], torch.as_tensor(encoded["xyz"][:n_poly_token_in_crop]), equal_nan=True)
            assert torch.allclose(poly_mask[0], torch.as_tensor(encoded["mask"][:n_poly_token_in_crop]), equal_nan=True)
        else:
            n_poly_token_in_crop = 0

        # Check if there are any non-polymers in the crop
        post_nonpoly_array = post_array[~is_poly(post_array)]
        if len(post_nonpoly_array) == 0:
            # ... early exit if there are no non-polymers
            data["encoded"]["xyz"] = poly_xyz
            data["encoded"]["mask"] = poly_mask
            data["symmetry_info"]["n_poly_isomorphisms"] = poly_xyz.shape[0]
            data["symmetry_info"]["max_automorphs"] = 0
            return data

        # Otherwise process non-poly info
        nonpoly_xyz, nonpoly_mask, automorph_info = self.handle_nonpoly_automorphisms(
            pre_nonpoly_array=pre_array[~is_poly(pre_array)],
            post_nonpoly_array=post_nonpoly_array,
            crop_tmask=crop_tmask,
            openbabel_data=data["openbabel"],
        )
        data["symmetry_info"]["nonpoly_automorph_info"] = automorph_info

        # ... sanity check
        n_nonpoly_token_in_crop = get_token_count(post_nonpoly_array)
        # assert torch.allclose(nonpoly_xyz[0], torch.tensor(encoded["xyz"][n_poly_token_in_crop:]), equal_nan=True)

        # Combine the poly and non-poly xyz & masks
        #  The poly_xyz & nonpoly_xyz are both padded to the same length (n_max_automorphs or n_max_poly_swaps),
        #  whichever is larger. The xyz's are padded with `nan`s (while the masks are 0).
        if n_poly_token_in_crop == 0:
            # ... early exit if there were no polymers
            encoded["xyz"] = nonpoly_xyz
            encoded["mask"] = nonpoly_mask
            data["symmetry_info"]["n_poly_isomorphisms"] = 0
            data["symmetry_info"]["max_automorphs"] = nonpoly_xyz.shape[0]
            return data

        n_max_poly_swaps = poly_xyz.shape[0]
        n_max_automorphs = nonpoly_xyz.shape[0]
        n_symmetries = max(n_max_automorphs, n_max_poly_swaps)
        n_token_in_crop = n_poly_token_in_crop + n_nonpoly_token_in_crop

        # ... initialize the xyz and mask
        xyz = torch.full((n_symmetries, n_token_in_crop, self.encoding.n_atoms_per_token, 3), np.nan)
        mask = torch.zeros((n_symmetries, n_token_in_crop, self.encoding.n_atoms_per_token), dtype=torch.bool)

        # ... fill in the xyz and mask
        xyz[:n_max_poly_swaps, :n_poly_token_in_crop] = poly_xyz
        xyz[:n_max_automorphs, n_poly_token_in_crop:] = nonpoly_xyz
        mask[:n_max_poly_swaps, :n_poly_token_in_crop] = poly_mask
        mask[:n_max_automorphs, n_poly_token_in_crop:] = nonpoly_mask

        # Update the encoded data, since this is what RF2AA will get as input
        encoded["xyz"] = xyz  # [n_symmetry, n_tokens, n_atoms_per_token, 3]
        encoded["mask"] = mask  # [n_symmetry, n_tokens, n_atoms_per_token]
        data["symmetry_info"]["n_poly_isomorphisms"] = n_max_poly_swaps
        data["symmetry_info"]["max_automorphs"] = n_max_automorphs

        return data


@cache_based_on_subset_of_args(cache_keys=["hash_key"])
def generate_automorphisms_from_atom_array_with_networkx(
    atom_array: AtomArray,
    max_automorphs: int = 1000,
    node_features: str | list = "element",
    ignore_bond_type: bool = True,
    hash_key: Hashable = None,
) -> np.ndarray:
    """Generate automorphisms of a molecular graph using NetworkX.

    In some cases, the automorphisms generated by RDKit or OpenBabel may be overly strict;
    e.g., they do not account for resonance. This function uses NetworkX to generate automorphisms
    of a molecular graph, which can be more flexible (but in some cases overly permissive).

    Args:
        atom_array (AtomArray): The input molecular structure as an AtomArray object.
        max_automorphs (int): The maximum number of automorphisms to generate. Default is 1000.
        node_features (str or list of str): The node-level features to use for coloring nodes.
            Can be a single feature (e.g., 'element') or a list of features (e.g., ['element', 'charge']).
            Default is 'element'.
        ignore_bond_type (bool): If True, the bond type is ignored when generating automorphisms. Must
            be true in order to detect some resonance-based automorphisms. Default is True.
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
    # ...convert the AtomArray to a NetworkX graph
    graph = atom_array.bonds.as_graph()

    if ignore_bond_type:
        # ...set all bond types to None (but preserve the edge existence)
        nx.set_edge_attributes(graph, None, "bond_type")

    # ...check if we're missing any atoms (e.g., disconnected ions within Heme groups)
    if len(graph.nodes) != len(atom_array):
        for idx in range(len(atom_array)):
            if idx not in graph.nodes:
                graph.add_node(idx)

    # node_features must be a list; convert to list if it is a string
    if isinstance(node_features, str):
        node_features = [node_features]

    # ...set node attributes based on the specified features
    # NOTE: Features must be present in the AtomArray annotations
    for feature in node_features:
        nx.set_node_attributes(
            graph, {idx: atom_array.get_annotation(feature)[idx] for idx in range(len(atom_array))}, feature
        )

    # ...build the automorphism generator
    matcher = iso.GraphMatcher(
        graph, graph, node_match=iso.categorical_node_match(node_features, [""] * len(node_features))
    )
    automorphism_generator = matcher.isomorphisms_iter()

    # List to store permutations; the first row is the identity permutation
    identity_permutation = list(range(len(atom_array)))
    permutations = [identity_permutation]

    for i, mapping in enumerate(automorphism_generator):
        # Early stopping if the number of automorphisms exceeds the maximum
        if i >= max_automorphs:
            break

        # ...convert the mapping dictionary to a permutation list
        permutation = [mapping[i] for i in identity_permutation]

        if permutation != identity_permutation:  # Skip the identity permutation
            permutations.append(permutation)

    return np.array(permutations)


def find_automorphisms_with_networkx(atom_array: AtomArray, max_automorphs: int = 1000) -> np.ndarray:
    """
    Finds automorphisms in an AtomArray using NetworkX, returning indices of atoms that can be permuted.

    Args:
        atom_array (AtomArray): The input AtomArray object. Must have the following annotations:
            `pn_unit_iid`, `is_polymer`, `res_id`, `res_name`, `atom_name`, and `element`.
        max_automorphs (int, optional): The maximum number of automorphisms to generate. Default is 1000.

    Returns:
        np.ndarray: A Python list of arrays, each containing indices of atoms that can be permuted within the global
                    frame of the input `atom_array`.

    Example:
        >>> automorphisms = find_automorphisms_with_networkx(atom_array)
        # Output:
        # [
        #     array([  # E.g., corresponding to the first residue
        #         [0, 1, 2, 3, 4, 5],  # The first row is the identity permutation
        #         [0, 1, 2, 3, 5, 4]   # Atoms with global indices 4 and 5 are swappable
        #     ]),
        #     array([  # E.g., corresponding to the second residue
        #         [6, 7, 8, 9, 10, 11],  # The first row is the identity permutation. Indices are global (within the AtomArray).
        #     ])
        # ]
        # Each sub-array represents indices of atoms that can be permuted within the global frame.
    """
    all_automorphs = []

    # ...iterate through pn_unit_iids
    for pn_unit_iid in np.unique(atom_array.pn_unit_iid):
        pn_unit_mask = atom_array.pn_unit_iid == pn_unit_iid

        # If a polymer, we find isomorphisms residue-wise (since we don't need to worry about multi-residue or multi-chain ligands)
        if atom_array.is_polymer[pn_unit_mask].all():
            # ...iterate through residues
            for res_id in np.unique(atom_array.res_id[pn_unit_mask]):
                # Global mask for the current residue
                residue_mask = pn_unit_mask & (atom_array.res_id == res_id)

                # Create a hashable key using residue name and atom names
                hash_key = (atom_array.res_name[residue_mask][0], tuple(atom_array.atom_name[residue_mask]))

                # ...find automorphisms
                automorphs = generate_automorphisms_from_atom_array_with_networkx(
                    atom_array[residue_mask],
                    max_automorphs=max_automorphs,
                    node_features=["element"],
                    ignore_bond_type=True,
                    hash_key=hash_key,
                )

                # ...get the indices of the atoms with respect to the global frame
                global_atom_indices = np.where(residue_mask)[0]
                automorphs = global_atom_indices[automorphs]

                all_automorphs.append(automorphs)
        # If a non-polymer, find automorphisms for the entire pn_unit (which may include multiple residues)
        else:
            # Create a hashable key that is all residue ID's concatenated and the atom names
            hash_key = (tuple(atom_array.element[pn_unit_mask]), tuple(atom_array.atom_name[pn_unit_mask]))

            # ...find automorphisms
            automorphs = generate_automorphisms_from_atom_array_with_networkx(
                atom_array[pn_unit_mask],
                max_automorphs=max_automorphs,
                node_features=["element"],
                ignore_bond_type=True,
                hash_key=hash_key,
            )

            # ...get the indices of the atoms with respect to the global frame
            global_atom_indices = np.where(pn_unit_mask)[0]
            automorphs = global_atom_indices[automorphs]

            all_automorphs.append(automorphs)

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
