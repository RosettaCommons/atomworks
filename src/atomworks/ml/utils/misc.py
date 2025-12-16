import collections
import hashlib
import logging
import numbers
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange

from atomworks.common import default
from atomworks.constants import NA_VALUES

logger = logging.getLogger(__name__)


# TODO: Distribute logically and rename away from `misc`
def dfill(a: np.ndarray) -> np.ndarray:
    """
    Takes an array and returns the indices at which the value changes, repeating each index until the next change occurs.

    Args:
        a (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: An array of indices where each index is repeated until a change in value occurs in the input array.

    Example:
        >>> short_list = np.array(list("aaabaaacaaadaaac"))
        >>> dfill(short_list)
        array([ 0,  0,  0,  3,  4,  4,  4,  7,  8,  8,  8, 11, 12, 12, 12, 15])
    """
    n = a.size
    b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [n]])
    return np.arange(n)[b[:-1]].repeat(np.diff(b))


def argunsort(s: np.ndarray) -> np.ndarray:
    """
    Returns the permutation necessary to undo a sort given the argsort array.

    An argsort array is an array of indices that sorts another array. This function allows you to get the argsort array, sort your array with it, and then undo the sort without the overhead of sorting again.

    Args:
        s (numpy.ndarray): The argsort array.

    Returns:
        numpy.ndarray: The permutation array that can be used to undo the sort.

    Example:
        >>> arr = np.array([3, 1, 2])
        >>> s = np.argsort(arr)
        >>> sorted_arr = arr[s]
        >>> undo_sort = argunsort(s)
        >>> original_arr = sorted_arr[undo_sort]
        >>> np.array_equal(original_arr, arr)
        True
    """
    n = s.size
    u = np.empty(n, dtype=np.int64)
    u[s] = np.arange(n)
    return u


def cumcount(a: np.ndarray) -> np.ndarray:
    """
    Helper function to compute the cumulative count of each unique element in an array.

    Source:
    https://stackoverflow.com/questions/40602269/how-to-use-numpy-to-get-the-cumulative-count-by-unique-values-in-linear-time
    """
    n = a.size
    s = a.argsort(kind="mergesort")
    i = argunsort(s)
    b = a[s]
    return (np.arange(n) - dfill(b))[i]


def hash_sequence(sequence: str) -> str:
    """
    Generate a SHA-256 hash for the given sequence and return a compressed string format of the hash.

    Args:
        sequence (str): The sequence to be hashed.

    Returns:
        str: The compressed hash string format.
    """
    sha256_hash = hashlib.sha256(sequence.encode()).hexdigest()
    compressed_name = sha256_hash[:11]  # Using first 11 characters for simplicity
    return compressed_name


@lru_cache(maxsize=1)
def _get_taxonomy_id_lookup_df(
    # TODO: Initialize the taxonomy_id_csv_path from Hydra
    taxonomy_id_csv_path: Path = Path("/projects/ml/RF2_allatom/data_loading/pdb_chain_taxonomy.csv.gz"),
) -> pd.DataFrame:
    """
    Loads a CSV file containing taxonomy IDs for PDB chains into a DataFrame.
    Maps {pdb_id, chain_id} : tax_id.
    The original CSV file is obtained from the SIFTS project, which maps PDB chains to UniProt sequences.
    See https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html for more detail, and most recent download links.
    For performance, we pickle the DataFrame after loading initially from the CSV.

    Args:
    - taxonomy_id_csv_path (Path): Path to the CSV file containing the taxonomy IDs.

    Returns:
    - pd.DataFrame: A DataFrame containing the taxonomy IDs for PDB chains.
    """
    # First, check if there is a pickle with the same name as the CSV
    pickle_path = taxonomy_id_csv_path.with_suffix(".pkl")
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    else:
        # NOTE: We may need to update the columns to keep if the SIFTS project changes their format
        columns_to_keep = ["PDB", "CHAIN", "TAX_ID", "SCIENTIFIC_NAME"]
        taxonomy_id_df = pd.read_csv(
            taxonomy_id_csv_path,
            usecols=columns_to_keep,
            compression="gzip",
            skiprows=1,
            keep_default_na=False,
            na_values=NA_VALUES,
        )

        # Pickle the dataframe for faster loading in the future
        taxonomy_id_df.to_pickle(pickle_path)
        return taxonomy_id_df


def get_msa_tax_id(pdb_id: str, chain_id: str) -> int:
    """
    Retrieves the taxonomy ID for a given PDB and chain ID combination.

    Parameters:
    - pdb_id (str): The PDB ID of the protein structure. E.g., "1A2K".
    - chain_id (str): The chain ID within the PDB structure. E.g., "A". Notably, no transformation ID.

    Returns:
    - str: The taxonomy ID corresponding to the combined PDB and chain ID (e.g., "79015").
    """
    df = _get_taxonomy_id_lookup_df()
    row = df[(df["PDB"] == pdb_id) & (df["CHAIN"] == chain_id)]

    if row.empty:
        return None
    return str(row["TAX_ID"].values[0])


def convert_pn_unit_iids_to_pn_unit_ids(pn_unit_iids: list[str]) -> list[str]:
    """
    Convert a list of pn_unit_iid strings to pn_unit_id strings.

    Example:
        >>> pn_unit_iids = ["B_1,C_1", "A_11,B_11"]
        >>> convert_pn_unit_iids_to_pn_unit_ids(pn_unit_iids)
        ['B,C', 'A,B']
    """
    pn_unit_ids = []
    for pn_unit_iid in pn_unit_iids:
        # Split by comma to get individual components
        components = pn_unit_iid.split(",")
        # Extract the first character of each component and join them with commas
        pn_unit_id = ",".join([component.split("_")[0] for component in components])
        pn_unit_ids.append(pn_unit_id)
    return pn_unit_ids


def extract_transformation_id_from_pn_unit_iid(pn_unit_iid: str) -> str:
    """
    Extracts the transformation ID from a pn_unit_iid string.

    Example:
        >>> extract_transformation_id_from_pn_unit_iid("A_1,B_1")
        '1'
    """
    segments = pn_unit_iid.split(",")
    transformation_ids = {segment.split("_")[1] for segment in segments}

    if len(transformation_ids) > 1:
        raise ValueError(
            f"Transformation IDs are different; must be the same for all chains within a pn_unit. IDs: {transformation_ids}"
        )

    return transformation_ids.pop()


def masked_mean(
    *,
    mask: torch.Tensor,
    value: torch.Tensor,
    axis: int | list[int] | None = None,
    drop_mask_channel: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute the masked mean of a tensor along specified axes.

    Parameters:
    - mask (torch.Tensor): A mask tensor with the same shape as `value` or with dimensions that can be broadcast to `value`.
    - value (torch.Tensor): The input tensor for which the masked mean is to be computed. If memory is a concern, can be float16 or even a bool - the sensitive parts of the computation are in float32.
    - axis (Optional[Union[int, List[int]]]): The axis or axes along which to compute the mean. If None, the mean is computed over all dimensions.
    - drop_mask_channel (bool): If True, drops the last channel of the mask (assumes the last dimension is a singleton).
    - eps (float): A small constant to avoid division by zero.

    Returns:
    - torch.Tensor: The masked mean of `value` along the specified axes. Given in full precision (float32).

    Example:
    >>> import torch
    >>> mask = torch.tensor([[1, 0], [1, 1]], dtype=bool)
    >>> value = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float16)
    >>> mask_mean(mask, value, axis=0)
    tensor([3., 5.]) # float32

    Reference:
        `AF2 Multimer Code <https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/model/utils.py#L79>`_
    """

    # Drop the last channel of the mask if specified
    if drop_mask_channel:
        mask = mask[..., 0]

    # Get the shapes of the mask and value tensors
    mask_shape = mask.shape
    value_shape = value.shape

    # Ensure the mask and value have the same number of dimensions
    assert len(mask_shape) == len(value_shape), "Mask and value must have the same number of dimensions."

    # Convert axis to a list if it's a single integer
    if isinstance(axis, numbers.Integral):
        axis = [axis]
    # If axis is None, compute the mean over all dimensions
    elif axis is None:
        axis = list(range(len(mask_shape)))
    # Ensure axis is an iterable
    assert isinstance(axis, collections.abc.Iterable), 'axis needs to be either an iterable, integer, or "None"'

    # Calculate the broadcast factor to account for broadcasting in the mask
    broadcast_factor = 1.0
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size, "Mask and value dimensions must match or be broadcastable."

    # Multiply the mask by the value...
    masked_values = mask * value  # If value is a boolean, equivalent to bitwise AND
    # ...and convert to sparse tensor to avoid memory issues
    masked_values_s = masked_values.to_sparse()

    # Compute the masked sum and convert back to a dense tensor, as we've reduced the dimensionality
    masked_sum = torch.sparse.sum(masked_values_s, dim=axis, dtype=torch.float16).to_dense()

    # Compute the mask sum, apply broadcast factor, and add epsilon
    mask_sum = torch.sum(mask, dim=axis, dtype=torch.float32) * broadcast_factor + eps

    # Compute the masked mean
    masked_mean = masked_sum / mask_sum

    return masked_mean


def grouped_sum(data: torch.Tensor, assignment: torch.Tensor, num_groups: int, as_float: bool = True) -> torch.Tensor:
    """
    Computes the sum along a tensor, given group indices.

    Args:
        data (torch.Tensor): A tensor whose groups are to be summed.
                             Shape: (N, ..., D), where N is the number of elements.
        assignment (torch.Tensor): A 1-D tensor containing group indices. Must be int64 (to be compatible with the scatter operation).
                                   Shape: (N,).
        num_groups (int): The number of groups.
        as_float (bool): If True, the input data will be converted to float before summing. If not True, then booleans will be added as booleans, not integers.

    Returns:
        torch.Tensor: A tensor of the same data type as the input `data`, containing
                      the sum of elements for each group (cluster).
                      Shape: (num_groups, ..., D).

    Example:
        >>> data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> assignment = torch.tensor([1, 1, 1, 1])
        >>> num_groups = 2
        >>> grouped_sum(data, assignment, num_groups)
        tensor([[ 6,  8],
                [10, 12]])
        # Explanation:
        # Group 0: [1, 2] + [5, 6] = [6, 8]
        # Group 1: [3, 4] + [7, 8] = [10, 12]
    """
    # Optionally, convert the data to float
    # NOTE: For booleans, not converting to floats will result in possibly unexpected behaviors (e.g., 1 + 1 = 1)
    if as_float:
        data = data.to(torch.float)

    # Reshape assignment to match the shape of data
    assignment = assignment.view(-1, *((1,) * (data.dim() - 1)))
    assignment = assignment.expand_as(data)

    # If assignment isn't int64, convert it to int64 (to be compatible with the scatter operation)
    if assignment.dtype != torch.int64:
        assignment = assignment.to(torch.int64)

    # Define the shape of the output tensor
    shape = [num_groups, *list(data.shape[1:])]

    # Create a zero tensor to accumulate the sums, and scatter-add the data
    csum = torch.zeros(*shape, dtype=data.dtype, device=data.device).scatter_add_(0, assignment, data)

    return csum


def grouped_count(
    data: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    groups: list[torch.Tensor] | None = None,
    n_tokens: int | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """
    Counts the occurrence of each token in a data tensor, optionally within specified groups and masked positions.
    (Time & memory-efficient implementation of `grouped_sum` accross one-hot-tokens)

    NOTE: The special case where `groups=None` and `mask=None` corresponds to one-hot token counting.

    Args:
        - data (torch.Tensor): The input tensor containing token data for which we want to count the occurence of each token.
        - mask (torch.Tensor | None): A boolean mask tensor with `True` values for all positions to include when conunting.
            If None, all positions are considered (i.e. mask = True for all positions).
        - groups (list[torch.Tensor] | None): A list of tensors specifying the group assignments for each dimension of the data tensor.
            If None, each position is its own group for each dimension.
        - n_tokens (int | None): The number of unique tokens. If None, it is inferred from the data tensor.

    Returns:
        - torch.Tensor: A tensor containing the count of each token in each group. The shape of the tensor is determined by the group sizes and the number of tokens.

    Example:
        >>> msa = torch.tensor(
        ...     [
        ...         [0, 1, 3, 1, 2],
        ...         [1, 0, 0, 3, 2],
        ...         [2, 2, 1, 0, 1],
        ...         [3, 1, 2, 2, 3],
        ...         [1, 0, 0, 0, 1],
        ...         [2, 1, 3, 3, 1],
        ...     ]
        ... )
        >>> groups = [
        ...     [0, 1, 2, 2, 1, 0],  # groups for dim=0 (=rows)
        ...     [0, 1, 2, 3, 4],  # groups for dim=1 (=cols)
        ... ]
        >>> group_counts = grouped_count(msa, mask=None, groups=groups)
        >>> group_counts[0]
        tensor([
            [1, 0, 1, 0],  # (corresponds to 0x1 & 2x1 at position 0 in rows 0 & 5)
            [0, 2, 0, 0],  # (corresponds to 1x2 at position 1 in rows 0 & 5)
            [0, 0, 0, 2],  # (corresponds to 3x2 at position 2 in rows 0 & 5)
            [0, 1, 0, 1],  # (corresponds to 1x1 & 3x1 at position 3 in rows 0 & 5)
            [0, 1, 1, 0]   # (corresponds to 2x1 & 1x1 at position 4 in rows 0 & 5)
        ])

    """

    n_tokens = default(n_tokens, data.max() + 1)
    mask = default(mask, torch.ones_like(data, dtype=torch.bool))
    groups = default(groups, [torch.arange(size, dtype=torch.long, device=data.device) for size in data.shape])

    # Check input validity
    assert len(groups) == data.dim(), "Number of groups must match the number of dimensions in the data tensor."
    assert all(
        len(group) == shape for group, shape in zip(groups, data.shape, strict=True)
    ), "The i-th assignments `groups` must have the same length as the i-th dimension of the data tensor."

    # Infer the group sizes (= number of unique groups in each dimension)
    group_sizes = [max(group) + 1 for group in groups]

    # ... initialize the (flattened) tensor to scatter the cluster statistics into
    flat_counts = torch.zeros(
        np.prod(group_sizes) * n_tokens, dtype=dtype, device=data.device
    )  # [n_group1 * n_group2 * ... * n_tokens]

    # ... infer the resulting strides for the flattened tensor
    strides = torch.cumprod(torch.tensor([n_tokens] + group_sizes[::-1], dtype=torch.long, device=data.device), dim=0)
    # ... note: strides are currently in reverse order, i.e. [n_tokens, n_group1, n_group2, ...], so we need to reverse them back
    strides = reversed(strides[:-1])

    # Create graded index tensor for each group
    data = data.clone()
    for group_idx, group, stride in zip(range(len(groups)), groups, strides, strict=False):
        # ... compute index offsets for each cluster
        #  will be of the form e.g. `n_group -> () n_group () ()`
        unsqueeze_pattern = "n_group -> " + " ".join(
            ["()"] * group_idx + ["n_group"] + ["()"] * (len(groups) - group_idx - 1)
        )
        # ... rearrange to `unsqueeze` all dimensions except the current group dimension and expand to the shape of the data tensor
        offset = rearrange(group * stride, unsqueeze_pattern).expand_as(data).expand_as(data)  # [n1, n2, ... n_tokens]
        # ... add the offset to the data tensor
        data += offset
    # ... subset to valid positions
    data = data[mask]  # [n_masked]

    # ... temporary tensor of ones to scatter (to count the number of times each token appears in each cluster)
    _one = torch.ones((1,), dtype=dtype, device=data.device)
    # ... scatter ONE's into the (flat) cluster statistics tensor to count the number of times each token appears in group
    flat_counts.scatter_add_(dim=0, index=data.view(-1), src=_one.expand_as(data))
    # ... reshape the flat counts tensor to the final desired shape
    return flat_counts.view(*group_sizes, n_tokens)  # [n_group1, n_group2, ..., n_tokens]


def randomly_select_items_with_weights(weight_dict: dict, n: int = 1) -> list:
    """Randomly select n items from a dictionary based on their weights."""
    choices, weights = zip(*weight_dict.items(), strict=False)
    weights = np.array(weights)
    probabilities = weights / sum(weights)
    chosen = np.random.choice(choices, size=n, p=probabilities)
    return chosen[0] if n == 1 else chosen
