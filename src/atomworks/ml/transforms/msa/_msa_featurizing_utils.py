import torch

from atomworks.ml.encoding_definitions import (
    TokenEncoding,
)
from atomworks.ml.utils.misc import grouped_count, grouped_sum


def transform_ins_counts(ins: torch.Tensor) -> torch.Tensor:
    """Transforms insertion counts into the range [0,1] using the function given in the AF2 Supplement"""
    return 2 / torch.pi * torch.arctan(ins / 3)


def uniformly_select_rows(
    n_rows: int, n_rows_to_select: int, preserve_first_index: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Selects row indices uniformly from a tensor.

    Args:
        n_rows (int): Total number of rows in the tensor.
        n_rows_to_select (int): Number of rows to select.
        preserve_first_index (bool, optional): If True, preserves index 0 in selection. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Selected indices and not selected indices.
    """
    if n_rows_to_select >= n_rows:
        # (If n_rows_to_select is greater than or equal to n_rows, select all rows)
        return torch.arange(n_rows), torch.tensor([], dtype=torch.int64)

    if preserve_first_index and n_rows_to_select < 1:
        raise ValueError("n_rows_to_select must be at least 1 when include_first_index is True")

    if preserve_first_index:
        # ...generate a permutation of indices, preserving the first index
        shuffled_indices = torch.randperm(n_rows - 1) + 1
        shuffled_indices = torch.cat((torch.tensor([0]), shuffled_indices))
    else:
        # ...generate a permutation of all indices
        shuffled_indices = torch.randperm(n_rows)

    # ...separate the shuffled indices into selected and not selected
    selected_indices = shuffled_indices[:n_rows_to_select]
    not_selected_indices = shuffled_indices[n_rows_to_select:]

    return selected_indices, not_selected_indices


def build_msa_index_can_be_masked(
    msa_is_padded_mask: torch.Tensor,
    token_idx_has_msa: torch.Tensor,
    encoded_msa: torch.Tensor,
    encoding: TokenEncoding,
) -> torch.Tensor:
    """
    Build the mask indicating where we can apply the BERT mask.

    For the QUERY sequence, we can apply the BERT mask to any position that:
    - Is a protein (i.e., not a small molecule, DNA, or RNA, which we currently do not mask)

    For the MSA, we can apply the BERT mask to any position that:
    - It is not padded due to unpaired sequences
    - It has an MSA (or at least a single-row MSA)
    - It is a protein (we currently do not apply the mask to DNA, RNA, or small molecules)
    """
    # ...do not apply a mask where there is padding, if we are ignoring padding
    index_can_be_masked = ~msa_is_padded_mask  # implicitly copies the mask, so we don't need to duplicate

    # ...outside of the query sequence, do not apply the mask where we do not have an MSA (e.g., DNA, small molecules)
    index_can_be_masked[1:, ~token_idx_has_msa] = False

    # ...do not apply a mask in any columns where the query sequence token index is greater than the "UNK" protein token (i.e., exclude columns for RNA, DNA, and small molecules)
    # NOTE: This exclusion is somewhoat brittle, and may be deprecated if we modify our encoding
    unk_amino_acid_index = encoding.token_to_idx["UNK"]
    greater_than_unk_mask = encoded_msa[0] > unk_amino_acid_index
    index_can_be_masked[:, greater_than_unk_mask] = False

    return index_can_be_masked


def build_indices_should_be_counted_masks(
    encoded_msa: torch.Tensor,  # [n_rows, n_tokens_across_chains] (int)
    mask_position: torch.Tensor,  # [n_rows, n_tokens_across_chains] (bool)
    token_idx_has_msa: torch.Tensor,  # [n_tokens_across_chains] (bool)
    tokens_to_ignore: torch.Tensor,  # [n_tokens_to_ignore] (int)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds mask for the full MSA to indicate which positions should be counted towards the agreement sum.

    Positions should be counted towards the agreement sum if:
    (1) They are not masked out by the BERT mask
    (2) They have an MSA (e.g., not small molecules)
    (3) They are not in the tokens to ignore (e.g., masks, gaps)
    """
    # ...do not count positions that were modified by the BERT mask
    index_should_be_counted_mask = ~mask_position

    # ...do not count positions that do not have MSAs (e.g., small molecules)
    index_should_be_counted_mask[:, ~token_idx_has_msa] = False

    # ...do not count positions that are in the tokens to ignore (e.g., DNA, RNA)
    ignore_mask_clust_reps = torch.isin(encoded_msa, tokens_to_ignore)

    index_should_be_counted_mask[ignore_mask_clust_reps] = False

    return index_should_be_counted_mask  # [n_rows, n_tokens_across_chains] (bool)


def mask_msa_like_bert(
    *,
    encoding: TokenEncoding,
    mask_behavior_probs: dict,
    mask_probability: float,
    full_msa_profile: torch.Tensor,
    encoded_msa: torch.Tensor,
    index_can_be_masked: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the MSA masking procedure described in the AlphaFold2 supplement, section 1.2.7, with some modifications to handle paired MSAs and small molecules.
    Only applies mask to indices where `index_can_be_masked` is True.

    From the AF2 supplement:
        (...)
        2. A mask is generated such that each position in a MSA cluster centre has a 15% probability of being
        included in the mask. Each element in the MSA that is included in the mask is replaced in the following
        way:
        • With 10% probability amino acids are replaced with a uniformly sampled random amino acid.
        • With 10% probability amino acids are replaced with an amino acid sampled from the MSA profile
            for a given position.
        • With 10% probability amino acids are not replaced.
        • With 70% probability amino acids are replaced with a special token (masked_msa_token).
        These masked positions are the prediction targets used in subsubsection 1.9.9. Note that this masking
        is used both at training time, and at inference time.
        (...)

    Args:
        encoding (TokenEncoding): Encoding object with `n_tokens` and `token_to_idx`.
        mask_behavior_probs (dict): Probabilities for each masking behavior:
            - "replace_with_random_aa": Probability of replacing with a random amino acid.
            - "replace_with_msa_profile": Probability of replacing with an amino acid from the MSA profile.
            - "do_not_replace": Probability of not replacing the amino acid.
            - The remaining probability is for replacing with a mask token.
        mask_probability (float): Probability of each position being masked.
        full_msa_profile (torch.Tensor): Tensor [n_tokens_across_chains, n_tokens] representing the MSA profile.
        encoded_msa (torch.Tensor): Tensor [n_rows, n_tokens_across_chains] representing the encoded MSA as token integers. Can be a subset of the full MSA (e.g., only the cluster representatives).
        index_can_be_masked (torch.Tensor): Boolean tensor [n_rows, n_tokens_across_chains] indicating whether a given index can have the BERT mask applied. I.e., padding, small molecules should not be masked.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - masked_msa (torch.Tensor): Tensor [n_rows, n_tokens_across_chains] representing the masked MSA, with the mask only applied to indices where `index_can_be_masked` is True.
            - mask_position (torch.Tensor): Boolean tensor [n_rows, n_tokens_across_chains] indicating positions where a mask was applied (i.e., one of the outcomes of the mask behavior)

    Reference:
        `AF2 Supplement <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_
    """
    # We start by defining the probabilities for each masking behavior:

    # ...create a tensor where the first 20 elements are 0.05 (corresponding to the amino acid tokens) and the rest are 0.0 (corresponding to all other tokens)
    # 0.05 = (1/20), e.g., uniform distribution over the amino acid tokens, 0 for all non-AA tokens (e.g., masked, unknown, atoms, nucleic acids, etc.)
    replace_with_random_aa = torch.tensor(
        [0.05] * 20 + [0.0] * (encoding.n_tokens - 20), dtype=torch.float32
    )  # [n_tokens] (float)

    # ...we've already computed the relevant distributions for the MSA profile, so we use directly
    replace_with_msa_profile = full_msa_profile  # [n_tokens_across_chains, n_tokens] (float)

    # ...re-compute the identity mapping from the encoding for flexibility (rather than passing both encoding and the one-hot tensor)
    do_not_replace = torch.nn.functional.one_hot(
        encoded_msa, encoding.n_tokens
    ).float()  # [n_rows, n_tokens_across_chains, n_tokens] (float)

    # ...get the index of the mask token, and create a tensor with a 1.0 at that index and 0.0 elsewhere
    mask_token_index = encoding.token_to_idx["<M>"]
    replace_with_mask_token = torch.tensor([0.0] * encoding.n_tokens, dtype=torch.float32)  # [n_tokens] (float)
    replace_with_mask_token[mask_token_index] = 1.0  # [n_tokens] (float)

    # ...calculate the mask token probability, which is 1 - sum of the other probabilities
    mask_behavior_probs = mask_behavior_probs.copy()  # Avoid modifying the original dictionary
    mask_behavior_probs["replace_with_mask_token"] = (
        1.0
        - mask_behavior_probs["replace_with_random_aa"]
        - mask_behavior_probs["replace_with_msa_profile"]
        - mask_behavior_probs["do_not_replace"]
    )
    assert mask_behavior_probs["replace_with_mask_token"] >= 0.0

    # ...finally, we can define the categorical probability distrution which we will sample from for each masked element
    categorical_probs = (
        mask_behavior_probs["replace_with_random_aa"]
        * replace_with_random_aa  # broadcast from [n_tokens] to [n_rows, n_tokens_across_chains, n_tokens]
        + mask_behavior_probs["replace_with_msa_profile"]
        * replace_with_msa_profile  # broadcast from [n_tokens_across_chains, n_tokens] to [n_rows, n_tokens_across_chains, n_tokens]
        + mask_behavior_probs["do_not_replace"] * do_not_replace
        + mask_behavior_probs["replace_with_mask_token"]
        * replace_with_mask_token  # broadcast from [n_tokens] to [n_rows, n_tokens_across_chains, n_tokens]
    )  # [n_rows, n_tokens_across_chains, n_tokens] (float)

    # Next, we generate a mask to indicate where to sample from `categorical_probs` (for each element, we sample from the discrete masking distribution with probability `mask_probability`)
    mask_position = torch.rand(encoded_msa.shape) < mask_probability  # [n_rows, n_tokens_across_chains] (bool)

    # ...apply the mask that restricts the positions where the mask can be applied (e.g., ignore padding, small molecules, etc.)
    mask_position &= index_can_be_masked

    # Finally, sample from the distribution defined by `categorical_probs` for each element...
    # TODO: Switch to gumbel-max sampling, AF-Multimer style (see: https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/model/modules_multimer.py#L120)
    sampler = torch.distributions.categorical.Categorical(probs=categorical_probs)
    sampled_token_indices = sampler.sample()

    # ...and apply the sampled tokens to the MSA, only at the positions that we marked for masking
    masked_msa = torch.where(mask_position, sampled_token_indices, encoded_msa)
    return masked_msa, mask_position


def assign_extra_rows_to_cluster_representatives(
    *,
    cluster_representatives_msa: torch.Tensor,  # [n_msa_cluster_representatives, n_tokens_across_chains] (int)
    clust_reps_should_be_counted_mask: torch.Tensor,  # [n_msa_cluster_representatives, n_tokens_across_chains] (bool)
    extra_msa: torch.Tensor,  # [n_not_selected_rows, n_tokens_across_chains] (int)
    extra_msa_should_be_counted_mask: torch.Tensor,  # [n_not_selected_rows, n_tokens_across_chains] (bool)
) -> torch.Tensor:
    """
    Assign sequences not included in the main MSA stack to the closest cluster representative by Hamming distance.
    Does not count values indicated by `cluster_representatives_should_be_counted_mask` and `extra_msa_should_be_counted_mask` towards the agreement sum.

    From the AF2 supplement:
        (...)
        3. The remaining sequences are assigned to their closest cluster by Hamming distance (ignoring masked
        out residues and gaps).
        (...)

    Args:
        cluster_representatives_msa (torch.Tensor): Integer tensor [n_msa_cluster_representatives, n_tokens_across_chains] representing the MSA cluster representatives as tokens.
        clust_reps_should_be_counted_mask (torch.Tensor): Boolean tensor [n_msa_cluster_representatives, n_tokens_across_chains] indicating which MSA indices cluster should be counted towards the agreement sum.
        extra_msa (torch.Tensor): Integer tensor [n_not_selected_rows, n_tokens_across_chains] representing the extra MSA sequences as tokens.
        extra_msa_should_be_counted_mask (torch.Tensor): Boolean tensor [n_not_selected_rows, n_tokens_across_chains] indicating which extra MSA indices should be counted towards the agreement sum.

    Returns:
        torch.Tensor: Integer tensor [n_not_selected_rows] indicating the assignment of each row in the extra MSA to the closest cluster representative.

    Example (simplified, excluding masks):
        If our cluster representative are:
        ```
        [
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
        ]
        ```
        And our extra rows are:
        ```
        [
            [2, 2, 1, 0, 0],
            [3, 3, 3, 2, 2],
            [1, 1, 3, 3, 3],
        ]
        Then the assignment would be:
        ```
        [1, 2, 0]
        ```
        For more detailed examples (including masks), see the test cases in `test_assign_extra_rows_to_cluster_representatives`.

    TODO: Implement specific weights for gaps, AF-Multimer-style (see: https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/data_transforms_multimer.py#L129)
    TODO: Implement using einsums, see: https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/data_transforms_multimer.py#L129
    """
    # Duplicate the cluster_representatives and extra_msa to avoid modifying the original tensors
    cluster_representatives_msa = cluster_representatives_msa.clone()
    extra_msa = extra_msa.clone()

    # Ignored positions should not be counted towards the agreement sum; thus, we choose two unequal negative numbers (negative to avoid collision with other tokens)
    cluster_representatives_msa[~clust_reps_should_be_counted_mask] = -2.0
    extra_msa[~extra_msa_should_be_counted_mask] = -1.0

    # Use 0-norm `cdist` to compute sequence identity percentage, which is equivalent to hamming distance, then invert to get the number of equal positions.
    agreement = torch.cdist(
        extra_msa.float(), cluster_representatives_msa.float(), p=0.0
    )  # [n_not_selected_rows, n_msa_cluster_representatives] (float)
    agreement = extra_msa.shape[1] - agreement  # [n_not_selected_rows, n_msa_cluster_representatives] (float)

    # Choose the closest cluster representative for each extra row
    assignment = torch.argmax(agreement, dim=-1)  # [n_not_selected_rows] (int)

    return assignment


def summarize_clusters(
    encoded_msa: torch.Tensor,  # [n_rows, n_tokens_across_chains] (int)
    msa_raw_ins: torch.Tensor,  # [n_rows, n_tokens_across_chains] (int)
    mask_position: torch.Tensor,  # [n_rows, n_tokens_across_chains] (bool)
    assignments: torch.Tensor,  # [n_not_selected_rows] (int)
    selected_indices: torch.Tensor,  # [n_msa_cluster_representatives] (int)
    not_selected_indices: torch.Tensor,  # [n_not_selected_rows] (int)
    msa_is_padded_mask: torch.Tensor,  # [n_rows, n_tokens_across_chains] (bool)
    n_tokens: int,  # Number of relevant tokens when one-hot encoding the MSA
    eps: float = 1e-6,  # Small value to avoid division by zero
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Summarizes extra MSA clusters (rows from the MSA that were not selected as cluster representatives) by calculating cluster profiles and insertion statistics.
    - Cluster profile: The average one-hot encoded representation of the tokens in the cluster (including the cluster representative), weighted by the number of valid (non-masked) residues at each position.
    - Insertion statistics: The average number of insertions at each position in the cluster (including the cluster representative), weighted by the number of valid (non-masked) residues at each position.

    Args:
        encoded_msa (torch.Tensor): Tensor [n_rows, n_tokens_across_chains] representing the encoded MSA as token integers.
        msa_raw_ins (torch.Tensor): Tensor [n_rows, n_tokens_across_chains] representing raw insertion counts in the MSA.
        mask_position (torch.Tensor): Boolean tensor [n_rows, n_tokens_across_chains] indicating positions where the BERT-style mask was applied.
        assignments (torch.Tensor): Integer tensor [n_not_selected_rows] indicating the assignment of each extra row to the closest cluster representative.
        selected_indices (torch.Tensor): Integer tensor [n_msa_cluster_representatives] indicating indices of cluster representatives in the MSA.
        not_selected_indices (torch.Tensor): Integer tensor [n_not_selected_rows] indicating indices of extra rows in the MSA.
        msa_is_padded_mask (torch.Tensor): Boolean tensor [n_rows, n_tokens_across_chains] indicating padded positions in the MSA.
        encoding (TokenEncoding): Encoding object with `n_tokens`.
        eps (float): Small value to avoid division by zero. Default is 1e-6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - msa_cluster_profiles (torch.Tensor): Tensor [n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] representing the cluster profiles
            - msa_cluster_ins (torch.Tensor): Tensor [n_msa_cluster_representatives, n_tokens_across_chains] representing the insertion statistics (mean insertions at this position)

    Examples:
        See the test cases in `test_featurize_msa`.

    Reference:
        `AlphaFold2 data_transforms.py <https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/model/tf/data_transforms.py#L292>`_
    """
    n_clust = selected_indices.shape[0]
    n_rows, n_seq = encoded_msa.shape
    n_extra = not_selected_indices.shape[0]
    assert n_rows == n_clust + n_extra, f"Expected n_rows == n_clust + n_extra, got {n_rows} != {n_clust} + {n_extra}"

    # Mask (a) where we applied the BERT mask and (b) where we have padding from unpaired sequences
    # Thus, a "True" value  in `inclusion_mask` indicates that the position should be included in the cluster profile calculations
    is_valid = ~(mask_position | msa_is_padded_mask)  # [n_rows, n_tokens_across_chains] (bool)

    # Get all assignments (including the cluster representatives)
    # ... map selected assignments to cluster index range `0, ... , n_clust`
    row_to_clust_idx = torch.empty(n_rows, dtype=assignments.dtype, device=assignments.device)
    row_to_clust_idx[selected_indices] = torch.arange(n_clust, dtype=assignments.dtype, device=assignments.device)
    row_to_clust_idx[not_selected_indices] = assignments  # [n_rows] (int)

    # ----------- Cluster profiles -----------
    # Compute cluster profile by counting the token occurrences in each cluster at each position
    clust_stats = grouped_count(
        encoded_msa,
        mask=is_valid,
        groups=[row_to_clust_idx, torch.arange(n_seq, dtype=row_to_clust_idx.dtype, device=row_to_clust_idx.device)],
        n_tokens=n_tokens,
        dtype=torch.float32,
    )  # [n_clust, n_seq, n_tokens] (float)
    # ... normalize into categorical probabilities
    num_clust_per_pos = clust_stats.sum(dim=-1)  # [n_clust, n_seq] (float)
    clust_stats /= num_clust_per_pos.unsqueeze(-1) + eps  # [n_clust, n_seq, n_tokens] (float)

    # ----------- Insertion statistics -----------
    # Count the number of insertions at each position in each cluster, including the cluster representatives
    ins_mean = grouped_sum(
        msa_raw_ins * is_valid, assignment=row_to_clust_idx, num_groups=n_clust
    )  # [n_clust, n_seq, n_tokens] (float)
    # ... normalize
    ins_mean /= num_clust_per_pos + eps  # [n_clust, n_seq] (float)

    return clust_stats, ins_mean
