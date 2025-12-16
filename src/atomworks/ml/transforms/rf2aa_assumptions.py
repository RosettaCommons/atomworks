from typing import Any

import numpy as np
import torch

from atomworks.ml.transforms.base import Transform


# Helper functions
def _is_atom(seq: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    return seq > 32


def _is_block_diagonal_with_full_blocks(array: torch.Tensor | np.ndarray) -> bool:
    # NOTE: This only
    # Check that matrix is 2D
    assert len(array.shape) == 2

    # Check that matrix is square
    n_rows, n_cols = array.shape
    assert n_rows == n_cols

    # Get occupied entries
    occupied = np.asarray(array != 0)

    # Check 1: A necessary condition for a full block diagonal structure is that
    #  the matrix retains it's occupancy pattern when squared.
    if not np.allclose(occupied, np.linalg.matrix_power(occupied, 2)):
        return False

    # Check 2: Explicitly go over the matrix and check that it is block diagonal
    start = 0
    end = np.where(occupied[0])[0][-1] + 1
    is_block_diagonal = True
    for row in occupied:
        if row[start] == 0:
            # case: jump to new block - increment block start & end
            start = end
            end = np.where(row)[0][-1] + 1

        # validate block structure
        is_block_diagonal &= (row[start:end] != 0).all()
        is_block_diagonal &= (start == 0) or (row[:start] == 0).all()
        is_block_diagonal &= (end == n_cols) or (row[end:] == 0).all()

        if not is_block_diagonal:
            # ... early stop
            return False

    # Create a mask for all occupied entries:
    return is_block_diagonal


def _are_all_blocks_the_same_size(array: torch.Tensor | np.ndarray) -> bool:
    assert _is_block_diagonal_with_full_blocks(array)
    block_diffs = np.unique(np.where(array.diff())[1]) + 1
    block_starts_ends = np.concatenate([[0], block_diffs, [array.shape[0]]])
    block_sizes = block_starts_ends[1:] - block_starts_ends[:-1]
    return np.all(block_sizes == block_sizes[0])


def _is_symmetric(array: torch.Tensor | np.ndarray) -> bool:
    # Check that matrix is symmetric
    return np.array_equal(array, array.T, equal_nan=True)


def _assert_shape(t: torch.Tensor | np.ndarray, s: tuple[int, ...]) -> None:
    assert tuple(t.shape) == s


def assert_satisfies_rf2aa_assumptions(sample: dict[str, Any]) -> None:
    """
    Asserts that the given sample satisfies the assumptions required for a
    successful forward and backward pass through RF2AA.
    """
    # Find out if there is a batch dimension:
    if sample["seq"].ndim == 3:
        # ... we have a batch dimension -- remove it
        sample = {k: v[0] for k, v in sample.items()}
    else:
        assert sample["seq"].ndim == 2, f"seq must have 2 or 3 dimensions, but has {sample['seq'].ndim}."

    # Extract the data
    seq = sample["seq"]
    msa = sample["msa"]
    msa_masked = sample["msa_masked"]
    msa_full = sample["msa_full"]
    mask_msa = sample["mask_msa"]
    true_crds = sample["xyz"]
    mask_crds = sample["mask"]
    idx_pdb = sample["idx_pdb"]
    xyz_t = sample["xyz_t"]
    t1d = sample["t1d"]
    mask_t = sample["mask_t"]
    xyz_prev = sample["xyz_prev"]
    mask_prev = sample["mask_prev"]
    same_chain = sample["same_chain"]
    unclamp = sample["unclamp"]
    negative = sample["negative"]
    atom_frames = sample["atom_frames"]
    bond_feats = sample["bond_feats"]
    dist_matrix = sample["dist_matrix"]
    chirals = sample["chirals"]
    ch_label = sample["ch_label"]
    symmgp = sample["symmgp"]
    task = sample["task"]
    item = sample["example_id"]

    # Check basic types
    assert isinstance(unclamp.item(), bool)
    assert isinstance(negative.item(), bool)
    assert symmgp == "C1", f"{item}: Got unexpected symmgp: {symmgp}"
    assert isinstance(task, str)
    assert isinstance(item, str)

    # Check basic shapes
    n_total = 36  # ... number of atoms per token

    n_recycles, N, L = msa.shape[:3]
    num_atoms = (_is_atom(seq[0]).sum()).item()
    _assert_shape(seq, (n_recycles, L))
    _assert_shape(msa, (n_recycles, N, L))
    _assert_shape(msa_masked, (n_recycles, N, L, 164))
    n_full = msa_full.shape[1]
    assert n_full > 0, f"{item}: n_full is {n_full}. But at least the query sequence should be present."
    _assert_shape(msa_full, (n_recycles, n_full, L, 83))
    _assert_shape(mask_msa, (n_recycles, N, L))
    n_symm = true_crds.shape[0]
    _assert_shape(true_crds, (n_symm, L, n_total, 3))
    _assert_shape(mask_crds, (n_symm, L, n_total))
    _assert_shape(idx_pdb, (L,))
    n_templ = xyz_t.shape[0]
    _assert_shape(xyz_t, (n_templ, L, n_total, 3))
    _assert_shape(t1d, (n_templ, L, 80))
    _assert_shape(mask_t, (n_templ, L, n_total))
    _assert_shape(xyz_prev, (L, n_total, 3))
    _assert_shape(mask_prev, (L, n_total))
    _assert_shape(same_chain, (L, L))
    _assert_shape(atom_frames, (num_atoms, 3, 2))
    _assert_shape(bond_feats, (L, L))
    _assert_shape(dist_matrix, (L, L))
    n_chirals = chirals.shape[0]
    _assert_shape(chirals, (n_chirals, 5))
    _assert_shape(ch_label, (L,))
    assert symmgp == "C1", f"{symmgp}"

    # Assert that the masking works correctly
    assert not true_crds[mask_crds].isnan().any()
    assert not xyz_t[mask_t].isnan().any()
    assert not xyz_prev[mask_prev].isnan().any()

    # Check 2D matrices are symmetric
    assert _is_symmetric(same_chain), f"{item}: same_chain is not symmetric"
    assert _is_symmetric(bond_feats), f"{item}: bond_feats is not symmetric"
    assert _is_symmetric(dist_matrix), f"{item}: dist_matrix is not symmetric"

    # Assert that the correspondence between chains is the same in ch_label and same_chain is valid
    ch_label_diffs = np.where(ch_label.diff())[0]
    same_chain_diffs = np.unique(np.where(same_chain.diff())[1])
    assert np.all(
        np.isin(ch_label_diffs, same_chain_diffs)
    ), f"{item}: ch_label_diffs: {ch_label_diffs}, same_chain_diffs: {same_chain_diffs}"

    # Assert that there are polymer tokens in the example:
    num_res_tokens = ((~_is_atom(seq[0])).sum()).item()
    assert (
        num_res_tokens > 0
    ), f"{item}: num_res_tokens: {num_res_tokens}. No polymer tokens at all. This would lead RF2AA to crash."
    assert num_res_tokens + num_atoms == L, f"{item}: num_res_tokens: {num_res_tokens}, num_atoms: {num_atoms}, L: {L}"

    if num_atoms > 0:
        # Assert that `same_chain` is block diagonal in the non-poly sector:
        assert _is_block_diagonal_with_full_blocks(
            same_chain[num_res_tokens:, num_res_tokens:]
        ), f"{item}: non-poly sector of `same_chain` is not block diagonal"

        # Assert that in the non-poly sector,
        for label in np.unique(ch_label[num_res_tokens:]):
            # ...all blocks where `ch_label` is the same are the same size:
            idxs = np.where(ch_label[num_res_tokens:] == label)[0]

            # NOTE: This will currently fail (on purpose) for cropped covalent modifications,
            #       where this assumption cannot be guaranteed with an AF3 like cropping strategy.
            same_chain_block = same_chain[num_res_tokens:, num_res_tokens:][np.ix_(idxs, idxs)]
            assert _are_all_blocks_the_same_size(
                same_chain_block
            ), f"{item}: `same_chain` block {label} is not the same size"

            # ... ensure there is no entirely unresolved `ch_label` segment in
            #     the non-poly sector:
            assert (
                mask_crds[0, idxs + num_res_tokens, :]
            ).any(), f"{item}: Entity with `chain_label` {label} is entirely unresolved in the non-poly sector."

            # ... ensure there is at least one resolved coordinate for each chain in each entity
            _block_size = same_chain_block[0].sum()
            assert len(idxs) % _block_size == 0
            for chain_idx in range(len(idxs) // _block_size):
                idxs_in_subblock = idxs[chain_idx * _block_size : (chain_idx + 1) * _block_size]
                assert (
                    mask_crds[0, idxs_in_subblock + num_res_tokens, :]
                ).any(), f"{item}: Chain {chain_idx} in block with `chain_label` {label} has no resolved coordinates in the non-poly sector."

    # Assert that there are no masks in `msa`:
    assert not (msa == 21).any(), f"{item}: There are masks in the ground truth `msa`."

    # Assert that there are no entirely unresolved chains in the poly sector:
    for label in np.unique(ch_label[:num_res_tokens]):
        idxs = np.where(ch_label[:num_res_tokens] == label)[0]
        assert (
            mask_crds[0, idxs, :]
        ).any(), f"{item}: Entity with `chain_label` {label} is entirely unresolved in the poly sector."

    # Ensure there is at least one resolved coordinate for each symmetry copy:
    #  mask_crds: (N_symm, L, NTOTAL)
    assert mask_crds.any(
        dim=(1, 2)
    ).all(), f"{item}: There are no resolved coordinates for at least one symmetry copy (neither poly nor non-poly)."

    # Ensure there is at least one resolved coordinate for each symmetry copy in the poly sector (excluding padding):
    n_symm_poly = mask_crds[:, :num_res_tokens].any(dim=(1, 2)).max().item()
    assert (
        n_symm_poly > 0
    ), f"{item}: There are no resolved coordinates for the poly sector of at least one symmetry copy (excluding padding)."

    # If the given symmetry copy has a poly swap, check that the N-CA-C of at least one residue is resolved, which
    #  is needed to construct the poly frames.
    symm_copies_has_at_least_one_resolved_N_CA_C = (mask_crds[:n_symm_poly, :num_res_tokens, :3].sum(dim=2) == 3).any(  # noqa: N806
        dim=1
    )
    problems = np.where(~symm_copies_has_at_least_one_resolved_N_CA_C)[0]
    assert (
        len(problems) == 0
    ), f"{item}: The following symmetry copies have no resolved N-CA-C coordinates for the poly sector: {problems}"


class AssertRF2AAAssumptions(Transform):
    """
    Assert that the given sample satisfies the assumptions required for a
    successful forward and backward pass through RF2AA.
    """

    def check_input(self, data: dict[str, Any]) -> None:
        # No specific input checks needed as the assertion function handles it
        pass

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        assert_satisfies_rf2aa_assumptions(data["feats"])
        return data
