"""Various geometry utility functions to deal with rigid body transformations in 3D."""

import numpy as np
import torch
from biotite.structure import AtomArray, rmsd, superimpose
from einops import einsum, rearrange
from torch.nn.functional import normalize

from atomworks.common import default


def get_torch_eps(dtype: torch.dtype) -> float:
    """Get the smallest positive representable value for a given torch dtype."""
    return torch.finfo(dtype).eps


def rigid_from_3_points(
    x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, eps: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rigid body transformation (R, t) that leads from the origin into the local frame
    via the Gram-Schmidt process.

    The local frame is centered at x2 with the x-axis pointing towards x3, the y-axis in the plane
    defined by x1, x2, and x3, and the z-axis perpendicular to this plane.

    E.g. if x1=N, x2=CA, x3=C, then the x-axis is the vector pointing CA -> C, the y-axis
    is in the N-CA-C plane and the z-axis is perpendicular to this plane.

    Args:
        x1: torch.Tensor of shape [..., 3], coordinates of the first point
        x2: torch.Tensor of shape [..., 3], coordinates of the second point (origin of local frame)
        x3: torch.Tensor of shape [..., 3], coordinates of the third point
        eps: float, small value to avoid division by zero

    Returns:
        R: torch.Tensor of shape [..., 3, 3], rotation matrix
        t: torch.Tensor of shape [..., 3], translation vector

    Reference:
        `AF2 supplementary, Algorithm 21 <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_

    Example:
        >>> x1 = torch.tensor([0.0, 0.0, 1.0])
        >>> x2 = torch.tensor([0.0, 0.0, 0.0])
        >>> x3 = torch.tensor([1.0, 0.0, 0.0])
        >>> R, t = rigid_from_3_points(x1, x2, x3)
        >>> print(R)
        tensor([[ 1., 0., 0.],
                [ 0., 0.,-1.],
                [ 0., 1., 0.]])
        >>> print(t)
        tensor([0., 0., 0.])
    """
    eps = default(eps, get_torch_eps(x1.dtype))

    # Compute the x-axis of the local frame (pointing from x2 to x3)
    x_axis = x3 - x2
    x_axis = normalize(x_axis, dim=-1, eps=eps)

    # Compute the y-axis of the local frame (in the plane defined by x1, x2, x3)
    xy_vec = x1 - x2
    y_axis = xy_vec - x_axis * torch.sum(x_axis * xy_vec, dim=-1, keepdim=True)
    y_axis = normalize(y_axis, dim=-1, eps=eps)

    # Compute the z-axis as the cross product of x_axis and y_axis
    #  (normalized & right-handed as a result)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)

    # Construct the rotation matrix
    rots = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    # The translation vector is simply x2
    trans = x2

    return rots, trans


def apply_rigid(
    rigid: tuple[torch.Tensor, torch.Tensor],
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a rigid body transformation to a set of points via (p -> R @ p + t).
    (i.e. first rotate then translate)

    Args:
        - rigid (tuple[torch.Tensor, torch.Tensor]): A tuple containing the rotation matrix (R) and
          translation vector (t) representing the rigid body transformation.
        - points (torch.Tensor): A tensor of shape [..., 3] representing the points to transform.

    Returns:
        - torch.Tensor: A tensor of shape [..., 3] representing the transformed points.

    NOTE: This transforms `p` from the local frame of the `rigid` to the global frame.
    """
    rots, trans = rigid
    return einsum(rots, points, "... i j, ... j -> ... i") + trans


def apply_batched_rigid(
    rigid: tuple[torch.Tensor, torch.Tensor],
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a batch of rigid body transformations to a set of batched points via (p -> R @ p + t).
    (i.e. first rotate then translate)

    Args:
        - rigid (tuple[torch.Tensor, torch.Tensor]): A tuple containing the rotation matrix (R) and
          translation vector (t) representing the rigid body transformation.
        - points (torch.Tensor): A tensor of shape [batch_size, ..., 3] representing the points to transform.

    Returns:
        - torch.Tensor: A tensor of shape [batch_size, ..., 3] representing the transformed points.

    NOTE: This transforms `p` from the local frame of the `rigid` to the global frame.
    """
    rots, trans = rigid
    batch, length, _ = points.shape
    assert rots.shape == (batch, 3, 3), "rotation dimension must match the points dimension"
    assert trans.shape == (batch, 3), "translation dimension must match the points dimension"
    trans = trans.unsqueeze(1).expand(-1, length, -1)
    return einsum(rots, points, "b i j, b l j -> b l i") + trans


def invert_rigid(rigid: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Invert a rigid body transformation (R, t) to (R^T, -R^T @ t).

    Args:
        - rigid (tuple[torch.Tensor, torch.Tensor]): A tuple containing the rotation matrix (R) and
          translation vector (t) representing the rigid body transformation.

    Returns:
        - tuple[torch.Tensor, torch.Tensor]: A tuple containing the inverted rotation matrix (R^T) and
          inverted translation vector (-R^T @ t).

    Example:
        >>> R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> t = torch.tensor([1, 2, 3])
        >>> R_inv, t_inv = invert_rigid((R, t))
        >>> print(R_inv)
        tensor([[ 0,  1,  0],
                [-1,  0,  0],
                [ 0,  0,  1]])
        >>> print(t_inv)
        tensor([-2,  1, -3])
    """
    rots, trans = rigid
    inv_rots = rearrange(rots, "... i j->... j i")
    inv_trans = -einsum(inv_rots, trans, "... i j, ... j->... i")
    return inv_rots, inv_trans


def compose_rigids(
    rigid1: tuple[torch.Tensor, torch.Tensor],
    rigid2: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compose two rigid body transformations (R1, t1) and (R2, t2) to (R2 @ R1, R2 @ t1 + t2).

    Args:
        - rigid1 (tuple[torch.Tensor, torch.Tensor]): First rigid body transformation (R1, t1).
        - rigid2 (tuple[torch.Tensor, torch.Tensor]): Second rigid body transformation (R2, t2).

    Returns:
        - tuple[torch.Tensor, torch.Tensor]: Composed rigid body transformation (R_composed, t_composed).

    Example:
        >>> R1, t1 = torch.eye(3), torch.tensor([1.0, 0.0, 0.0])
        >>> R2, t2 = torch.eye(3), torch.tensor([0.0, 1.0, 0.0])
        >>> R_composed, t_composed = compose_rigids((R1, t1), (R2, t2))
        >>> print(R_composed)
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
        >>> print(t_composed)
        tensor([1., 1., 0.])
    """
    rots1, trans1 = rigid1
    rots2, trans2 = rigid2

    rots_composed = einsum(rots2, rots1, "... i j, ... j k->... i k")
    trans_composed = einsum(rots2, trans1, "... i j, ... j->... i") + trans2

    return rots_composed, trans_composed


def apply_inverse_rigid(
    rigid: tuple[torch.Tensor, torch.Tensor],
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the inverse of a rigid body transformation to a set of points via (p -> R^T @ (p - t)).

    Args:
        - rigid (tuple[torch.Tensor, torch.Tensor]): A tuple containing the rotation matrix (R) and
          translation vector (t) of the rigid body transformation.
        - points (torch.Tensor): The points to transform, with shape (..., 3).

    Returns:
        - torch.Tensor: The transformed points, with the same shape as the input points.
    """
    inv_rigid = invert_rigid(rigid)
    return apply_rigid(inv_rigid, points)


def get_random_rots(batch_size: int, **tensor_kwargs) -> torch.Tensor:
    """
    Generate random 3D rotation matrices.

    Args:
        - batch_size (int): Number of rotation matrices to generate.
        - device (torch.device | None): Device to place the tensors on. Defaults to None.

    Returns:
        - torch.Tensor: Batch of random rotation matrices with shape (batch_size, 3, 3).

    Example:
        >>> R = get_random_rots(5)
        >>> print(R.shape)
        torch.Size([5, 3, 3])
        >>> print(torch.allclose(torch.det(R), torch.ones(5)))
        True
    """
    # Generate random matrices
    rand_mat = torch.randn(batch_size, 3, 3, **tensor_kwargs)

    # Compute QR decomposition
    q_decomp, _ = torch.linalg.qr(rand_mat)

    # Ensure proper rotation (determinant = 1)
    det = torch.det(q_decomp)
    q_decomp *= det.unsqueeze(-1).unsqueeze(-1).sign()

    return q_decomp


def get_random_rigid(batch_size: int, scale: float = 1.0, **tensor_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random rigid body transformations (R, t).

    Args:
        batch_size: Number of rigid transformations to generate.
        scale: Scale factor for the translation vectors. Defaults to 1.0.
        **tensor_kwargs: Additional keyword arguments to pass to tensor creation functions.

    Returns:
        A rigid tuple containing:

            - rots: Batch of random rotation matrices with shape (batch_size, 3, 3).
            - trans: Batch of random translation vectors with shape (batch_size, 3).

    Note:
        If batch_size is 1, the output tensors are squeezed to remove the batch dimension.
    """
    rots = get_random_rots(batch_size, **tensor_kwargs)
    trans = scale * torch.randn(batch_size, 3, **tensor_kwargs)
    if batch_size == 1:
        rots, trans = rots.squeeze(0), trans.squeeze(0)
    return rots, trans


def random_rigid_augmentation(coord_atom_lvl: torch.Tensor, batch_size: int, s: float = 1.0) -> torch.Tensor:
    """
    Apply random rigid body transformations to atomic coordinates.

    Generates random rigid body transformations (rotation and translation)
    for a batch of atomic coordinates and applies these transformations to the input coordinates.

    Args:
        coord_atom_lvl (torch.Tensor): A tensor containing atomic coordinates to be transformed.
                                       The shape is expected to be (batch_size, num_atoms, 3).
        batch_size (int): The number of transformations to generate and apply, corresponding to
                          the number of coordinate sets in `coord_atom_lvl`.
        s (float, optional): The translational scale in Angstrom. Random translations will be drawn from N(0, s), i.e. with standard deviation `s`. The rotational degree of freedom is sampled uniformly random.
                             Defaults to 1.0.

    Returns:
        torch.Tensor: A tensor of the same shape as `coord_atom_lvl`, containing the transformed
                      atomic coordinates.
    """
    rigid = get_random_rigid(batch_size, scale=s)

    # (`get_random_rigid` squeezes dimension for batch_size=1)
    if batch_size == 1:
        rigid = rigid[0].unsqueeze(0), rigid[1].unsqueeze(0)

    return apply_batched_rigid(rigid, coord_atom_lvl)


def masked_center(
    coord_atom_lvl: np.ndarray | torch.Tensor, mask_atom_lvl: np.ndarray | torch.Tensor = None
) -> np.ndarray | torch.Tensor:
    """Center the coordinates of the atoms in coord_atom_lvl around the origin using the mask mask_atom_lvl.

    Supports both NumPy and PyTorch tensors.
    """
    if mask_atom_lvl is None:
        mask_atom_lvl = (
            np.ones(coord_atom_lvl.shape[0], dtype=bool)
            if isinstance(coord_atom_lvl, np.ndarray)
            else torch.ones(coord_atom_lvl.shape[0], dtype=torch.bool)
        )

    atoms = coord_atom_lvl[mask_atom_lvl]
    center = atoms.mean(axis=0) if isinstance(coord_atom_lvl, np.ndarray) else atoms.mean(dim=0)
    coord_atom_lvl = coord_atom_lvl - center

    return coord_atom_lvl


def align_atom_arrays(mbl_sele: AtomArray, tgt_sele: AtomArray, mbl_full: AtomArray) -> tuple[AtomArray, float]:
    """
    Computes the transformation that aligns mbl_sele to tgt_sele,
    then applies that transformation to mbl_full and returns it along with aligment rmsd

    Args:
        mbl_sele (AtomArray): An atom array containing atomic coordinates of the array to
                              be transformed, pre-masked to contain only the portion to be aligned.
        tgt_sele (AtomArray): An atom array containing coordinates for mbl_sele to be aligned to.
                              Must be the same size as mbl_sele; should be the same residues / molecules.
        mbl_full (AtomArray): The full atom array to be transformed based on the alignment between
                              mbl_sele and tgt_sle.

    Returns:
        AtomArray: an atom array of the same shape as mbl_full, containing the transformed coordinates.
        float: the RMSD between mbl_sele and tgt_sele following alignment.
    """
    mbl_fitted, xform = superimpose(tgt_sele, mbl_sele)
    mbl_full_xformed = xform.apply(mbl_full)
    return mbl_full_xformed, rmsd(mbl_fitted, tgt_sele)
