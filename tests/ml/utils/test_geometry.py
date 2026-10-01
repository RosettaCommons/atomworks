import numpy as np
import pytest
import torch

from atomworks.ml.utils.geometry import (
    apply_inverse_rigid,
    apply_rigid,
    compose_rigids,
    get_random_rigid,
    get_random_rots,
    get_torch_eps,
    invert_rigid,
    rigid_from_3_points,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_get_torch_eps():
    assert get_torch_eps(torch.float32) == torch.finfo(torch.float32).eps
    assert get_torch_eps(torch.float64) == torch.finfo(torch.float64).eps


def test_rigid_from_3_points_random(device):
    batch_size = 10
    x1 = torch.randn(batch_size, 3, device=device)
    x2 = torch.randn(batch_size, 3, device=device)
    x3 = torch.randn(batch_size, 3, device=device)

    rot, t = rigid_from_3_points(x1, x2, x3)

    # Check if R is a valid rotation matrix
    assert torch.allclose(torch.det(rot), torch.ones(batch_size, device=device), atol=1e-6)
    assert torch.allclose(
        torch.matmul(rot, rot.transpose(-1, -2)),
        torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
        atol=1e-4,
    )

    # Check if the transformation preserves the relative positions of the points
    transformed_x1 = apply_inverse_rigid((rot, t), x1)
    transformed_x2 = apply_inverse_rigid((rot, t), x2)
    transformed_x3 = apply_inverse_rigid((rot, t), x3)

    # Check that x2 is at the origin of the local frames
    assert torch.allclose(transformed_x2, torch.zeros_like(transformed_x2), atol=1e-5)

    # Check that x1 is in the x-y plane of the local frames
    assert torch.allclose(transformed_x1[:, 2], torch.zeros_like(transformed_x1[:, 2]), atol=1e-5)

    # Check that x3 is along the x-axis of the local frames
    assert torch.allclose(transformed_x3[:, 1:], torch.zeros_like(transformed_x3[:, 1:]), atol=1e-5)
    assert torch.all(transformed_x3[:, 0] > 0)

    # Check that the distances between points are preserved
    assert torch.allclose(torch.norm(transformed_x1 - transformed_x2, dim=-1), torch.norm(x1 - x2, dim=-1), atol=1e-4)
    assert torch.allclose(torch.norm(transformed_x1 - transformed_x3, dim=-1), torch.norm(x1 - x3, dim=-1), atol=1e-4)
    assert torch.allclose(torch.norm(transformed_x2 - transformed_x3, dim=-1), torch.norm(x2 - x3, dim=-1), atol=1e-4)


def test_apply_rigid_and_inverse(device):
    num_points = 5
    rot, t = get_random_rigid(1, device=device)
    points = torch.randn(num_points, 3, device=device)

    # Test: apply_inverse_rigid(apply_rigid(points)) = points
    transformed = apply_rigid((rot, t), points)
    inv_transformed = apply_inverse_rigid((rot, t), transformed)

    assert torch.allclose(inv_transformed, points, atol=1e-6)


def test_compose_and_invert_rigids(device):
    batch_size = 10
    rot1, t1 = get_random_rigid(batch_size, device=device)
    rot2, t2 = get_random_rigid(batch_size, device=device)

    # Test: inverse(compose(rot1,rot2)) = compose(inverse(rot2), inverse(rot1))
    composed = compose_rigids((rot1, t1), (rot2, t2))
    inv_composed = invert_rigid(composed)

    inv_rot2, inv_t2 = invert_rigid((rot2, t2))
    inv_rot1, inv_t1 = invert_rigid((rot1, t1))
    composed_inv = compose_rigids((inv_rot2, inv_t2), (inv_rot1, inv_t1))

    assert torch.allclose(inv_composed[0], composed_inv[0], atol=1e-6)
    assert torch.allclose(inv_composed[1], composed_inv[1], atol=1e-6)

    # Test: compose(rot, inverse(rot)) = identity
    identity_composed_rot, identity_composed_t = compose_rigids((rot1, t1), invert_rigid((rot1, t1)))

    identity_rot = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    identity_t = torch.zeros(batch_size, 3, device=device)

    assert torch.allclose(identity_composed_rot, identity_rot, atol=1e-6)
    assert torch.allclose(identity_composed_t, identity_t, atol=1e-6)


def test_get_random_rots_and_rigids(device):
    batch_size = 5
    scale = 2.0

    # Test random rotations
    rot = get_random_rots(batch_size, device=device)

    assert rot.shape == (batch_size, 3, 3)
    assert torch.allclose(torch.det(rot), torch.ones(batch_size, device=device), atol=1e-6)
    assert torch.allclose(
        torch.matmul(rot, rot.transpose(-1, -2)),
        torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
        atol=1e-6,
    )

    # Test random rigid transformations
    rot, t = get_random_rigid(batch_size, scale=scale, device=device)

    assert rot.shape == (batch_size, 3, 3)
    assert t.shape == (batch_size, 3)
    assert torch.allclose(torch.det(rot), torch.ones(batch_size, device=device), atol=1e-6)

    # Test single random rigid transformation
    r_single, t_single = get_random_rigid(1, device=device)

    assert r_single.shape == (3, 3)
    assert t_single.shape == (3,)
    assert torch.allclose(torch.det(r_single), torch.tensor(1.0, device=device), atol=1e-6)


@pytest.mark.parametrize("mode", ["numpy", "torch"])
def test_get_random_rots_is_uniform(mode, device):
    batch_size = 8192
    numpy_state = np.random.get_state()
    try:
        np.random.seed(42)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(42)
            rotations = get_random_rots(batch_size, mode=mode, device=device)
    finally:
        np.random.set_state(numpy_state)

    if isinstance(rotations, torch.Tensor):
        rotations = rotations.cpu().numpy()

    identity = np.broadcast_to(np.eye(3), rotations.shape)
    np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), identity, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-6)
    np.testing.assert_allclose(rotations.mean(axis=0), 0.0, atol=0.03)
    np.testing.assert_allclose((rotations**2).mean(axis=0), 1.0 / 3.0, atol=0.03)
