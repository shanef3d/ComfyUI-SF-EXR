import math

import numpy as np
import pytest
import torch

from sf_exr import control


def test_depth_near_is_white_and_background_black():
    z = torch.tensor([[[1.0, 2.0], [4.0, math.inf]]])
    inv = control.depth_to_control(z, 1.0, 4.0, "inverse")[..., 0]
    assert torch.allclose(inv, torch.tensor([[[1.0, 1 / 3], [0.0, 0.0]]]))
    lin = control.depth_to_control(z, 1.0, 4.0, "linear")[..., 0]
    assert torch.allclose(lin, torch.tensor([[[1.0, 2 / 3], [0.0, 0.0]]]))
    assert control.depth_to_control(z, 1.0, 4.0).shape == (1, 2, 2, 3)
    with pytest.raises(ValueError, match="greater than near"):
        control.depth_to_control(z, 4.0, 1.0)


def test_depth_range_ignores_background_and_strays():
    z = torch.full((2, 100, 100), 1e10)
    z[:, :, :50] = torch.linspace(2.0, 10.0, 50)
    z[0, 0, 0] = 0.001  # one stray near sample
    near, far = control.depth_range(z)
    assert 1.9 < near < 2.2 and 9.8 < far <= 10.0
    with pytest.raises(ValueError, match="no valid pixels"):
        control.depth_range(torch.full((1, 4, 4), math.inf))


def test_normals_camera_convention():
    n = torch.zeros(1, 1, 3, 3)
    n[0, 0, 0] = torch.tensor([0.0, 0.0, 2.0])   # facing camera (not unit length)
    n[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0])   # facing right
    out = control.normals_to_control(n)       # third pixel is background
    assert torch.allclose(out[0, 0, 0], torch.tensor([0.5, 0.5, 1.0]))
    assert torch.allclose(out[0, 0, 1], torch.tensor([1.0, 0.5, 0.5]))
    assert torch.allclose(out[0, 0, 2], torch.tensor([0.5, 0.5, 1.0]))
    up = torch.tensor([[[[0.0, 1.0, 0.2]]]])
    assert out.shape == (1, 1, 3, 3)
    assert control.normals_to_control(up, flip_y=True)[0, 0, 0, 1] < 0.5


def test_normals_from_left_handed_camera_are_turned_toward_it():
    n = torch.tensor([[[[0.0, 0.0, -1.0], [0.3, 0.0, -0.9]]]])
    out = control.normals_to_control(n)
    assert (out[..., 2] > 0.5).all()


def test_world_to_camera_rotation():
    # camera turned 90 degrees about Y: world +X becomes camera +Z (row-vector matrix, as OpenEXR stores it)
    m = np.eye(4)
    m[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    rot = control.camera_rotation([("worldToCamera", "matrix", tuple(m.flatten()))])
    n = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    out = control.normals_to_control(n, [rot])
    assert torch.allclose(out[0, 0, 0], torch.tensor([0.5, 0.5, 1.0]), atol=1e-6)
    assert control.camera_rotation([("Camera", "string", "main")]) is None


def test_id_edges_follow_object_boundaries():
    ids = torch.zeros(1, 6, 6)
    ids[:, :, 3:] = 1.5
    e = control.id_edges(ids, thickness=1)[0, ..., 0]
    assert e[:, 2:4].all() and not e[:, :2].any() and not e[:, 4:].any()
    assert control.id_edges(ids, thickness=2)[0, ..., 0][:, 1:5].all()


def test_depth_edges_mark_jumps_and_silhouettes():
    z = torch.full((1, 4, 8), 5.0)
    z[:, :, 4:] = 5.1      # small step: not an edge
    z[:, :, 6:] = 20.0     # big jump: edge
    z[:, 0, :] = math.inf  # background row: silhouette
    e = control.depth_edges(z, thickness=1)[0, ..., 0]
    assert not e[2, 3:5].any()
    assert e[2, 5] and e[2, 6]
    assert e[0].all() and e[1].all()


def test_edges_from_a_real_cryptomatte(stack):
    group = next(iter(stack.header.crypto.values()))
    ids = stack.get(group["layers"][0])[..., 0]
    e = control.id_edges(ids, thickness=1)
    assert e.shape[-1] == 3 and 0 < e[..., 0].float().mean() < 0.5
