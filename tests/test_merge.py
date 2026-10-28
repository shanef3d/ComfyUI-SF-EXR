import pytest
import torch
from conftest import H, W, write_scene

from sf_exr import sequence
from sf_exr.merge import OPERATIONS, bbox_mask, merge
from sf_exr.stack import ExrStack, frame_bboxes


def px(*rgba):
    """One premultiplied pixel as ([1,1,1,3] colour, [1,1,1] alpha)."""
    t = torch.tensor(rgba, dtype=torch.float32).view(1, 1, 1, 4)
    return t[..., :3], t[..., 3]


A, a = px(0.4, 0.2, 0.0, 0.5)    # half-transparent foreground (premultiplied)
B, b = px(0.2, 0.6, 1.0, 1.0)    # opaque background


@pytest.mark.parametrize("op,rgb,alpha", [
    ("over", [0.5, 0.5, 0.5], 1.0),         # A + B(1-a)
    ("under", [0.2, 0.6, 1.0], 1.0),        # A(1-b) + B, b = 1
    ("plus", [0.6, 0.8, 1.0], 1.5),
    ("minus", [0.2, -0.4, -1.0], -0.5),
    ("from", [-0.2, 0.4, 1.0], 0.5),
    ("multiply", [0.08, 0.12, 0.0], 0.5),
    ("in", [0.4, 0.2, 0.0], 0.5),
    ("out", [0.0, 0.0, 0.0], 0.0),
    ("mask", [0.1, 0.3, 0.5], 0.5),
    ("stencil", [0.1, 0.3, 0.5], 0.5),
    ("copy", [0.4, 0.2, 0.0], 0.5),
    ("max", [0.4, 0.6, 1.0], 1.0),
    ("min", [0.2, 0.2, 0.0], 0.5),
    ("difference", [0.2, 0.4, 1.0], 0.5),
])
def test_operations_match_nuke(op, rgb, alpha):
    out, oa = merge(A, a, B, b, op)
    assert torch.allclose(out[0, 0, 0], torch.tensor(rgb), atol=1e-6)
    assert torch.allclose(oa[0, 0, 0], torch.tensor(alpha), atol=1e-6)


def test_screen_falls_back_to_max_for_hdr():
    hdr, ha = px(3.0, 0.5, 0.5, 1.0)
    out, _ = merge(hdr, ha, B, b, "screen")
    assert torch.allclose(out[0, 0, 0], torch.tensor([3.0, 0.8, 1.0]))


def test_divide_by_zero_is_zero():
    zero, za = px(0.0, 0.0, 0.0, 1.0)
    out, _ = merge(A, a, zero, za, "divide")
    assert torch.equal(out, torch.zeros_like(out))


def test_opacity_and_mask_blend_back_to_b():
    half, _ = merge(A, a, B, b, "plus", opacity=0.5)
    assert torch.allclose(half[0, 0, 0], torch.tensor([0.4, 0.7, 1.0]))
    masked, ma = merge(A, a, B, b, "plus", opacity=1.0, mask=torch.zeros(1, 1, 1))
    assert torch.equal(masked, B) and torch.equal(ma, b)


def test_full_opacity_is_exact():
    A4, B4 = torch.rand(1, 8, 8, 3), torch.rand(1, 8, 8, 3)
    ones = torch.ones(1, 8, 8)
    rgb, _ = merge(A4, ones, B4, ones, "multiply")
    assert torch.equal(rgb, A4 * B4)
    rgb, _ = merge(A4, ones, B4, ones, "copy")
    assert torch.equal(rgb, A4)


@pytest.mark.parametrize("mode,area", [("union", 40 * 30), ("intersection", 10 * 10),
                                       ("A", 20 * 20), ("B", 30 * 20)])
def test_bbox_modes(mode, area):
    box_a, box_b = (0, 0, 20, 20), (10, 10, 40, 30)
    keep = bbox_mask([box_a], [box_b], mode, 30, 40)
    assert int(keep.sum()) == area


def test_bbox_from_data_window(tmp_path):
    cropped = write_scene(str(tmp_path / "crop.exr"), window=(8, 6, 20, 10))
    full = write_scene(str(tmp_path / "full.exr"))
    st_crop = ExrStack.from_sequence(sequence.detect(cropped, "single"))
    st_full = ExrStack.from_sequence(sequence.detect(full, "single"))
    assert frame_bboxes(st_crop) == [(8, 6, 28, 16)]
    assert frame_bboxes(st_full) == [(0, 0, W, H)]
    assert frame_bboxes(st_crop, W * 2, H * 2) == [(16, 12, 56, 32)]  # scaled to another resolution


def test_every_operation_runs_on_batches():
    A4, B4 = torch.rand(2, 8, 8, 3), torch.rand(2, 8, 8, 3)
    a4, b4 = torch.rand(2, 8, 8), torch.rand(2, 8, 8)
    for op in OPERATIONS:
        rgb, alpha = merge(A4, a4, B4, b4, op, 0.7, torch.rand(2, 8, 8))
        assert rgb.shape == (2, 8, 8, 3) and alpha.shape == (2, 8, 8), op
        assert torch.isfinite(rgb).all(), op
