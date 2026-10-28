import numpy as np
import pytest
import torch
from conftest import H, W

from sf_exr import masks
from sf_exr.stack import pick_layer


def crypto(stack, names):
    group = stack.header.crypto["RL.CryptoObject"]
    ids = masks.resolve_names(group["manifest"], names)
    return masks.crypto_mask([stack.get(n) for n in group["layers"]], list(ids.values()))[0]


def test_name_hash_matches_blender_manifest():
    # "head" -> 95a53fc9 is taken from a real Blender cryptomatte manifest
    assert masks.name_to_hex("head") == "95a53fc9"


def test_crypto_coverage_sums_across_ranks(stack):
    cube = crypto(stack, "cube")
    assert torch.allclose(cube[1:, : W // 2], torch.ones(H - 1, W // 2))
    assert torch.allclose(cube[1:, W // 2:], torch.full((H - 1, W // 2), 0.25))
    sphere = crypto(stack, "sphere")
    assert torch.allclose(sphere[1:, W // 2:], torch.full((H - 1, W // 2), 0.75))
    assert float(sphere[0].max()) == 0.0  # empty background row


def test_crypto_wildcards_and_unknown_names(stack):
    manifest = stack.header.crypto["RL.CryptoObject"]["manifest"]
    assert set(masks.resolve_names(manifest, "s*")) == {"sphere"}
    assert set(masks.resolve_names(manifest, "CUBE, sphere")) == {"cube", "sphere"}
    assert masks.resolve_names(manifest, "notInManifest") == {"notInManifest": masks.name_to_hex("notInManifest")}


def test_names_at_pixel(stack):
    group = stack.header.crypto["RL.CryptoObject"]
    ranks = [stack.get(n) for n in group["layers"]]
    assert masks.names_at(ranks, group["manifest"], 0, W - 1, H - 1) == [("sphere", 0.75), ("cube", 0.25)]


def test_depth_band_hard_and_soft(stack):
    depth = stack.get(pick_layer(list(stack.layers.values()), "depth"))[..., 0]
    hard = masks.depth_mask(depth, 1.5, 2.5, 0.0, normalized=False)[0]
    assert float(hard[1:, : W // 2].min()) == 1.0 and float(hard[:, W // 2:].max()) == 0.0
    assert float(hard[0].max()) == 0.0  # background depth (1e10) is never selected
    soft = masks.depth_mask(depth, 1.5, 2.5, 10.0, normalized=False)[0]
    assert 0.0 < float(soft[1, -1]) < 1.0


@pytest.mark.parametrize("shape", ["sphere", "cube", "cylinder"])
def test_position_shapes_select_inside(stack, shape):
    pos = stack.get("RL.Position")
    center = (1.0, 2.0, 1.0)  # inside the left half (z = 1)
    m = masks.position_mask(pos, shape, center, (0.55, 0.55, 0.55), 0.0, axis="Y")[0]
    inside = m.numpy() > 0.5
    assert inside[20, 10] and not inside[20, 60]
    assert not inside[0].any()  # empty background has position 0,0,0 and is ignored


def test_combine_ops():
    a, b = torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0, 0.0, 1.0])
    assert masks.combine(a, b, "add").tolist() == [1.0, 1.0, 1.0]
    assert masks.combine(a, b, "subtract").tolist() == [0.0, 1.0, 0.0]
    assert masks.combine(a, b, "intersect").tolist() == [1.0, 0.0, 0.0]
    assert np.allclose(masks.combine(None, b, "subtract").numpy(), 0.0)
