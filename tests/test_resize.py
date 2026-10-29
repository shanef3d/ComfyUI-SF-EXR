import numpy as np
import pytest
import torch

from sf_exr import exr_io, masks, resize, sequence
from sf_exr.stack import ExrStack, frame_layers


def crypto(stack, name):
    group = next(iter(stack.header.crypto.values()))
    ranks = [stack.get(n) for n in group["layers"]]
    return masks.crypto_mask(ranks, [group["manifest"][name]])


def test_target_sizes_follow_resize_image_mask():
    t = resize.target
    assert t(1920, 1080, {"resize_type": "scale by multiplier", "multiplier": 0.5}) == (960, 540, False)
    assert t(1920, 1080, {"resize_type": "scale dimensions", "width": 1000, "height": 0, "crop": "center"}) == \
        (1000, 562, True)
    assert t(1920, 1080, {"resize_type": "scale longer dimension", "longer_size": 1000}) == (1000, 562, False)
    assert t(1080, 1920, {"resize_type": "scale shorter dimension", "shorter_size": 540}) == (540, 960, False)
    assert t(1920, 1080, {"resize_type": "scale total pixels", "megapixels": 1.0})[:2] == (1365, 768)
    assert t(1925, 1083, {"resize_type": "scale to multiple", "multiple": 8}) == (1920, 1080, True)
    assert t(64, 48, {"resize_type": "match size", "crop": "disabled"}, (32, 32)) == (32, 32, False)
    assert resize.crop_box(1920, 1080, 1000, 1000) == (420, 0, 1080, 1080)


@pytest.mark.parametrize("method", resize.METHODS)
def test_color_filters_keep_hdr(method):
    t = torch.full((1, 40, 60, 3), 50.0)
    for size in ((30, 20), (120, 80)):
        out = resize.resample(t, *size, method)
        assert out.shape == (1, size[1], size[0], 3)
        assert torch.allclose(out, torch.full_like(out, 50.0), rtol=1e-4)  # no clipping to 0-1


def test_lanczos_is_sharper_than_bilinear_on_upscale():
    t = torch.zeros(1, 8, 8, 1)
    t[:, :, 4:] = 1.0
    lz = resize.resample(t, 32, 32, "lanczos")[0, 0, :, 0]
    bl = resize.resample(t, 32, 32, "bilinear")[0, 0, :, 0]
    assert (lz[17] - lz[14]) > (bl[17] - bl[14])  # steeper step across the edge


def test_cryptomatte_downscale_keeps_mattes(stack):
    small = resize.resize_stack(stack, 32, 24, False, "area")
    for name in ("cube", "sphere"):
        want = resize.resample(crypto(stack, name).unsqueeze(-1), 32, 24, "area")[..., 0]
        got = crypto(small, name)
        assert got.shape == (1, 24, 32)
        assert (got - want).abs().max() < 1e-5, name


def test_cryptomatte_odd_downscale_and_upscale(stack):
    odd = resize.resize_stack(stack, 27, 19, False, "area")
    for name in ("cube", "sphere"):
        m = crypto(odd, name)
        assert 0.0 <= float(m.min()) and float(m.max()) <= 1.0 + 1e-6
    big = resize.resize_stack(stack, 128, 96, False, "lanczos")
    up = crypto(big, "cube")
    assert set(torch.unique(up).tolist()) <= set(torch.unique(crypto(stack, "cube")).tolist())


def test_data_layers_keep_real_values(stack):
    small = resize.resize_stack(stack, 21, 17, False, "bicubic")
    src = set(torch.unique(stack.get("RL.Depth")).tolist())
    assert set(torch.unique(small.get("RL.Depth")).tolist()) <= src  # never an in-between depth
    blended = resize.resize_stack(stack, 21, 17, False, "bilinear", "same as color")
    assert not set(torch.unique(blended.get("RL.Depth")).tolist()) <= src


def test_motion_vectors_scale_with_the_picture():
    v = torch.tensor([[[[4.0, -2.0, 1.0, 8.0]]]]).repeat(1, 4, 4, 1)
    out = resize.scale_vectors(v, 0.5, 2.0)
    assert out[0, 0, 0].tolist() == [2.0, -4.0, 0.5, 16.0]


def test_stretch_sets_pixel_aspect_and_crop_keeps_it(stack):
    stretched = resize.resize_stack(stack, 32, 48, False, "area")  # half width, same height
    par = dict((n, v) for n, _, v in stretched.header.attribs)["PixelAspectRatio"]
    assert par == pytest.approx(2.0)
    cropped = resize.resize_stack(stack, 32, 48, True, "area")
    assert dict((n, v) for n, _, v in cropped.header.attribs).get("PixelAspectRatio", 1.0) == pytest.approx(1.0)


def test_resized_stack_saves_and_reloads(stack, tmp_path):
    small = resize.resize_stack(stack, 32, 24, False, "lanczos")
    window = exr_io.output_window(small.header)
    assert window == (0, 0, 32, 24)
    path = str(tmp_path / "small.exr")
    exr_io.write_exr(path, frame_layers(small, 0, "source", window), small.header.attribs, "zip", "auto")
    back = ExrStack.from_sequence(sequence.detect(path, "single"))
    assert back.size == (24, 32)
    assert set(back.layers) == set(stack.layers)
    assert (crypto(back, "cube") - crypto(small, "cube")).abs().max() < 1e-6
    hdr = back.get(back.resolve("auto"))
    assert float(hdr[..., 0].max()) > 3.0  # the 0-4 HDR ramp survives
    assert set(torch.unique(back.get("RL.Depth")).tolist()) <= {2.0, 8.0, np.float32(1e10).item()}
