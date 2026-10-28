"""Synthetic multi-layer EXRs for the tests: beauty, depth, position and a two-object cryptomatte.

Layout (64 x 48): the left half is a "cube" 2 units away, the right half a "sphere" 8 units away whose edge
pixels also hold 25% cube coverage in the second cryptomatte rank. The top row is empty background.
"""
import json
import os
import sys

import numpy as np
import OpenImageIO as oiio
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sf_exr.masks import hex_to_float, name_to_hex  # noqa: E402

W, H = 64, 48
NAMES = ["cube", "sphere"]


def scene() -> tuple[list[str], list, np.ndarray, list]:
    left = np.zeros((H, W), bool)
    left[:, : W // 2] = True
    bg = np.zeros((H, W), bool)
    bg[0, :] = True

    beauty = np.zeros((H, W, 4), np.float32)
    beauty[..., 0] = np.linspace(0, 4, W)[None, :]  # HDR ramp up to 4.0
    beauty[..., 1] = 0.5
    beauty[..., 3] = 1.0

    depth = np.where(left, 2.0, 8.0).astype(np.float32)
    depth[bg] = 1e10

    ys, xs = np.mgrid[0:H, 0:W]
    pos = np.stack([xs * 0.1, ys * 0.1, np.where(left, 1.0, 3.0)], -1).astype(np.float32)
    pos[bg] = 0.0

    cube, sphere = hex_to_float(name_to_hex("cube")), hex_to_float(name_to_hex("sphere"))
    rank0 = np.zeros((H, W, 4), np.float32)
    rank0[..., 0] = np.where(left, cube, sphere)
    rank0[..., 1] = np.where(left, 1.0, 0.75)
    rank0[..., 2] = np.where(left, 0.0, cube)
    rank0[..., 3] = np.where(left, 0.0, 0.25)
    rank0[bg] = 0.0
    rank1 = np.zeros((H, W, 4), np.float32)

    chans, types, data = [], [], []
    for name, sufs, arr, t in (
        ("", ["R", "G", "B", "A"], beauty, oiio.HALF),
        ("RL.CryptoObject00", ["r", "g", "b", "a"], rank0, oiio.FLOAT),
        ("RL.CryptoObject01", ["r", "g", "b", "a"], rank1, oiio.FLOAT),
        ("RL.Depth", ["Z"], depth[..., None], oiio.FLOAT),
        ("RL.Position", ["X", "Y", "Z"], pos, oiio.FLOAT),
    ):
        chans += [f"{name}.{s}" if name else s for s in sufs]
        types += [t] * len(sufs)
        data.append(arr)
    manifest = {n: name_to_hex(n) for n in NAMES}
    attribs = [("cryptomatte/abc1234/name", "RL.CryptoObject"), ("cryptomatte/abc1234/hash", "MurmurHash3_32"),
               ("cryptomatte/abc1234/conversion", "uint32_to_float32"),
               ("cryptomatte/abc1234/manifest", json.dumps(manifest))]
    return chans, types, np.concatenate(data, -1), attribs


def write_scene(path: str) -> str:
    chans, types, pixels, attribs = scene()
    spec = oiio.ImageSpec(W, H, len(chans), oiio.FLOAT)
    spec.channelnames = tuple(chans)
    spec.channelformats = tuple(types)
    spec.attribute("compression", "zip")
    for k, v in attribs:
        spec.attribute(k, v)
    out = oiio.ImageOutput.create(path)
    assert out.open(path, spec), oiio.geterror()
    assert out.write_image(np.ascontiguousarray(pixels))
    out.close()
    return path


@pytest.fixture
def exr_file(tmp_path) -> str:
    return write_scene(str(tmp_path / "shot.exr"))


@pytest.fixture
def exr_sequence(tmp_path) -> str:
    """Frames 1001, 1002 and 1004 (1003 missing)."""
    for f in (1001, 1002, 1004):
        write_scene(str(tmp_path / f"shot.{f}.exr"))
    return str(tmp_path / "shot.1001.exr")


@pytest.fixture
def stack(exr_file):
    from sf_exr import sequence
    from sf_exr.stack import ExrStack
    return ExrStack.from_sequence(sequence.detect(exr_file))
