import os

import numpy as np
import pytest
import torch

from sf_exr import exr_io, masks, sequence
from sf_exr.stack import ExrStack, pick_layer


def write(stack, path, compression, layout="auto", pixel_type=None):
    layers = [exr_io.WriteLayer(n, l.suffixes, stack.get(n)[0].numpy(), pixel_type or l.pixel_type, l.kind)
              for n, l in stack.layers.items()]
    return exr_io.write_exr(path, layers, stack.header.attribs, compression, layout)


def reload(path):
    return ExrStack.from_sequence(sequence.detect(path, "single"))


def head_mask(stack, name="cube"):
    base, group = next(iter(stack.header.crypto.items()))
    return masks.crypto_mask([stack.get(n) for n in group["layers"]], [group["manifest"][name]])


def test_header_groups_and_classifies_layers(exr_file):
    h = exr_io.read_header(exr_file)
    kinds = {n: l.kind for n, l in h.layers.items()}
    assert kinds == {"rgba": "color", "RL.CryptoObject00": "crypto", "RL.CryptoObject01": "crypto",
                     "RL.Depth": "data", "RL.Position": "data"}
    assert h.crypto["RL.CryptoObject"]["layers"] == ["RL.CryptoObject00", "RL.CryptoObject01"]
    assert set(h.crypto["RL.CryptoObject"]["manifest"]) == {"cube", "sphere"}
    assert h.layers["rgba"].suffixes == ["R", "G", "B", "A"]
    assert h.colorspace == "Linear Rec.709 (sRGB)"


def test_roles_found_by_name(exr_file):
    layers = list(exr_io.read_header(exr_file).layers.values())
    assert [pick_layer(layers, r) for r in ("beauty", "depth", "position")] == ["rgba", "RL.Depth", "RL.Position"]


def test_zip_roundtrip_is_exact(stack, tmp_path):
    back = reload(write(stack, str(tmp_path / "rt.exr"), "zip"))
    for n in stack.layers:
        assert torch.equal(back.get(n), stack.get(n)), n
    assert len(back.header.parts) == 1


def test_dwaa_keeps_float_data_and_stays_single_part(stack, tmp_path):
    back = reload(write(stack, str(tmp_path / "dwaa.exr"), "dwaa"))
    assert len(back.header.parts) == 1
    assert torch.equal(back.get("RL.Depth"), stack.get("RL.Depth"))
    assert torch.equal(head_mask(back), head_mask(stack))


def test_pxr24_splits_so_data_stays_lossless(stack, tmp_path):
    back = reload(write(stack, str(tmp_path / "pxr24.exr"), "pxr24"))
    assert len(back.header.parts) > 1
    assert torch.equal(back.get("RL.Position"), stack.get("RL.Position"))
    assert torch.equal(head_mask(back), head_mask(stack))


def test_cryptomatte_is_always_written_as_float(stack, tmp_path):
    back = reload(write(stack, str(tmp_path / "half.exr"), "zip", pixel_type="half"))
    assert back.layers["RL.CryptoObject00"].pixel_type == "float"
    assert torch.equal(head_mask(back), head_mask(stack))


def test_display_window_padding(exr_file):
    h = exr_io.read_header(exr_file)
    data = exr_io.read_layers(exr_file, [h.layers["RL.Depth"]], h)["RL.Depth"]
    assert data.shape == (h.height, h.width, 1)


def test_sequence_detection_and_gaps(exr_sequence):
    seq = sequence.detect(exr_sequence)
    assert seq.frames == [1001, 1002, 1004] and seq.missing == [1003]
    assert os.path.basename(seq.pattern) == "shot.####.exr"
    with pytest.raises(ValueError, match="Missing frames 1003"):
        seq.select(missing_frames="error")
    held = seq.select(missing_frames="hold")
    assert held.frames == [1001, 1002, 1003, 1004] and held.paths[2] == held.paths[1]
    assert sequence.detect(exr_sequence.replace("1001", "####")).frames == [1001, 1002, 1004]
    assert sequence.detect(exr_sequence, "single").frames == [None]


def test_sequence_stack_batches_frames(exr_sequence):
    st = ExrStack.from_sequence(sequence.detect(exr_sequence).select(missing_frames="hold"))
    assert st.get("RL.Depth").shape[0] == 4
    assert np.allclose(st.get("RL.Depth")[2].numpy(), st.get("RL.Depth")[1].numpy())
