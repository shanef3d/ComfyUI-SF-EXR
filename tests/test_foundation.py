"""Shot metadata, overscan and chunked sequences survive a load -> save round trip."""
import numpy as np
import OpenImageIO as oiio
import pytest
import torch
from conftest import OVERSCAN_VALUE, H, W, write_scene

from sf_exr import exr_io, sequence
from sf_exr.stack import ExrStack, frame_layers

SHOT_META = [("PixelAspectRatio", "float", 2.0), ("screenWindowWidth", "float", 1.5),
             ("screenWindowCenter", "float2", (0.1, 0.2)), ("smpte:TimeCode", "timecode", (0x01020304, 0))]


def load(path, mode="single"):
    return ExrStack.from_sequence(sequence.detect(path, mode))


def save(stack, path, i=0, compression="zip"):
    """What SF EXR Save does for one frame."""
    window = exr_io.output_window(stack.header)
    display = (*stack.header.origin, stack.header.width, stack.header.height)
    return exr_io.write_exr(path, frame_layers(stack, i, "source", window), stack.header.attribs, compression,
                            "auto", 45.0, window, display)


def raw_spec(path):
    inp = oiio.ImageInput.open(path)
    spec = inp.spec()
    pixels = inp.read_image(oiio.FLOAT)
    inp.close()
    return spec, np.asarray(pixels)


def test_shot_metadata_survives_save(tmp_path):
    src = write_scene(str(tmp_path / "meta.exr"), extra=SHOT_META)
    out, _ = raw_spec(save(load(src), str(tmp_path / "out.exr")))
    assert out.getattribute("PixelAspectRatio") == 2.0
    assert out.getattribute("screenWindowWidth") == 1.5
    assert np.allclose(out.getattribute("screenWindowCenter"), (0.1, 0.2))
    assert tuple(out.getattribute("smpte:TimeCode")) == (0x01020304, 0)


def test_overscan_is_kept_but_hidden_from_the_graph(tmp_path):
    src = write_scene(str(tmp_path / "os.exr"), window=(-4, -2, W + 8, H + 4))
    st = load(src)
    assert st.size == (H, W)  # nodes see the display window
    assert st.get("RL.Depth").shape[1:3] == (H, W)
    out, px = raw_spec(save(st, str(tmp_path / "out.exr")))
    assert (out.x, out.y, out.width, out.height) == (-4, -2, W + 8, H + 4)
    assert (out.full_width, out.full_height) == (W, H)
    _, src_px = raw_spec(src)
    assert np.array_equal(px, src_px)


def test_edited_layer_keeps_its_overscan(tmp_path):
    src = write_scene(str(tmp_path / "os.exr"), window=(-4, -2, W + 8, H + 4))
    st = load(src)
    depth = torch.full_like(st.get("RL.Depth"), 5.0)
    edited = st.set("RL.Depth", depth, ["Z"], "data", "float")
    back = load(save(edited, str(tmp_path / "out.exr")))
    hdr = back.header
    full = exr_io.read_layers(hdr.path, [hdr.layers["RL.Depth"]], hdr, exr_io.output_window(hdr))["RL.Depth"]
    assert np.all(full[2:2 + H, 4:4 + W] == 5.0)          # the edit inside the display window
    assert np.all(full[0, :] == OVERSCAN_VALUE)            # overscan rows above it, untouched
    assert np.all(full[:, 0] == OVERSCAN_VALUE)


def test_autocropped_file_grows_to_display_window(tmp_path):
    src = write_scene(str(tmp_path / "crop.exr"), window=(8, 6, 20, 10))
    st = load(src)
    assert exr_io.output_window(st.header) == (0, 0, W, H)
    out, _ = raw_spec(save(st, str(tmp_path / "out.exr")))
    assert (out.x, out.y, out.width, out.height) == (0, 0, W, H)


def test_save_streams_unedited_layers(stack, tmp_path):
    """Writing a stack must not pull every unedited layer of the sequence into memory."""
    save(stack, str(tmp_path / "out.exr"))
    assert stack._cache.data == {}


def test_chunks(exr_sequence):
    seq = sequence.detect(exr_sequence).select(missing_frames="hold")  # 1001-1004 with 1003 held
    assert seq.chunk_count(2) == 2 and seq.chunk_count(0) == 1
    assert seq.chunk(2, 0).frames == [1001, 1002]
    assert seq.chunk(2, 1).frames == [1003, 1004]
    assert seq.chunk(3, 1).frames == [1004]
    assert seq.chunk(0, 5).frames == seq.frames  # size 0 = everything, index ignored
    with pytest.raises(ValueError, match="past the end"):
        seq.chunk(2, 2)


def test_chunked_save_keeps_frame_numbers(exr_sequence, tmp_path):
    seq = sequence.detect(exr_sequence).select(missing_frames="hold").chunk(2, 1)
    st = ExrStack.from_sequence(seq)
    assert st.seq.frames == [1003, 1004] and st.get("RL.Depth").shape[0] == 2
