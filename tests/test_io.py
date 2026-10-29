import os

import numpy as np
import pytest
import torch

from sf_exr import exr_io, masks, sequence, upload
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


def test_frame_list_parsing():
    parse = sequence.parse_frames
    assert parse("", 1001, 1004) == [1001, 1002, 1003, 1004]
    assert parse("", 1001, 1004, 2) == [1001, 1003]
    assert parse("0", 0, 99) == [0]
    assert parse("15, 20", 0, 99) == [15, 20]
    assert parse("1-10, 50-52", 0, 99) == [*range(1, 11), 50, 51, 52]
    assert parse("1-9", 0, 99, 4) == [1, 5, 9]
    assert parse("1-9, 4", 0, 99, 4) == [1, 4, 5, 9]  # single frames ignore the step
    assert parse("first-3, 97-last", 0, 99) == [0, 1, 2, 3, 97, 98, 99]
    assert parse("-10--8, -1-1", -10, 99) == [-10, -9, -8, -1, 0, 1]
    assert parse("20, 15, 15", 0, 99) == [15, 20]
    with pytest.raises(ValueError, match="backwards"):
        parse("20-15", 0, 99)
    for bad in ("1001-", "1-10x2", "abc"):
        with pytest.raises(ValueError, match="Can't read frames"):
            parse(bad, 0, 99)


def test_sequence_frame_lists(exr_sequence):
    seq = sequence.detect(exr_sequence)  # 1001, 1002, 1004 on disk
    assert seq.select("1001, 1004").frames == [1001, 1004]
    assert seq.select("1002").paths == [seq.paths[1]]
    with pytest.raises(ValueError, match="Missing frames 1003"):
        seq.select("1003")
    held = seq.select("", 2, "hold")  # 1001, 1003 -> 1003 holds the nearest earlier frame on disk, 1002
    assert held.frames == [1001, 1003] and held.paths[1] == seq.paths[1]
    assert seq.select("1000", missing_frames="hold").paths == [seq.paths[0]]


def test_negative_and_hyphenated_frame_numbers(tmp_path):
    for name in ("shot.-002.exr", "shot.-001.exr", "shot.0000.exr", "shot.0001.exr",
                 "render-0010.exr", "render-0011.exr"):
        (tmp_path / name).write_bytes(b"")
    for start in ("shot.0000.exr", "shot.-001.exr", "shot.####.exr"):
        seq = sequence.detect(str(tmp_path / start))
        assert seq.frames == [-2, -1, 0, 1], start
        assert os.path.basename(seq.pattern) == "shot.####.exr"
    assert sequence.detect(str(tmp_path / "shot.0000.exr")).select("-1-0").frames == [-1, 0]
    assert sequence.detect(str(tmp_path / "render-0010.exr")).frames == [10, 11]  # a hyphen in the name, not a sign


def test_lone_numbered_frame_keeps_its_number(tmp_path):
    (tmp_path / "shot.0050.exr").write_bytes(b"")
    (tmp_path / "plate.exr").write_bytes(b"")
    lone = sequence.detect(str(tmp_path / "shot.0050.exr"))
    assert lone.frames == [50] and lone.padding == 4
    assert os.path.basename(lone.pattern) == "shot.####.exr"
    assert lone.select("").frames == [50]
    assert sequence.detect(str(tmp_path / "plate.exr")).frames == [None]
    assert sequence.detect(str(tmp_path / "shot.0050.exr"), "single").frames == [None]


def test_upload_folders_stay_inside_input(tmp_path):
    root = tmp_path / upload.ROOT
    folder, name = upload.folder_for(str(tmp_path), "shot beauty")
    assert name == "shot beauty" and os.path.dirname(folder) == os.path.realpath(root)
    for bad in ("..", "../../etc", "a/b", ""):
        folder_b, name_b = upload.folder_for(str(tmp_path), bad)
        assert os.path.dirname(folder_b) == os.path.realpath(root), bad
    assert upload.file_in(folder, "shot.0001.exr").endswith("shot.0001.exr")
    for bad in ("../shot.0001.exr", "a/shot.exr", "notes.txt", "shot.exr.py"):
        with pytest.raises(ValueError):
            upload.file_in(folder, bad)
    os.makedirs(folder)
    for n in ("shot.0001.exr", "shot.0002.exr", "readme.txt"):
        (tmp_path / upload.ROOT / name / n).write_bytes(b"")
    assert upload.clear(folder) == 2
    assert os.listdir(folder) == ["readme.txt"]


def test_browse_listing_collapses_sequences(tmp_path):
    (tmp_path / "v002").mkdir()
    (tmp_path / ".hidden").mkdir()
    for name in ("shot.0001.exr", "shot.0002.exr", "shot.0003.exr", "shot.0005.exr", "plate.exr", "notes.txt",
                 "lone.0050.exr", "pre.-001.exr", "pre.0000.exr", "pre.0001.exr"):
        (tmp_path / name).write_bytes(b"")
    listing = sequence.list_dir(str(tmp_path))
    by_name = {e["name"]: e for e in listing["entries"]}
    assert listing["entries"][0] == {"type": "dir", "name": "v002", "path": str(tmp_path / "v002")}
    assert ".hidden" not in by_name and "notes.txt" not in by_name
    shot = by_name["shot.####.exr"]
    assert (shot["type"], shot["first"], shot["last"], shot["count"], shot["missing"]) == ("seq", 1, 5, 4, 1)
    assert sequence.detect(shot["path"]).frames == [1, 2, 3, 5]  # the listed pattern loads the same frames
    assert by_name["pre.####.exr"]["first"] == -1
    assert by_name["lone.0050.exr"]["type"] == "file" and by_name["plate.exr"]["type"] == "file"


def test_network_shared_comfyui_limits_folders(tmp_path, monkeypatch):
    from sf_exr import access
    allowed_dir, other = tmp_path / "shows", tmp_path / "private"
    allowed_dir.mkdir()
    other.mkdir()
    (other / "secret.exr").write_bytes(b"")
    assert access.allowed(str(other / "secret.exr"))  # local-only ComfyUI may read anywhere
    monkeypatch.setattr(access, "_listen", lambda: "0.0.0.0")
    monkeypatch.setenv(access.ENV, str(allowed_dir))
    assert access.allowed(str(allowed_dir / "sh010" / "beauty.1001.exr"))
    assert not access.allowed(str(other / "secret.exr"))
    with pytest.raises(PermissionError, match="SF_EXR_ROOTS"):
        sequence.detect(str(other / "secret.exr"))
    with pytest.raises(PermissionError):
        sequence.list_dir(str(other))
    assert sequence.list_dir(str(allowed_dir))["parent"] is None  # can't step up out of the allowed folder


def test_sequence_stack_batches_frames(exr_sequence):
    st = ExrStack.from_sequence(sequence.detect(exr_sequence).select(missing_frames="hold"))
    assert st.get("RL.Depth").shape[0] == 4
    assert np.allclose(st.get("RL.Depth")[2].numpy(), st.get("RL.Depth")[1].numpy())
