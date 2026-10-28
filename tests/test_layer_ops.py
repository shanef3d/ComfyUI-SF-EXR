import pytest
import torch

from sf_exr import exr_io, layer_ops, masks, sequence
from sf_exr.stack import ExrStack


def rename(stack, strip="none", find="", replace="", mapping=""):
    return layer_ops.apply_ops(stack, [{"type": "rename", "strip": strip, "find": find, "replace": replace,
                                        "mapping": mapping}])


def keep(stack, mode, layers):
    return layer_ops.apply_ops(stack, [{"type": "keep", "mode": mode, "layers": layers}])


def cube(stack):
    base, group = next(iter(stack.header.crypto.items()))
    return masks.crypto_mask([stack.get(n) for n in group["layers"]], [group["manifest"]["cube"]])


def test_strip_prefix_moves_cryptomatte_metadata(stack):
    r = rename(stack, strip="first segment")
    assert list(r.layers) == ["rgba", "CryptoObject00", "CryptoObject01", "Depth", "Position"]
    assert list(r.header.crypto) == ["CryptoObject"]
    assert [v for n, _, v in r.header.attribs if n.endswith("/name")] == ["CryptoObject"]
    assert torch.equal(cube(r), cube(stack))
    assert list(stack.header.crypto) == ["RL.CryptoObject"]  # the source is untouched


def test_mapping_layers_channels_and_crypto_base(stack):
    r = rename(stack, mapping="RL.Depth = depth\nRL.Position.X = PX\nRL.CryptoObject = objID")
    assert r.layers["depth"].suffixes == ["Z"]
    assert r.layers["RL.Position"].suffixes == ["PX", "Y", "Z"]
    assert list(r.header.crypto) == ["objID"] and "objID00" in r.layers


def test_regex_find_replace(stack):
    r = rename(stack, find=r"^RL\.", replace="beauty_rl.")
    assert "beauty_rl.Depth" in r.layers and list(r.header.crypto) == ["beauty_rl.CryptoObject"]


@pytest.mark.parametrize("mapping,match", [
    ("RL.Depth = x\nRL.Position = x", "two layers"),
    ("RL.CryptoObject00 = foo", "shared name"),
    ("RL.CryptoObject00.r = R", "can't be renamed"),
    ("Nope = x", "no layer"),
])
def test_rename_errors(stack, mapping, match):
    with pytest.raises(ValueError, match=match):
        rename(stack, mapping=mapping)


def test_keep_and_remove(stack):
    assert list(keep(stack, "keep", "rgba, *Depth").layers) == ["rgba", "RL.Depth"]
    k = keep(stack, "remove", "*01")
    assert k.header.crypto["RL.CryptoObject"]["layers"] == ["RL.CryptoObject00"]
    gone = keep(stack, "remove", "RL.CryptoObject")
    assert not gone.header.crypto
    assert not [n for n, _, _ in gone.header.attribs if n.startswith("cryptomatte/")]
    with pytest.raises(ValueError, match="every layer"):
        keep(stack, "remove", "*")
    with pytest.raises(ValueError, match="No layer matches"):
        keep(stack, "keep", "typo*")


def test_renamed_file_saves_and_reloads(stack, tmp_path):
    r = keep(rename(stack, strip="first segment"), "remove", "Position")
    layers = [exr_io.WriteLayer(n, l.suffixes, r.get(n)[0].numpy(), l.pixel_type, l.kind) for n, l in r.layers.items()]
    path = exr_io.write_exr(str(tmp_path / "out.exr"), layers, r.header.attribs, "zip")
    back = ExrStack.from_sequence(sequence.detect(path, "single"))
    assert set(back.layers) == {"rgba", "CryptoObject00", "CryptoObject01", "Depth"}
    assert torch.equal(cube(back), cube(stack))


def test_browser_view_matches_nodes(stack, exr_file):
    op = {"type": "rename", "strip": "first segment", "find": "", "replace": "", "mapping": "RL.Depth = depth"}
    info = exr_io.read_header(exr_file).as_dict()
    view = layer_ops.apply_to_info(info, [op, {"type": "keep", "mode": "remove", "layers": "Position"}])
    nodes = keep(layer_ops.apply_ops(stack, [op]), "remove", "Position")
    assert [l["name"] for l in view["layers"]] == list(nodes.layers)
    assert view["roles"]["depth"] == "depth"
