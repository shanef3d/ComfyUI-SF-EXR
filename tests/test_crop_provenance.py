import json

import pytest
import torch

from sf_exr import crop, exr_io, provenance, sequence
from sf_exr.stack import ExrStack, frame_layers


def test_mask_box_covers_every_frame_with_padding():
    m = torch.zeros(2, 100, 200)
    m[0, 40:50, 60:80] = 1.0
    m[1, 45:70, 70:90] = 0.5  # the object moves; one box covers both frames
    assert crop.mask_box(m, 0) == (60, 40, 30, 30)
    assert crop.mask_box(m, 10) == (50, 30, 50, 50)
    assert crop.mask_box(m, 500) == (0, 0, 200, 100)  # clamped to the image
    with pytest.raises(ValueError, match="empty"):
        crop.mask_box(torch.zeros(1, 8, 8), 4)


def test_model_size():
    assert crop.model_size(300, 200, 1.0, 16) == (1248, 832)  # ~1 MP, multiples of 16, aspect kept
    assert crop.model_size(301, 199, 0.0, 16) == (304, 192)   # keep size, snapped
    w, h = crop.model_size(4000, 100, 1.0, 64)
    assert w % 64 == 0 and h % 64 == 0 and h >= 64


def test_scale_keeps_hdr_and_clamps_display_pixels():
    hdr = torch.full((1, 10, 10, 3), 40.0)
    assert torch.allclose(crop.scale(hdr, 33, 27, hdr=True), torch.full((1, 27, 33, 3), 40.0), rtol=1e-4)
    edge = torch.zeros(1, 8, 8, 3)
    edge[:, :, 4:] = 1.0
    up = crop.scale(edge, 40, 40)
    assert float(up.min()) >= 0.0 and float(up.max()) <= 1.0


def test_crop_round_trip_puts_pixels_back_in_place():
    plate = torch.rand(1, 60, 80, 3) * 5.0
    m = torch.zeros(1, 60, 80)
    m[:, 20:30, 30:50] = 1.0
    box = crop.mask_box(m, 4)
    tw, th = crop.model_size(box[2], box[3], 0.0, 1)
    region = crop.scale(crop.cut(plate, box), tw, th, hdr=True)
    back = crop.scale(region, box[2], box[3], hdr=True)
    assert torch.allclose(back, crop.cut(plate, box))


def test_summary_finds_models_seeds_and_prompts():
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 1234, "model": ["1", 0], "steps": 20}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red jacket", "clip": ["4", 0]}},
        "4": {"class_type": "LoraLoader", "inputs": {"lora_name": "style.safetensors", "strength_model": 1.0}},
    }
    s = provenance.summary(prompt)
    assert "models: qwen_image_2.1_bf16.safetensors, style.safetensors" in s
    assert "seeds: 1234" in s and "prompt: a red jacket" in s
    assert provenance.summary(None) == ""


def test_metadata_is_written_and_replaced(stack, tmp_path):
    prompt = {"2": {"class_type": "KSampler", "inputs": {"seed": 7}}}
    workflow = {"nodes": [{"id": 1, "type": "KSampler"}] * 200}  # a sizeable workflow
    attribs = provenance.strip(stack.header.attribs) + provenance.attribs(prompt, {"workflow": workflow})
    path = str(tmp_path / "meta.exr")
    exr_io.write_exr(path, frame_layers(stack, 0, "source", exr_io.output_window(stack.header)), attribs, "zip")
    back = ExrStack.from_sequence(sequence.detect(path, "single"))
    found = {n: v for n, _, v in back.header.attribs}
    assert json.loads(found["comfyui/prompt"]) == prompt
    assert json.loads(found["comfyui/workflow"]) == workflow
    assert found["comfyui/summary"] == "seeds: 7"
    # saving again replaces the old values instead of keeping both; off removes them
    again = provenance.strip(back.header.attribs)
    assert not any(n.startswith("comfyui/") for n, _, _ in again)
    assert any(n.startswith("cryptomatte/") for n, _, _ in again)
