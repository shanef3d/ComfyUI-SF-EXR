"""OCIO color transforms through OpenImageIO's built-in OCIO (no separate OCIO install needed)."""
import os
from functools import lru_cache

import numpy as np
import OpenImageIO as oiio
import torch

CONFIGS = {
    "ACES CG (built-in)": "ocio://default",
    "ACES Studio (built-in)": "ocio://studio-config-latest",
    "$OCIO environment": "$OCIO",
    "custom file": "custom",
}

PRESETS = {
    "ACEScg → Linear Rec.709": ("ACEScg", "Linear Rec.709 (sRGB)"),
    "Linear Rec.709 → ACEScg": ("Linear Rec.709 (sRGB)", "ACEScg"),
    "ACES2065-1 → ACEScg": ("ACES2065-1", "ACEScg"),
    "ACEScg → ACES2065-1": ("ACEScg", "ACES2065-1"),
    "ACEScg → ACEScct": ("ACEScg", "ACEScct"),
    "ACEScct → ACEScg": ("ACEScct", "ACEScg"),
    "Linear Rec.709 → sRGB": ("Linear Rec.709 (sRGB)", "sRGB Encoded Rec.709 (sRGB)"),
    "sRGB → Linear Rec.709": ("sRGB Encoded Rec.709 (sRGB)", "Linear Rec.709 (sRGB)"),
}

# Encodings used to hand scene-linear pixels to image models and take them back
MODEL_ENCODINGS = {
    "sRGB (ACES SDR view)": ("display", "sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"),
    "sRGB (un-tone-mapped)": ("display", "sRGB - Display", "Un-tone-mapped"),
    "ACEScct (log)": ("space", "ACEScct", ""),
}


def config_path(config: str, custom_path: str = "") -> str:
    key = CONFIGS.get(config, config)
    if key == "$OCIO":
        return os.environ.get("OCIO", "") or "ocio://default"
    if key == "custom":
        if not custom_path or not (custom_path.startswith("ocio://") or os.path.isfile(custom_path)):
            raise ValueError(f"OCIO config not found: {custom_path!r}")
        return custom_path
    return key


@lru_cache(maxsize=8)
def _config(path: str) -> oiio.ColorConfig:
    return oiio.ColorConfig(path)


def colorspaces(path: str = "ocio://default") -> list[str]:
    return list(_config(path).getColorSpaceNames())


def displays(path: str = "ocio://default") -> list[str]:
    return list(_config(path).getDisplayNames())


def views(path: str = "ocio://default", display: str = "") -> list[str]:
    cfg = _config(path)
    out = []
    for d in ([display] if display else cfg.getDisplayNames()):
        out += [v for v in cfg.getViewNames(d) if v not in out]
    return out


def _per_frame(images: torch.Tensor, fn) -> torch.Tensor:
    out = []
    for frame in images.float().cpu().numpy():
        h, w, c = frame.shape
        src = oiio.ImageBuf(oiio.ImageSpec(w, h, c, oiio.FLOAT))
        src.set_pixels(oiio.ROI(), np.ascontiguousarray(frame))
        dst = oiio.ImageBuf()
        if not fn(dst, src):
            raise RuntimeError(f"OCIO transform failed: {dst.geterror() or oiio.geterror()}")
        out.append(dst.get_pixels(oiio.FLOAT).reshape(h, w, c))
    return torch.from_numpy(np.stack(out))


def convert(images: torch.Tensor, src: str, dst: str, config: str = "ocio://default") -> torch.Tensor:
    if src == dst:
        return images
    return _per_frame(images, lambda d, s: oiio.ImageBufAlgo.colorconvert(
        d, s, src, dst, unpremult=False, colorconfig=config))


def display(images: torch.Tensor, src: str, display_name: str, view: str, inverse: bool = False,
            config: str = "ocio://default") -> torch.Tensor:
    return _per_frame(images, lambda d, s: oiio.ImageBufAlgo.ociodisplay(
        d, s, display_name, view, fromspace=src, unpremult=False, inverse=inverse, colorconfig=config))


def to_model(images: torch.Tensor, src: str, encoding: str, exposure: float) -> torch.Tensor:
    """Scene-linear -> 0..1 pixels an image model expects."""
    x = images * (2.0 ** exposure)
    kind, a, b = MODEL_ENCODINGS[encoding]
    x = display(x, src, a, b) if kind == "display" else convert(x, src, a)
    return x.clamp(0.0, 1.0)


def from_model(images: torch.Tensor, dst: str, encoding: str, exposure: float) -> torch.Tensor:
    """Inverse of to_model: model output back to scene-linear."""
    kind, a, b = MODEL_ENCODINGS[encoding]
    x = display(images, dst, a, b, inverse=True) if kind == "display" else convert(images, a, dst)
    return x / (2.0 ** exposure)
