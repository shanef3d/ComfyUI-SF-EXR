"""Crop to mask: hand an image model only the masked region, at the size it works best at, and put the result
back into the full-resolution plate. One box covers the mask on every frame, so a sequence keeps a steady crop."""
import torch

from .resize import resample


def mask_box(mask: torch.Tensor, padding: int) -> tuple[int, int, int, int]:
    """(x, y, width, height) around everything the mask touches on any frame, grown by `padding` pixels and
    kept inside the image."""
    hit = (mask > 1e-3).any(0)
    if not bool(hit.any()):
        raise ValueError("The mask is empty, so there's nothing to crop to.")
    ys = torch.nonzero(hit.any(1)).flatten()
    xs = torch.nonzero(hit.any(0)).flatten()
    h, w = hit.shape
    x0, x1 = max(0, int(xs[0]) - padding), min(w, int(xs[-1]) + 1 + padding)
    y0, y1 = max(0, int(ys[0]) - padding), min(h, int(ys[-1]) + 1 + padding)
    return x0, y0, x1 - x0, y1 - y0


def model_size(w: int, h: int, megapixels: float, multiple: int) -> tuple[int, int]:
    """Size to give the model: the crop scaled to `megapixels` (0 keeps its size), sides a multiple of `multiple`."""
    if megapixels > 0:
        s = (megapixels * 1024 * 1024 / (w * h)) ** 0.5
        w, h = w * s, h * s
    m = max(1, int(multiple))
    return max(m, round(w / m) * m), max(m, round(h / m) * m)


def cut(t: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    """[F,H,W,...] -> the box region."""
    x, y, w, h = box
    return t[:, y:y + h, x:x + w]


def scale(t: torch.Tensor, w: int, h: int, hdr: bool = False) -> torch.Tensor:
    """Resize [F,H,W,C] (or a [F,H,W] mask) with area when shrinking; lanczos when enlarging HDR, bicubic for 0-1
    pixels (clamped so the filter can't push them outside 0-1)."""
    is_mask = t.dim() == 3
    x = t.unsqueeze(-1) if is_mask else t
    shrink = w * h < x.shape[1] * x.shape[2]
    out = resample(x, w, h, "area" if shrink else ("lanczos" if hdr else "bicubic"))
    if not hdr:
        out = out.clamp(0.0, 1.0)
    return out[..., 0] if is_mask else out
