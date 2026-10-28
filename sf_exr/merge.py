"""Nuke-style merge operations on premultiplied RGBA. A is the foreground, B the background; a and b are their
alphas. Every operation runs on the colour channels and the alpha the same way, as Nuke does for rgba."""
import torch

T = torch.Tensor


def _screen(A: T, B: T) -> T:
    # Nuke: A+B-AB while both stay in 0..1, otherwise max(A, B) so HDR values don't fold over
    in_range = (A >= 0) & (A <= 1) & (B >= 0) & (B <= 1)
    return torch.where(in_range, A + B - A * B, torch.maximum(A, B))


def _overlay(A: T, B: T) -> T:
    return torch.where(B < 0.5, 2 * A * B, 1 - 2 * (1 - A) * (1 - B))


def _divide(A: T, B: T) -> T:
    out = A / torch.where(B == 0, torch.ones_like(B), B)
    return torch.where((B == 0) | ((A < 0) & (B < 0)), torch.zeros_like(A), out)


def _multiply(A: T, B: T) -> T:
    return torch.where((A < 0) & (B < 0), A, A * B)


# name -> f(A, B, a, b)
OPERATIONS = {
    "over": lambda A, B, a, b: A + B * (1 - a),
    "under": lambda A, B, a, b: A * (1 - b) + B,
    "plus": lambda A, B, a, b: A + B,
    "minus": lambda A, B, a, b: A - B,
    "from": lambda A, B, a, b: B - A,
    "multiply": lambda A, B, a, b: _multiply(A, B),
    "screen": lambda A, B, a, b: _screen(A, B),
    "overlay": lambda A, B, a, b: _overlay(A, B),
    "hard-light": lambda A, B, a, b: _overlay(B, A),
    "max": lambda A, B, a, b: torch.maximum(A, B),
    "min": lambda A, B, a, b: torch.minimum(A, B),
    "average": lambda A, B, a, b: (A + B) / 2,
    "difference": lambda A, B, a, b: (A - B).abs(),
    "divide": lambda A, B, a, b: _divide(A, B),
    "hypot": lambda A, B, a, b: torch.sqrt(A * A + B * B),
    "in": lambda A, B, a, b: A * b,
    "out": lambda A, B, a, b: A * (1 - b),
    "atop": lambda A, B, a, b: A * b + B * (1 - a),
    "xor": lambda A, B, a, b: A * (1 - b) + B * (1 - a),
    "mask": lambda A, B, a, b: B * a,
    "stencil": lambda A, B, a, b: B * (1 - a),
    "copy": lambda A, B, a, b: A,
}


BBOX_MODES = ["union", "intersection", "A", "B"]
Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels, end-exclusive


def bbox_mask(boxes_a: list[Box], boxes_b: list[Box], mode: str, h: int, w: int) -> T:
    """[N,H,W] 1 inside the output bounding box of each frame, like Nuke's 'set bbox to'."""
    out = torch.zeros(len(boxes_b), h, w)
    for i, (ba, bb) in enumerate(zip(boxes_a, boxes_b)):
        if mode == "A":
            box = ba
        elif mode == "B":
            box = bb
        elif mode == "intersection":
            box = (max(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), min(ba[3], bb[3]))
        else:
            box = (min(ba[0], bb[0]), min(ba[1], bb[1]), max(ba[2], bb[2]), max(ba[3], bb[3]))
        x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
        if x1 > x0 and y1 > y0:
            out[i, y0:y1, x0:x1] = 1.0
    return out


def merge(A: T, a: T, B: T, b: T, operation: str, opacity: float = 1.0, mask: T | None = None) -> tuple[T, T]:
    """A/B: [N,H,W,3] colour, a/b: [N,H,W] alpha. Returns (rgb, alpha) blended back toward B by opacity * mask."""
    fn = OPERATIONS[operation]
    a4, b4 = a.unsqueeze(-1), b.unsqueeze(-1)
    rgb = fn(A, B, a4, b4)
    alpha = fn(a4, b4, a4, b4)[..., 0]
    mix = torch.full_like(b, float(opacity)) if mask is None else mask.clamp(0, 1) * float(opacity)
    # lerp is exact at both ends, so opacity 1 returns the operation's result bit for bit
    rgb = torch.lerp(B, rgb, mix.unsqueeze(-1).expand_as(rgb))
    alpha = torch.lerp(b, alpha, mix)
    return rgb, alpha
