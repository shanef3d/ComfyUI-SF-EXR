"""Resize every layer of an EXR the way its contents need: color with the chosen filter (in float, HDR-safe),
data passes without blending values across edges, motion vectors rescaled with the image, and cryptomatte by
re-ranking each ID's coverage so the mattes still work."""
from dataclasses import replace

import torch
import torch.nn.functional as F

from .stack import ExrStack, StackLayer

METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
DATA_MODES = ["nearest (keep values)", "same as color"]
# Pixel-space motion vector passes (the values are distances in pixels, so they scale with the image)
VECTOR_NAMES = ("vector", "vectors", "motion", "motionvector", "motionvectors", "mv")


# ----------------------------------------------------------------- sizes (same rules as Resize Image/Mask)

def crop_box(w: int, h: int, tw: int, th: int) -> tuple[int, int, int, int]:
    """(x, y, width, height) of the centered region with the target's aspect, as comfy.utils.common_upscale crops."""
    old, new = w / h, tw / th
    x = y = 0
    if old > new:
        x = round((w - w * (new / old)) / 2)
    elif old < new:
        y = round((h - h * (old / new)) / 2)
    return x, y, w - 2 * x, h - 2 * y


def target(w: int, h: int, mode: dict, match_size: tuple[int, int] | None = None) -> tuple[int, int, bool]:
    """(width, height, center_crop) for a Resize Image/Mask style resize_type choice."""
    kind = mode["resize_type"]
    if kind == "scale by multiplier":
        m = float(mode["multiplier"])
        return max(1, round(w * m)), max(1, round(h * m)), False
    if kind == "scale dimensions":
        tw, th = int(mode["width"]), int(mode["height"])
        if tw == 0 and th == 0:
            return w, h, False
        if tw == 0:
            tw = max(1, round(w * th / h))
        elif th == 0:
            th = max(1, round(h * tw / w))
        return tw, th, mode.get("crop", "disabled") == "center"
    if kind == "scale longer dimension":
        s = int(mode["longer_size"])
        return (round(w / h * s), s, False) if h > w else (s, round(h / w * s), False)
    if kind == "scale shorter dimension":
        s = int(mode["shorter_size"])
        return (round(w / h * s), s, False) if h < w else (s, round(h / w * s), False)
    if kind == "scale width":
        tw = int(mode["width"])
        return tw, max(1, round(h * tw / w)), False
    if kind == "scale height":
        th = int(mode["height"])
        return max(1, round(w * th / h)), th, False
    if kind == "scale total pixels":
        s = (float(mode["megapixels"]) * 1024 * 1024 / (w * h)) ** 0.5
        return max(1, round(w * s)), max(1, round(h * s)), False
    if kind == "match size":
        mw, mh = match_size
        return mw, mh, mode.get("crop", "disabled") == "center"
    if kind == "scale to multiple":
        m = int(mode["multiple"])
        tw, th = (w // m) * m, (h // m) * m
        if m <= 1 or tw == 0 or th == 0:
            return w, h, False
        return tw, th, True
    raise ValueError(f"Unsupported resize type: {kind}")


# ----------------------------------------------------------------- filters

def _lanczos_matrix(n_in: int, n_out: int, a: int = 3) -> torch.Tensor:
    """[n_out, n_in] Lanczos-a weights; widened when shrinking so it also antialiases."""
    scale = n_out / n_in
    stretch = min(scale, 1.0)
    centers = (torch.arange(n_out, dtype=torch.float64) + 0.5) / scale - 0.5
    x = (torch.arange(n_in, dtype=torch.float64)[None, :] - centers[:, None]) * stretch
    w = torch.sinc(x) * torch.sinc(x / a) * (x.abs() < a)
    w = w / w.sum(1, keepdim=True).clamp(min=1e-12)
    return w.float()


def _lanczos(x: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    """x [N,C,H,W] float, any range (unlike comfy's lanczos, which goes through 8-bit PIL and clips HDR)."""
    wy = _lanczos_matrix(x.shape[-2], th)
    wx = _lanczos_matrix(x.shape[-1], tw)
    return torch.einsum("yh,nchw,xw->ncyx", wy, x, wx)


def resample(t: torch.Tensor, tw: int, th: int, method: str) -> torch.Tensor:
    """[F,H,W,C] float -> [F,th,tw,C] with one filter. Bilinear/bicubic antialias when shrinking."""
    h, w = t.shape[1], t.shape[2]
    if (h, w) == (th, tw):
        return t
    x = t.permute(0, 3, 1, 2).float()
    if method == "lanczos":
        out = _lanczos(x, th, tw)
    elif method in ("bilinear", "bicubic"):
        out = F.interpolate(x, size=(th, tw), mode=method, align_corners=False, antialias=th < h or tw < w)
    else:
        out = F.interpolate(x, size=(th, tw), mode=method)
    return out.permute(0, 2, 3, 1).contiguous()


def scale_vectors(t: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """Motion vectors in pixels: X channels (0, 2) scale with the width, Y channels (1, 3) with the height."""
    out = t.clone()
    for c in range(min(4, t.shape[-1])):
        out[..., c] *= sx if c % 2 == 0 else sy
    return out


# ----------------------------------------------------------------- cryptomatte

def _nearest_index(n_in: int, n_out: int) -> torch.Tensor:
    return ((torch.arange(n_out, dtype=torch.float64) + 0.5) * n_in / n_out).floor().long().clamp(max=n_in - 1)


def resample_crypto(ranks: list[torch.Tensor], tw: int, th: int) -> list[torch.Tensor]:
    """Cryptomatte rank layers ([F,H,W,4], two (id, coverage) pairs each) at a new size. Enlarging repeats pixels.
    Shrinking adds up each ID's coverage over the source pixels that land in an output pixel and keeps the
    strongest IDs, so mattes stay soft and correct; IDs are never blended."""
    h, w = ranks[0].shape[1], ranks[0].shape[2]
    if tw > w or th > h:  # enlarge the growing direction(s) by repetition first
        iy = _nearest_index(h, max(th, h)) if th > h else torch.arange(h)
        ix = _nearest_index(w, max(tw, w)) if tw > w else torch.arange(w)
        ranks = [r[:, iy][:, :, ix] for r in ranks]
        h, w = ranks[0].shape[1], ranks[0].shape[2]
    if (h, w) == (th, tw):
        return [r.clone() for r in ranks]

    n_pairs = 2 * len(ranks)
    out_y = (torch.arange(h) * th // h)
    out_x = (torch.arange(w) * tw // w)
    out_idx = (out_y[:, None] * tw + out_x[None, :]).reshape(-1)              # [H*W] output pixel per source pixel
    per_out = torch.bincount(out_idx, minlength=th * tw).float()             # source pixels per output pixel
    result = []
    for f in range(ranks[0].shape[0]):
        ids = torch.cat([r[f, ..., c] for r in ranks for c in (0, 2)]).reshape(n_pairs, -1)
        cov = torch.cat([r[f, ..., c] for r in ranks for c in (1, 3)]).reshape(n_pairs, -1)
        bits = ids.contiguous().view(torch.int32).long() & 0xFFFFFFFF
        where = out_idx.expand(n_pairs, -1)
        keep = cov > 0
        key = where[keep] * (1 << 32) + bits[keep]
        uniq, inv = torch.unique(key, return_inverse=True)
        total = torch.zeros(uniq.shape[0]).scatter_add_(0, inv, cov[keep].float())
        pix = uniq >> 32
        total = total / per_out[pix]
        # strongest first within each output pixel
        order = torch.argsort(total, descending=True, stable=True)
        order = order[torch.argsort(pix[order], stable=True)]
        pix, idv, total = pix[order], (uniq[order] & 0xFFFFFFFF), total[order]
        _, counts = torch.unique_consecutive(pix, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(pix.shape[0]) - torch.repeat_interleave(starts, counts)
        sel = rank < n_pairs
        ids_out = torch.zeros(n_pairs, th * tw, dtype=torch.int32)
        cov_out = torch.zeros(n_pairs, th * tw)
        signed = torch.where(idv >= (1 << 31), idv - (1 << 32), idv).int()
        ids_out[rank[sel], pix[sel]] = signed[sel]
        cov_out[rank[sel], pix[sel]] = total[sel]
        ids_f = ids_out.view(torch.float32).reshape(n_pairs, th, tw)
        cov_out = cov_out.reshape(n_pairs, th, tw)
        frame = [torch.stack([ids_f[2 * k], cov_out[2 * k], ids_f[2 * k + 1], cov_out[2 * k + 1]], -1)
                 for k in range(len(ranks))]
        result.append(frame)
    return [torch.stack([fr[k] for fr in result]) for k in range(len(ranks))]


# ----------------------------------------------------------------- whole stack

def resize_stack(stack: ExrStack, tw: int, th: int, center_crop: bool, method: str,
                 data_mode: str = DATA_MODES[0]) -> ExrStack:
    """Every layer at tw x th. Overscan outside the display window is dropped (it can't be scaled with the
    picture), and the pixel aspect ratio follows any change of shape so anamorphic plates stay correct."""
    h, w = stack.size
    x0, y0, cw, ch = crop_box(w, h, tw, th) if center_crop else (0, 0, w, h)
    sx, sy = tw / cw, th / ch

    def cropped(name):
        return stack.get(name)[:, y0:y0 + ch, x0:x0 + cw]

    edits, done = {}, set()
    for group in stack.header.crypto.values():
        names = [n for n in group["layers"] if n in stack.layers]
        if names:
            for n, t in zip(names, resample_crypto([cropped(n) for n in names], tw, th)):
                edits[n] = t
            done.update(names)
    for n, layer in stack.layers.items():
        if n in done:
            continue
        t = cropped(n)
        if layer.kind == "color" or (layer.kind == "data" and data_mode == "same as color"):
            out = resample(t, tw, th, method)
        else:  # data and unmatched cryptomatte: exact source values only
            out = resample(t, tw, th, "nearest-exact")
        if layer.kind == "data" and n.split(".")[-1].lower() in VECTOR_NAMES:
            out = scale_vectors(out, sx, sy)
        edits[n] = out

    attribs = []
    for name, typ, value in stack.header.attribs:
        if name == "PixelAspectRatio" and abs(sx - sy) > 1e-9:
            value = float(value) * sy / sx
        attribs.append((name, typ, value))
    if abs(sx - sy) > 1e-9 and not any(a[0] == "PixelAspectRatio" for a in attribs):
        attribs.append(("PixelAspectRatio", "float", sy / sx))
    ox, oy = stack.header.origin
    header = replace(stack.header, width=tw, height=th, data_window=(0, 0, tw, th),
                     origin=(round(ox * sx), round(oy * sy)), attribs=attribs)
    layers = {n: StackLayer(n, l.suffixes, l.kind, l.pixel_type, None, None) for n, l in stack.layers.items()}
    return ExrStack(stack.seq, header, layers, stack._cache, {n: edits[n].float().cpu() for n in layers})
