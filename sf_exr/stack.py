"""SF_EXR: every layer of a file or sequence, loaded lazily, plus what's needed to write it back."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from . import exr_io
from .sequence import Sequence

BEAUTY_NAMES = ("combined", "beauty", "rgba", "main", "image", "color")


@dataclass
class StackLayer:
    name: str
    suffixes: list[str]
    kind: str          # color | data | crypto
    pixel_type: str    # half | float | uint
    source: exr_io.LayerInfo | None = None   # set while the layer still comes from the file
    origin: exr_io.LayerInfo | None = None   # the file layer an edit replaced; its overscan is kept on save


class _SourceCache:
    """Pixels read from disk, shared by every copy of a stack so each layer is read once."""

    def __init__(self, seq: Sequence, header: exr_io.Header):
        self.seq, self.header = seq, header
        self.data: dict[str, torch.Tensor] = {}

    def get(self, info: exr_io.LayerInfo) -> torch.Tensor:
        if info.name not in self.data:
            def read(path):
                hdr = exr_io.read_header(path)
                li = hdr.layers.get(info.name)
                if li is None:
                    raise ValueError(f"Layer {info.name} is missing from {path}")
                return exr_io.read_layers(path, [li], hdr)[info.name]

            # OpenImageIO releases the GIL while decoding, so frames read in parallel
            with ThreadPoolExecutor(max_workers=min(8, len(self.seq.paths))) as pool:
                frames = list(pool.map(read, self.seq.paths))
            shapes = {f.shape for f in frames}
            if len(shapes) > 1:
                raise ValueError(f"Frames of {self.seq.pattern} have different sizes: {sorted(shapes)}")
            self.data[info.name] = torch.from_numpy(np.stack(frames))
        return self.data[info.name]


@dataclass
class ExrStack:
    seq: Sequence
    header: exr_io.Header
    layers: dict[str, StackLayer]
    _cache: _SourceCache
    _edits: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_sequence(cls, seq: Sequence) -> "ExrStack":
        header = exr_io.read_header(seq.paths[0])
        layers = {n: StackLayer(n, li.suffixes, li.kind, li.pixel_type, li) for n, li in header.layers.items()}
        return cls(seq, header, layers, _SourceCache(seq, header))

    # ------------------------------------------------------------- reading
    @property
    def frame_count(self) -> int:
        return len(self.seq.paths)

    @property
    def size(self) -> tuple[int, int]:
        return self.header.height, self.header.width

    def names(self, kind: str | None = None) -> list[str]:
        return [n for n, l in self.layers.items() if kind is None or l.kind == kind]

    def get(self, name: str) -> torch.Tensor:
        """[frames, H, W, C] float32."""
        if name not in self.layers:
            raise ValueError(f"No layer '{name}'. Layers: {', '.join(self.layers)}")
        if name in self._edits:
            return self._edits[name]
        return self._cache.get(self.layers[name].source)

    def resolve(self, name: str, prefer: str = "beauty") -> str:
        """'auto' picks a sensible layer for the purpose: beauty, depth, position, normal."""
        if name and name != "auto":
            if name in self.layers:
                return name
            low = {n.lower(): n for n in self.layers}
            if name.lower() in low:
                return low[name.lower()]
            raise ValueError(f"No layer '{name}'. Layers: {', '.join(self.layers)}")
        return pick_layer(list(self.layers.values()), prefer) or next(iter(self.layers))

    # ------------------------------------------------------------- editing
    def copy(self) -> "ExrStack":
        return ExrStack(self.seq, self.header, dict(self.layers), self._cache, dict(self._edits))

    def set(self, name: str, pixels: torch.Tensor, suffixes: list[str], kind: str, pixel_type: str) -> "ExrStack":
        out = self.copy()
        prev = self.layers.get(name)
        origin = (prev.source or prev.origin) if prev else None
        out.layers[name] = StackLayer(name, suffixes, kind, pixel_type, None, origin)
        out._edits[name] = pixels.float().cpu()
        return out

    def renamed(self, name_map: dict[str, str], suffix_maps: dict[str, dict[str, str]],
                crypto_map: dict[str, str]) -> "ExrStack":
        """New names for layers/channels. Cryptomatte metadata follows its layers so the mattes keep working.
        Source layers keep pointing at their names in the file."""
        layers = {}
        for old, l in self.layers.items():
            new = name_map.get(old, old)
            sufs = [suffix_maps.get(old, {}).get(s, s) for s in l.suffixes]
            layers[new] = replace(l, name=new, suffixes=sufs)
        crypto = {crypto_map.get(b, b): {**g, "layers": [name_map.get(r, r) for r in g["layers"]]}
                  for b, g in self.header.crypto.items()}
        attribs = [(n, t, crypto_map.get(v, v) if _is_crypto_name(n) else v) for n, t, v in self.header.attribs]
        header = replace(self.header, crypto=crypto, attribs=attribs)
        edits = {name_map.get(k, k): v for k, v in self._edits.items()}
        return ExrStack(self.seq, header, layers, self._cache, edits)

    def subset(self, keep: list[str]) -> "ExrStack":
        """Only the given layers. Cryptomatte groups that lose all their layers drop their metadata too."""
        keep_set = set(keep)
        layers = {n: l for n, l in self.layers.items() if n in keep_set}
        crypto, dropped = {}, set()
        for base, g in self.header.crypto.items():
            ranks = [r for r in g["layers"] if r in keep_set]
            if ranks:
                crypto[base] = {**g, "layers": ranks}
            else:
                dropped.add(base)
        keys = {n.split("/")[1] for n, _, v in self.header.attribs if _is_crypto_name(n) and v in dropped}
        attribs = [a for a in self.header.attribs
                   if not (a[0].startswith("cryptomatte/") and a[0].split("/")[1] in keys)]
        header = replace(self.header, crypto=crypto, attribs=attribs)
        edits = {k: v for k, v in self._edits.items() if k in keep_set}
        return ExrStack(self.seq, header, layers, self._cache, edits)

    def __repr__(self):
        return f"ExrStack({self.seq.pattern}, {self.frame_count} frame(s), layers={list(self.layers)})"


def resolve_pixel_type(mode: str, layer) -> str:
    if mode == "source":
        return layer.pixel_type
    if mode == "auto":
        return "half" if layer.kind == "color" else "float"
    return mode


def frame_layers(stack: ExrStack, i: int, pixel_type: str, window) -> list[exr_io.WriteLayer]:
    """Layers of frame i for writing. Unedited layers stream from the source file (never held for the whole
    sequence); edited ones are pasted over the file's overscan so nothing outside the display window is lost."""
    path = stack.seq.paths[i]
    hdr = exr_io.read_header(path)
    display = (0, 0, stack.header.width, stack.header.height)
    wanted = {}
    for l in stack.layers.values():
        info = l.source or (l.origin if window != display else None)
        if info is not None:
            li = hdr.layers.get(info.name)
            if li is None:
                raise ValueError(f"Layer {info.name} is missing from {path}")
            wanted[info.name] = li
    data = exr_io.read_layers(path, list(wanted.values()), hdr, window) if wanted else {}

    out = []
    for n, l in stack.layers.items():
        if l.source is not None:
            px = data[l.source.name]
        else:
            edited = stack.get(n)[i].numpy()
            if window == display:
                px = edited
            else:
                base = data.get(l.origin.name) if l.origin else None
                if base is None or base.shape[2] != edited.shape[2]:
                    base = np.zeros((window[3], window[2], edited.shape[2]), np.float32)
                px = exr_io.paste_display(base.copy(), edited, window)
        out.append(exr_io.WriteLayer(n, l.suffixes, px, resolve_pixel_type(pixel_type, l), l.kind))
    return out


def frame_bboxes(stack: ExrStack, w: int | None = None, h: int | None = None) -> list[tuple[int, int, int, int]]:
    """Each frame's data window as (x0, y0, x1, y1) in display-window pixels, scaled to w x h if given (default:
    the stack's size, which differs from the file's after SF EXR Resize)."""
    boxes = []
    for path in stack.seq.paths:
        fh = exr_io.read_header(path)
        sx = (w or stack.header.width) / fh.width
        sy = (h or stack.header.height) / fh.height
        dx, dy, dw, dh = fh.data_window
        boxes.append((round(dx * sx), round(dy * sy), round((dx + dw) * sx), round((dy + dh) * sy)))
    return boxes


def _is_crypto_name(attr: str) -> bool:
    return attr.startswith("cryptomatte/") and attr.endswith("/name")


def pick_layer(layers: list, prefer: str) -> str | None:
    """Choose a layer by role from its name; works on LayerInfo or StackLayer."""
    def leaf(l):
        return l.name.split(".")[-1].lower()

    if prefer == "beauty":
        for want in BEAUTY_NAMES:
            for l in layers:
                if l.kind == "color" and leaf(l) == want:
                    return l.name
        return next((l.name for l in layers if l.kind == "color"), None)
    words = {"depth": ("depth", "z", "zdepth"), "position": ("position", "p", "pw", "worldpos", "pref"),
             "normal": ("normal", "n", "nw")}[prefer]
    for l in layers:
        if leaf(l) in words:
            return l.name
    for l in layers:
        if any(w in leaf(l) for w in words if len(w) > 2):
            return l.name
    return None


def layer_to_image(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B,H,W,C] layer -> IMAGE [B,H,W,3] and MASK [B,H,W] (alpha, the single channel, or ones)."""
    c = t.shape[-1]
    if c == 1:
        return t.repeat(1, 1, 1, 3), t[..., 0]
    if c == 2:
        return torch.cat([t, torch.zeros_like(t[..., :1])], -1), torch.ones_like(t[..., 0])
    if c == 3:
        return t, torch.ones_like(t[..., 0])
    return t[..., :3], t[..., 3]


def fit(image: torch.Tensor, frames: int, h: int, w: int) -> torch.Tensor:
    """Match an IMAGE/MASK batch to the stack: resize to its resolution, repeat a single frame."""
    is_mask = image.dim() == 3
    x = image.unsqueeze(-1) if is_mask else image
    if x.shape[1:3] != (h, w):
        x = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2).float(), size=(h, w), mode="bilinear",
                                            align_corners=False).permute(0, 2, 3, 1)
    if x.shape[0] == 1 and frames > 1:
        x = x.repeat(frames, 1, 1, 1)
    elif x.shape[0] != frames:
        raise ValueError(f"Batch has {x.shape[0]} frames but the EXR has {frames}.")
    return x[..., 0] if is_mask else x
