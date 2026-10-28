"""Reading and writing layered images with OpenImageIO: header inspection, layer grouping, pixels in and out."""
import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import OpenImageIO as oiio

log = logging.getLogger("SF EXR")

DEFAULT_LAYER = "rgba"
LOSSY = {"pxr24", "b44", "b44a", "dwaa", "dwab"}
COMPRESSIONS = ["zip", "zips", "piz", "rle", "dwaa", "dwab", "b44", "b44a", "pxr24", "none"]
HALF_MAX = 65504.0

# Channel order within a layer; anything unlisted sorts alphabetically after these
_ORDER = ["R", "r", "red", "G", "g", "green", "B", "b", "blue", "A", "a", "alpha",
          "X", "x", "Y", "y", "Z", "z", "W", "w", "U", "u", "V", "v"]
_DATA_WORDS = ("depth", "position", "normal", "uv", "vector", "velocity", "motion", "mist",
               "index", "objectid", "object_id", "matid", "worldpos", "pref")
_DATA_NAMES = {"z", "p", "n", "pw", "pref", "uv"}
# Header attributes that describe pixel layout; everything else is copied to written files
_STRUCTURAL = {"channels", "compression", "dataWindow", "displayWindow", "chunkCount", "tiles", "type", "name",
               "version", "lineOrder", "openexr:chunkCount", "openexr:lineOrder", "openexr:dwaCompressionLevel",
               "openexr:zipCompressionLevel", "Compression", "tiff:Compression", "planarconfig", "oiio:subimages",
               "oiio:subimagename", "oiio:BitsPerSample", "oiio:ColorSpace", "DateTime"}

# colorInteropID / oiio:ColorSpace values -> colorspace names in OCIO's built-in config
COLOR_INTEROP = {
    "lin_rec709_scene": "Linear Rec.709 (sRGB)", "lin_rec709": "Linear Rec.709 (sRGB)",
    "lin_srgb": "Linear Rec.709 (sRGB)", "linear": "Linear Rec.709 (sRGB)", "scene_linear": "Linear Rec.709 (sRGB)",
    "lin_ap1_scene": "ACEScg", "acescg": "ACEScg", "lin_ap1": "ACEScg",
    "lin_ap0_scene": "ACES2065-1", "aces2065-1": "ACES2065-1", "lin_ap0": "ACES2065-1",
    "srgb_rec709_scene": "sRGB Encoded Rec.709 (sRGB)", "srgb": "sRGB Encoded Rec.709 (sRGB)",
    "srgb_rec709_display": "sRGB Encoded Rec.709 (sRGB)",
    "lin_rec2020_scene": "Linear Rec.2020", "lin_p3d65_scene": "Linear P3-D65",
}


@dataclass
class LayerInfo:
    name: str
    part: int
    channels: list[str]          # full channel names as stored in the file
    suffixes: list[str]          # short names: R, G, B, A, Z, X ...
    indices: list[int]           # channel indices within the part
    types: list[str]             # half / float / uint per channel
    kind: str = "color"          # color | data | crypto

    @property
    def pixel_type(self) -> str:
        return "float" if "float" in self.types else ("uint" if "uint" in self.types else "half")

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "channels": self.suffixes, "type": self.pixel_type}


@dataclass
class Header:
    path: str
    width: int
    height: int
    origin: tuple[int, int]                         # display window x, y
    data_window: tuple[int, int, int, int]          # x, y, w, h relative to the file
    parts: list[str]
    compression: str
    multipart: bool
    layers: dict[str, LayerInfo]
    crypto: dict[str, dict] = field(default_factory=dict)   # group -> {layers, manifest, ...}
    attribs: list[tuple[str, str, object]] = field(default_factory=list)
    colorspace: str = ""
    file_format: str = "openexr"

    def as_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height, "format": self.file_format,
            "compression": self.compression, "multipart": self.multipart, "colorspace": self.colorspace,
            "layers": [l.as_dict() for l in self.layers.values()],
            "crypto": {k: {"layers": v["layers"], "names": sorted(v["manifest"])} for k, v in self.crypto.items()},
        }


def _type_name(t) -> str:
    s = str(t)
    return "half" if s == "half" else ("uint" if s.startswith("uint") else "float")


def _suffix_key(s: str):
    return (_ORDER.index(s), "") if s in _ORDER else (len(_ORDER), s.lower())


def classify(name: str, suffixes: list[str]) -> str:
    leaf = name.split(".")[-1].lower()
    if leaf in _DATA_NAMES or any(w in leaf for w in _DATA_WORDS):
        return "data"
    if [s.lower() for s in suffixes] == ["z"]:
        return "data"
    return "color"


def _attr_value(spec, p):
    try:
        return spec.getattribute(p.name)
    except Exception:
        return p.value


_header_cache: dict[tuple[str, float, int], Header] = {}


def read_header(path: str) -> Header:
    """Parse layers, crypto manifests and metadata without reading pixels. Cached per file and mtime."""
    st = os.stat(path)
    key = (path, st.st_mtime, st.st_size)
    if key in _header_cache:
        return _header_cache[key]

    inp = oiio.ImageInput.open(path)
    if not inp:
        raise RuntimeError(f"Cannot open {path}: {oiio.geterror()}")
    try:
        specs = []
        while inp.seek_subimage(len(specs), 0):
            specs.append(inp.spec())
        file_format = inp.format_name()
    finally:
        inp.close()

    first = specs[0]
    if first.deep:
        raise RuntimeError(f"Deep images are not supported: {path}")
    parts = [s.getattribute("name") or f"part{i}" for i, s in enumerate(specs)]
    multipart = len(specs) > 1

    layers: dict[str, LayerInfo] = {}
    for pi, spec in enumerate(specs):
        fmts = [_type_name(t) for t in spec.channelformats] or [_type_name(spec.format)] * spec.nchannels
        groups: dict[str, list[int]] = {}
        for ci, ch in enumerate(spec.channelnames):
            layer = ch.rsplit(".", 1)[0] if "." in ch else ""
            groups.setdefault(layer, []).append(ci)
        for layer, idx in groups.items():
            name = layer or (parts[pi] if multipart and parts[pi] not in layers else DEFAULT_LAYER)
            if name in layers:
                name = f"{parts[pi]}.{name}"
            chans = [spec.channelnames[i] for i in idx]
            sufs = [c.rsplit(".", 1)[-1] for c in chans]
            order = sorted(range(len(idx)), key=lambda k: _suffix_key(sufs[k]))
            layers[name] = LayerInfo(name, pi, [chans[k] for k in order], [sufs[k] for k in order],
                                     [idx[k] for k in order], [fmts[idx[k]] for k in order],
                                     classify(name, [sufs[k] for k in order]))

    attribs, raw = [], {}
    for p in first.extra_attribs:
        raw[p.name] = _attr_value(first, p)
        if p.name not in _STRUCTURAL and not p.name.startswith("oiio:"):
            attribs.append((p.name, str(p.type), raw[p.name]))

    crypto = _crypto_groups(path, raw, layers)
    colorspace = ""
    for key_name in ("colorInteropID", "oiio:ColorSpace"):
        v = str(raw.get(key_name, "")).strip()
        if v:
            colorspace = COLOR_INTEROP.get(v.lower(), v)
            break
    if not colorspace and file_format == "openexr":
        colorspace = "Linear Rec.709 (sRGB)"

    header = Header(
        path=path, width=first.full_width or first.width, height=first.full_height or first.height,
        origin=(first.full_x, first.full_y),
        data_window=(first.x - first.full_x, first.y - first.full_y, first.width, first.height),
        parts=parts, compression=str(raw.get("compression", "none")).split(":")[0], multipart=multipart,
        layers=layers, crypto=crypto, attribs=attribs, colorspace=colorspace, file_format=file_format,
    )
    if len(_header_cache) > 256:
        _header_cache.clear()
    _header_cache[key] = header
    return header


def _crypto_groups(path: str, raw: dict, layers: dict[str, LayerInfo]) -> dict[str, dict]:
    """Group cryptomatte rank layers using the file's own cryptomatte/<key>/name metadata."""
    groups = {}
    keys = {k.split("/")[1] for k in raw if k.startswith("cryptomatte/") and k.count("/") == 2}
    for key in sorted(keys):
        base = str(raw.get(f"cryptomatte/{key}/name", ""))
        if not base:
            continue
        manifest = {}
        text = raw.get(f"cryptomatte/{key}/manifest")
        sidecar = raw.get(f"cryptomatte/{key}/manif_file")
        try:
            if text:
                manifest = json.loads(str(text))
            elif sidecar:
                with open(os.path.join(os.path.dirname(path), str(sidecar)), encoding="utf-8") as f:
                    manifest = json.load(f)
        except Exception as e:  # a broken manifest still leaves the IDs usable by name hashing
            log.warning(f"Cryptomatte manifest for {base} unreadable: {e}")
        ranks = sorted(n for n in layers if n.startswith(base) and n[len(base):].isdigit())
        if not ranks:
            continue
        for n in ranks:
            layers[n].kind = "crypto"
        groups[base] = {"layers": ranks, "manifest": manifest,
                        "hash": str(raw.get(f"cryptomatte/{key}/hash", "MurmurHash3_32"))}

    # Files written without metadata: fall back to the CryptoXxx00 naming convention
    for name, info in layers.items():
        if info.kind != "crypto" and "crypto" in name.lower() and name[-2:].isdigit():
            base = name[:-2]
            g = groups.setdefault(base, {"layers": [], "manifest": {}, "hash": "MurmurHash3_32"})
            g["layers"] = sorted(set(g["layers"]) | {name})
            info.kind = "crypto"
    return groups


def output_window(header: Header) -> tuple[int, int, int, int]:
    """Region to write, relative to the display window: the display window plus any overscan the file had.
    A data window smaller than the display (auto-cropped renders) grows to the display so edits anywhere survive."""
    dx, dy, dw, dh = header.data_window
    x0, y0 = min(0, dx), min(0, dy)
    return x0, y0, max(header.width, dx + dw) - x0, max(header.height, dy + dh) - y0


def has_overscan(header: Header) -> bool:
    return output_window(header) != (0, 0, header.width, header.height)


def read_layers(path: str, layers: list[LayerInfo], header: Header,
                window: tuple[int, int, int, int] | None = None) -> dict[str, np.ndarray]:
    """Read the given layers of one file as float32 arrays covering `window` (relative to the display window;
    default: exactly the display window)."""
    window = window or (0, 0, header.width, header.height)
    inp = oiio.ImageInput.open(path)
    if not inp:
        raise RuntimeError(f"Cannot open {path}: {oiio.geterror()}")
    out = {}
    try:
        for li in layers:
            spec = inp.spec(li.part, 0)
            lo, hi = min(li.indices), max(li.indices) + 1
            # Layer channels sit next to each other in sorted EXR channel lists, so read one contiguous span
            arr = inp.read_image(li.part, 0, lo, hi, oiio.FLOAT)
            if arr is None:
                raise RuntimeError(f"Cannot read {li.name} from {path}: {inp.geterror()}")
            arr = np.asarray(arr, dtype=np.float32).reshape(spec.height, spec.width, hi - lo)
            arr = arr[..., [i - lo for i in li.indices]]
            out[li.name] = _place(arr, spec.x - header.origin[0], spec.y - header.origin[1], window)
    finally:
        inp.close()
    return out


def _place(arr: np.ndarray, ax: int, ay: int, window: tuple[int, int, int, int]) -> np.ndarray:
    """Put pixels whose top-left sits at (ax, ay) onto a canvas covering `window`; outside stays 0."""
    wx, wy, ww, wh = window
    if (ax, ay, arr.shape[1], arr.shape[0]) == (wx, wy, ww, wh):
        return arr
    canvas = np.zeros((wh, ww, arr.shape[2]), np.float32)
    x0, y0 = ax - wx, ay - wy
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    w = min(arr.shape[1] - sx0, ww - dx0)
    h = min(arr.shape[0] - sy0, wh - dy0)
    if w > 0 and h > 0:
        canvas[dy0:dy0 + h, dx0:dx0 + w] = arr[sy0:sy0 + h, sx0:sx0 + w]
    return canvas


def paste_display(canvas: np.ndarray, display: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    """Write display-window pixels into a canvas that covers `window`."""
    x0, y0 = -window[0], -window[1]
    canvas[y0:y0 + display.shape[0], x0:x0 + display.shape[1]] = display
    return canvas


@dataclass
class WriteLayer:
    name: str
    suffixes: list[str]
    pixels: np.ndarray       # H x W x C float32
    pixel_type: str          # half | float | uint
    kind: str = "color"


def _channel_name(layer: str, suffix: str) -> str:
    return suffix if layer in ("", DEFAULT_LAYER) else f"{layer}.{suffix}"


def _typedesc(t: str):
    return {"half": oiio.HALF, "float": oiio.FLOAT, "uint": oiio.UINT}[t]


def _compression(name: str, dwa_level: float) -> str:
    return f"{name}:{int(dwa_level)}" if name in ("dwaa", "dwab") else name


def _apply_attribs(spec, attribs):
    for name, tname, value in attribs:
        try:
            spec.attribute(name, oiio.TypeDesc(tname), value)
        except Exception:
            try:
                spec.attribute(name, value)
            except Exception:
                pass


def damaged_by(layer: WriteLayer, compression: str) -> bool:
    """Would this codec lose data in a non-color layer? PXR24 rounds float; B44 is lossy on all half
    channels; DWA is lossy only on half channels named R/G/B/Y/RY/BY. Everything else stays lossless."""
    if layer.kind == "color":
        return False
    if compression == "pxr24":
        return layer.pixel_type == "float"
    if compression in ("b44", "b44a"):
        return layer.pixel_type == "half"
    if compression in ("dwaa", "dwab"):
        return layer.pixel_type == "half" and any(s.upper() in ("R", "G", "B", "Y", "RY", "BY") for s in layer.suffixes)
    return False


def plan_parts(layers: list[WriteLayer], layout: str, compression: str) -> list[tuple[str, list[WriteLayer], str]]:
    """Split layers into EXR parts. Lossy compression never touches data or cryptomatte layers:
    when it would, the file becomes multi-part and those layers get zip."""
    needs_split = any(damaged_by(l, compression) for l in layers)
    if layout == "multi-part" or (layout != "single-part" and needs_split):
        if layout != "multi-part" and needs_split:
            log.info("SF EXR: writing multi-part so data and cryptomatte layers stay lossless")
        return [(l.name, [l], "zip" if damaged_by(l, compression) else compression) for l in layers]
    if needs_split:
        log.warning("SF EXR: lossy compression in a single-part file damages data/cryptomatte layers; using zip")
        compression = "zip"
    return [("", layers, compression)]


def write_exr(path: str, layers: list[WriteLayer], attribs=None, compression: str = "zip",
              layout: str = "auto", dwa_level: float = 45.0, window: tuple[int, int, int, int] | None = None,
              display: tuple[int, int, int, int] | None = None) -> str:
    """Write layers as one multi-layer EXR, or multi-part when the layout or compression requires it.
    `display` is the display window (x, y, w, h); `window` is where the pixels sit relative to it (overscan)."""
    if not layers:
        raise ValueError("Nothing to write: the stack has no layers.")
    h, w = layers[0].pixels.shape[:2]
    for l in layers:
        if l.kind == "crypto":
            l.pixel_type = "float"  # cryptomatte IDs are float bit patterns; half would destroy them
        if l.pixel_type == "half" and l.pixels.size and np.nanmax(np.abs(l.pixels)) > HALF_MAX:
            log.warning(f"SF EXR: layer {l.name} exceeds the half-float maximum ({HALF_MAX}); values clip to inf")

    parts = plan_parts(layers, layout, compression)
    specs, pixels = [], []
    for part_name, part_layers, comp in parts:
        names, types, data = [], [], []
        for l in part_layers:
            names += [_channel_name(l.name, s) for s in l.suffixes]
            types += [_typedesc(l.pixel_type)] * len(l.suffixes)
            data.append(l.pixels.reshape(h, w, -1).astype(np.float32))
        spec = oiio.ImageSpec(w, h, len(names), oiio.FLOAT)
        if display:
            fx, fy, fw, fh = display
            wx, wy = (window or (0, 0, w, h))[:2]
            spec.x, spec.y = fx + wx, fy + wy
            spec.full_x, spec.full_y, spec.full_width, spec.full_height = fx, fy, fw, fh
        spec.channelnames = tuple(names)
        spec.channelformats = tuple(types)
        _apply_attribs(spec, attribs or [])
        spec.attribute("compression", _compression(comp, dwa_level))
        if part_name:
            spec.attribute("name", part_name)
        specs.append(spec)
        pixels.append(np.ascontiguousarray(np.concatenate(data, axis=2)))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = oiio.ImageOutput.create(path)
    if not out:
        raise RuntimeError(f"Cannot create {path}: {oiio.geterror()}")
    try:
        ok = out.open(path, specs) if len(specs) > 1 else out.open(path, specs[0])
        if not ok:
            raise RuntimeError(f"Cannot open {path} for writing: {out.geterror()}")
        for i, px in enumerate(pixels):
            if i > 0 and not out.open(path, specs[i], "AppendSubimage"):
                raise RuntimeError(f"Cannot add part {i} to {path}: {out.geterror()}")
            if not out.write_image(px):
                raise RuntimeError(f"Cannot write {path}: {out.geterror()}")
    finally:
        out.close()
    return path
