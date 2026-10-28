"""HTTP routes the browser uses to read headers, previews and picks without queueing the workflow."""
import asyncio
import io as pyio
import json
import logging
import os
from collections import OrderedDict

import numpy as np
from aiohttp import web
from PIL import Image

from . import color, exr_io, layer_ops, masks
from . import sequence as seq_mod
from .nodes_mask import crypto_group, crypto_ranks, evaluate
from .stack import ExrStack, layer_to_image, pick_layer

log = logging.getLogger("SF EXR")
_stacks: "OrderedDict[tuple, ExrStack]" = OrderedDict()


def _frame_stack(path: str, mode: str, frame: int) -> ExrStack:
    seq = seq_mod.detect(path, mode)
    if seq.is_sequence and seq.frames[0] is not None:
        f = frame if frame in seq.frames else seq.frames[0]
        seq = seq.select(f, f, 1, "hold")
    key = (seq.paths[0], tuple(os.path.getmtime(p) for p in seq.paths))
    if key not in _stacks:
        _stacks[key] = ExrStack.from_sequence(seq)
        while len(_stacks) > 4:
            _stacks.popitem(last=False)
    _stacks.move_to_end(key)
    return _stacks[key]


def _png(arr: np.ndarray, size: int) -> bytes:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
    if size and max(img.size) > size:
        img.thumbnail((size, size), Image.BILINEAR)
    buf = pyio.BytesIO()
    img.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def _visualize(stack: ExrStack, name: str) -> np.ndarray:
    """A displayable picture of any layer: beauty through the ACES view, data normalized, cryptomatte as ID colors."""
    layer = stack.layers[name]
    t = stack.get(name)[0]
    if layer.kind == "crypto":
        ids = t[..., 0].numpy().view(np.uint32)
        rgb = np.stack([((ids >> s) & 255) / 255.0 for s in (0, 8, 16)], -1).astype(np.float32)
        return rgb * np.clip(t[..., 1].numpy(), 0, 1)[..., None]
    rgb = layer_to_image(t.unsqueeze(0))[0][0]
    if layer.kind == "color":
        space = stack.header.colorspace or "Linear Rec.709 (sRGB)"
        try:
            return color.display(rgb.unsqueeze(0), space, "sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)")[0].numpy()
        except Exception:
            return np.clip(rgb.numpy(), 0, 1) ** (1 / 2.2)
    x = rgb.numpy().copy()
    finite = np.isfinite(x) & (np.abs(x) < 1e9)
    lo, hi = (x[finite].min(), x[finite].max()) if finite.any() else (0.0, 1.0)
    x[~finite] = hi
    return (x - lo) / max(hi - lo, 1e-8)


def _ops(text: str | None) -> list[dict]:
    return json.loads(text) if text else []


def _source(src) -> ExrStack:
    """One frame of the file the browser points at, with the graph's rename/keep operations applied."""
    ops = src.get("ops") or []
    if isinstance(ops, str):
        ops = _ops(ops)
    st = _frame_stack(src.get("path", ""), src.get("sequence", "auto"), int(src.get("frame", 0) or 0))
    return layer_ops.apply_ops(st, ops)


def _error(e: Exception, status: int = 200):
    return web.json_response({"error": str(e)}, status=status)


def register(routes):
    @routes.get("/sf_exr/inspect")
    async def inspect(request):
        q = request.rel_url.query
        try:
            def work():
                seq = seq_mod.detect(q.get("path", ""), q.get("sequence", "auto"))
                hdr = exr_io.read_header(seq.paths[0])
                info = hdr.as_dict()
                info.update(path=seq.paths[0], pattern=seq.pattern, frames=len(seq.paths),
                            first=seq.frames[0], last=seq.frames[-1], missing=seq.missing[:200],
                            roles={r: pick_layer(list(hdr.layers.values()), r)
                                   for r in ("beauty", "depth", "position", "normal")})
                return layer_ops.apply_to_info(info, _ops(q.get("ops")))
            return web.json_response(await asyncio.to_thread(work))
        except Exception as e:
            return _error(e)

    @routes.get("/sf_exr/colorspaces")
    async def colorspaces(request):
        q = request.rel_url.query
        try:
            cfg = color.config_path(q.get("config", "ACES CG (built-in)"), q.get("custom", ""))
            return web.json_response({"spaces": color.colorspaces(cfg), "displays": color.displays(cfg),
                                      "views": color.views(cfg)})
        except Exception as e:
            return _error(e)

    @routes.get("/sf_exr/preview")
    async def preview(request):
        q = request.rel_url.query
        try:
            def work():
                st = _source(q)
                name = st.resolve(q.get("layer", "auto"))
                return _png(_visualize(st, name), int(q.get("size", 768)))
            return web.Response(body=await asyncio.to_thread(work), content_type="image/png")
        except Exception as e:
            return _error(e, 400)

    @routes.post("/sf_exr/pick")
    async def pick(request):
        """Values of every pass at one pixel (x, y as 0-1 fractions of the image)."""
        try:
            body = await request.json()

            def work():
                st = _source(body)
                h, w = st.size
                x = min(w - 1, max(0, int(float(body["x"]) * w)))
                y = min(h - 1, max(0, int(float(body["y"]) * h)))
                out = {"x": x, "y": y, "crypto": {}}
                for g in st.header.crypto:
                    _, group = crypto_group(st, g)
                    out["crypto"][g] = masks.names_at(crypto_ranks(st, group), group["manifest"], 0, x, y)
                for role in ("depth", "position", "normal"):
                    name = pick_layer(list(st.layers.values()), role)
                    if name:
                        v = st.get(name)[0, y, x].tolist()
                        out[role] = {"layer": name, "value": v[0] if role == "depth" else v[:3]}
                return out
            return web.json_response(await asyncio.to_thread(work))
        except Exception as e:
            return _error(e)

    @routes.post("/sf_exr/ranges")
    async def ranges(request):
        """Value ranges the picker's sliders map onto: depth near/far, position bounds, and the up axis."""
        try:
            body = await request.json()

            def work():
                st = _source(body)
                out = {"up": "Z" if any("blender" in str(v).lower() for n, _, v in st.header.attribs
                                        if n.lower() == "software") else "Y"}
                name = pick_layer(list(st.layers.values()), "depth")
                if name:
                    d = st.get(name)[0, ..., 0].numpy()
                    d = d[np.isfinite(d) & (np.abs(d) < 1e9)]
                    if d.size:
                        out["depth"] = {"layer": name, "min": float(d.min()), "max": float(d.max())}
                name = pick_layer(list(st.layers.values()), "position")
                if name:
                    p = st.get(name)[0, ..., :3].numpy().reshape(-1, 3)
                    p = p[np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)]
                    if p.size:
                        lo, hi = p.min(0), p.max(0)
                        out["position"] = {"layer": name, "min": lo.tolist(), "max": hi.tolist(),
                                           "size": float(np.linalg.norm(hi - lo))}
                return out
            return web.json_response(await asyncio.to_thread(work))
        except Exception as e:
            return _error(e)

    @routes.post("/sf_exr/mask_preview")
    async def mask_preview(request):
        try:
            body = await request.json()

            def work():
                st = _source(body)
                m = evaluate(st, body.get("rules", []))[0]
                if body.get("invert"):
                    m = 1.0 - m
                return _png(m.numpy(), int(body.get("size", 768)))
            return web.Response(body=await asyncio.to_thread(work), content_type="image/png")
        except Exception as e:
            return _error(e, 400)

    log.info("SF EXR routes registered")
