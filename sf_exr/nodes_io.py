"""SF EXR load / layer / save / convert nodes."""
import os

import numpy as np
import OpenImageIO as oiio
import torch
from comfy_api.latest import io

from . import exr_io, layer_ops
from . import sequence as seq_mod
from .stack import ExrStack, fit, frame_layers, layer_to_image, resolve_pixel_type

CATEGORY = "SF EXR"
ExrType = io.Custom("SF_EXR")

SEQUENCE_MODES = ["auto", "single"]
LAYOUTS = ["auto", "single-part", "multi-part"]
PIXEL_TYPES = ["source", "auto", "half", "float"]
IMAGE_FORMATS = {
    "png 8-bit": ("png", oiio.UINT8, 0), "png 16-bit": ("png", oiio.UINT16, 0),
    "tiff 8-bit": ("tif", oiio.UINT8, 0), "tiff 16-bit": ("tif", oiio.UINT16, 0),
    "tiff 32-bit float": ("tif", oiio.FLOAT, 0), "jpg": ("jpg", oiio.UINT8, 0),
    "dpx 10-bit": ("dpx", oiio.UINT16, 10), "exr half": ("exr", oiio.HALF, 0), "exr float": ("exr", oiio.FLOAT, 0),
}


def dynamic_layer(id: str = "layer", tooltip: str = "") -> io.Combo.Input:
    """Dropdown filled in the browser from the file header (see web/sf_exr.js). 'auto' picks by role."""
    return io.Combo.Input(id, options=["auto"], default="auto", tooltip=tooltip or None)


def output_dir(prefix: str, folder: str) -> tuple[str, str]:
    """(directory, file prefix) under ComfyUI's output folder, or under an absolute folder."""
    if folder.strip():
        base = seq_mod.resolve_path(folder)
    else:
        import folder_paths
        base = folder_paths.get_output_directory()
    sub, name = os.path.split(prefix.replace("\\", "/"))
    return os.path.join(base, sub), name or "render"


def next_counter(directory: str, name: str, ext: str) -> int:
    if not os.path.isdir(directory):
        return 1
    used = [0]
    for f in os.listdir(directory):
        if f.startswith(name + "_") and f.endswith(ext):
            digits = f[len(name) + 1:-len(ext)].strip("_")
            if digits.isdigit():
                used.append(int(digits))
    return max(used) + 1


def file_names(directory: str, name: str, ext: str, frames: list, padding: int) -> list[str]:
    """Sequence frames keep their numbers; single images get the next free counter."""
    if len(frames) > 1 or frames[0] is not None:
        nums = [f if f is not None else i + 1 for i, f in enumerate(frames)]
        return [os.path.join(directory, f"{name}.{n:0{padding}d}.{ext}") for n in nums]
    return [os.path.join(directory, f"{name}_{next_counter(directory, name, '.' + ext):05d}.{ext}")]


def stack_fingerprint(path: str, mode: str) -> str:
    try:
        seq = seq_mod.detect(path, mode)
        return "|".join(f"{p}:{os.path.getmtime(p)}" for p in seq.paths)
    except Exception as e:
        return f"error:{e}"


class SFExrLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrLoad",
            display_name="SF EXR Load",
            category=CATEGORY,
            description="Load an EXR (single, multi-layer or multi-part) or any image file. Point at one frame of a "
                        "numbered sequence, or use a #### pattern, to load the whole sequence.",
            search_aliases=["exr", "openexr", "multilayer", "image sequence", "load exr"],
            inputs=[
                io.String.Input("path", default="", tooltip="Absolute path, or relative to ComfyUI's input folder."),
                io.Combo.Input("sequence", options=SEQUENCE_MODES, default="auto"),
                io.Int.Input("frame_start", default=0, min=0, max=10_000_000, tooltip="0 = first frame"),
                io.Int.Input("frame_end", default=0, min=0, max=10_000_000, tooltip="0 = last frame"),
                io.Int.Input("frame_step", default=1, min=1, max=1000),
                io.Combo.Input("missing_frames", options=["error", "hold"], default="error"),
                dynamic_layer(tooltip="Layer for the image output; 'auto' picks the beauty."),
            ],
            outputs=[
                ExrType.Output(display_name="exr"),
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="alpha"),
                io.String.Output(display_name="layers"),
                io.String.Output(display_name="colorspace"),
                io.Int.Output(display_name="frames"),
            ],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def fingerprint_inputs(cls, path, sequence, frame_start, frame_end, frame_step, missing_frames, layer):
        return stack_fingerprint(path, sequence)

    @classmethod
    def execute(cls, path, sequence, frame_start, frame_end, frame_step, missing_frames, layer):
        if not path.strip():
            raise ValueError("Set a path to an EXR or image file.")
        seq = seq_mod.detect(path, sequence).select(frame_start, frame_end, frame_step, missing_frames)
        stack = ExrStack.from_sequence(seq)
        image, alpha = layer_to_image(stack.get(stack.resolve(layer)))
        return io.NodeOutput(stack, image, alpha, "\n".join(stack.layers), stack.header.colorspace, stack.frame_count)


class SFExrGetLayer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrGetLayer",
            display_name="SF EXR Get Layer",
            category=CATEGORY,
            description="Take one layer out of the EXR as IMAGE and MASK. Values stay as stored (scene-linear, unclamped).",
            inputs=[ExrType.Input("exr"), dynamic_layer()],
            outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask"),
                     io.String.Output(display_name="name")],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def execute(cls, exr, layer):
        name = exr.resolve(layer)
        image, mask = layer_to_image(exr.get(name))
        return io.NodeOutput(image, mask, name)


class SFExrSetLayer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrSetLayer",
            display_name="SF EXR Set Layer",
            category=CATEGORY,
            description="Replace a layer, or add a new one, from an IMAGE and/or MASK. Images are resized to the EXR.",
            inputs=[
                ExrType.Input("exr"),
                dynamic_layer(tooltip="Layer to replace; 'auto' is the beauty."),
                io.String.Input("new_layer", default="", optional=True, tooltip="Name for a new layer instead."),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True, tooltip="Written to the alpha channel."),
                io.Combo.Input("pixel_type", options=["same", "half", "float"], default="same", optional=True),
            ],
            outputs=[ExrType.Output(display_name="exr")],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def execute(cls, exr, layer, new_layer="", image=None, mask=None, pixel_type="same"):
        if image is None and mask is None:
            return io.NodeOutput(exr)
        h, w = exr.size
        n = exr.frame_count
        name = new_layer.strip() or exr.resolve(layer)
        existing = exr.layers.get(name)
        img = fit(image, n, h, w) if image is not None else None
        msk = fit(mask, n, h, w) if mask is not None else None

        if existing:
            data = exr.get(name).clone()
            c = data.shape[-1]
            if img is not None:
                k = min(3, c)
                data[..., :k] = img[..., :k] if c > 1 else img[..., :1]
            if msk is not None:
                if c >= 4:
                    data[..., 3] = msk
                elif c == 1 and img is None:
                    data[..., 0] = msk
            suffixes, kind, ptype = existing.suffixes, existing.kind, existing.pixel_type
        else:
            parts, suffixes = [], []
            if img is not None:
                parts.append(img[..., :3])
                suffixes += ["R", "G", "B"]
            if msk is not None:
                parts.append(msk.unsqueeze(-1))
                suffixes.append("A")
            data = torch.cat(parts, -1)
            kind = "color" if img is not None else "data"
            ptype = "half" if kind == "color" else "float"
        if pixel_type != "same":
            ptype = pixel_type
        return io.NodeOutput(exr.set(name, data, suffixes, kind, ptype))


class SFExrRenameLayers(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrRenameLayers",
            display_name="SF EXR Rename Layers",
            category=CATEGORY,
            description="Rename layers and channels, e.g. to match Nuke or studio naming. Mapping lines win over "
                        "the strip and find/replace rules. Cryptomatte metadata is updated with its layers.",
            search_aliases=["rename aov", "rename pass", "conform layers"],
            inputs=[
                ExrType.Input("exr"),
                io.Combo.Input("strip_prefix", options=layer_ops.STRIP_MODES, default="none",
                               tooltip="first segment: ViewLayer.Depth -> Depth"),
                io.String.Input("find", default="", tooltip="Regular expression"),
                io.String.Input("replace", default=""),
                io.String.Input("mapping", default="", multiline=True,
                                tooltip="One per line: old = new. Channels too: Depth.R = Z"),
            ],
            outputs=[ExrType.Output(display_name="exr")],
        )

    @classmethod
    def execute(cls, exr, strip_prefix, find, replace, mapping):
        op = {"type": "rename", "strip": strip_prefix, "find": find, "replace": replace, "mapping": mapping}
        return io.NodeOutput(layer_ops.apply_ops(exr, [op]))


class SFExrKeepLayers(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrKeepLayers",
            display_name="SF EXR Keep/Remove Layers",
            category=CATEGORY,
            description="Keep only the listed layers, or remove them. Names or wildcards, comma or line separated; "
                        "a cryptomatte name covers all its 00, 01 … layers.",
            search_aliases=["remove layers", "drop aov", "filter layers", "delete pass"],
            inputs=[
                ExrType.Input("exr"),
                io.Combo.Input("mode", options=layer_ops.KEEP_MODES, default="remove"),
                io.String.Input("layers", default="", multiline=True, tooltip="e.g. *Mist, *Crypto*02"),
            ],
            outputs=[ExrType.Output(display_name="exr")],
        )

    @classmethod
    def execute(cls, exr, mode, layers):
        if not layers.strip():
            return io.NodeOutput(exr)
        return io.NodeOutput(layer_ops.apply_ops(exr, [{"type": "keep", "mode": mode, "layers": layers}]))


class SFExrSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrSave",
            display_name="SF EXR Save",
            category=CATEGORY,
            description="Write the EXR as multi-layer EXR (or multi-part). Sequences keep their frame numbers. Cryptomatte is always "
                        "32-bit float, and lossy compression never touches data or cryptomatte layers.",
            is_output_node=True,
            inputs=[
                ExrType.Input("exr"),
                io.String.Input("filename_prefix", default="sf_exr/render"),
                io.String.Input("output_folder", default="", tooltip="Empty = ComfyUI output folder."),
                io.Combo.Input("layout", options=LAYOUTS, default="auto"),
                io.Combo.Input("compression", options=["source"] + exr_io.COMPRESSIONS, default="source"),
                io.Float.Input("dwa_level", default=45.0, min=0.0, max=1000.0, step=1.0, advanced=True),
                io.Combo.Input("pixel_type", options=PIXEL_TYPES, default="source",
                               tooltip="auto = half for color layers, float for data."),
                io.Int.Input("frame_padding", default=4, min=1, max=10, advanced=True),
            ],
            outputs=[io.String.Output(display_name="path")],
        )

    @classmethod
    def execute(cls, exr, filename_prefix, output_folder, layout, compression, dwa_level, pixel_type, frame_padding):
        directory, name = output_dir(filename_prefix, output_folder)
        comp = exr.header.compression if compression == "source" else compression
        if comp not in exr_io.COMPRESSIONS:
            comp = "zip"
        if layout == "auto" and exr.header.multipart:
            layout = "multi-part"
        paths = file_names(directory, name, "exr", exr.seq.frames, exr.seq.padding or frame_padding)
        window = exr_io.output_window(exr.header)
        display = (*exr.header.origin, exr.header.width, exr.header.height)
        for i, path in enumerate(paths):
            exr_io.write_exr(path, frame_layers(exr, i, pixel_type, window), exr.header.attribs, comp, layout,
                             dwa_level, window, display)
        return io.NodeOutput(paths[0] if len(paths) == 1 else os.path.dirname(paths[0]))


class SFExrConvert(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrConvert",
            display_name="SF EXR Convert",
            category=CATEGORY,
            description="Re-write EXR files with new compression, bit depth or layout, file to file. All layers and "
                        "metadata are kept. Works on single files and sequences.",
            is_output_node=True,
            inputs=[
                io.String.Input("path", default=""),
                io.Combo.Input("sequence", options=SEQUENCE_MODES, default="auto"),
                io.String.Input("output_folder", default="", tooltip="Empty = ComfyUI output folder/sf_exr_convert."),
                io.String.Input("filename_suffix", default="", tooltip="Added before the frame number."),
                io.Combo.Input("layout", options=LAYOUTS, default="auto"),
                io.Combo.Input("compression", options=exr_io.COMPRESSIONS, default="zip"),
                io.Float.Input("dwa_level", default=45.0, min=0.0, max=1000.0, step=1.0, advanced=True),
                io.Combo.Input("pixel_type", options=PIXEL_TYPES, default="source"),
            ],
            outputs=[io.String.Output(display_name="folder")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return stack_fingerprint(kwargs.get("path", ""), kwargs.get("sequence", "auto")) + repr(sorted(kwargs.items()))

    @classmethod
    def execute(cls, path, sequence, output_folder, filename_suffix, layout, compression, dwa_level, pixel_type):
        seq = seq_mod.detect(path, sequence)
        if output_folder.strip():
            directory = seq_mod.resolve_path(output_folder)
        else:
            import folder_paths
            directory = os.path.join(folder_paths.get_output_directory(), "sf_exr_convert")
        for src in seq.paths:
            hdr = exr_io.read_header(src)
            window = exr_io.output_window(hdr)
            data = exr_io.read_layers(src, list(hdr.layers.values()), hdr, window)
            layers = [exr_io.WriteLayer(n, li.suffixes, data[n], resolve_pixel_type(pixel_type, li), li.kind)
                      for n, li in hdr.layers.items()]
            stem, _ = os.path.splitext(os.path.basename(src))
            m = seq_mod.FRAME_RE.match(os.path.basename(src)) if seq.is_sequence else None
            if m:  # render.1001.exr + "_dwaa" -> render_dwaa.1001.exr
                head = m.group("head")
                base = head.rstrip("._")
                out_name = base + filename_suffix + head[len(base):] + m.group("frame") + ".exr"
            else:
                out_name = stem + filename_suffix + ".exr"
            lay = "multi-part" if (layout == "auto" and hdr.multipart) else layout
            exr_io.write_exr(os.path.join(directory, out_name), layers, hdr.attribs, compression, lay, dwa_level,
                             window, (*hdr.origin, hdr.width, hdr.height))
        return io.NodeOutput(directory)


class SFImageSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFImageSave",
            display_name="SF Image Save",
            category=CATEGORY,
            description="Save images as PNG, TIFF, JPG, DPX or single-layer EXR at a chosen bit depth. Integer formats "
                        "clamp to 0-1, so convert scene-linear images with SF OCIO Display first.",
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="sf_exr/image"),
                io.String.Input("output_folder", default="", tooltip="Empty = ComfyUI output folder."),
                io.Combo.Input("format", options=list(IMAGE_FORMATS), default="png 16-bit"),
                io.Boolean.Input("dither", default=True, tooltip="Reduces banding in 8-bit output."),
                io.Int.Input("quality", default=95, min=1, max=100, tooltip="JPG only."),
                io.Int.Input("start_frame", default=1, min=0, max=10_000_000, tooltip="Numbering for batches."),
                io.Mask.Input("alpha", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, images, filename_prefix, output_folder, format, dither, quality, start_frame, alpha=None):
        ext, td, bits = IMAGE_FORMATS[format]
        directory, name = output_dir(filename_prefix, output_folder)
        os.makedirs(directory, exist_ok=True)
        frames = [start_frame + i for i in range(images.shape[0])] if images.shape[0] > 1 else [None]
        paths = file_names(directory, name, ext, frames, 4)
        px = images.float().cpu()
        if alpha is not None and ext in ("png", "tif", "exr"):
            px = torch.cat([px, fit(alpha, px.shape[0], px.shape[1], px.shape[2]).unsqueeze(-1)], -1)
        results = []
        for path, frame in zip(paths, px.numpy()):
            if td not in (oiio.HALF, oiio.FLOAT):
                frame = np.clip(frame, 0.0, 1.0)
            h, w, c = frame.shape
            spec = oiio.ImageSpec(w, h, c, td)
            if bits:
                spec.attribute("oiio:BitsPerSample", bits)
            if dither and td == oiio.UINT8:
                spec.attribute("oiio:dither", 1)
            if ext == "jpg":
                spec.attribute("Compression", f"jpeg:{quality}")
            if ext == "exr":
                spec.attribute("compression", "zip")
            out = oiio.ImageOutput.create(path)
            if not out or not out.open(path, spec):
                raise RuntimeError(f"Cannot write {path}: {oiio.geterror()}")
            out.write_image(np.ascontiguousarray(frame, dtype=np.float32))
            out.close()
            results.append(path)
        return io.NodeOutput(ui=_preview_ui(results) if ext in ("png", "jpg") else None)


def _preview_ui(paths: list[str]) -> dict | None:
    try:
        import folder_paths
        root = os.path.abspath(folder_paths.get_output_directory())
    except ImportError:
        return None
    images = []
    for p in paths:
        p = os.path.abspath(p)
        if not p.startswith(root):
            return None
        sub, fname = os.path.split(os.path.relpath(p, root))
        images.append({"filename": fname, "subfolder": sub, "type": "output"})
    return {"images": images}


NODES = [SFExrLoad, SFExrGetLayer, SFExrSetLayer, SFExrRenameLayers, SFExrKeepLayers, SFExrSave, SFExrConvert,
         SFImageSave]
