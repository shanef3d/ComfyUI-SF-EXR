"""SF EXR load / layer / save / convert nodes."""
import os

import numpy as np
import OpenImageIO as oiio
import torch
from comfy_api.latest import io

from . import exr_io, layer_ops, provenance, resize
from . import sequence as seq_mod
from .stack import ExrStack, fit, frame_layers, layer_to_image, resolve_pixel_type

CATEGORY = "SF EXR"
ExrType = io.Custom("SF_EXR")
# The hand-off between an image and its place in the EXR: which layer, its colorspace and how it was encoded.
# SF EXR to Image makes one for model-ready pixels, SF EXR Load for the layer as stored.
ModelContextType = io.Custom("SF_EXR_MODEL_CONTEXT")
RAW_ENCODING = "as stored"


def raw_context(stack: ExrStack, layer: str) -> dict:
    return {"stack": stack, "layer": layer, "space": stack.header.colorspace or "Linear Rec.709 (sRGB)",
            "encoding": RAW_ENCODING, "exposure": 0.0}

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
                io.Combo.Input("sequence", options=SEQUENCE_MODES, default="auto", advanced=True),
                io.String.Input("frames", default="", placeholder="all",
                                tooltip="Empty = all frames. Frames and ranges separated by commas: "
                                        "1001-1100, 0, 15, 20, first-1050, 1001-last. Negative numbers are pre-roll."),
                io.Int.Input("frame_step", default=1, min=1, max=1000, tooltip="Every Nth frame of each range.",
                             advanced=True),
                io.Combo.Input("missing_frames", options=["error", "hold"], default="error", advanced=True),
                dynamic_layer(tooltip="Layer for the image output; 'auto' picks the beauty."),
            ],
            outputs=[
                ExrType.Output(display_name="exr"),
                io.Image.Output(display_name="selected layer",
                                tooltip="The chosen layer as stored: scene-linear, unclamped."),
                io.Mask.Output(display_name="alpha"),
                ModelContextType.Output(display_name="context",
                                        tooltip="For SF EXR Info, or SF Image to EXR to put an edited copy of the "
                                                "selected layer back."),
            ],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def fingerprint_inputs(cls, path, sequence, frames, frame_step, missing_frames, layer):
        return stack_fingerprint(path, sequence)

    @classmethod
    def execute(cls, path, sequence, frames, frame_step, missing_frames, layer):
        if not path.strip():
            raise ValueError("Set a path to an EXR or image file.")
        seq = seq_mod.detect(path, sequence).select(frames, frame_step, missing_frames)
        stack = ExrStack.from_sequence(seq)
        name = stack.resolve(layer)
        image, alpha = layer_to_image(stack.get(name))
        return io.NodeOutput(stack, image, alpha, raw_context(stack, name))


def info_text(stack: ExrStack, layer: str, space: str) -> str:
    """SF EXR Info's readable summary."""
    frames = [f for f in stack.seq.frames if f is not None]
    if frames:
        span = f"{frames[0]}–{frames[-1]} · {stack.frame_count} frames"
        if stack.seq.missing:
            span += f" · {len(stack.seq.missing)} missing"
    else:
        span = "single image"
    h, w = stack.size
    return "\n".join([
        f"File        {os.path.basename(stack.seq.pattern)}",
        f"Frames      {span}",
        f"Size        {w} × {h}",
        f"Colorspace  {space or 'unknown'}",
        f"Layer       {layer}",
        f"Layers      {len(stack.layers)}: {', '.join(stack.layers)}",
    ])


class SFExrInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrInfo",
            display_name="SF EXR Info",
            category=CATEGORY,
            description="The file, frame range, size, colorspace and layers behind a context, as outputs. 'info' is "
                        "a readable summary; connect it to Preview as Text to see it.",
            search_aliases=["exr info", "frame count", "frame range", "resolution", "layers", "metadata"],
            inputs=[ModelContextType.Input("context")],
            outputs=[
                io.Int.Output(display_name="frame_count"),
                io.Int.Output(display_name="first_frame", tooltip="0 for a single image."),
                io.Int.Output(display_name="last_frame", tooltip="0 for a single image."),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.String.Output(display_name="layer"),
                io.String.Output(display_name="layers", tooltip="One layer name per line."),
                io.String.Output(display_name="colorspace"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, context):
        stack, layer, space = context["stack"], context["layer"], context.get("space", "")
        frames = [f for f in stack.seq.frames if f is not None]
        h, w = stack.size
        return io.NodeOutput(stack.frame_count, frames[0] if frames else 0, frames[-1] if frames else 0, w, h,
                             layer, "\n".join(stack.layers), space, info_text(stack, layer, space))


MAX_SIZE = 16384


class SFExrResize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        def crop():
            return io.Combo.Input("crop", options=["disabled", "center"], default="center",
                                  tooltip="How to handle aspect ratio mismatch: 'disabled' stretches to fit (and "
                                          "sets the pixel aspect ratio), 'center' crops to keep the aspect.")
        return io.Schema(
            node_id="SFExrResize",
            display_name="SF EXR Resize",
            category=CATEGORY,
            description="Resize every layer of an EXR, each the way its contents need: color layers with the chosen "
                        "filter in full float (HDR is never clipped), depth / position / normal passes without "
                        "blending values across edges, motion vectors rescaled, and cryptomatte re-ranked so the "
                        "mattes keep working. Same options as Resize Image/Mask.",
            search_aliases=["resize", "scale", "exr resize", "downres", "proxy", "half res", "change size"],
            inputs=[
                ExrType.Input("exr"),
                io.DynamicCombo.Input("resize_type", options=[
                    io.DynamicCombo.Option("scale dimensions", [
                        io.Int.Input("width", default=1920, min=0, max=MAX_SIZE,
                                     tooltip="Target width in pixels. 0 = from height, keeping the aspect ratio."),
                        io.Int.Input("height", default=1080, min=0, max=MAX_SIZE,
                                     tooltip="Target height in pixels. 0 = from width, keeping the aspect ratio."),
                        crop(),
                    ]),
                    io.DynamicCombo.Option("scale by multiplier", [
                        io.Float.Input("multiplier", default=0.5, min=0.01, max=8.0, step=0.01,
                                       tooltip="Scale factor (2.0 doubles the size, 0.5 halves it)."),
                    ]),
                    io.DynamicCombo.Option("scale longer dimension", [
                        io.Int.Input("longer_size", default=1024, min=1, max=MAX_SIZE),
                    ]),
                    io.DynamicCombo.Option("scale shorter dimension", [
                        io.Int.Input("shorter_size", default=1024, min=1, max=MAX_SIZE),
                    ]),
                    io.DynamicCombo.Option("scale width", [
                        io.Int.Input("width", default=1920, min=1, max=MAX_SIZE),
                    ]),
                    io.DynamicCombo.Option("scale height", [
                        io.Int.Input("height", default=1080, min=1, max=MAX_SIZE),
                    ]),
                    io.DynamicCombo.Option("scale total pixels", [
                        io.Float.Input("megapixels", default=1.0, min=0.01, max=64.0, step=0.01,
                                       tooltip="Target total megapixels (1.0 ≈ 1024×1024). Aspect ratio is kept."),
                    ]),
                    io.DynamicCombo.Option("match size", [
                        io.MultiType.Input("match", [io.Image, io.Mask, ExrType],
                                           tooltip="Resize to the size of this image, mask or EXR."),
                        crop(),
                    ]),
                    io.DynamicCombo.Option("scale to multiple", [
                        io.Int.Input("multiple", default=8, min=1, max=MAX_SIZE,
                                     tooltip="Width and height become divisible by this (e.g. 8, 16 or 64 for "
                                             "latents); the excess is cropped from the center."),
                    ]),
                ], tooltip="How to resize: exact dimensions, a scale factor, matching another image, etc."),
                io.Combo.Input("scale_method", options=resize.METHODS, default="area",
                               tooltip="Filter for color layers. 'area' is best for shrinking, 'lanczos' for "
                                       "enlarging. Cryptomatte always keeps exact IDs."),
                io.Combo.Input("data_layers", options=resize.DATA_MODES, default=resize.DATA_MODES[0],
                               advanced=True,
                               tooltip="Depth, position and normal passes. 'nearest' copies real values so edges "
                                       "never get in-between depths; 'same as color' filters them like color."),
            ],
            outputs=[ExrType.Output(display_name="exr")],
        )

    @classmethod
    def execute(cls, exr: ExrStack, resize_type: dict, scale_method: str, data_layers: str = resize.DATA_MODES[0]):
        h, w = exr.size
        match = resize_type.get("match")
        if isinstance(match, ExrStack):
            match_size = (match.size[1], match.size[0])
        elif match is not None:
            match_size = (int(match.shape[2]), int(match.shape[1]))
        else:
            match_size = None
        tw, th, center = resize.target(w, h, resize_type, match_size)
        if (tw, th) == (w, h) and not center:
            return io.NodeOutput(exr)
        return io.NodeOutput(resize.resize_stack(exr, tw, th, center, scale_method, data_layers))


class SFExrGetLayer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrGetLayer",
            display_name="SF EXR Get Layer",
            category=CATEGORY,
            description="Take one layer out of the EXR as IMAGE and MASK. Values stay as stored (scene-linear, unclamped).",
            inputs=[ExrType.Input("exr"), dynamic_layer()],
            outputs=[io.Image.Output(display_name="image"),
                     io.Mask.Output(display_name="alpha", tooltip="The layer's alpha, or its only channel."),
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
                io.Boolean.Input("save_metadata", default=True, display_name="save metadata",
                                 tooltip="Write the ComfyUI prompt, workflow and a summary (models, seeds, prompt "
                                         "text) into the EXR header, so each frame can be traced and re-made. Off "
                                         "also removes any ComfyUI metadata the source file carried."),
            ],
            outputs=[io.String.Output(display_name="path")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, exr, filename_prefix, output_folder, layout, compression, dwa_level, pixel_type, frame_padding,
                save_metadata=True):
        attribs = provenance.strip(exr.header.attribs)
        if save_metadata and not _metadata_disabled():
            attribs += provenance.attribs(cls.hidden.prompt, cls.hidden.extra_pnginfo)
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
            exr_io.write_exr(path, frame_layers(exr, i, pixel_type, window), attribs, comp, layout,
                             dwa_level, window, display)
        return io.NodeOutput(paths[0] if len(paths) == 1 else os.path.dirname(paths[0]))


def _metadata_disabled() -> bool:
    """ComfyUI's --disable-metadata switch wins over the node toggle, as it does for core's Save Image."""
    try:
        from comfy.cli_args import args
        return bool(args.disable_metadata)
    except Exception:
        return False


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


NODES = [SFExrLoad, SFExrInfo, SFExrResize, SFExrGetLayer, SFExrSetLayer, SFExrRenameLayers, SFExrKeepLayers, SFExrSave, SFExrConvert,
         SFImageSave]
