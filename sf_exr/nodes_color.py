"""SF EXR color nodes: OCIO transforms and the hand-off to and from image models."""
import torch
from comfy_api.latest import io

from . import color
from . import crop as crop_mod
from .nodes_io import CATEGORY, RAW_ENCODING, ExrType, ModelContextType, dynamic_layer
from .stack import fit, layer_to_image

_SPACES = color.colorspaces()
_DISPLAYS = color.displays()
_VIEWS = color.views()
AUTO_SPACE = "auto (from file)"


def _config_inputs():
    return [
        io.Combo.Input("config", options=list(color.CONFIGS), default="ACES CG (built-in)", advanced=True),
        io.String.Input("custom_config", default="", optional=True, advanced=True,
                        tooltip="Path to a config.ocio when config is 'custom file'."),
    ]


class SFOcioTransform(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFOcioTransform",
            display_name="SF OCIO Transform",
            category=CATEGORY,
            description="Convert between colorspaces with OCIO. Only use on color layers, never on depth, "
                        "position, normal or cryptomatte data.",
            search_aliases=["ocio", "aces", "colorspace", "acescg", "color convert"],
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("preset", options=["custom"] + list(color.PRESETS), default="Linear Rec.709 → ACEScg"),
                io.Combo.Input("from_space", options=_SPACES, default="Linear Rec.709 (sRGB)"),
                io.Combo.Input("to_space", options=_SPACES, default="ACEScg"),
                *_config_inputs(),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def validate_inputs(cls, from_space, to_space):
        return True

    @classmethod
    def execute(cls, image, preset, from_space, to_space, config, custom_config=""):
        src, dst = color.PRESETS.get(preset, (from_space, to_space))
        return io.NodeOutput(color.convert(image, src, dst, color.config_path(config, custom_config)))


class SFOcioDisplay(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFOcioDisplay",
            display_name="SF OCIO Display",
            category=CATEGORY,
            description="Apply an OCIO display/view (tone mapping) for previews and 8/16-bit output. "
                        "Inverse turns a display image back into scene-linear.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("from_space", options=_SPACES, default="Linear Rec.709 (sRGB)"),
                io.Combo.Input("display", options=_DISPLAYS, default="sRGB - Display"),
                io.Combo.Input("view", options=_VIEWS, default="ACES 2.0 - SDR 100 nits (Rec.709)"),
                io.Boolean.Input("inverse", default=False),
                *_config_inputs(),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def validate_inputs(cls, from_space, display, view):
        return True

    @classmethod
    def execute(cls, image, from_space, display, view, inverse, config, custom_config=""):
        out = color.display(image, from_space, display, view, inverse, color.config_path(config, custom_config))
        return io.NodeOutput(out if inverse else out.clamp(0.0, 1.0))


class SFExrToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            # node_id predates the rename; kept so saved workflows still load.
            node_id="SFPrepareForModel",
            display_name="SF EXR to Image",
            category=CATEGORY,
            description="Turn a scene-linear layer into the 0-1 image a diffusion model expects. Pair with "
                        "SF Image to EXR to put the result back into the EXR. With a mask, 'crop to mask' hands the "
                        "model only the masked region at the size it works best at; the return pastes it back "
                        "into the full-resolution plate.",
            search_aliases=["prepare for model", "exr to image", "linear to srgb", "crop to mask", "inpaint crop"],
            inputs=[
                ExrType.Input("exr"),
                dynamic_layer(tooltip="'auto' is the beauty."),
                io.Combo.Input("source_space", options=[AUTO_SPACE] + _SPACES, default=AUTO_SPACE),
                io.Combo.Input("encoding", options=list(color.MODEL_ENCODINGS), default="sRGB (ACES SDR view)",
                               tooltip="ACEScct keeps more highlight range; sRGB looks like a normal photo."),
                io.Float.Input("exposure", default=0.0, min=-10.0, max=10.0, step=0.1, tooltip="Stops; undone on return."),
                io.DynamicCombo.Input("crop", options=[
                    io.DynamicCombo.Option("full frame", []),
                    io.DynamicCombo.Option("crop to mask", [
                        io.Int.Input("padding", default=64, min=0, max=4096,
                                     tooltip="Pixels of context around the mask, so the model sees its surroundings."),
                        io.Float.Input("megapixels", default=1.0, min=0.0, max=16.0, step=0.05,
                                       tooltip="Size given to the model (1.0 ≈ 1024×1024); 0 keeps the crop's size."),
                        io.Int.Input("multiple", default=16, min=1, max=256,
                                     tooltip="Width and height become multiples of this (16 suits most models)."),
                    ]),
                ], tooltip="'crop to mask' needs the crop mask input. One crop covers it on every frame."),
                # Last so saved workflows' input slots don't shift; id stays "mask", only the label changed
                io.Mask.Input("mask", display_name="crop mask", optional=True,
                              tooltip="The area to edit. Crop to mask crops to it, it comes out matching 'image' "
                                      "for inpainting, and SF Image to EXR blends through it."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                ModelContextType.Output(display_name="context"),
                io.Mask.Output(display_name="alpha", tooltip="The layer's own alpha channel."),
                io.Mask.Output(display_name="crop mask", tooltip="The crop mask matching 'image' (cropped and "
                                                                 "scaled with it); empty when none is connected."),
            ],
        )

    @classmethod
    def validate_inputs(cls, layer, source_space):
        return True

    @classmethod
    def execute(cls, exr, layer, source_space, encoding, exposure, crop=None, mask=None):
        name = exr.resolve(layer)
        if source_space == AUTO_SPACE:
            space = exr.header.colorspace or "Linear Rec.709 (sRGB)"
        else:
            space = source_space
        rgb, alpha = layer_to_image(exr.get(name))
        n, h, w = rgb.shape[0], rgb.shape[1], rgb.shape[2]
        plate_mask = fit(mask, n, h, w).clamp(0.0, 1.0) if mask is not None else None
        ctx = {"stack": exr, "layer": name, "space": space, "encoding": encoding, "exposure": exposure,
               "mask": plate_mask}

        if (crop or {}).get("crop") == "crop to mask":
            if plate_mask is None:
                raise ValueError("'crop to mask' needs a mask: connect one to the crop mask input.")
            box = crop_mod.mask_box(plate_mask, int(crop["padding"]))
            tw, th = crop_mod.model_size(box[2], box[3], float(crop["megapixels"]), int(crop["multiple"]))
            rgb = crop_mod.scale(crop_mod.cut(rgb, box), tw, th, hdr=True)
            alpha = crop_mod.scale(crop_mod.cut(alpha, box), tw, th)
            out_mask = crop_mod.scale(crop_mod.cut(plate_mask, box), tw, th)
            ctx["box"] = box
        else:
            out_mask = plate_mask if plate_mask is not None else torch.zeros(n, h, w)
        image = color.to_model(rgb, space, encoding, exposure)
        return io.NodeOutput(image, ctx, alpha, out_mask)


class SFImageToExr(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            # node_id predates the rename; kept so saved workflows still load.
            node_id="SFReturnFromModel",
            display_name="SF Image to EXR",
            category=CATEGORY,
            description="Convert a model's output back to scene-linear, resize it to the plate (or to the crop it "
                        "came from), and composite it into the original layer through the mask. Pixels outside the "
                        "mask keep their original values.",
            search_aliases=["return from model", "image to exr", "srgb to linear", "paste crop"],
            inputs=[
                ModelContextType.Input("context"),
                io.Image.Input("image"),
                io.Mask.Input("mask", display_name="crop mask", optional=True,
                              tooltip="Where to use the model output, at plate or crop size. Empty = SF EXR to "
                                      "Image's crop mask, or everywhere."),
            ],
            outputs=[
                ExrType.Output(display_name="exr"),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, context, image, mask=None):
        stack, name = context["stack"], context["layer"]
        orig = stack.get(name)
        n, h, w = orig.shape[0], orig.shape[1], orig.shape[2]
        x, y, bw, bh = context.get("box") or (0, 0, w, h)

        edited = image[..., :3]
        if edited.shape[0] == 1 and n > 1:
            edited = edited.repeat(n, 1, 1, 1)
        if edited.shape[0] != n:
            raise ValueError(f"Batch has {edited.shape[0]} frames but the EXR has {n}.")
        raw = context["encoding"] == RAW_ENCODING  # context from SF EXR Load: the edit is already scene-linear
        edited = crop_mod.scale(edited.float(), bw, bh, hdr=raw)
        if raw:
            linear = edited
        else:
            linear = color.from_model(edited, context["space"], context["encoding"], context["exposure"])

        if mask is not None and mask.shape[-2:] == (h, w):
            m = crop_mod.cut(fit(mask, n, h, w), (x, y, bw, bh))   # plate-size mask
        elif mask is not None:
            m = crop_mod.scale(fit(mask, n, mask.shape[-2], mask.shape[-1]), bw, bh)  # crop-size mask
        elif context.get("mask") is not None:
            m = crop_mod.cut(context["mask"], (x, y, bw, bh))
        else:
            m = torch.ones(n, bh, bw)
        m = m.unsqueeze(-1).clamp(0.0, 1.0)

        data = orig.clone()
        region = data[:, y:y + bh, x:x + bw]
        if data.shape[-1] >= 3:
            region[..., :3] = region[..., :3] * (1.0 - m) + linear * m
        else:
            region[..., :1] = region[..., :1] * (1.0 - m) + linear.mean(-1, keepdim=True) * m
        layer = stack.layers[name]
        out = stack.set(name, data, layer.suffixes, layer.kind, layer.pixel_type)
        return io.NodeOutput(out, layer_to_image(data)[0])


NODES = [SFOcioTransform, SFOcioDisplay, SFExrToImage, SFImageToExr]
