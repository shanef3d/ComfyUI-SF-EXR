"""SF Merge: Nuke-style merge for EXR layers and plain images."""
import torch
from comfy_api.latest import io

from .merge import BBOX_MODES, OPERATIONS, bbox_mask, merge
from .nodes_io import CATEGORY, ExrType, dynamic_layer
from .stack import ExrStack, fit, frame_bboxes


def _rgba(value, layer: str, n=None, h=None, w=None):
    """(rgb [N,H,W,3], alpha [N,H,W], stack-or-None, layer name) from an EXR or an IMAGE.
    Images carry alpha as a 4th channel (ComfyUI's Join Image with Alpha); RGB images are opaque."""
    if isinstance(value, ExrStack):
        name = value.resolve(layer) if (layer in ("", "auto") or layer in value.layers) else value.resolve("auto")
        t, stack = value.get(name), value
    else:
        t, stack, name = value.float(), None, None
    c = t.shape[-1]
    rgb = t[..., :3] if c >= 3 else t[..., :1].repeat(1, 1, 1, 3)
    alpha = t[..., 3] if c >= 4 else torch.ones_like(t[..., 0])
    if n is not None:
        rgb, alpha = fit(rgb, n, h, w), fit(alpha, n, h, w)
    return rgb, alpha, stack, name


class SFMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        b_type = io.MatchType.Template("B", allowed_types=[io.Image, ExrType])
        return io.Schema(
            node_id="SFMerge",
            display_name="SF Merge",
            category=CATEGORY,
            description="Merge A onto B like Nuke's Merge. A and B can each be an EXR or an image; the output is the "
                        "same type as B. With an EXR, the chosen layer is merged and every other layer passes through. "
                        "Colour is treated as premultiplied by its alpha. Images bring alpha as a 4th channel "
                        "(use Join Image with Alpha); RGB images count as opaque.",
            search_aliases=["merge", "over", "composite", "blend", "comp"],
            inputs=[
                io.MultiType.Input("A", types=[io.Image, ExrType], optional=True,
                                   tooltip="Unconnected = black and transparent, so 'over' passes B through."),
                io.MatchType.Input("B", template=b_type),
                io.Mask.Input("mask", optional=True, tooltip="Limits the merge; outside it B shows through."),
                io.Combo.Input("operation", options=list(OPERATIONS), default="over"),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01, display_mode=io.NumberDisplay.slider),
                io.Combo.Input("bbox", options=BBOX_MODES, default="union", display_name="set bbox to",
                               tooltip="Where the result exists; outside it is black and transparent. "
                                       "An EXR's box is its data window, an image's is the whole frame."),
                dynamic_layer(tooltip="EXR layer to merge; 'auto' is the beauty."),
            ],
            outputs=[
                io.MatchType.Output(template=b_type, display_name="output"),
                io.Mask.Output(display_name="mask", tooltip="The merged alpha."),
            ],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def execute(cls, B, operation, opacity, bbox, layer, A=None, mask=None):
        b_rgb, b_a, b_stack, b_name = _rgba(B, layer)
        n, h, w = b_rgb.shape[0], b_rgb.shape[1], b_rgb.shape[2]
        if A is None:  # like Nuke: a missing input is black and transparent
            a_rgb, a_a = torch.zeros_like(b_rgb), torch.zeros_like(b_a)
        else:
            a_rgb, a_a, _, _ = _rgba(A, layer, n, h, w)
        m = fit(mask, n, h, w) if mask is not None else None
        rgb, alpha = merge(a_rgb, a_a, b_rgb, b_a, operation, opacity, m)
        full = [(0, 0, w, h)] * n
        box_a = (frame_bboxes(A, w, h) if A.frame_count == n else full) if isinstance(A, ExrStack) else full
        box_b = frame_bboxes(B, w, h) if isinstance(B, ExrStack) else full
        if box_a != full or box_b != full:
            keep = bbox_mask(box_a, box_b, bbox, h, w)
            rgb, alpha = rgb * keep.unsqueeze(-1), alpha * keep

        if b_stack is None:
            # An RGBA B stays RGBA; an RGB B stays RGB (its alpha is still on the mask output)
            image = torch.cat([rgb, alpha.unsqueeze(-1)], -1) if B.shape[-1] >= 4 else rgb
            return io.NodeOutput(image, alpha)
        data = b_stack.get(b_name).clone()
        c = data.shape[-1]
        if c >= 3:
            data[..., :3] = rgb
            if c >= 4:
                data[..., 3] = alpha
        else:
            data[..., 0] = rgb.mean(-1)
        info = b_stack.layers[b_name]
        return io.NodeOutput(b_stack.set(b_name, data, info.suffixes, info.kind, info.pixel_type), alpha)


NODES = [SFMerge]
