"""SF Merge: Nuke-style merge for EXR layers and plain images."""
import torch
from comfy_api.latest import io

from .merge import BBOX_MODES, OPERATIONS, bbox_mask, merge
from .nodes_io import CATEGORY, ExrType, dynamic_layer
from .stack import ExrStack, fit, frame_bboxes


def _rgba(value, layer: str, alpha_in, n=None, h=None, w=None):
    """(rgb [N,H,W,3], alpha [N,H,W], stack-or-None, layer name) from an EXR or an IMAGE."""
    if isinstance(value, ExrStack):
        name = value.resolve(layer) if (layer in ("", "auto") or layer in value.layers) else value.resolve("auto")
        t = value.get(name)
        c = t.shape[-1]
        rgb = t[..., :3] if c >= 3 else t[..., :1].repeat(1, 1, 1, 3)
        alpha = t[..., 3] if c >= 4 else torch.ones_like(t[..., 0])
        stack = value
    else:
        rgb, alpha, stack, name = value[..., :3].float(), None, None, None
        alpha = torch.ones_like(rgb[..., 0])
    if alpha_in is not None:
        alpha = fit(alpha_in, rgb.shape[0], rgb.shape[1], rgb.shape[2])
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
                        "Colour is treated as premultiplied by its alpha.",
            search_aliases=["merge", "over", "composite", "blend", "comp"],
            inputs=[
                io.MultiType.Input("A", types=[io.Image, ExrType]),
                io.MatchType.Input("B", template=b_type),
                io.Mask.Input("mask", optional=True, tooltip="Limits the merge; outside it B shows through."),
                io.Combo.Input("operation", options=list(OPERATIONS), default="over"),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01, display_mode=io.NumberDisplay.slider),
                io.Combo.Input("bbox", options=BBOX_MODES, default="union", display_name="set bbox to",
                               tooltip="Where the result exists; outside it is black and transparent. "
                                       "An EXR's box is its data window, an image's is the whole frame."),
                dynamic_layer(tooltip="EXR layer to merge; 'auto' is the beauty."),
                io.Mask.Input("A_alpha", optional=True, advanced=True, tooltip="Alpha for an image A (EXR layers use their own)."),
                io.Mask.Input("B_alpha", optional=True, advanced=True, tooltip="Alpha for an image B (EXR layers use their own)."),
            ],
            outputs=[
                io.MatchType.Output(template=b_type, display_name="output"),
                io.Mask.Output(display_name="alpha"),
            ],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def execute(cls, A, B, operation, opacity, bbox, layer, mask=None, A_alpha=None, B_alpha=None):
        b_rgb, b_a, b_stack, b_name = _rgba(B, layer, B_alpha)
        n, h, w = b_rgb.shape[0], b_rgb.shape[1], b_rgb.shape[2]
        a_rgb, a_a, _, _ = _rgba(A, layer, A_alpha, n, h, w)
        m = fit(mask, n, h, w) if mask is not None else None
        rgb, alpha = merge(a_rgb, a_a, b_rgb, b_a, operation, opacity, m)
        full = [(0, 0, w, h)] * n
        box_a = (frame_bboxes(A, w, h) if A.frame_count == n else full) if isinstance(A, ExrStack) else full
        box_b = frame_bboxes(B, w, h) if isinstance(B, ExrStack) else full
        if box_a != full or box_b != full:
            keep = bbox_mask(box_a, box_b, bbox, h, w)
            rgb, alpha = rgb * keep.unsqueeze(-1), alpha * keep

        if b_stack is None:
            return io.NodeOutput(rgb, alpha)
        src = b_stack.get(b_name)
        data = src.clone()
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
