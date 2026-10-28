"""SF EXR mask nodes: masks from cryptomatte, depth, world position and normals, plus the interactive picker."""
import json

import torch
from comfy_api.latest import io

from . import masks
from .nodes_io import CATEGORY, ExrType, dynamic_layer
from .stack import ExrStack

OPS = ["add", "subtract", "intersect"]


def crypto_group(stack: ExrStack, name: str) -> tuple[str, dict]:
    groups = stack.header.crypto
    if not groups:
        raise ValueError("This file has no cryptomatte layers.")
    if not name or name == "auto":
        key = next(iter(groups))
    elif name in groups:
        key = name
    else:
        key = next((g for g in groups if name.rstrip("0123456789") == g or name.lower() in g.lower()), None)
        if key is None:
            raise ValueError(f"No cryptomatte '{name}'. Available: {', '.join(groups)}")
    return key, groups[key]


def crypto_ranks(stack: ExrStack, group: dict) -> list[torch.Tensor]:
    return [stack.get(n) for n in group["layers"]]


def rule_mask(stack: ExrStack, rule: dict) -> torch.Tensor:
    """One selection rule -> MASK [frames, H, W]."""
    kind = rule.get("type")
    if kind == "crypto":
        _, group = crypto_group(stack, rule.get("layer", "auto"))
        names = rule.get("names", [])
        ids = masks.resolve_names(group["manifest"], ",".join(names) if isinstance(names, list) else str(names))
        return masks.crypto_mask(crypto_ranks(stack, group), list(ids.values()))
    if kind == "depth":
        d = stack.get(stack.resolve(rule.get("layer", "auto"), "depth"))[..., 0]
        if "z" in rule:
            # z = focus depth, range = in-focus width, smooth = 0 hard edge .. 1 fades out over another range/2
            z, rng = float(rule["z"]), max(float(rule.get("range", 1)), 0.0)
            near, far, falloff = z - rng / 2, z + rng / 2, float(rule.get("smooth", 0.5)) * rng / 2
        else:
            near, far, falloff = float(rule.get("near", 0)), float(rule.get("far", 1)), float(rule.get("falloff", 0))
        return masks.depth_mask(d, near, far, falloff, bool(rule.get("normalized", False)))
    if kind == "position":
        p = stack.get(stack.resolve(rule.get("layer", "auto"), "position"))
        if "range" in rule:
            # range = radius (sphere, cylinder) or half-size (cube); smooth fades out over another radius
            r = max(float(rule["range"]), 0.0)
            size, falloff = [r, r, r], float(rule.get("smooth", 0.3)) * r
        else:
            size = rule.get("size", [1, 1, 1])
            size, falloff = size if isinstance(size, list) else [size] * 3, float(rule.get("falloff", 0))
        return masks.position_mask(p, rule.get("shape", "sphere"), tuple(rule.get("center", [0, 0, 0])),
                                   tuple(size), falloff, axis=rule.get("axis", "Y"))
    if kind == "normal":
        n = stack.get(stack.resolve(rule.get("layer", "auto"), "normal"))
        return masks.normal_mask(n, tuple(rule.get("direction", [0, 1, 0])), float(rule.get("angle", 45)),
                                 float(rule.get("falloff", 10)))
    raise ValueError(f"Unknown mask rule type: {kind!r}")


def evaluate(stack: ExrStack, rules: list[dict]) -> torch.Tensor:
    out = None
    for rule in rules:
        if rule.get("enabled", True):
            out = masks.combine(out, rule_mask(stack, rule), rule.get("op", "add"))
    if out is None:
        h, w = stack.size
        out = torch.zeros(stack.frame_count, h, w)
    return out


def _finish(mask: torch.Tensor, invert: bool) -> torch.Tensor:
    return 1.0 - mask if invert else mask


class SFMaskFromCryptomatte(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFMaskFromCryptomatte",
            display_name="SF Mask from Cryptomatte",
            category=CATEGORY,
            description="Matte objects or materials by name. Names come from the file's cryptomatte manifest; "
                        "separate several with commas, and use * wildcards.",
            search_aliases=["cryptomatte", "matte", "id mask", "object mask"],
            inputs=[
                ExrType.Input("exr"),
                io.Combo.Input("cryptomatte", options=["auto"], default="auto"),
                io.String.Input("names", default="", multiline=True, tooltip="e.g. head, eyes, fern*"),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[io.Mask.Output(display_name="mask"), io.String.Output(display_name="matched")],
        )

    @classmethod
    def validate_inputs(cls, cryptomatte):
        return True

    @classmethod
    def execute(cls, exr, cryptomatte, names, invert):
        _, group = crypto_group(exr, cryptomatte)
        ids = masks.resolve_names(group["manifest"], names)
        mask = masks.crypto_mask(crypto_ranks(exr, group), list(ids.values()))
        return io.NodeOutput(_finish(mask, invert), ", ".join(ids))


class SFMaskFromDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFMaskFromDepth",
            display_name="SF Mask from Depth",
            category=CATEGORY,
            description="Select everything between a near and far distance, with a soft falloff on both sides.",
            inputs=[
                ExrType.Input("exr"),
                dynamic_layer(tooltip="'auto' finds the depth/Z pass."),
                io.Float.Input("near", default=0.0, min=-1e6, max=1e6, step=0.01),
                io.Float.Input("far", default=10.0, min=-1e6, max=1e6, step=0.01),
                io.Float.Input("falloff", default=0.5, min=0.0, max=1e6, step=0.01),
                io.Boolean.Input("normalized", default=False, tooltip="Treat depth as 0-1 between nearest and farthest."),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def execute(cls, exr, layer, near, far, falloff, normalized, invert):
        rule = {"type": "depth", "layer": layer, "near": near, "far": far, "falloff": falloff, "normalized": normalized}
        return io.NodeOutput(_finish(rule_mask(exr, rule), invert))


class SFMaskFromPosition(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFMaskFromPosition",
            display_name="SF Mask from Position",
            category=CATEGORY,
            description="Select a region of the 3D scene from the world-position pass: inside a sphere or box, or on "
                        "one side of a plane. The mask follows the region across a whole sequence.",
            inputs=[
                ExrType.Input("exr"),
                dynamic_layer(tooltip="'auto' finds the position/P pass."),
                io.Combo.Input("shape", options=["sphere", "box", "cylinder", "plane"], default="sphere"),
                io.Combo.Input("axis", options=list(masks.AXES), default="Y", tooltip="Cylinder axis (the up axis)."),
                io.Float.Input("center_x", default=0.0, min=-1e6, max=1e6, step=0.01),
                io.Float.Input("center_y", default=0.0, min=-1e6, max=1e6, step=0.01),
                io.Float.Input("center_z", default=0.0, min=-1e6, max=1e6, step=0.01),
                io.Float.Input("size_x", default=1.0, min=-1e6, max=1e6, step=0.01,
                               tooltip="Sphere/cylinder: radius. Box: half-size. Plane: normal."),
                io.Float.Input("size_y", default=1.0, min=-1e6, max=1e6, step=0.01),
                io.Float.Input("size_z", default=1.0, min=-1e6, max=1e6, step=0.01),
                io.Float.Input("falloff", default=0.25, min=0.0, max=1e6, step=0.01),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def execute(cls, exr, layer, shape, axis, center_x, center_y, center_z, size_x, size_y, size_z, falloff, invert):
        rule = {"type": "position", "layer": layer, "shape": shape, "axis": axis,
                "center": [center_x, center_y, center_z], "size": [size_x, size_y, size_z], "falloff": falloff}
        return io.NodeOutput(_finish(rule_mask(exr, rule), invert))


class SFMaskFromNormal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFMaskFromNormal",
            display_name="SF Mask from Normal",
            category=CATEGORY,
            description="Select surfaces facing a direction, e.g. 0,1,0 for upward-facing in Y-up scenes.",
            inputs=[
                ExrType.Input("exr"),
                dynamic_layer(tooltip="'auto' finds the normal/N pass."),
                io.Float.Input("direction_x", default=0.0, min=-1.0, max=1.0, step=0.01),
                io.Float.Input("direction_y", default=1.0, min=-1.0, max=1.0, step=0.01),
                io.Float.Input("direction_z", default=0.0, min=-1.0, max=1.0, step=0.01),
                io.Float.Input("angle", default=45.0, min=0.0, max=180.0, step=1.0),
                io.Float.Input("falloff", default=10.0, min=0.0, max=180.0, step=1.0),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def execute(cls, exr, layer, direction_x, direction_y, direction_z, angle, falloff, invert):
        rule = {"type": "normal", "layer": layer, "direction": [direction_x, direction_y, direction_z],
                "angle": angle, "falloff": falloff}
        return io.NodeOutput(_finish(rule_mask(exr, rule), invert))


class SFMaskPicker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFMaskPicker",
            display_name="SF Mask Picker",
            category=CATEGORY,
            description="Pick a mask by clicking on the image: cryptomatte objects, a depth band, or a 3D region. "
                        "The picks are saved as settings and rebuilt at full resolution for every frame.",
            search_aliases=["mask picker", "cryptomatte picker", "interactive mask"],
            inputs=[
                ExrType.Input("exr"),
                io.String.Input("rules", default='{"rules": []}', multiline=True,
                                tooltip="Set by clicking in the preview; editable as JSON."),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, exr, rules, invert):
        data = json.loads(rules or "{}")
        return io.NodeOutput(_finish(evaluate(exr, data.get("rules", [])), invert))


NODES = [SFMaskFromCryptomatte, SFMaskFromDepth, SFMaskFromPosition, SFMaskFromNormal, SFMaskPicker]
