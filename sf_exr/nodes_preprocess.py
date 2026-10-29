"""SF EXR to Preprocessor: render passes as control images (depth, normal, edges) for ControlNet-style models."""
import torch
from comfy_api.latest import io

from . import control as ctl
from . import exr_io
from .nodes_io import CATEGORY, ExrType, dynamic_layer
from .nodes_mask import crypto_group, crypto_ranks
from .stack import ExrStack, pick_layer

CURVES = ["inverse", "linear"]


def _role_layer(exr: ExrStack, layer: str, role: str) -> str:
    """The chosen layer, or the pass found by name for 'auto' (never silently the beauty)."""
    if layer and layer != "auto":
        return exr.resolve(layer)
    name = pick_layer(list(exr.layers.values()), role)
    if not name:
        raise ValueError(f"No {role} pass found by name. Pick its layer in the layer dropdown.")
    return name


def _choice(value, key: str):
    """A nested DynamicCombo arrives as a dict holding its own choice; tolerate a plain value too."""
    return value.get(key) if isinstance(value, dict) else value


def depth_control(exr: ExrStack, layer: str, opts: dict) -> torch.Tensor:
    z = exr.get(_role_layer(exr, layer, "depth"))[..., 0]
    curve = opts.get("curve", "inverse")
    rng = opts.get("range", "whole shot")
    mode = _choice(rng, "range")
    if mode == "manual":
        return ctl.depth_to_control(z, float(rng["near"]), float(rng["far"]), curve)
    if mode == "each frame":
        return torch.cat([ctl.depth_to_control(z[i:i + 1], *ctl.depth_range(z[i:i + 1]), curve)
                          for i in range(z.shape[0])])
    return ctl.depth_to_control(z, *ctl.depth_range(z), curve)


def normal_control(exr: ExrStack, layer: str, opts: dict) -> torch.Tensor:
    n = exr.get(_role_layer(exr, layer, "normal"))
    if n.shape[-1] < 3:
        raise ValueError("The normal pass needs 3 channels.")
    rotations = None
    if opts.get("space") == "world to camera":
        rotations = []
        for path in exr.seq.paths:
            r = ctl.camera_rotation(exr_io.read_header(path).attribs)
            if r is None:
                raise ValueError(f"{path} has no worldToCamera matrix (Blender doesn't write one). "
                                 "Render a camera-space normal pass, or use 'as stored'.")
            rotations.append(r)
    return ctl.normals_to_control(n, rotations, bool(opts.get("flip_y", False)))


def edges_control(exr: ExrStack, layer: str, opts: dict) -> torch.Tensor:
    thickness = int(opts.get("thickness", 2))
    source = opts.get("source", "auto")
    chosen = exr.layers.get(layer) if layer and layer != "auto" else None
    if chosen:
        use_crypto = chosen.kind == "crypto" if source == "auto" else source == "cryptomatte"
    else:
        use_crypto = source == "cryptomatte" or (source == "auto" and bool(exr.header.crypto))
    if use_crypto:
        _, group = crypto_group(exr, layer if chosen is not None and chosen.kind == "crypto" else "auto")
        ids = crypto_ranks(exr, group)[0][..., 0]  # strongest rank's ID per pixel
        return ctl.id_edges(ids, thickness)
    z = exr.get(_role_layer(exr, layer if chosen is not None and chosen.kind != "crypto" else "auto", "depth"))
    return ctl.depth_edges(z[..., 0], thickness)


class SFExrToPreprocessor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFExrToPreprocessor",
            display_name="SF EXR to Preprocessor",
            category=CATEGORY,
            description="Turn a render pass into the control image a ControlNet-style model expects: depth (near = "
                        "white), camera-space normals, or edges from cryptomatte object boundaries. Rendered passes "
                        "are exact, so they guide better than depth or edges estimated from the beauty.",
            search_aliases=["controlnet", "control", "preprocessor", "depth map", "normal map", "lineart", "canny",
                            "pass to control"],
            inputs=[
                ExrType.Input("exr"),
                dynamic_layer(tooltip="'auto' finds the depth / normal pass, or the cryptomatte for edges."),
                io.DynamicCombo.Input("control", options=[
                    io.DynamicCombo.Option("depth", [
                        io.Combo.Input("curve", options=CURVES, default="inverse",
                                       tooltip="inverse (1/z) matches the estimated depth most depth models were "
                                               "trained on; linear keeps distances evenly spaced."),
                        io.DynamicCombo.Input("range", options=[
                            io.DynamicCombo.Option("whole shot", []),
                            io.DynamicCombo.Option("each frame", []),
                            io.DynamicCombo.Option("manual", [
                                io.Float.Input("near", default=1.0, min=0.0, max=1e9, step=0.01),
                                io.Float.Input("far", default=100.0, min=0.0, max=1e9, step=0.01),
                            ]),
                        ], tooltip="whole shot keeps one near/far for every frame so sequences don't flicker."),
                    ]),
                    io.DynamicCombo.Option("normal", [
                        io.Combo.Input("space", options=["as stored", "world to camera"], default="as stored",
                                       tooltip="Control models want camera-space normals. 'world to camera' uses "
                                               "the file's worldToCamera matrix, frame by frame."),
                        io.Boolean.Input("flip_y", default=False,
                                         tooltip="For models that expect Y down (DirectX style)."),
                    ]),
                    io.DynamicCombo.Option("edges", [
                        io.Combo.Input("source", options=["auto", "cryptomatte", "depth"], default="auto",
                                       tooltip="auto uses cryptomatte object boundaries when the file has them, "
                                               "else depth jumps."),
                        io.Int.Input("thickness", default=2, min=1, max=16),
                    ]),
                ]),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def validate_inputs(cls, layer):
        return True

    @classmethod
    def execute(cls, exr: ExrStack, layer: str, control: dict):
        build = {"depth": depth_control, "normal": normal_control, "edges": edges_control}.get(control["control"])
        if build is None:
            raise ValueError(f"Unknown control type: {control['control']!r}")
        return io.NodeOutput(build(exr, layer, control))


NODES = [SFExrToPreprocessor]
