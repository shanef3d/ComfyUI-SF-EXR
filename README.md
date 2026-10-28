# SF EXR

Load, edit, mask and save multi-layer EXR files and image sequences in ComfyUI.

- Layer dropdowns fill from the file header as soon as you set a path; nothing has to run first.
- Point at one frame of a numbered sequence (any format) and the whole sequence loads. `####`, `%04d` and `@` patterns work too.
- Masks from cryptomatte names, depth bands, world-position regions and normals, or click them in **SF Mask Picker**.
- OCIO through OpenImageIO's built-in ACES configs; no separate OCIO install.
- Saves keep the source layout, metadata (timecode, pixel aspect), overscan and frame numbers. Cryptomatte is always
  written as lossless 32-bit float.
- Long shots run a chunk at a time: set `chunk_size` on SF EXR Load and queue one run per chunk.

## Install

**ComfyUI-Manager:** search for **SF EXR** and install. Or with comfy-cli: `comfy node install comfyui-sf-exr`.

**Manually:** clone into `ComfyUI/custom_nodes`, then install the one dependency into ComfyUI's Python:

```
git clone https://github.com/shanef3d/ComfyUI-SF-EXR
python -m pip install -r ComfyUI-SF-EXR/requirements.txt
```

`OpenImageIO` ships prebuilt wheels for Windows, Linux and macOS. Torch and numpy come with ComfyUI.
An example workflow is in `example_workflows/` and appears in ComfyUI's Templates browser.

## Nodes

| Node | Does |
|---|---|
| SF EXR Load | File or sequence → `SF_EXR`, plus one layer as IMAGE/MASK |
| SF EXR Get Layer | EXR → one layer as IMAGE/MASK (scene-linear, unclamped) |
| SF EXR Set Layer | Replace or add a layer from IMAGE/MASK |
| SF EXR Rename Layers | Strip prefixes, find/replace, or `old = new` lines (layers and channels); cryptomatte metadata follows |
| SF EXR Keep/Remove Layers | Keep or drop layers by name or wildcard |
| SF Merge | Nuke-style merge of A onto B (over, plus, multiply, screen, in, out, …) with opacity, mask and "set bbox to"; works on EXR layers or images, output matches B |
| SF EXR Save | EXR → multi-layer or multi-part file |
| SF EXR Convert | Re-write EXRs with new compression, bit depth or layout, file to file |
| SF Image Save | PNG / TIFF / JPG / DPX / single-layer EXR at a chosen bit depth |
| SF OCIO Transform | Colorspace conversion (presets for common ACES moves) |
| SF OCIO Display | Display/view transform, or its inverse |
| SF Prepare for Model | Scene-linear layer → 0–1 image for a diffusion model |
| SF Return from Model | Model output → scene-linear, composited into the layer through a mask |
| SF Mask from Cryptomatte | Names or wildcards from the manifest → MASK |
| SF Mask from Depth | Near/far band with falloff |
| SF Mask from Position | Sphere, box, cylinder or plane in world space |
| SF Mask from Normal | Surfaces facing a direction |
| SF Mask Picker | Click cryptomatte objects, depth bands or 3D shapes on a preview; tune each with Z / Range / Smooth or Shape / Range / Smooth; preview any layer or the final mask |

## Typical edit

```
SF EXR Load → SF Prepare for Model → (edit model) → SF Return from Model → SF EXR Save
                    SF Mask Picker ─────────── mask ──────┘
```

Return from Model resizes the edit to the plate, converts it back to scene-linear and composites it through the
mask, so pixels outside the mask keep their original HDR values. Other passes (depth, cryptomatte) pass through
unchanged; they still describe the original render where the edit changed shapes.

## Compression rules

Save and Convert never let a lossy codec damage data. DWAA/DWAB are only lossy on half-float R/G/B channels and
B44 on half channels, so float data layers stay single-part. When a codec would damage a data or cryptomatte
layer (PXR24 on float, B44 on half data), the file is written multi-part and those layers get ZIP.
`compression: source` keeps the original codec; with a lossy source (DWAA) each save loses a little more.

## Layer names across renderers

Cryptomatte layers are found from the file's own `cryptomatte/<id>/name` metadata, so any renderer that follows
the Cryptomatte spec works. Depth, position and normal passes are found by name (`Z`/`depth`, `P`/`position`,
`N`/`normal`, including names like `VRayZDepth`); pick the layer explicitly in the dropdown, or use
**SF EXR Rename Layers**, when a renderer uses something else.

## Development

```
pip install -r requirements.txt numpy pytest ruff torch
ruff check .
pytest tests
```

Tests build small synthetic EXRs (beauty, depth, position, cryptomatte) on the fly; no render files are committed.
Releases: bump `version` in `pyproject.toml` and add a `CHANGELOG.md` entry; pushing that to `main` publishes to
the Comfy Registry.

## License

MIT
