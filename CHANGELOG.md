# Changelog

## Unreleased

- **SF EXR Resize:** the options of Resize Image/Mask (dimensions with crop, multiplier, longer/shorter side,
  width, height, megapixels, match size, multiple of N) applied to every layer the way it needs. Color layers use
  the chosen filter in float, so HDR values are never clipped (unlike core's lanczos, which goes through 8-bit);
  bilinear and bicubic antialias when shrinking. Depth, position and normal passes take exact source values
  (optional: filter them like color). Pixel motion vectors scale with the picture. Cryptomatte coverage is summed
  per ID and re-ranked, so mattes match an area-scaled matte exactly and IDs are never blended. Stretching updates
  the pixel aspect ratio; overscan is dropped.
- **SF EXR Load outputs:** `image` is now **selected layer**; `layers`, `colorspace` and `frame_count` are replaced by
  a **context** output. Saved workflows that used the old outputs get an SF EXR Info node inserted, with their
  links moved onto it.
- **SF EXR Info:** from any context (SF EXR Load or SF EXR to Image): frame count, first and last frame, width,
  height, layer, layers, colorspace, and an `info` summary to show with Preview as Text.
- **SF Image to EXR takes SF EXR Load's context too:** an edited copy of the selected layer (already scene-linear)
  goes back into the EXR without any color conversion.
- **SF EXR to Preprocessor:** render passes as ControlNet-style control images, exact instead of estimated from the
  beauty. Depth (near = white, inverse or linear, one near/far for the whole shot so sequences don't flicker, or per
  frame, or manual), normals (camera space; world-space passes converted with the file's worldToCamera matrix;
  left-handed cameras detected), and edges (cryptomatte object boundaries, or depth jumps when there's no
  cryptomatte).
- **Renamed:** SF Prepare for Model is now **SF EXR to Image**, and SF Return from Model is now **SF Image to EXR**.
  Saved workflows still load; search finds the old names too.
- **Frame lists on SF EXR Load:** `frame_start` / `frame_end` are replaced by one `frames` field (empty = all,
  `1001-1100`, `0`, `15, 20`, `1-10, 50-60`, `first-1050`, `1001-last`); `frame_step` stays. Frame 0 can now be
  picked on its own. Saved workflows are converted when opened. The `frames` count output is now `frame_count`.
- **Pre-roll frames:** negative frame numbers (`shot.-010.exr`, printf-style padding) are part of the sequence.
  A hyphen that is part of the name (`shot-0010.exr`) still reads as frame 10.
- **Hold** now reuses the nearest earlier frame on disk, which also covers stepped and listed frames.
- **Browse on SF EXR Load:** a file browser for the ComfyUI machine's folders, with sequences collapsed to one line
  (pattern, range, count, missing frames), recent folders and shortcuts. Files are read in place. Dropping a single
  image on the node copies it into `input/sf_exr/` for quick tests.
- **Mask Picker timeline:** for sequences, a frame slider over the frames SF EXR Load will run, with missing frames
  marked; ← / → over the preview step a frame (Shift: 10). The frame shown is saved with the node but never sent
  with the prompt.
- **SF EXR Load layout:** `sequence`, `frame_step` and `missing_frames` move to the advanced section; the `frames`
  label shows the shot's range.
- **Automatic refresh:** headers and previews are re-read when you return to the ComfyUI tab (previews redraw only
  if the files changed), and right-click → Reload files does it by hand.
- **Shared servers:** when ComfyUI is started with `--listen`, SF EXR only reads, browses and writes inside ComfyUI's
  input / output / temp folders and the folders in `SF_EXR_ROOTS`. This also covers the preview routes, which
  previously read any path they were sent.
- **A lone numbered frame keeps its number:** loading only `shot.0050.exr` now saves as frame 50 instead of the
  next counter.

## 0.2.0

- **SF Merge:** Nuke-style merge with 22 operations (over, under, plus, minus, from, multiply, screen, overlay,
  hard-light, max, min, average, difference, divide, hypot, in, out, atop, xor, mask, stencil, copy), an opacity
  slider, a mask, and "set bbox to" (union / intersection / A / B). A and B can each be an EXR or an image; the
  output takes B's type, and with an EXR only the chosen layer changes.
- **Shot metadata kept on save:** timecode, pixel aspect ratio and screen window were dropped before; anamorphic
  plates now stay anamorphic.
- **Overscan kept:** pixels outside the display window survive Save and Convert, including on edited layers
  (the edit is pasted over the original overscan). Auto-cropped renders are written at full display size so
  edits anywhere are kept.
- **Lighter on memory:** Save streams untouched layers from disk one frame at a time instead of holding the
  whole sequence, and sequence frames load in parallel.

## 0.1.0

First release.

- **SF EXR Load** — single, multi-layer and multi-part EXR plus any image format; numbered sequences found from
  one frame or a `####` / `%04d` / `@` pattern; missing frames error or hold.
- **SF EXR Get / Set Layer**, **Rename Layers**, **Keep/Remove Layers** — renames keep cryptomatte metadata in sync.
- **SF EXR Save / Convert**, **SF Image Save** — keep layout, metadata and frame numbers; cryptomatte always
  lossless 32-bit float; lossy codecs never touch data layers (auto multi-part when needed).
- **SF OCIO Transform / Display**, **SF Prepare for Model / Return from Model** — scene-linear ↔ model hand-off
  with masked composite back into the plate.
- **SF Mask from Cryptomatte / Depth / Position / Normal** and the interactive **SF Mask Picker**.
- Layer dropdowns and picker previews come from the file header in the browser; nothing has to run first.
