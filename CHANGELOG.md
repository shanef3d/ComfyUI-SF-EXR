# Changelog

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
