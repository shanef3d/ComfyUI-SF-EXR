# Changelog

## 0.2.0

- **Shot metadata kept on save:** timecode, pixel aspect ratio and screen window were dropped before; anamorphic
  plates now stay anamorphic.
- **Overscan kept:** pixels outside the display window survive Save and Convert, including on edited layers
  (the edit is pasted over the original overscan). Auto-cropped renders are written at full display size so
  edits anywhere are kept.
- **Long sequences:** SF EXR Load has `chunk_size` / `chunk` (steps forward each run) and a `chunks` output, so a
  shot can be processed a chunk per queued run. Save streams untouched layers from disk instead of holding the
  whole sequence in memory, and frames load in parallel.

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
