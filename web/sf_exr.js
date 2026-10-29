// SF EXR browser helpers: fill layer dropdowns from the file header, and the interactive Mask Picker.
// Everything here talks to the pack's /sf_exr/* routes; nothing queues the workflow.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOAD = "SFExrLoad";
const INFO = "SFExrInfo";
// node type -> [widget name, which layers it lists]
const LAYER_WIDGETS = {
  SFExrLoad: ["layer", "all"],
  SFExrGetLayer: ["layer", "all"],
  SFExrSetLayer: ["layer", "all"],
  SFPrepareForModel: ["layer", "color"],
  SFMaskFromDepth: ["layer", "data"],
  SFMaskFromPosition: ["layer", "data"],
  SFMaskFromNormal: ["layer", "data"],
  SFMaskFromCryptomatte: ["cryptomatte", "crypto"],
  SFMerge: ["layer", "all"],
  SFExrToPreprocessor: ["layer", "all"],
};
const UPSTREAM = { exr: true, context: true, B: true, A: true };
// Nodes between a Load and its users that change layer names; the routes replay them so dropdowns match
const LAYER_OPS = {
  SFExrRenameLayers: (n) => ({
    type: "rename",
    strip: widget(n, "strip_prefix")?.value ?? "none",
    find: widget(n, "find")?.value ?? "",
    replace: widget(n, "replace")?.value ?? "",
    mapping: widget(n, "mapping")?.value ?? "",
  }),
  SFExrKeepLayers: (n) => ({ type: "keep", mode: widget(n, "mode")?.value ?? "remove", layers: widget(n, "layers")?.value ?? "" }),
};

const infoCache = new Map();

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

// Walk back through exr/context links (and reroutes) to the SF EXR Load node feeding this one,
// collecting rename/keep operations on the way (nearest first).
function findLoad(node, ops = [], seen = new Set(), start = node) {
  if (!node || seen.has(node.id)) return null;
  seen.add(node.id);
  if (node.comfyClass === LOAD) return { load: node, ops };
  if (node !== start && LAYER_OPS[node.comfyClass]) {
    const op = LAYER_OPS[node.comfyClass](node);
    if (op.type === "rename" || op.layers.trim()) ops = [...ops, op];
  }
  for (let i = 0; i < (node.inputs?.length ?? 0); i++) {
    const inp = node.inputs[i];
    if (inp.link == null) continue;
    if (!UPSTREAM[inp.name] && node.comfyClass !== "Reroute") continue;
    const found = findLoad(node.getInputNode(i), ops, seen, start);
    if (found) return found;
  }
  return null;
}

function sourceOf(node) {
  const found = findLoad(node);
  if (!found) return null;
  const path = widget(found.load, "path")?.value?.trim();
  if (!path) return null;
  const ops = [...found.ops].reverse();
  return {
    path,
    sequence: widget(found.load, "sequence")?.value ?? "auto",
    frames: widget(found.load, "frames")?.value ?? "",
    step: widget(found.load, "frame_step")?.value ?? 1,
    ops,
  };
}

async function inspect(src) {
  const ops = JSON.stringify(src.ops);
  const key = `${src.sequence}|${src.path}|${ops}|${src.frames}|${src.step}`;
  if (!infoCache.has(key)) {
    const q = new URLSearchParams({ path: src.path, sequence: src.sequence, frames: src.frames, step: String(src.step) });
    if (src.ops.length) q.set("ops", ops);
    infoCache.set(key, api.fetchApi(`/sf_exr/inspect?${q}`).then((r) => r.json()));
  }
  const info = await infoCache.get(key);
  if (info.error) infoCache.delete(key);
  return info;
}

function setOptions(w, values) {
  const list = ["auto", ...values.filter((v) => v !== "auto")];
  // A saved workflow may name a layer this file lacks; keep it so the value isn't silently changed
  if (w.value && !list.includes(w.value)) list.push(w.value);
  w.options.values = list;
}

function setPlaceholder(node, name, text) {
  const w = widget(node, name);
  const input = w?.inputEl ?? w?.element;
  if (input) input.placeholder = text;
}

async function refreshNode(node) {
  const spec = LAYER_WIDGETS[node.comfyClass];
  const src = sourceOf(node);
  if (!src) {
    node.sfPicker?.clear("");
    return;
  }
  const info = await inspect(src);
  if (info.error) {
    node.sfError = info.error;
    node.sfPicker?.clear(info.error); // never leave the previous shot on screen
    return;
  }
  node.sfError = null;
  if (node.comfyClass === LOAD) {
    // The shot's own range next to `frames` (canvas text boxes can't show placeholder text)
    const w = widget(node, "frames");
    if (w) {
      const range = info.first == null ? "" : info.frames > 1 ? `${info.first}-${info.last}` : `${info.first}`;
      w.label = range ? `frames  ${range}` : "frames";
      w.options.placeholder = range || "all";
    }
  }
  if (spec) {
    const w = widget(node, spec[0]);
    if (w) {
      const names =
        spec[1] === "crypto"
          ? Object.keys(info.crypto)
          : info.layers.filter((l) => spec[1] === "all" || l.kind === spec[1]).map((l) => l.name);
      setOptions(w, names);
    }
  }
  if (node.comfyClass === "SFMaskFromCryptomatte") {
    setPlaceholder(node, "names", Object.values(info.crypto).flatMap((g) => g.names).join(", "));
  }
  if (LAYER_OPS[node.comfyClass]) {
    const names = info.layers.map((l) => l.name);
    if (node.comfyClass === "SFExrKeepLayers") setPlaceholder(node, "layers", names.join("\n"));
    else setPlaceholder(node, "mapping", names.map((n) => `${n} = `).join("\n"));
  }
  node.sfPicker?.load(src, info);
  node.setDirtyCanvas(true, true);
}

function refreshAll() {
  for (const node of app.graph?._nodes ?? []) {
    if (LAYER_WIDGETS[node.comfyClass] || LAYER_OPS[node.comfyClass] || node.comfyClass === "SFMaskPicker") {
      refreshNode(node);
    }
  }
}

let refreshTimer = null;
function scheduleRefresh(clearCache) {
  if (clearCache) infoCache.clear();
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(refreshAll, 300);
}

// ------------------------------------------------------------------ Mask Picker

const MODES = [
  ["crypto", "Crypto"],
  ["depth", "Depth"],
  ["position", "Position"],
];
const OPS = [
  ["add", "+"],
  ["subtract", "−"],
  ["intersect", "∩"],
];
const SHAPES = ["sphere", "cube", "cylinder"];

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  Object.assign(e, attrs);
  if (attrs.style) e.style.cssText = attrs.style;
  for (const c of children) e.append(c);
  return e;
}

function fmt(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
}

// Label + slider + number box. The slider can work in a mapped 0..1 space while the number shows real units.
function slider(label, { min, max, value, onChange }) {
  const span = Math.max(max - min, 1e-6);
  const range = el("input", { type: "range", min: "0", max: "1000", step: "1" });
  const num = el("input", { type: "number", step: "any", className: "sf-exr-num" });
  const set = (v) => {
    range.value = String(Math.round(((v - min) / span) * 1000));
    num.value = fmt(v);
  };
  set(value);
  range.oninput = () => {
    const v = min + (Number(range.value) / 1000) * span;
    num.value = fmt(v);
    onChange(v);
  };
  num.onchange = () => {
    const v = Number(num.value);
    if (Number.isFinite(v)) {
      set(v);
      onChange(v);
    }
  };
  return el("div", { className: "sf-exr-row" }, [el("span", { className: "sf-exr-label", textContent: label }), range, num]);
}

function select(options, value, onChange, title = "") {
  const s = el("select", { className: "sf-exr-select", title });
  for (const [v, label] of options) s.append(el("option", { value: v, textContent: label }));
  s.value = value;
  s.onchange = () => onChange(s.value);
  return s;
}

function toggle(label, title, onChange) {
  const b = el("button", { textContent: label, title, className: "sf-exr-btn" });
  b.onclick = () => {
    b.classList.toggle("on");
    onChange(b.classList.contains("on"));
  };
  return b;
}

class Picker {
  constructor(node) {
    this.node = node;
    this.mode = "crypto";
    this.op = "add";
    this.selected = -1;
    this.showLayer = false;
    this.showMask = false;
    this.src = null;
    this.info = null;
    this.ranges = {};
    this.preview = new Image();
    this.mask = null;
    this.rulesWidget = widget(node, "rules");
    this.selection = [];
    this.missing = new Set();
    this.frame = null; // frame on screen; display only, never part of the prompt
    this.build();
    this.preview.onload = () => this.draw();
    this.preview.onerror = () => this.previewFailed();
  }

  // No source, or it can't be read: blank the preview so the previous shot isn't mistaken for this one
  clear(message) {
    // srcKey (the last shot shown) is kept: if that shot comes back it returns to its frame, any other starts at the first
    this.key = null;
    this.src = null;
    this.info = null;
    this.mask = null;
    this.selection = [];
    this.frame = null;
    this.timeline.style.display = "none";
    this.preview.removeAttribute("src");
    this.blank();
    this.status.textContent = message;
  }

  blank() {
    const ctx = this.canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.node.setDirtyCanvas(true, true);
  }

  async previewFailed() {
    const url = this.preview.src;
    if (!url) return;
    this.blank();
    const j = await fetch(url).then((r) => r.json()).catch(() => ({}));
    if (this.preview.src === url) this.status.textContent = j.error ?? "Preview failed";
  }

  get rules() {
    try {
      return JSON.parse(this.rulesWidget.value || "{}").rules ?? [];
    } catch {
      return [];
    }
  }

  set rules(rules) {
    this.rulesWidget.value = JSON.stringify({ rules }, null, 1);
    if (this.selected >= rules.length) this.selected = rules.length - 1;
    this.renderRules();
    this.updateMask();
  }

  // Edit one rule in place; the mask preview refreshes shortly after the last change
  patch(i, changes) {
    const rules = this.rules;
    Object.assign(rules[i], changes);
    this.rulesWidget.value = JSON.stringify({ rules }, null, 1);
    this.renderList();
    clearTimeout(this.patchTimer);
    this.patchTimer = setTimeout(() => this.updateMask(), 120);
  }

  build() {
    const btn = (label, title, onclick) => el("button", { textContent: label, title, onclick, className: "sf-exr-btn" });
    this.modeBtns = MODES.map(([m, label]) => btn(label, "", () => this.setMode(m)));
    this.opBtns = OPS.map(([o, label]) => btn(label, o, () => this.setOp(o)));
    this.layerSelect = select([["auto", "auto"]], "auto", () => this.showLayer && this.loadPreview(), "Preview layer");
    this.layerBtn = toggle("Preview layer", "", (on) => ((this.showLayer = on), this.loadPreview()));
    this.groupSelect = select([], "", () => {}, "Cryptomatte");
    this.canvas = el("canvas", { style: "width:100%;display:block;cursor:crosshair;border-radius:4px;background:#000" });
    this.canvas.onpointerdown = (e) => this.click(e);
    this.buildTimeline();
    this.list = el("div", { className: "sf-exr-chips" });
    this.editor = el("div", { className: "sf-exr-editor" });
    this.status = el("div", { className: "sf-exr-status" });
    this.root = el("div", { className: "sf-exr-picker" }, [
      el("div", { className: "sf-exr-row" }, [
        ...this.modeBtns,
        el("span", { style: "flex:1" }),
        ...this.opBtns,
        btn("Clear", "", () => ((this.selected = -1), (this.rules = []))),
      ]),
      el("div", { className: "sf-exr-row" }, [this.layerBtn, this.layerSelect, el("span", { style: "flex:1" }), this.groupSelect]),
      this.canvas,
      this.timeline,
      this.list,
      this.editor,
      this.status,
    ]);
    this.setMode("crypto");
    this.setOp("add");
    this.renderRules();
  }

  // One row under the preview, only for sequences: slider over the frames SF EXR Load will run, a frame
  // number box, and red ticks for frames missing on disk. ← / → step a frame over the preview, Shift steps 10.
  buildTimeline() {
    this.timeRange = el("input", { type: "range", min: "0", max: "0", step: "1", value: "0" });
    this.timeNum = el("input", { type: "number", step: "1", className: "sf-exr-num" });
    this.ticks = el("canvas", { className: "sf-exr-ticks", height: 3 });
    this.timeRange.oninput = () => this.setFrameIndex(Number(this.timeRange.value), true);
    this.timeRange.onchange = () => this.setFrameIndex(Number(this.timeRange.value), false);
    this.timeNum.onchange = () => {
      const f = Number(this.timeNum.value);
      const i = this.selection.findIndex((s) => s >= f);
      this.setFrameIndex(i < 0 ? this.selection.length - 1 : i, false);
    };
    this.timeline = el("div", { className: "sf-exr-row", style: "display:none" }, [
      el("div", { className: "sf-exr-timeline" }, [this.timeRange, this.ticks]),
      this.timeNum,
    ]);
    this.canvas.onpointerenter = () => (this.hover = true);
    this.canvas.onpointerleave = () => (this.hover = false);
    document.addEventListener("keydown", (e) => {
      if (!this.hover || this.selection.length < 2 || !this.root.isConnected) return;
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      e.stopPropagation();
      const step = (e.shiftKey ? 10 : 1) * (e.key === "ArrowLeft" ? -1 : 1);
      this.setFrameIndex(this.index() + step, false);
    }, true);
  }

  index() {
    return Math.max(0, this.selection.indexOf(this.frame));
  }

  // Take the Load node's frame selection; keep the frame on screen if it is still in it. Returns true when
  // the frame on screen changed.
  updateTimeline(info) {
    const sel = info.selection ?? [];
    this.selection = sel;
    this.missing = new Set(info.selection_missing ?? []);
    const before = this.frame;
    if (sel.length < 2) {
      this.timeline.style.display = "none";
      this.frame = sel[0] ?? null;
      return this.frame !== before;
    }
    this.timeline.style.display = "";
    this.timeRange.max = String(sel.length - 1);
    const want = this.frame ?? this.node.properties?.sfFrame ?? null;
    let i = want == null ? 0 : sel.findIndex((f) => f >= want);
    if (i < 0) i = sel.length - 1;
    this.showFrame(i);
    this.drawTicks();
    return this.frame !== before;
  }

  showFrame(i) {
    this.frame = this.selection[i];
    this.node.properties ??= {};
    this.node.properties.sfFrame = this.frame;
    this.timeRange.value = String(i);
    this.timeNum.value = String(this.frame);
    this.timeNum.classList.toggle("missing", this.missing.has(this.frame));
    this.timeNum.title = this.missing.has(this.frame) ? "Missing on disk; showing the frame held for it" : "";
  }

  drawTicks() {
    const c = this.ticks;
    c.width = Math.max(1, c.clientWidth || 300);
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const n = this.selection.length;
    if (n < 2 || !this.missing.size) return;
    ctx.fillStyle = "#f66";
    this.selection.forEach((f, i) => {
      if (this.missing.has(f)) ctx.fillRect(Math.round((i / (n - 1)) * (c.width - 2)), 0, 2, c.height);
    });
  }

  // Scrubbing draws a small preview while dragging, the full size once let go.
  setFrameIndex(i, scrubbing) {
    if (this.selection.length < 2) return;
    i = Math.min(this.selection.length - 1, Math.max(0, i));
    if (this.selection[i] === this.frame && !scrubbing && !this.scrubbed) return;
    this.showFrame(i);
    clearTimeout(this.scrubTimer);
    this.scrubbed = scrubbing;
    const size = scrubbing ? 384 : 1024;
    const run = () => {
      this.loadPreview(size);
      this.updateMask(size);
    };
    if (scrubbing) this.scrubTimer = setTimeout(run, 40);
    else run();
  }

  setMode(m) {
    this.mode = m;
    this.modeBtns.forEach((b, i) => b.classList.toggle("on", MODES[i][0] === m));
    this.groupSelect.style.display = m === "crypto" && this.groupSelect.options.length > 1 ? "" : "none";
  }

  setOp(o) {
    this.op = o;
    this.opBtns.forEach((b, i) => b.classList.toggle("on", OPS[i][0] === o));
  }

  async load(src, info) {
    // stamp changes when the files are re-rendered, so a refresh redraws only when something changed
    const key = `${src.sequence}|${src.path}|${JSON.stringify(src.ops)}|${info.stamp}`;
    const srcKey = `${src.sequence}|${src.path}`;
    if (this.srcKey && this.srcKey !== srcKey) {
      // A different shot starts at its first frame (a reopened workflow keeps its saved frame)
      this.frame = null;
      if (this.node.properties) delete this.node.properties.sfFrame;
    }
    this.srcKey = srcKey;
    this.src = src;
    this.info = info;
    this.status.textContent = "";
    const frameChanged = this.updateTimeline(info);
    if (key === this.key) {
      if (frameChanged) {
        this.loadPreview();
        this.updateMask();
      }
      return;
    }
    this.key = key;

    const keepLayer = this.layerSelect.value;
    this.layerSelect.replaceChildren(...[["auto", "auto"], ...info.layers.map((l) => [l.name, l.name])].map(([v, t]) =>
      el("option", { value: v, textContent: t })));
    this.layerSelect.value = info.layers.some((l) => l.name === keepLayer) ? keepLayer : "auto";
    const groups = Object.keys(info.crypto);
    const keepGroup = this.groupSelect.value;
    this.groupSelect.replaceChildren(...groups.map((g) => el("option", { value: g, textContent: g.split(".").pop() })));
    if (groups.includes(keepGroup)) this.groupSelect.value = keepGroup;
    this.setMode(this.mode);

    const r = await api.fetchApi("/sf_exr/ranges", { method: "POST", body: JSON.stringify(this.req()) });
    this.ranges = await r.json();
    this.loadPreview();
    this.renderRules();
    this.updateMask();
  }

  req(extra = {}) {
    const frame = this.frame == null ? {} : { frame: this.frame };
    return { path: this.src.path, sequence: this.src.sequence, ops: this.src.ops, ...frame, ...extra };
  }

  loadPreview(size = 1024) {
    if (!this.src) return;
    const layer = this.showLayer ? this.layerSelect.value : "auto";
    const q = new URLSearchParams({ path: this.src.path, sequence: this.src.sequence, layer, size: String(size) });
    if (this.frame != null) q.set("frame", String(this.frame));
    if (this.src.ops.length) q.set("ops", JSON.stringify(this.src.ops));
    this.preview.src = api.apiURL(`/sf_exr/preview?${q}`);
  }

  draw() {
    const img = this.preview;
    if (!img.naturalWidth) return;
    const c = this.canvas;
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    if (this.showMask) {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, c.width, c.height);
      if (this.mask?.naturalWidth) ctx.drawImage(this.mask, 0, 0, c.width, c.height);
    } else {
      ctx.drawImage(img, 0, 0);
      if (this.mask?.naturalWidth) ctx.drawImage(this.tinted(c.width, c.height), 0, 0);
    }
    this.node.setDirtyCanvas(true, true);
  }

  tinted(w, h) {
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const o = off.getContext("2d");
    o.drawImage(this.mask, 0, 0, w, h);
    const d = o.getImageData(0, 0, w, h);
    for (let i = 0; i < d.data.length; i += 4) {
      const v = d.data[i];
      d.data[i] = 255;
      d.data[i + 1] = 60;
      d.data[i + 2] = 40;
      d.data[i + 3] = v * 0.55;
    }
    o.putImageData(d, 0, 0);
    return off;
  }

  async updateMask(size = 1024) {
    if (!this.src) return;
    const seq = (this.seq = (this.seq || 0) + 1);
    const body = this.req({ rules: this.rules, size, invert: widget(this.node, "invert")?.value });
    const r = await api.fetchApi("/sf_exr/mask_preview", { method: "POST", body: JSON.stringify(body) });
    if (seq !== this.seq) return;
    if (!r.ok) {
      this.status.textContent = (await r.json().catch(() => ({}))).error ?? "";
      this.mask = null; // drop the old overlay rather than drawing it over a different image
      this.draw();
      return;
    }
    this.status.textContent = "";
    const url = URL.createObjectURL(await r.blob());
    const m = new Image();
    m.onload = () => {
      this.mask = m;
      this.draw();
      URL.revokeObjectURL(url);
    };
    m.src = url;
  }

  async click(e) {
    if (!this.src) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    const r = await api.fetchApi("/sf_exr/pick", { method: "POST", body: JSON.stringify(this.req({ x, y })) });
    const pick = await r.json();
    if (pick.error) return (this.status.textContent = pick.error);
    const op = e.shiftKey ? "subtract" : this.op;
    const rules = this.rules;

    if (this.mode === "crypto") {
      const group = this.groupSelect.value || Object.keys(pick.crypto)[0];
      const hits = pick.crypto[group] ?? [];
      if (!hits.length) return (this.status.textContent = "No cryptomatte here");
      const name = hits[0][0];
      let i = rules.findIndex((r) => r.type === "crypto" && r.layer === group && r.op === op);
      if (i < 0) i = rules.push({ type: "crypto", layer: group, names: [], op }) - 1;
      const names = rules[i].names;
      rules[i].names = names.includes(name) ? names.filter((n) => n !== name) : [...names, name];
      this.selected = rules[i].names.length ? i : -1;
      this.rules = rules.filter((r) => r.type !== "crypto" || r.names.length);
    } else if (this.mode === "depth") {
      if (!pick.depth) return (this.status.textContent = "No depth pass");
      const d = this.ranges.depth;
      const span = d ? d.max - d.min : Math.abs(pick.depth.value);
      rules.push({ type: "depth", layer: pick.depth.layer, z: pick.depth.value, range: span * 0.05 || 0.1, smooth: 0.5, op });
      this.selected = rules.length - 1;
      this.rules = rules;
    } else {
      if (!pick.position) return (this.status.textContent = "No position pass");
      const size = this.ranges.position?.size || 10;
      rules.push({ type: "position", layer: pick.position.layer, shape: "sphere", axis: this.ranges.up || "Y",
                   center: pick.position.value, range: size * 0.05, smooth: 0.3, op });
      this.selected = rules.length - 1;
      this.rules = rules;
    }
  }

  label(r) {
    if (r.type === "crypto") return r.names.join(", ");
    if (r.type === "depth") return r.z != null ? `depth ${fmt(r.z)}` : `depth ${fmt(r.near)}–${fmt(r.far)}`;
    if (r.type === "position") return `${r.shape ?? "sphere"} ${fmt(r.range ?? r.size?.[0] ?? 0)}`;
    return r.type;
  }

  renderRules() {
    this.renderList();
    this.renderEditor();
  }

  renderList() {
    this.list.replaceChildren(
      ...this.rules.map((r, i) => {
        const sym = OPS.find(([o]) => o === r.op)?.[1] ?? "+";
        const x = el("span", { textContent: "×", className: "sf-exr-x" });
        x.onclick = (e) => {
          e.stopPropagation();
          if (this.selected === i) this.selected = -1;
          else if (this.selected > i) this.selected--;
          this.rules = this.rules.filter((_, j) => j !== i);
        };
        const chip = el("span", { className: "sf-exr-chip" + (i === this.selected ? " on" : ""), textContent: `${sym} ${this.label(r)} ` }, [x]);
        chip.onclick = () => {
          this.selected = this.selected === i ? -1 : i;
          this.renderRules();
        };
        return chip;
      })
    );
  }

  renderEditor() {
    const i = this.selected;
    const r = this.rules[i];
    if (!r) return this.editor.replaceChildren();
    const rows = [
      el("div", { className: "sf-exr-row" }, [
        el("span", { className: "sf-exr-label", textContent: "Mode" }),
        select(OPS.map(([o, s]) => [o, `${s} ${o}`]), r.op ?? "add", (v) => this.patch(i, { op: v })),
      ]),
    ];
    if (r.type === "depth") {
      if (r.z == null) {  // rules from older workflows store near/far; convert once so the sliders own them
        const conv = { z: (r.near + r.far) / 2, range: r.far - r.near, smooth: 0.5 };
        Object.assign(r, conv);
        this.patch(i, conv);
      }
      const d = this.ranges.depth ?? { min: 0, max: Math.max(r.z * 2, 1) };
      const span = Math.max(d.max - d.min, 1e-6);
      rows.push(
        slider("Z", { min: d.min, max: d.max, value: r.z, onChange: (v) => this.patch(i, { z: v }) }),
        slider("Range", { min: 0, max: span, value: r.range, onChange: (v) => this.patch(i, { range: v }) }),
        slider("Smooth", { min: 0, max: 1, value: r.smooth, onChange: (v) => this.patch(i, { smooth: v }) })
      );
    } else if (r.type === "position") {
      if (r.range == null) {
        const conv = { range: r.size?.[0] ?? 1, smooth: 0.3 };
        Object.assign(r, conv);
        this.patch(i, conv);
      }
      const size = this.ranges.position?.size || Math.max(r.range * 4, 1);
      const axis = select([["X", "X"], ["Y", "Y"], ["Z", "Z"]], r.axis ?? this.ranges.up ?? "Y", (v) => this.patch(i, { axis: v }), "Axis");
      axis.style.display = r.shape === "cylinder" ? "" : "none";
      rows.push(
        el("div", { className: "sf-exr-row" }, [
          el("span", { className: "sf-exr-label", textContent: "Shape" }),
          select(SHAPES.map((s) => [s, s]), r.shape === "box" ? "cube" : r.shape ?? "sphere", (v) => {
            axis.style.display = v === "cylinder" ? "" : "none";
            this.patch(i, { shape: v });
          }),
          axis,
        ]),
        slider("Range", { min: 0, max: size / 2, value: r.range, onChange: (v) => this.patch(i, { range: v }) }),
        slider("Smooth", { min: 0, max: 1, value: r.smooth, onChange: (v) => this.patch(i, { smooth: v }) })
      );
    }
    this.editor.replaceChildren(...rows);
  }
}

const CSS = `
.sf-exr-picker{display:flex;flex-direction:column;gap:6px;font:12px sans-serif;color:var(--input-text,#ddd)}
.sf-exr-row{display:flex;gap:4px;align-items:center}
.sf-exr-row input[type=range]{flex:1;min-width:40px}
.sf-exr-btn,.sf-exr-select,.sf-exr-num{background:var(--comfy-input-bg,#222);color:inherit;border:1px solid var(--border-color,#444);
  border-radius:4px;padding:2px 6px;font:inherit}
.sf-exr-btn{cursor:pointer}
.sf-exr-btn.on{background:var(--p-primary-color,#3b82f6);border-color:transparent;color:#fff}
.sf-exr-select{max-width:140px}
.sf-exr-num{width:64px}
.sf-exr-label{width:48px;opacity:.8}
.sf-exr-chips{display:flex;flex-wrap:wrap;gap:4px}
.sf-exr-chip{background:var(--comfy-input-bg,#222);border:1px solid transparent;border-radius:10px;padding:1px 8px;cursor:pointer}
.sf-exr-chip.on{border-color:var(--p-primary-color,#3b82f6)}
.sf-exr-x{cursor:pointer;opacity:.6}
.sf-exr-x:hover{opacity:1}
.sf-exr-editor{display:flex;flex-direction:column;gap:4px}
.sf-exr-status{color:#f88;min-height:1em}
.sf-exr-timeline{flex:1;display:flex;flex-direction:column;gap:1px;min-width:40px}
.sf-exr-timeline input[type=range]{width:100%;margin:0}
.sf-exr-ticks{width:100%;height:3px;display:block}
.sf-exr-num.missing{color:#f88}
.sf-exr-overlay{position:fixed;inset:0;z-index:10000;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.45)}
.sf-exr-browse{display:flex;flex-direction:column;gap:8px;width:min(760px,calc(100vw - 32px));height:min(600px,calc(100vh - 64px));
  padding:12px;box-sizing:border-box;background:var(--comfy-menu-bg,#1e1e1e);border:1px solid var(--border-color,#444);
  border-radius:8px;font:13px sans-serif;color:var(--input-text,#ddd)}
.sf-exr-browse-path{flex:1;min-width:0;background:var(--comfy-input-bg,#222);color:inherit;border:1px solid var(--border-color,#444);
  border-radius:4px;padding:3px 6px;font:inherit}
.sf-exr-browse .sf-exr-select{max-width:160px}
.sf-exr-wrap{flex-wrap:wrap}
.sf-exr-browse-list{flex:1;overflow:auto;border:1px solid var(--border-color,#444);border-radius:4px}
.sf-exr-entry{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:16px;padding:4px 10px;cursor:pointer;white-space:nowrap}
.sf-exr-entry>span:first-child{overflow:hidden;text-overflow:ellipsis}
.sf-exr-entry:hover{background:var(--comfy-input-bg,#2a2a2a)}
.sf-exr-entry.dir{opacity:.85}
.sf-exr-entry .meta{opacity:.7;font-variant-numeric:tabular-nums;text-align:right}
.sf-exr-entry .meta.warn{color:#f88;opacity:1}
.sf-exr-empty{cursor:default;opacity:.6}
`;

// --- Files: browse reads files where they are on the ComfyUI server; nothing is copied. Dropping one image on
// SF EXR Load copies it into input/sf_exr/<name>/ for quick tests; sequences and folders go through browse.

const IMAGE_EXT = [".exr", ".sxr", ".dpx", ".cin", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".hdr", ".tga", ".bmp", ".webp"];
const FRAME_NAME = /^(.*?)(\d+)(\.[A-Za-z0-9]+)$/;
const RECENT_KEY = "sf_exr.recent";

function isImageName(name) {
  const dot = name.lastIndexOf(".");
  return dot > 0 && IMAGE_EXT.includes(name.slice(dot).toLowerCase());
}

// Same rule as sequence.py: the last digit run before the extension is the frame; a "-" right after
// "." or "_" (or at the start) is a minus sign.
function splitFrame(name) {
  const m = FRAME_NAME.exec(name);
  if (!m) return null;
  let [, head, digits, tail] = m;
  let frame = Number(digits);
  if (head.endsWith("-") && (head.length === 1 || /[._]$/.test(head.slice(0, -1)))) {
    head = head.slice(0, -1);
    frame = -frame;
  }
  return { head, frame, tail };
}

function notify(detail, severity = "error") {
  const toast = app.extensionManager?.toast;
  if (toast) toast.add({ severity, summary: "SF EXR Load", detail, life: 6000 });
  else console.warn("[SF EXR]", detail);
}

function setWidget(node, name, value) {
  const w = widget(node, name);
  if (!w) return;
  w.value = value;
  w.callback?.(value);
}

// Point SF EXR Load at a file or sequence; a new source starts from all of its frames.
function setSource(node, path) {
  setWidget(node, "frames", "");
  setWidget(node, "path", path);
}

async function dropStill(node, file) {
  const f = splitFrame(file.name);
  const folder = (f ? f.head.replace(/[._ -]+$/, "") : file.name.replace(/\.[^.]+$/, "")) || "image";
  const body = new FormData();
  body.append("file", file, file.name);
  const r = await api.fetchApi(`/sf_exr/upload?folder=${encodeURIComponent(folder)}&reset=1`, { method: "POST", body });
  const j = await r.json().catch(() => ({ error: `Upload failed (${r.status})` }));
  if (j.error) return notify(`${file.name}: ${j.error}`);
  setSource(node, j.path);
}

function onLoadDrop(node, e) {
  const items = [...(e.dataTransfer?.items ?? [])].filter((i) => i.kind === "file");
  if (!items.length) return false;
  // The DataTransfer is only readable during the event
  const entry = items[0].webkitGetAsEntry?.();
  const file = items[0].getAsFile();
  if (items.length > 1 || entry?.isDirectory) {
    notify("Drop a single image here. Use browse for sequences and folders; it reads them in place.", "warn");
  } else if (!file || !isImageName(file.name)) {
    notify(`${file?.name ?? "That file"} isn't an image SF EXR Load reads.`, "warn");
  } else {
    dropStill(node, file);
  }
  return true;
}

function recentDirs() {
  try {
    const v = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

function rememberDir(dir) {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify([dir, ...recentDirs().filter((d) => d !== dir)].slice(0, 8)));
  } catch {}
}

function folderOf(path) {
  const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return i > 0 ? path.slice(0, i) : "";
}

// Server-side file browser: lists the ComfyUI machine's folders with sequences collapsed to one line.
class Browser {
  constructor(node) {
    this.node = node;
    this.dir = null;
    this.parent = null;
    const btn = (label, title, onclick) => el("button", { textContent: label, title, onclick, className: "sf-exr-btn" });
    this.pathInput = el("input", { className: "sf-exr-browse-path", spellcheck: false });
    this.pathInput.onkeydown = (e) => {
      e.stopPropagation();
      if (e.key === "Enter") this.open(this.pathInput.value.trim().replace(/^["']|["']$/g, ""));
      if (e.key === "Escape") this.close();
    };
    this.recent = el("select", { className: "sf-exr-select", title: "Recent" });
    this.recent.onchange = () => this.recent.value && this.open(this.recent.value);
    this.upBtn = btn("↑", "Up", () => this.parent && this.open(this.parent));
    this.places = el("div", { className: "sf-exr-row sf-exr-wrap" });
    this.list = el("div", { className: "sf-exr-browse-list" });
    this.status = el("div", { className: "sf-exr-status" });
    this.panel = el("div", { className: "sf-exr-browse" }, [
      el("div", { className: "sf-exr-row" }, [this.upBtn, this.pathInput, this.recent, btn("×", "Close", () => this.close())]),
      this.places,
      this.list,
      this.status,
    ]);
    this.overlay = el("div", { className: "sf-exr-overlay" }, [this.panel]);
    this.overlay.onpointerdown = (e) => e.target === this.overlay && this.close();
    this.onKey = (e) => e.key === "Escape" && this.close();
  }

  show() {
    document.body.append(this.overlay);
    document.addEventListener("keydown", this.onKey);
    const current = widget(this.node, "path")?.value?.trim() ?? "";
    this.open(folderOf(current) || recentDirs()[0] || "", true);
  }

  close() {
    this.overlay.remove();
    document.removeEventListener("keydown", this.onKey);
  }

  async open(dir, fallback = false) {
    const r = await api.fetchApi(`/sf_exr/browse?${new URLSearchParams({ dir })}`);
    const j = await r.json().catch(() => ({ error: `Browse failed (${r.status})` }));
    if (j.error) {
      if (fallback && dir) return this.open("", false); // e.g. the last folder was moved; start at input
      this.status.textContent = j.error;
      return;
    }
    this.status.textContent = "";
    this.dir = j.dir;
    this.parent = j.parent;
    this.pathInput.value = j.dir;
    this.upBtn.disabled = !j.parent;
    this.recent.replaceChildren(
      el("option", { value: "", textContent: "Recent" }),
      ...recentDirs().map((d) => el("option", { value: d, textContent: d }))
    );
    this.places.replaceChildren(
      ...(j.places ?? []).map((p) => {
        const b = el("button", { textContent: p.name, title: p.path, className: "sf-exr-btn" });
        b.onclick = () => this.open(p.path);
        return b;
      })
    );
    this.list.replaceChildren(...j.entries.map((e) => this.row(e)));
    if (!j.entries.length) this.list.append(el("div", { className: "sf-exr-entry sf-exr-empty", textContent: "No images or folders" }));
  }

  row(e) {
    const cells = [el("span", { textContent: e.type === "dir" ? `${e.name}/` : e.name })];
    if (e.type === "seq") {
      cells.push(el("span", { className: "meta", textContent: `${e.first}–${e.last}` }));
      cells.push(el("span", { className: e.missing ? "meta warn" : "meta", textContent: e.missing ? `${e.count} · ${e.missing} missing` : `${e.count}` }));
    } else {
      cells.push(el("span"), el("span"));
    }
    const row = el("div", { className: `sf-exr-entry ${e.type}`, title: e.path }, cells);
    row.onclick = () => (e.type === "dir" ? this.open(e.path) : this.choose(e.path));
    return row;
  }

  choose(path) {
    rememberDir(this.dir);
    setSource(this.node, path);
    this.close();
  }
}

// --- Workflow migration: SF EXR Load's frame_start / frame_end (0 = first / last) became one `frames` list.
// Runs on the saved JSON before the graph is built, so widget values and link slots land in the right places.

function framesFromRange(start, end) {
  const s = Number(start) || 0;
  const e = Number(end) || 0;
  if (!s && !e) return "";
  if (s && s === e) return String(s);
  return `${s || "first"}-${e || "last"}`;
}

const linkField = (l, key, i) => (Array.isArray(l) ? l[i] : l[key]);
const LINK_INDEX = { id: 0, origin_id: 1, origin_slot: 2, target_id: 3, target_slot: 4 };

function migrateLoadNode(node, graph) {
  const named = node.widgets_values_named;
  const vals = node.widgets_values;
  const oldArray = Array.isArray(vals) && vals.length === 7 && typeof vals[2] === "number" && typeof vals[3] === "number";
  const oldObject = vals && !Array.isArray(vals) && "frame_start" in vals;
  let start;
  let end;
  if (named && "frame_start" in named) [start, end] = [named.frame_start, named.frame_end];
  else if (oldArray) [start, end] = [vals[2], vals[3]];
  else if (oldObject) [start, end] = [vals.frame_start, vals.frame_end];
  else return;
  const frames = framesFromRange(start, end);

  if (oldArray) vals.splice(2, 2, frames);
  for (const obj of [oldObject ? vals : null, named]) {
    if (!obj) continue;
    const rebuilt = {};
    for (const [k, v] of Object.entries(obj)) {
      if (k === "frame_start") rebuilt.frames = frames;
      else if (k !== "frame_end") rebuilt[k] = v;
    }
    for (const k of Object.keys(obj)) delete obj[k];
    Object.assign(obj, rebuilt);
  }

  // Input slots: frame_start becomes frames; frame_end goes, so later slots shift down by one.
  const inputs = node.inputs ?? [];
  const links = graph.links ?? [];
  const dropLink = (id) => {
    if (id == null) return;
    const i = links.findIndex((l) => linkField(l, "id", 0) === id);
    if (i < 0) return;
    const [l] = links.splice(i, 1);
    const origin = graph.nodes?.find((n) => n.id === linkField(l, "origin_id", 1));
    const out = origin?.outputs?.[linkField(l, "origin_slot", 2)];
    if (out?.links) out.links = out.links.filter((x) => x !== id);
  };
  const si = inputs.findIndex((i) => i.name === "frame_start");
  if (si >= 0) {
    dropLink(inputs[si].link); // an INT can't feed the new STRING input
    inputs[si] = { localized_name: "frames", name: "frames", type: "STRING", widget: { name: "frames" }, link: null };
  }
  const ei = inputs.findIndex((i) => i.name === "frame_end");
  if (ei >= 0) {
    dropLink(inputs[ei].link);
    inputs.splice(ei, 1);
    for (const l of links) {
      if (linkField(l, "target_id", 3) !== node.id) continue;
      const slot = linkField(l, "target_slot", 4);
      if (slot > ei) {
        if (Array.isArray(l)) l[LINK_INDEX.target_slot] = slot - 1;
        else l.target_slot = slot - 1;
      }
    }
  }
}

// SF EXR Load's outputs were exr, image, alpha, layers, colorspace, frames (later frame_count). They are now
// exr, selected layer, alpha, context; the file details moved to SF EXR Info. Links from the old detail outputs
// are moved onto an SF EXR Info node inserted next to the Load and fed by its context.
const CONTEXT_TYPE = "SF_EXR_MODEL_CONTEXT";
const INFO_OUTPUTS = [
  ["frame_count", "INT"], ["first_frame", "INT"], ["last_frame", "INT"], ["width", "INT"], ["height", "INT"],
  ["layer", "STRING"], ["layers", "STRING"], ["colorspace", "STRING"], ["info", "STRING"],
];
const OLD_DETAIL_TO_INFO = { layers: 6, colorspace: 7, frames: 0, frame_count: 0 };

function migrateLoadOutputs(node, graph) {
  const outs = node.outputs;
  if (!outs || outs.length < 4 || outs[3]?.type === CONTEXT_TYPE) return;
  if (!outs.slice(3).some((o) => o.name in OLD_DETAIL_TO_INFO)) return;

  const links = (graph.links ??= []);
  const linkById = new Map(links.map((l) => [linkField(l, "id", 0), l]));
  const moved = []; // [link, info slot]
  for (const o of outs.slice(3)) {
    for (const id of o.links ?? []) {
      const l = linkById.get(id);
      if (l && o.name in OLD_DETAIL_TO_INFO) moved.push([l, OLD_DETAIL_TO_INFO[o.name]]);
    }
  }
  const image = outs[1];
  if (image?.name === "image") {
    image.name = "selected layer";
    if (image.localized_name) image.localized_name = "selected layer";
  }
  const context = { name: "context", localized_name: "context", type: CONTEXT_TYPE, links: [] };
  outs.splice(3, outs.length - 3, context);
  if (!moved.length) return;

  const asArrays = links.length ? Array.isArray(links[0]) : graph.last_link_id !== undefined;
  const nodeId = Math.max(0, graph.last_node_id ?? 0, graph.state?.lastNodeId ?? 0, ...(graph.nodes ?? []).map((n) => n.id)) + 1;
  let linkId = Math.max(0, graph.last_link_id ?? 0, graph.state?.lastLinkId ?? 0, ...links.map((l) => linkField(l, "id", 0)));
  const makeLink = (origin, oslot, target, tslot, type) => {
    linkId += 1;
    links.push(asArrays ? [linkId, origin, oslot, target, tslot, type]
      : { id: linkId, origin_id: origin, origin_slot: oslot, target_id: target, target_slot: tslot, type });
    return linkId;
  };

  const x = (node.pos?.[0] ?? 0) + (node.size?.[0] ?? 300) + 40;
  const info = {
    id: nodeId, type: INFO, pos: [x, node.pos?.[1] ?? 0], size: [240, 230], flags: {}, order: node.order ?? 0,
    mode: 0, inputs: [{ name: "context", localized_name: "context", type: CONTEXT_TYPE, link: null }],
    outputs: INFO_OUTPUTS.map(([name, type]) => ({ name, localized_name: name, type, links: [] })),
    properties: { "Node name for S&R": INFO }, widgets_values: [],
  };
  graph.nodes.push(info);
  const feed = makeLink(node.id, 3, nodeId, 0, CONTEXT_TYPE);
  context.links.push(feed);
  info.inputs[0].link = feed;
  for (const [l, slot] of moved) {
    if (Array.isArray(l)) [l[1], l[2]] = [nodeId, slot];
    else [l.origin_id, l.origin_slot] = [nodeId, slot];
    info.outputs[slot].links.push(linkField(l, "id", 0));
  }
  if (graph.last_node_id !== undefined) graph.last_node_id = nodeId;
  if (graph.last_link_id !== undefined) graph.last_link_id = linkId;
  if (graph.state) {
    graph.state.lastNodeId = Math.max(graph.state.lastNodeId ?? 0, nodeId);
    graph.state.lastLinkId = Math.max(graph.state.lastLinkId ?? 0, linkId);
  }
}

function migrateWorkflow(graphData) {
  const graphs = [graphData, ...(graphData?.definitions?.subgraphs ?? [])];
  for (const g of graphs) {
    for (const n of [...(g?.nodes ?? [])]) {
      if (n.type !== LOAD) continue;
      migrateLoadNode(n, g);
      migrateLoadOutputs(n, g);
    }
  }
}

app.registerExtension({
  name: "SF.EXR",

  setup() {
    document.head.append(el("style", { textContent: CSS }));
    // Coming back to ComfyUI (e.g. after re-rendering) re-reads headers; previews redraw only if files changed
    window.addEventListener("focus", () => scheduleRefresh(true));
    document.addEventListener("visibilitychange", () => document.visibilityState === "visible" && scheduleRefresh(true));
  },

  async beforeConfigureGraph(graphData) {
    try {
      migrateWorkflow(graphData);
    } catch (e) {
      console.warn("[SF EXR] could not update SF EXR Load frame settings", e);
    }
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const cls = nodeData.name;
    if (!LAYER_WIDGETS[cls] && !LAYER_OPS[cls] && cls !== "SFMaskPicker") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      if (cls === LOAD) {
        for (const name of ["path", "sequence", "frames", "frame_step"]) {
          const w = widget(this, name);
          const cb = w?.callback;
          if (w) w.callback = (...a) => (cb?.apply(w, a), scheduleRefresh(name === "path" || name === "sequence"));
        }
        // Browse button under the path; never saved with the workflow, so widget values keep their positions
        const btn = this.addWidget("button", "browse", null, () => new Browser(this).show(), { serialize: false });
        btn.serialize = false;
        this.widgets.splice(this.widgets.indexOf(btn), 1);
        this.widgets.splice(this.widgets.indexOf(widget(this, "path")) + 1, 0, btn);
      }
      if (LAYER_OPS[cls]) {
        for (const w of this.widgets ?? []) {
          const cb = w.callback;
          w.callback = (...a) => (cb?.apply(w, a), scheduleRefresh(false));
        }
      }
      if (cls === "SFMaskPicker") {
        const rules = widget(this, "rules");
        if (rules) {
          rules.hidden = true;
          rules.computeSize = () => [0, -4];
          if (rules.element) rules.element.style.display = "none";
        }
        this.sfPicker = new Picker(this);
        const inv = widget(this, "invert");
        const cb = inv?.callback;
        if (inv) inv.callback = (...a) => (cb?.apply(inv, a), this.sfPicker.updateMask());
        // Display-only toggle under invert; never sent with the prompt or saved in the workflow
        const finalMask = this.addWidget("toggle", "preview final mask", false, (v) => {
          this.sfPicker.showMask = !!v;
          this.sfPicker.draw();
        }, { serialize: false });
        finalMask.serialize = false;
        this.widgets.splice(this.widgets.indexOf(finalMask), 1);
        this.widgets.splice(this.widgets.indexOf(inv) + 1, 0, finalMask);
        this.addDOMWidget("picker", "sf_exr_picker", this.sfPicker.root, { serialize: false });
        this.setSize([Math.max(this.size[0], 380), Math.max(this.size[1], 560)]);
      }
      setTimeout(() => refreshNode(this), 50);
      return r;
    };

    if (cls === LOAD) {
      nodeType.prototype.onDragOver = (e) => !!e.dataTransfer?.types?.includes("Files");
      nodeType.prototype.onDragDrop = function (e) {
        return onLoadDrop(this, e);
      };
    }

    // Right-click → Reload files: re-read headers and previews after a re-render (also automatic on tab focus)
    const getExtra = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_canvas, options) {
      const r = getExtra?.apply(this, arguments);
      options?.push({ content: "Reload files", callback: () => scheduleRefresh(true) });
      return r;
    };

    const onConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConn?.apply(this, arguments);
      setTimeout(() => refreshNode(this), 50);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (this.sfPicker) this.sfPicker.renderRules();
      scheduleRefresh(false);
      return r;
    };
  },
});
