// SF EXR browser helpers: fill layer dropdowns from the file header, and the interactive Mask Picker.
// Everything here talks to the pack's /sf_exr/* routes; nothing queues the workflow.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOAD = "SFExrLoad";
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
  return { path, sequence: widget(found.load, "sequence")?.value ?? "auto", ops };
}

async function inspect(src) {
  const ops = JSON.stringify(src.ops);
  const key = `${src.sequence}|${src.path}|${ops}`;
  if (!infoCache.has(key)) {
    const q = new URLSearchParams({ path: src.path, sequence: src.sequence });
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
  if (!src) return;
  const info = await inspect(src);
  if (info.error) {
    node.sfError = info.error;
    return;
  }
  node.sfError = null;
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

// `chunk` only steps forward after a run while chunking is on. With chunk_size 0 it must stay fixed: a value
// that changes every run would make the Load node, and every AI model after it, re-run instead of caching.
function syncChunkStepping(node, sizeChanged) {
  const size = widget(node, "chunk_size")?.value ?? 0;
  const chunk = widget(node, "chunk");
  const control = widget(node, "control_after_generate");
  if (!chunk || !control) return;
  if (size <= 0) {
    control.value = "fixed";
    chunk.value = 0;
  } else if (sizeChanged) {
    control.value = "increment";
  }
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
    this.build();
    this.preview.onload = () => this.draw();
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
      this.list,
      this.editor,
      this.status,
    ]);
    this.setMode("crypto");
    this.setOp("add");
    this.renderRules();
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
    const key = `${src.sequence}|${src.path}|${JSON.stringify(src.ops)}`;
    this.src = src;
    this.info = info;
    if (key === this.key) return;
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
    return { path: this.src.path, sequence: this.src.sequence, ops: this.src.ops, ...extra };
  }

  loadPreview() {
    if (!this.src) return;
    const layer = this.showLayer ? this.layerSelect.value : "auto";
    const q = new URLSearchParams({ path: this.src.path, sequence: this.src.sequence, layer, size: "1024" });
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

  async updateMask() {
    if (!this.src) return;
    const seq = (this.seq = (this.seq || 0) + 1);
    const body = this.req({ rules: this.rules, size: 1024, invert: widget(this.node, "invert")?.value });
    const r = await api.fetchApi("/sf_exr/mask_preview", { method: "POST", body: JSON.stringify(body) });
    if (seq !== this.seq) return;
    if (!r.ok) {
      this.status.textContent = (await r.json().catch(() => ({}))).error ?? "";
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
`;

app.registerExtension({
  name: "SF.EXR",

  setup() {
    document.head.append(el("style", { textContent: CSS }));
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const cls = nodeData.name;
    if (!LAYER_WIDGETS[cls] && !LAYER_OPS[cls] && cls !== "SFMaskPicker") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      if (cls === LOAD) {
        for (const name of ["path", "sequence"]) {
          const w = widget(this, name);
          const cb = w?.callback;
          if (w) w.callback = (...a) => (cb?.apply(w, a), scheduleRefresh(true));
        }
        const size = widget(this, "chunk_size");
        const cb = size?.callback;
        if (size) size.callback = (...a) => (cb?.apply(size, a), syncChunkStepping(this, true));
        syncChunkStepping(this, false);
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

    const onConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConn?.apply(this, arguments);
      setTimeout(() => refreshNode(this), 50);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (cls === LOAD) syncChunkStepping(this, false);
      if (this.sfPicker) this.sfPicker.renderRules();
      scheduleRefresh(false);
      return r;
    };
  },
});
