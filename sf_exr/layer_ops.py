"""Layer renaming and filtering, planned on names alone so nodes and the browser routes agree on the result."""
import fnmatch
import re
from types import SimpleNamespace

STRIP_MODES = ["none", "first segment", "all but last"]
KEEP_MODES = ["keep", "remove"]


def parse_mapping(text: str) -> list[tuple[str, str]]:
    pairs = []
    for line in text.replace(",", "\n").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Rename mapping line needs 'old = new': {line!r}")
        left, right = (s.strip() for s in line.split("=", 1))
        pairs.append((left, right))
    return pairs


def rename_plan(suffixes: dict[str, list[str]], kinds: dict[str, str], crypto: dict[str, list[str]],
                strip: str, find: str, replace: str, mapping: str):
    """-> (layer name map, channel name maps, cryptomatte base map). Explicit mapping lines win over the rules."""
    names = list(suffixes)
    explicit, suffix_maps = {}, {}
    for left, right in parse_mapping(mapping):
        if left in suffixes:
            explicit[left] = right
            continue
        layer, _, suf = left.rpartition(".")
        if layer in suffixes and suf in suffixes[layer]:
            if kinds.get(layer) == "crypto":
                raise ValueError(f"Cryptomatte channels can't be renamed ({left}); rename the layer instead.")
            suffix_maps.setdefault(layer, {})[suf] = right.rpartition(".")[2]
        elif left in crypto:  # renaming a cryptomatte by its base name renames every rank layer
            for r in crypto[left]:
                explicit[r] = right + r[len(left):]
        else:
            raise ValueError(f"Rename: no layer or channel '{left}'. Layers: {', '.join(names)}")

    try:
        pattern = re.compile(find) if find else None
    except re.error as e:
        raise ValueError(f"Rename: bad find pattern {find!r}: {e}") from e
    name_map = {}
    for n in names:
        if n in explicit:
            new = explicit[n]
        else:
            new = n
            if strip == "first segment" and "." in new:
                new = new.split(".", 1)[1]
            elif strip == "all but last" and "." in new:
                new = new.rsplit(".", 1)[1]
            if pattern:
                new = pattern.sub(replace, new)
        if not new:
            raise ValueError(f"Rename turns '{n}' into an empty name.")
        name_map[n] = new

    seen = {}
    for old, new in name_map.items():
        if new in seen:
            raise ValueError(f"Rename gives two layers the name '{new}': {seen[new]} and {old}.")
        seen[new] = old

    crypto_map = {}
    for base, ranks in crypto.items():
        digits = [r[len(base):] for r in ranks]
        new_ranks = [name_map[r] for r in ranks]
        new_base = new_ranks[0][:len(new_ranks[0]) - len(digits[0])]
        if not new_base or any(nr != new_base + d for nr, d in zip(new_ranks, digits)):
            raise ValueError(f"Cryptomatte layers must keep one shared name plus their 00, 01 … numbers: "
                             f"{', '.join(ranks)} would become {', '.join(new_ranks)}.")
        crypto_map[base] = new_base
    return name_map, suffix_maps, crypto_map


def keep_plan(names: list[str], crypto: dict[str, list[str]], mode: str, patterns: str) -> list[str]:
    """Names to keep. Patterns: comma/newline separated names or wildcards; a cryptomatte base name covers its ranks."""
    tokens = [t.strip() for t in patterns.replace("\n", ",").split(",") if t.strip()]
    matched = set()
    for tok in tokens:
        hits = {n for n in names if fnmatch.fnmatch(n.lower(), tok.lower())}
        for base, ranks in crypto.items():
            if fnmatch.fnmatch(base.lower(), tok.lower()):
                hits |= set(ranks)
        if not hits:
            raise ValueError(f"No layer matches '{tok}'. Layers: {', '.join(names)}")
        matched |= hits
    kept = [n for n in names if (n in matched) == (mode == "keep")]
    if not kept:
        raise ValueError("That would remove every layer.")
    return kept


def _plan_for(op: dict, suffixes, kinds, crypto):
    if op.get("type") == "rename":
        return "rename", rename_plan(suffixes, kinds, crypto, op.get("strip", "none"), op.get("find", ""),
                                     op.get("replace", ""), op.get("mapping", ""))
    if op.get("type") == "keep":
        return "keep", keep_plan(list(suffixes), crypto, op.get("mode", "keep"), op.get("layers", ""))
    raise ValueError(f"Unknown layer operation {op!r}")


def apply_ops(stack, ops: list[dict]):
    """Apply rename/keep operations to an ExrStack."""
    for op in ops:
        suffixes = {n: l.suffixes for n, l in stack.layers.items()}
        kinds = {n: l.kind for n, l in stack.layers.items()}
        crypto = {b: g["layers"] for b, g in stack.header.crypto.items()}
        kind, plan = _plan_for(op, suffixes, kinds, crypto)
        stack = stack.renamed(*plan) if kind == "rename" else stack.subset(plan)
    return stack


def apply_to_info(info: dict, ops: list[dict]) -> dict:
    """The same operations on an /inspect result, without reading any pixels."""
    from .stack import pick_layer

    layers = [dict(l) for l in info["layers"]]
    crypto = {b: dict(g) for b, g in info["crypto"].items()}
    for op in ops:
        suffixes = {l["name"]: l["channels"] for l in layers}
        kinds = {l["name"]: l["kind"] for l in layers}
        kind, plan = _plan_for(op, suffixes, kinds, {b: g["layers"] for b, g in crypto.items()})
        if kind == "rename":
            name_map, suffix_maps, crypto_map = plan
            for l in layers:
                sm = suffix_maps.get(l["name"], {})
                l["channels"] = [sm.get(s, s) for s in l["channels"]]
                l["name"] = name_map.get(l["name"], l["name"])
            crypto = {crypto_map.get(b, b): {**g, "layers": [name_map.get(r, r) for r in g["layers"]]}
                      for b, g in crypto.items()}
        else:
            keep = set(plan)
            layers = [l for l in layers if l["name"] in keep]
            crypto = {b: {**g, "layers": [r for r in g["layers"] if r in keep]} for b, g in crypto.items()}
            crypto = {b: g for b, g in crypto.items() if g["layers"]}
    objs = [SimpleNamespace(name=l["name"], kind=l["kind"]) for l in layers]
    return {**info, "layers": layers, "crypto": crypto,
            "roles": {r: pick_layer(objs, r) for r in ("beauty", "depth", "position", "normal")}}
