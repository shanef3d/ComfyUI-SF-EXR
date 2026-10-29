"""ComfyUI provenance in EXR headers: the prompt, the workflow and a short readable summary (models, seeds, prompt
text), so any saved frame can be traced and re-made. Same content core's Save Image puts in PNGs."""
import json

PREFIX = "comfyui/"
MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin", ".sft", ".onnx")
SEED_INPUTS = {"seed", "noise_seed"}
TEXT_INPUTS = {"text", "prompt", "positive", "negative"}
TEXT_LIMIT = 500


def strip(attribs: list) -> list:
    """Attributes without any ComfyUI metadata (e.g. inherited from a file SF EXR Load read)."""
    return [a for a in attribs if not str(a[0]).startswith(PREFIX)]


def summary(prompt: dict | None) -> str:
    """Models, seeds and prompt text found in an API-format prompt, one kind per line."""
    models, seeds, texts = [], [], []
    for node in (prompt or {}).values():
        for key, value in (node.get("inputs") or {}).items():
            if isinstance(value, list):  # a link, not a value
                continue
            if isinstance(value, str) and value.lower().endswith(MODEL_EXTENSIONS) and value not in models:
                models.append(value)
            elif key in SEED_INPUTS and isinstance(value, int) and value not in seeds:
                seeds.append(value)
            elif key in TEXT_INPUTS and isinstance(value, str) and value.strip():
                text = value.strip()
                texts.append(text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT] + "…")
    lines = []
    if models:
        lines.append("models: " + ", ".join(models))
    if seeds:
        lines.append("seeds: " + ", ".join(str(s) for s in seeds))
    lines += [f"prompt: {t}" for t in texts]
    return "\n".join(lines)


def attribs(prompt: dict | None, extra: dict | None) -> list:
    """EXR string attributes: comfyui/prompt, comfyui/<each extra_pnginfo key> (the workflow), comfyui/summary
    and comfyui/version."""
    out = []
    if prompt:
        out.append((PREFIX + "prompt", "string", json.dumps(prompt)))
    for key, value in (extra or {}).items():
        out.append((PREFIX + str(key), "string", json.dumps(value)))
    text = summary(prompt)
    if text:
        out.append((PREFIX + "summary", "string", text))
    try:
        from comfyui_version import __version__
        out.append((PREFIX + "version", "string", str(__version__)))
    except ImportError:
        pass
    return out
