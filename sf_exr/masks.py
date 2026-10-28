"""Mask math for render passes: cryptomatte decoding, depth bands, world-position shapes, normal angles."""
import fnmatch
import struct

import numpy as np
import torch

# ---------------------------------------------------------------- cryptomatte

def mmh3_32(data: bytes, seed: int = 0) -> int:
    c1, c2, h, n = 0xCC9E2D51, 0x1B873593, seed, len(data)
    for i in range(0, n - n % 4, 4):
        k = int.from_bytes(data[i:i + 4], "little")
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        h ^= (k * c2) & 0xFFFFFFFF
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    k, tail = 0, data[n - n % 4:]
    for i, b in enumerate(tail):
        k |= b << (8 * i)
    if tail:
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        h ^= (k * c2) & 0xFFFFFFFF
    h ^= n
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    return h ^ (h >> 16)


def name_to_hex(name: str) -> str:
    """Cryptomatte ID for a name: MurmurHash3, with the float exponent nudged off 0/255 (no NaN, inf or denormals)."""
    h = mmh3_32(name.encode("utf-8"))
    exp = (h >> 23) & 255
    if exp in (0, 255):
        h ^= 1 << 23
    return f"{h:08x}"


def hex_to_float(hexid: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(hexid, 16)))[0]


def resolve_names(manifest: dict[str, str], query: str) -> dict[str, str]:
    """Comma/newline separated names or wildcards -> {name: hex id}. Unknown literal names are hashed."""
    picked = {}
    lower = {k.lower(): k for k in manifest}
    for token in [t.strip() for t in query.replace("\n", ",").split(",")]:
        if not token:
            continue
        if any(c in token for c in "*?["):
            for name in manifest:
                if fnmatch.fnmatch(name.lower(), token.lower()):
                    picked[name] = manifest[name]
        elif token.lower() in lower:
            name = lower[token.lower()]
            picked[name] = manifest[name]
        else:
            picked[token] = name_to_hex(token)
    return picked


def rank_pairs(ranks: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Each RGBA rank layer holds two (id, coverage) pairs."""
    pairs = []
    for r in ranks:
        for i in range(0, r.shape[-1] - 1, 2):
            pairs.append((r[..., i], r[..., i + 1]))
    return pairs


def crypto_mask(ranks: list[torch.Tensor], hex_ids: list[str]) -> torch.Tensor:
    """Sum the coverage of every rank whose ID matches. IDs compare bit-exact as float32."""
    pairs = rank_pairs(ranks)
    out = torch.zeros_like(pairs[0][1], dtype=torch.float32)
    if not hex_ids:
        return out
    targets = torch.tensor(np.array([hex_to_float(h) for h in hex_ids], np.float32))
    for ids, cov in pairs:
        hit = torch.isin(ids.float(), targets)
        out += torch.where(hit, cov.float(), torch.zeros_like(out))
    return out.clamp(0.0, 1.0)


def names_at(ranks: list[torch.Tensor], manifest: dict[str, str], frame: int, x: int, y: int) -> list[tuple[str, float]]:
    """Names covering one pixel, strongest first."""
    by_id = {np.float32(hex_to_float(h)).tobytes(): n for n, h in manifest.items()}
    found = []
    for ids, cov in rank_pairs(ranks):
        c = float(cov[frame, y, x])
        if c <= 0:
            continue
        key = np.float32(float(ids[frame, y, x])).tobytes()
        found.append((by_id.get(key, f"#{struct.unpack('<I', key)[0]:08x}"), c))
    return sorted(found, key=lambda t: -t[1])


# ---------------------------------------------------------------- pass masks

def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    if e1 <= e0:
        return (x >= e0).float()
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def depth_mask(depth: torch.Tensor, near: float, far: float, falloff: float, normalized: bool) -> torch.Tensor:
    """1 between near and far, easing to 0 over `falloff` on either side. Empty background (inf/0) counts as far away."""
    d = depth.float().clone()
    bad = ~torch.isfinite(d) | (d >= 1e9)
    if normalized:
        good = d[~bad]
        lo, hi = (float(good.min()), float(good.max())) if good.numel() else (0.0, 1.0)
        d = (d - lo) / max(hi - lo, 1e-8)
        d[bad] = 1e9
    else:
        d[bad] = 1e9
    rise = _smoothstep(near - falloff, near, d) if falloff > 0 else (d >= near).float()
    fall = 1.0 - (_smoothstep(far, far + falloff, d) if falloff > 0 else (d > far).float())
    return (rise * fall).clamp(0.0, 1.0)


AXES = {"X": 0, "Y": 1, "Z": 2}


def position_mask(p: torch.Tensor, shape: str, center: tuple, size: tuple, falloff: float,
                  ignore_empty: bool = True, axis: str = "Y") -> torch.Tensor:
    """World-position selection: inside a sphere / box (cube) / infinite cylinder along an axis,
    or on the positive side of a plane through center."""
    p = p[..., :3].float()
    c = torch.tensor(center, dtype=torch.float32)
    if shape == "sphere":
        dist = torch.linalg.vector_norm(p - c, dim=-1) - float(size[0])
    elif shape in ("box", "cube"):
        q = (p - c).abs() - torch.tensor(size, dtype=torch.float32).abs()
        dist = torch.linalg.vector_norm(q.clamp(min=0), dim=-1) + q.max(dim=-1).values.clamp(max=0)
    elif shape == "cylinder":
        keep = [i for i in range(3) if i != AXES.get(axis, 1)]
        dist = torch.linalg.vector_norm((p - c)[..., keep], dim=-1) - float(size[0])
    else:  # plane: size is the normal
        n = torch.tensor(size, dtype=torch.float32)
        n = n / max(float(torch.linalg.vector_norm(n)), 1e-8)
        dist = -((p - c) * n).sum(-1)
    m = 1.0 - _smoothstep(0.0, falloff, dist) if falloff > 0 else (dist <= 0).float()
    if ignore_empty:
        m = m * (p.abs().sum(-1) > 0).float()  # renderers write 0,0,0 where nothing was hit
    return m


def normal_mask(n: torch.Tensor, direction: tuple, angle: float, falloff: float) -> torch.Tensor:
    """Surfaces whose normal points within `angle` degrees of a direction."""
    n = n[..., :3].float()
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp(min=1e-8)
    d = torch.tensor(direction, dtype=torch.float32)
    d = d / max(float(torch.linalg.vector_norm(d)), 1e-8)
    ang = torch.rad2deg(torch.arccos((n * d).sum(-1).clamp(-1.0, 1.0)))
    m = 1.0 - _smoothstep(angle, angle + falloff, ang) if falloff > 0 else (ang <= angle).float()
    return m * (n.abs().sum(-1) > 0).float()


def combine(a: torch.Tensor | None, b: torch.Tensor, op: str) -> torch.Tensor:
    if a is None:
        return b if op != "subtract" else torch.zeros_like(b)
    if op == "subtract":
        return (a - b).clamp(0.0, 1.0)
    if op == "intersect":
        return a * b
    return torch.maximum(a, b)
