"""Render passes as the control images ControlNet-style models expect: depth (near = white), normals (camera
space, RGB = XYZ * 0.5 + 0.5) and edges (white lines on black). All take and return [frames, H, W, C] tensors."""
import numpy as np
import torch

# Depth beyond this (or inf / NaN / <= 0) is background: renderers write 1e10 or inf where nothing was hit
_FAR_CUTOFF = 1e9
_FLAT_NORMAL = (0.5, 0.5, 1.0)


def _valid_depth(z: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(z) & (z > 0) & (z < _FAR_CUTOFF)


def depth_range(z: torch.Tensor, low: float = 0.005, high: float = 0.995) -> tuple[float, float]:
    """Robust near/far of the hit pixels (percentiles, so a stray near sample doesn't flatten the rest)."""
    v = z[_valid_depth(z)]
    if v.numel() == 0:
        raise ValueError("The depth pass has no valid pixels (everything is background).")
    v = v[:: max(1, v.numel() // 4_000_000)].float()  # quantile is exact enough on a few million samples
    near, far = torch.quantile(v, torch.tensor([low, high])).tolist()
    if far <= near:
        far = near + max(abs(near) * 1e-3, 1e-6)
    return near, far


def depth_to_control(z: torch.Tensor, near: float, far: float, curve: str = "inverse") -> torch.Tensor:
    """[F,H,W] distances -> [F,H,W,3] with near = white, far and background = black. 'inverse' (1/z) matches
    estimated depth (Depth Anything, MiDaS) that depth ControlNets were trained on; 'linear' keeps even spacing."""
    if far <= near:
        raise ValueError(f"Depth far ({far}) must be greater than near ({near}).")
    valid = _valid_depth(z)
    zc = z.clamp(min=near, max=far)
    if curve == "inverse":
        near = max(near, 1e-6)
        zc = zc.clamp(min=near)
        out = (1.0 / zc - 1.0 / far) / (1.0 / near - 1.0 / far)
    else:
        out = (far - zc) / (far - near)
    out = torch.where(valid, out.clamp(0.0, 1.0), torch.zeros_like(out))
    return out.unsqueeze(-1).repeat(1, 1, 1, 3).float()


def camera_rotation(attribs) -> np.ndarray | None:
    """Rotation part of the EXR worldToCamera matrix (row-vector convention, as OpenEXR stores it), or None."""
    for name, _, value in attribs:
        if name.lower() == "worldtocamera":
            m = np.asarray(value, dtype=np.float64).reshape(4, 4)
            return m[:3, :3]
    return None


def normals_to_control(n: torch.Tensor, rotations: list[np.ndarray] | None = None, flip_y: bool = False) -> torch.Tensor:
    """[F,H,W,3] normals -> [F,H,W,3] colors (X right, Y up, Z toward camera; background flat). rotations, one per
    frame, turn world-space normals into camera space. Surfaces seen by the camera face it, so if most point away
    the pass came from a left-handed camera and Z is flipped."""
    n = n[..., :3].float()
    length = n.norm(dim=-1, keepdim=True)
    valid = (torch.isfinite(n).all(-1, keepdim=True)) & (length > 1e-6)
    n = torch.where(valid, n / length.clamp(min=1e-6), torch.zeros_like(n))
    if rotations is not None:
        rot = torch.tensor(np.stack(rotations), dtype=torch.float32)  # [F,3,3]
        n = torch.einsum("fhwi,fij->fhwj", n, rot)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = valid[..., 0]
    if v.any() and n[..., 2][v].mean() < 0:
        n = n * torch.tensor([1.0, 1.0, -1.0])
    if flip_y:
        n = n * torch.tensor([1.0, -1.0, 1.0])
    color = n * 0.5 + 0.5
    return torch.where(valid, color, torch.tensor(_FLAT_NORMAL)).clamp(0.0, 1.0)


def _dilate(edges: torch.Tensor, thickness: int) -> torch.Tensor:
    x = edges.unsqueeze(1).float()
    for _ in range(max(0, int(thickness) - 1)):
        x = torch.nn.functional.max_pool2d(x, 3, stride=1, padding=1)
    return x[:, 0]


def id_edges(ids: torch.Tensor, thickness: int = 2) -> torch.Tensor:
    """[F,H,W] cryptomatte IDs (the strongest rank) -> [F,H,W,3] white lines where the object changes. IDs are
    compared bit-exact, so neighboring objects with close hash values still separate."""
    bits = ids.float().contiguous().view(torch.int32)
    e = torch.zeros_like(bits, dtype=torch.bool)
    h = bits[:, :, 1:] != bits[:, :, :-1]
    v = bits[:, 1:, :] != bits[:, :-1, :]
    e[:, :, 1:] |= h
    e[:, :, :-1] |= h
    e[:, 1:, :] |= v
    e[:, :-1, :] |= v
    return _dilate(e, thickness).unsqueeze(-1).repeat(1, 1, 1, 3)


def depth_edges(z: torch.Tensor, thickness: int = 2, jump: float = 0.05) -> torch.Tensor:
    """[F,H,W] distances -> [F,H,W,3] white lines where depth jumps by more than `jump` (relative), plus
    silhouettes against the background."""
    valid = _valid_depth(z)
    zf = torch.where(valid, z.float(), torch.zeros_like(z, dtype=torch.float32))

    def jumps(a, b, va, vb):
        rel = (a - b).abs() / torch.minimum(a, b).clamp(min=1e-6)
        return (va != vb) | (va & vb & (rel > jump))

    e = torch.zeros_like(valid)
    h = jumps(zf[:, :, 1:], zf[:, :, :-1], valid[:, :, 1:], valid[:, :, :-1])
    v = jumps(zf[:, 1:, :], zf[:, :-1, :], valid[:, 1:, :], valid[:, :-1, :])
    e[:, :, 1:] |= h
    e[:, :, :-1] |= h
    e[:, 1:, :] |= v
    e[:, :-1, :] |= v
    return _dilate(e, thickness).unsqueeze(-1).repeat(1, 1, 1, 3)
