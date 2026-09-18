"""Colour math for DiffHDR: transfer functions, log curve, primaries, tonemapping."""

import math

import numpy as np
import torch

LOG_MAX = 65536.0
LOG_GAMMA = 2.2
_LOG_NORM = math.log(LOG_GAMMA * LOG_MAX + 1.0)

# CIE xy chromaticities: (rx, ry, gx, gy, bx, by, wx, wy)
CHROMATICITIES = {
    "linear_rec709": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
    "acescg": (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767),
    "aces2065_1": (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767),
}
COLORSPACES = tuple(CHROMATICITIES)

_BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]])


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Applies the sRGB EOTF (gamma-coded [0,1] to linear)."""
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Applies the sRGB OETF (linear to gamma-coded). Input is clamped to [0,1]."""
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1 / 2.4) - 0.055)


def lin_to_log(x: torch.Tensor) -> torch.Tensor:
    """Encodes linear radiance with the DiffHDR log-gamma curve into [0,1].

    Args:
        x: Linear values >= 0. Values above 65536 are clamped.

    Returns:
        Log-encoded tensor of the same shape.
    """
    x = torch.clamp(x, max=LOG_MAX)
    return torch.pow(torch.log(LOG_GAMMA * x + 1.0) / _LOG_NORM, 1.0 / LOG_GAMMA)


def log_to_lin(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`lin_to_log`. Input must be >= 0."""
    return (torch.exp(torch.pow(x, LOG_GAMMA) * _LOG_NORM) - 1.0) / LOG_GAMMA


def luma709(rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 luma of a channel-last RGB tensor ``[..., 3]`` -> ``[...]``."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _rgb_to_xyz(chroma: tuple) -> np.ndarray:
    rx, ry, gx, gy, bx, by, wx, wy = chroma
    prim = np.array([[rx / ry, gx / gy, bx / by],
                     [1.0, 1.0, 1.0],
                     [(1 - rx - ry) / ry, (1 - gx - gy) / gy, (1 - bx - by) / by]])
    white = np.array([wx / wy, 1.0, (1 - wx - wy) / wy])
    scale = np.linalg.solve(prim, white)
    return prim * scale


def _white_xyz(chroma: tuple) -> np.ndarray:
    wx, wy = chroma[6], chroma[7]
    return np.array([wx / wy, 1.0, (1 - wx - wy) / wy])


def primaries_matrix(target: str) -> np.ndarray:
    """3x3 matrix converting linear Rec.709 RGB to ``target`` primaries (Bradford CAT).

    Args:
        target: One of :data:`COLORSPACES`.

    Returns:
        ``float64`` array ``M`` so that ``rgb_target = M @ rgb_rec709``.
    """
    if target == "linear_rec709":
        return np.eye(3)
    src, dst = CHROMATICITIES["linear_rec709"], CHROMATICITIES[target]
    cone_src = _BRADFORD @ _white_xyz(src)
    cone_dst = _BRADFORD @ _white_xyz(dst)
    cat = np.linalg.inv(_BRADFORD) @ np.diag(cone_dst / cone_src) @ _BRADFORD
    return np.linalg.inv(_rgb_to_xyz(dst)) @ cat @ _rgb_to_xyz(src)


def convert_primaries(img: torch.Tensor, target: str) -> torch.Tensor:
    """Converts a channel-last linear Rec.709 image to ``target`` primaries."""
    if target == "linear_rec709":
        return img
    mat = torch.tensor(primaries_matrix(target), dtype=img.dtype, device=img.device)
    return img @ mat.T


def tonemap(hdr: torch.Tensor, exposure: float = 0.0, operator: str = "reinhard") -> torch.Tensor:
    """Maps linear HDR to display sRGB in [0,1].

    Args:
        hdr: Linear image ``[..., 3]``.
        exposure: Exposure offset in stops applied before the operator.
        operator: ``"reinhard"`` (per channel ``x/(1+x)``) or ``"clip"``.

    Returns:
        sRGB-encoded tensor in [0,1].
    """
    x = hdr.clamp_min(0) * (2.0 ** exposure)
    if operator == "reinhard":
        x = x / (1.0 + x)
    elif operator != "clip":
        raise ValueError(f"Unknown tonemap operator: {operator}")
    return linear_to_srgb(x)
