"""Exposure mask detection ported from DiffHDR (utils/color_utils.py, infer_hdri.py)."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .color import linear_to_srgb, luma709, srgb_to_linear

OVER_THR = 0.95
UNDER_THR = 0.01
SOFTNESS = 0.02
CLIP_EPS = 0.02
EMA_ALPHA = 0.7
K_SMOOTH = 9
K_OPEN_CLOSE = 9
BIN_THR = 0.2


@dataclass
class VideoMasks:
    """Binary masks ``[F,H,W]`` (1 = regenerate)."""

    combined: torch.Tensor
    over: torch.Tensor
    under: torch.Tensor


def _smoothstep(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    t = ((x - edge0) / (edge1 - edge0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(m[None, None], k, 1, k // 2)[0, 0]


def _erode(m: torch.Tensor, k: int) -> torch.Tensor:
    return -F.max_pool2d(-m[None, None], k, 1, k // 2)[0, 0]


def _lin(v: float) -> float:
    return srgb_to_linear(torch.tensor(float(v))).item()


def soft_exposure_masks(frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft over/under exposure masks for one sRGB frame.

    Args:
        frame: ``[H,W,3]`` sRGB in [0,1].

    Returns:
        ``(over, under)`` soft masks ``[H,W]`` in [0,1].
    """
    frame = frame.clamp(0, 1)
    y = luma709(srgb_to_linear(frame))
    over_l = _smoothstep(y, _lin(OVER_THR - SOFTNESS), _lin(OVER_THR + SOFTNESS))
    ch_clip = ((frame >= 1.0 - CLIP_EPS).float().sum(dim=-1) >= 2).float()
    over = (0.7 * over_l + 0.3 * ch_clip).clamp(0, 1)

    under_l = 1.0 - _smoothstep(y, _lin(max(0.0, UNDER_THR - SOFTNESS)), _lin(UNDER_THR + SOFTNESS))
    all_low = (frame <= UNDER_THR).all(dim=-1).float()
    under = (0.7 * under_l + 0.3 * all_low).clamp(0, 1)

    over = _erode(_dilate(over, 3), 3).clamp(0, 1)      # 3x3 closing
    under = _erode(_dilate(under, 3), 3).clamp(0, 1)
    return over, under


def stabilize(soft: torch.Tensor, prev_ema: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporal EMA + box blur + open/close. Returns ``(stabilised, ema_state)``."""
    ema = soft if prev_ema is None else EMA_ALPHA * soft + (1 - EMA_ALPHA) * prev_ema
    state = ema.clone()
    out = F.avg_pool2d(ema[None, None], K_SMOOTH, 1, K_SMOOTH // 2)[0, 0]
    out = _dilate(_erode(out, K_OPEN_CLOSE), K_OPEN_CLOSE)
    out = _erode(_dilate(out, K_OPEN_CLOSE), K_OPEN_CLOSE)
    return out.clamp(0, 1), state


def video_masks(images: torch.Tensor, use_over: bool = True, use_under: bool = False) -> VideoMasks:
    """Binary exposure masks for a clip.

    Args:
        images: ``[F,H,W,3]`` sRGB in [0,1].
        use_over: Include over-exposed regions in ``combined``.
        use_under: Include under-exposed regions in ``combined``.

    Returns:
        VideoMasks with ``combined``, ``over`` and ``under`` of shape ``[F,H,W]``.

    Raises:
        ValueError: If both flags are False.
    """
    if not (use_over or use_under):
        raise ValueError("Enable mask_overexposed and/or mask_underexposed, or connect a mask.")
    overs, unders = [], []
    prev_o = prev_u = None
    for frame in images.float():
        o, u = soft_exposure_masks(frame)
        o, prev_o = stabilize(o, prev_o)
        u, prev_u = stabilize(u, prev_u)
        overs.append((o > BIN_THR).float())
        unders.append((u > BIN_THR).float())
    over, under = torch.stack(overs), torch.stack(unders)
    combined = torch.zeros_like(over)
    if use_over:
        combined = torch.maximum(combined, over)
    if use_under:
        combined = torch.maximum(combined, under)
    return VideoMasks(combined=combined, over=over, under=under)


def paint_underexposed(images: torch.Tensor, under: torch.Tensor) -> torch.Tensor:
    """Returns a copy of ``images`` with under-exposed pixels set to sRGB 0.5."""
    out = images.clone()
    out[under.bool()] = 0.5
    return out


def pano_mask(image: torch.Tensor, over_thr: float = OVER_THR) -> torch.Tensor:
    """Over-exposure mask for an equirectangular sRGB image ``[H,W,3]`` -> ``[H,W]``."""
    image = image.clamp(0, 1).float()
    luma_srgb = linear_to_srgb(luma709(srgb_to_linear(image)))
    over = _smoothstep(luma_srgb, over_thr - 0.02, over_thr + 0.02)
    ch_clip = _smoothstep(image.max(dim=-1).values, 0.93, 0.99)
    return (torch.maximum(over, ch_clip) > BIN_THR).float()
