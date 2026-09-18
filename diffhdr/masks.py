"""Exposure mask detection ported from DiffHDR (utils/color_utils.py, infer_hdri.py).

The results are bit-identical to the reference implementation (see
``tests/test_masks.py::test_matches_reference``); only the way the work is
scheduled differs. Frames are processed in batches instead of one at a time,
the 2-D morphology is evaluated as two 1-D passes (``max`` is separable, so
this changes nothing numerically) and a mask that the caller did not ask for is
not computed at all.
"""

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

# Frames per morphology batch are derived from this pixel budget, so memory
# stays bounded for long clips and for resolutions above 720p.
CHUNK_PIXELS = 32 * 720 * 1280


@dataclass
class VideoMasks:
    """Binary masks ``[F,H,W]`` (1 = regenerate).

    A mask whose flag was not set in :func:`video_masks` is returned as zeros
    because it is never computed.
    """

    combined: torch.Tensor
    over: torch.Tensor
    under: torch.Tensor


def _smoothstep(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    t = ((x - edge0) / (edge1 - edge0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _max_pool(x: torch.Tensor, k: int) -> torch.Tensor:
    """Separable max pooling on ``[N,1,H,W]``; identical to ``max_pool2d(x, k, 1, k // 2)``."""
    p = k // 2
    return F.max_pool2d(F.max_pool2d(x, (1, k), 1, (0, p)), (k, 1), 1, (p, 0))


def _dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    return _max_pool(m.reshape(-1, 1, *m.shape[-2:]), k).reshape(m.shape)


def _erode(m: torch.Tensor, k: int) -> torch.Tensor:
    return (-_max_pool(-m.reshape(-1, 1, *m.shape[-2:]), k)).reshape(m.shape)


def _lin(v: float) -> float:
    return srgb_to_linear(torch.tensor(float(v))).item()


def _chunk(height: int, width: int) -> int:
    """Frames per batch for the given frame size."""
    return max(1, CHUNK_PIXELS // max(1, height * width))


def _soft_masks(frames: torch.Tensor, want_over: bool = True, want_under: bool = True):
    """Soft over/under exposure masks for a batch of sRGB frames.

    Args:
        frames: ``[F,H,W,3]`` sRGB in [0,1].
        want_over: Compute the over-exposure mask.
        want_under: Compute the under-exposure mask.

    Returns:
        tuple: ``(over, under)`` soft masks ``[F,H,W]``, each None when not requested.
    """
    frames = frames.clamp(0, 1)
    y = luma709(srgb_to_linear(frames))
    over = under = None
    if want_over:
        over_l = _smoothstep(y, _lin(OVER_THR - SOFTNESS), _lin(OVER_THR + SOFTNESS))
        ch_clip = ((frames >= 1.0 - CLIP_EPS).float().sum(dim=-1) >= 2).float()
        over = (0.7 * over_l + 0.3 * ch_clip).clamp(0, 1)
        over = _erode(_dilate(over, 3), 3).clamp(0, 1)      # 3x3 closing
    if want_under:
        under_l = 1.0 - _smoothstep(y, _lin(max(0.0, UNDER_THR - SOFTNESS)), _lin(UNDER_THR + SOFTNESS))
        all_low = (frames <= UNDER_THR).all(dim=-1).float()
        under = (0.7 * under_l + 0.3 * all_low).clamp(0, 1)
        under = _erode(_dilate(under, 3), 3).clamp(0, 1)
    return over, under


def soft_exposure_masks(frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft over/under exposure masks for one sRGB frame.

    Args:
        frame: ``[H,W,3]`` sRGB in [0,1].

    Returns:
        ``(over, under)`` soft masks ``[H,W]`` in [0,1].
    """
    over, under = _soft_masks(frame[None])
    return over[0], under[0]


def _morph(ema: torch.Tensor) -> torch.Tensor:
    """Box blur plus opening and closing on a batch of EMA states ``[F,H,W]``."""
    flat = ema.reshape(-1, 1, *ema.shape[-2:])
    out = F.avg_pool2d(flat, K_SMOOTH, 1, K_SMOOTH // 2).reshape(ema.shape)
    out = _dilate(_erode(out, K_OPEN_CLOSE), K_OPEN_CLOSE)
    out = _erode(_dilate(out, K_OPEN_CLOSE), K_OPEN_CLOSE)
    return out.clamp(0, 1)


def stabilize(soft: torch.Tensor, prev_ema: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporal EMA + box blur + open/close. Returns ``(stabilised, ema_state)``."""
    ema = soft if prev_ema is None else EMA_ALPHA * soft + (1 - EMA_ALPHA) * prev_ema
    return _morph(ema[None])[0], ema.clone()


def _stabilize_sequence(soft: torch.Tensor) -> torch.Tensor:
    """Runs the temporal EMA over ``[F,H,W]`` and binarises the stabilised result."""
    ema = torch.empty_like(soft)
    prev = None
    for i in range(soft.shape[0]):
        ema[i] = soft[i] if prev is None else EMA_ALPHA * soft[i] + (1 - EMA_ALPHA) * prev
        prev = ema[i]
    out = torch.empty_like(soft)
    step = _chunk(soft.shape[-2], soft.shape[-1])
    for start in range(0, soft.shape[0], step):
        out[start:start + step] = _morph(ema[start:start + step])
    return (out > BIN_THR).float()


def video_masks(images: torch.Tensor, use_over: bool = True, use_under: bool = False) -> VideoMasks:
    """Binary exposure masks for a clip.

    Args:
        images: ``[F,H,W,3]`` sRGB in [0,1].
        use_over: Include over-exposed regions in ``combined``.
        use_under: Include under-exposed regions in ``combined``.

    Returns:
        VideoMasks with ``combined``, ``over`` and ``under`` of shape ``[F,H,W]``.
        A mask whose flag is False is not computed and comes back as zeros.

    Raises:
        ValueError: If both flags are False.
    """
    if not (use_over or use_under):
        raise ValueError("Enable mask_overexposed and/or mask_underexposed, or connect a mask.")
    images = images.float()
    count, height, width = images.shape[:3]
    zeros = torch.zeros((count, height, width), dtype=torch.float32, device=images.device)

    soft_over = torch.empty_like(zeros) if use_over else None
    soft_under = torch.empty_like(zeros) if use_under else None
    step = _chunk(height, width)
    for start in range(0, count, step):
        over, under = _soft_masks(images[start:start + step], use_over, use_under)
        if use_over:
            soft_over[start:start + step] = over
        if use_under:
            soft_under[start:start + step] = under

    over = _stabilize_sequence(soft_over) if use_over else zeros
    under = _stabilize_sequence(soft_under) if use_under else zeros
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
