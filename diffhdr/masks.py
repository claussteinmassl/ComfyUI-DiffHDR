"""Exposure mask detection ported from DiffHDR (utils/color_utils.py, infer_hdri.py).

The results are bit-identical to the reference implementation (see
``tests/test_masks.py::test_matches_reference``); only the way the work is
scheduled differs. Frames are processed in batches instead of one at a time,
the 2-D morphology is evaluated as two 1-D passes (``max`` is separable, so
this changes nothing numerically), a mask that the caller did not ask for is
not computed at all, and the EMA/morphology/binarisation stages reuse one
``[F,H,W]`` buffer per mask instead of allocating one each.

The over/under luma thresholds can be moved away from the reference values
(:data:`OVER_THR`, :data:`UNDER_THR`). At those defaults every result is unchanged.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .color import linear_to_srgb, luma709, srgb_to_linear
from .frames import block_size

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
    """Binary masks ``[F,H,W]`` (1 = regenerate).

    A mask whose flag was not set in :func:`video_masks` is returned as a zero-stride
    view of a single zero because it is never computed. When only one flag is set,
    ``combined`` is that mask itself, not a copy; none of the four may be modified
    in place by the caller.

    ``paint`` marks the under-exposed pixels to paint mid-grey. It is the under mask at
    the reference threshold, limited to ``under``, so raising the under threshold
    regenerates more shadow area while the shadow detail there stays visible to the
    model. At the reference threshold it is ``under`` itself.
    """

    combined: torch.Tensor
    over: torch.Tensor
    under: torch.Tensor
    paint: torch.Tensor


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


def _soft_masks(frames: torch.Tensor, want_over: bool = True, want_under: bool = True,
                over_thr: float = OVER_THR, under_thr: float = UNDER_THR):
    """Soft over/under exposure masks for a batch of sRGB frames.

    Args:
        frames: ``[F,H,W,3]`` sRGB in [0,1].
        want_over: Compute the over-exposure mask.
        want_under: Compute the under-exposure mask.
        over_thr: sRGB luma above which a pixel counts as over-exposed. The per-channel
            clip test (``1 - CLIP_EPS`` at the reference threshold) moves by the same
            amount, capped at 1.0 so fully clipped channels always count.
        under_thr: sRGB level below which a pixel counts as under-exposed.

    Returns:
        tuple: ``(over, under)`` soft masks ``[F,H,W]``, each None when not requested.
    """
    frames = frames.clamp(0, 1)
    y = luma709(srgb_to_linear(frames))
    over = under = None
    if want_over:
        over_l = _smoothstep(y, _lin(over_thr - SOFTNESS), _lin(over_thr + SOFTNESS))
        ch_thr = min(1.0, 1.0 - CLIP_EPS + (over_thr - OVER_THR))
        ch_clip = ((frames >= ch_thr).float().sum(dim=-1) >= 2).float()
        over = (0.7 * over_l + 0.3 * ch_clip).clamp(0, 1)
        over = _erode(_dilate(over, 3), 3).clamp(0, 1)      # 3x3 closing
    if want_under:
        under_l = 1.0 - _smoothstep(y, _lin(max(0.0, under_thr - SOFTNESS)), _lin(under_thr + SOFTNESS))
        all_low = (frames <= under_thr).all(dim=-1).float()
        under = (0.7 * under_l + 0.3 * all_low).clamp(0, 1)
        under = _erode(_dilate(under, 3), 3).clamp(0, 1)
    return over, under


def soft_exposure_masks(frame: torch.Tensor, over_thr: float = OVER_THR,
                        under_thr: float = UNDER_THR) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft over/under exposure masks for one sRGB frame.

    Args:
        frame: ``[H,W,3]`` sRGB in [0,1].
        over_thr: Over-exposure luma threshold (sRGB).
        under_thr: Under-exposure threshold (sRGB).

    Returns:
        ``(over, under)`` soft masks ``[H,W]`` in [0,1].
    """
    over, under = _soft_masks(frame[None], over_thr=over_thr, under_thr=under_thr)
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
    """Temporal EMA, morphology and binarisation, all in place in ``soft``.

    The EMA recurrence only ever reads the previous frame's state, and the morphology
    is purely spatial, so the whole stage runs in the caller's buffer instead of
    allocating a separate EMA, morphology and binarisation copy of the sequence.

    Args:
        soft: ``[F,H,W]`` soft mask, owned by the caller and overwritten here.

    Returns:
        torch.Tensor: ``soft``, now holding the binarised stabilised mask.
    """
    for i in range(1, soft.shape[0]):
        soft[i] = EMA_ALPHA * soft[i] + (1 - EMA_ALPHA) * soft[i - 1]
    step = block_size(soft.shape[-2], soft.shape[-1])
    for start in range(0, soft.shape[0], step):
        chunk = soft[start:start + step]
        chunk.copy_(_morph(chunk) > BIN_THR)
    return soft


def _detect(images: torch.Tensor, use_over: bool, use_under: bool, over_thr: float,
            under_thr: float) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Stabilised binary over/under masks ``[F,H,W]``, each None when not requested."""
    count, height, width = images.shape[:3]
    shape = (count, height, width)
    kwargs = {"dtype": torch.float32, "device": images.device}

    # One full [F,H,W] buffer per requested mask: the soft mask is written here and the
    # EMA, morphology and binarisation stages then run inside it.
    over = torch.empty(shape, **kwargs) if use_over else None
    under = torch.empty(shape, **kwargs) if use_under else None
    step = block_size(height, width)
    for start in range(0, count, step):
        soft_over, soft_under = _soft_masks(images[start:start + step], use_over, use_under,
                                            over_thr, under_thr)
        if use_over:
            over[start:start + step] = soft_over
        if use_under:
            under[start:start + step] = soft_under

    if use_over:
        _stabilize_sequence(over)
    if use_under:
        _stabilize_sequence(under)
    return over, under


def video_masks(images: torch.Tensor, use_over: bool = True, use_under: bool = False,
                over_threshold: float = OVER_THR, under_threshold: float = UNDER_THR) -> VideoMasks:
    """Binary exposure masks for a clip.

    Args:
        images: ``[F,H,W,3]`` sRGB in [0,1].
        use_over: Include over-exposed regions in ``combined``.
        use_under: Include under-exposed regions in ``combined``.
        over_threshold: sRGB luma above which a pixel is regenerated. Lower values also
            regenerate highlights that are bright but not yet clipped.
        under_threshold: sRGB level below which a pixel is regenerated. Higher values
            also regenerate shadows that are dark but not yet crushed.

    Returns:
        VideoMasks with ``combined``, ``over``, ``under`` and ``paint`` of shape ``[F,H,W]``.
        A mask whose flag is False is not computed and comes back as a zero-stride
        view of a single zero, which costs no memory. With only one flag set,
        ``combined`` is the requested mask itself rather than a copy of it.
        ``images`` is never modified.

    Raises:
        ValueError: If both flags are False.
    """
    if not (use_over or use_under):
        raise ValueError("Enable mask_overexposed and/or mask_underexposed, or connect a mask.")
    images = images.float()
    over, under = _detect(images, use_over, use_under, over_threshold, under_threshold)
    paint = under
    if use_under and under_threshold != UNDER_THR:
        _, paint = _detect(images, False, True, OVER_THR, UNDER_THR)
        paint.mul_(under)
    combined = torch.maximum(over, under) if (use_over and use_under) else (over if use_over else under)
    zeros = torch.zeros((1, 1, 1), dtype=torch.float32, device=images.device).expand(combined.shape)
    return VideoMasks(combined=combined,
                      over=over if use_over else zeros,
                      under=under if use_under else zeros,
                      paint=paint if use_under else zeros)


def paint_underexposed(images: torch.Tensor, under: torch.Tensor) -> torch.Tensor:
    """Returns a copy of ``images`` with under-exposed pixels set to sRGB 0.5."""
    out = images.clone()
    out[under.bool()] = 0.5
    return out


def pano_mask(image: torch.Tensor, over_thr: float = OVER_THR) -> torch.Tensor:
    """Over-exposure mask for an equirectangular sRGB image ``[H,W,3]`` -> ``[H,W]``.

    Args:
        image: ``[H,W,3]`` sRGB in [0,1].
        over_thr: sRGB luma threshold. The per-channel clip ramp (0.93-0.99 at the
            reference threshold) moves by the same amount.
    """
    image = image.clamp(0, 1).float()
    luma_srgb = linear_to_srgb(luma709(srgb_to_linear(image)))
    over = _smoothstep(luma_srgb, over_thr - 0.02, over_thr + 0.02)
    shift = over_thr - OVER_THR
    ch_clip = _smoothstep(image.max(dim=-1).values, 0.93 + shift, 0.99 + shift)
    return (torch.maximum(over, ch_clip) > BIN_THR).float()
