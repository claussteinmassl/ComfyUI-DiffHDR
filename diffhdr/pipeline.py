"""Model-agnostic DiffHDR orchestration around an injected window function."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from . import color, frames, masks, windows

# window_fn(control_log [F,H,W,3], mask [F,H,W], reference_log [1,H,W,3] | None) -> log frames [F,H,W,3]
WindowFn = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


@dataclass
class Result:
    """Pipeline output: linear HDR ``[F,H,W,3]`` and the mask used ``[F,H,W]``."""

    hdr: torch.Tensor
    mask: torch.Tensor


def encode_control(images: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] -> log-encoded control frames."""
    return color.lin_to_log(color.srgb_to_linear(images.float().clamp(0, 1)))


def decode_output(log_frames: torch.Tensor) -> torch.Tensor:
    """Log frames -> linear HDR (negative values clamped first)."""
    return color.log_to_lin(log_frames.float().clamp_min(0))


def prepare_reference(reference: torch.Tensor, mask0: torch.Tensor, ev: float) -> torch.Tensor:
    """Builds the log-encoded VACE reference image.

    Args:
        reference: ``[1,H,W,3]`` sRGB reference, already at processing size.
        mask0: ``[H,W]`` mask of frame 0 (1 = regenerate). Reference content is kept inside it, white elsewhere.
        ev: Exposure boost in stops applied in linear space.
    """
    m = mask0.float()[None, ..., None]
    ref = reference[:1].float().clamp(0, 1) * m + (1.0 - m)
    return color.lin_to_log(color.srgb_to_linear(ref) * (2.0 ** ev))


def _prepare(images, user_mask, mask_overexposed, mask_underexposed):
    images = images.float().clamp(0, 1)
    if user_mask is not None:
        mask = frames.pad_frames((user_mask.float() > 0.5).float(), images.shape[0])[: images.shape[0]]
        return images, mask
    vm = masks.video_masks(images, use_over=mask_overexposed, use_under=mask_underexposed)
    if mask_underexposed:
        images = masks.paint_underexposed(images, vm.under)
    return images, vm.combined


def run_video(images: torch.Tensor, window_fn: WindowFn, *, user_mask: Optional[torch.Tensor] = None,
              mask_overexposed: bool = True, mask_underexposed: bool = False,
              reference: Optional[torch.Tensor] = None, reference_ev: float = 5.0,
              window_size: int = 33, window_stride: int = 16, use_prev_window_reference: bool = False,
              on_window: Optional[Callable[[int, int], None]] = None) -> Result:
    """Runs image, short-video or long-video reconstruction.

    Args:
        images: ``[F,H,W,3]`` sRGB at processing size (H, W multiples of 16).
        window_fn: Generates log frames for one window.
        user_mask: Optional ``[F,H,W]`` mask overriding detection.
        mask_overexposed: Detect over-exposed regions.
        mask_underexposed: Detect under-exposed regions and paint them to 0.5.
        reference: Optional ``[1,H,W,3]`` sRGB reference image.
        reference_ev: EV boost for the reference.
        window_size: Frames per window (``4k+1``).
        window_stride: Stride between windows.
        use_prev_window_reference: Feed the previous window's output as reference.
        on_window: Progress callback ``(index, count)``.

    Returns:
        Result with linear HDR and the mask used.
    """
    total = images.shape[0]
    images, mask = _prepare(images, user_mask, mask_overexposed, mask_underexposed)
    control = encode_control(images)
    ref_log = prepare_reference(reference, mask[0], reference_ev) if reference is not None else None

    if total == 1:
        n = frames.image_mode_num_frames(images.shape[1], images.shape[2])
        log = window_fn(control.expand(n, -1, -1, -1).contiguous(), mask.expand(n, -1, -1).contiguous(), ref_log)
        return Result(hdr=decode_output(log[:1]), mask=mask)

    size = frames.snap_4n1(window_size)
    plan = windows.plan_windows(total, size, window_stride)
    blender = windows.StreamingBlender(size, window_stride)
    finished = []
    for index, win in enumerate(plan):
        if on_window is not None:
            on_window(index, len(plan))
        length = win.end - win.start
        log = window_fn(frames.pad_frames(control[win.start:win.end], size),
                        frames.pad_frames(mask[win.start:win.end], size), ref_log)[:length].float().clamp_min(0)
        if use_prev_window_reference:
            ref_log = log[min(window_stride, length - 1)][None].clone()
        finished.append(blender.add(color.log_to_lin(log), win))
    finished.append(blender.flush())
    return Result(hdr=torch.cat([f for f in finished if f.shape[0] > 0]), mask=mask)


def run_pano(image: torch.Tensor, window_fn: WindowFn, *, user_mask: Optional[torch.Tensor] = None) -> Result:
    """Runs panorama reconstruction on one equirectangular frame ``[1,H,W,3]``."""
    image = image[:1].float().clamp(0, 1)
    if user_mask is not None:
        mask = (user_mask[:1].float() > 0.5).float()
    else:
        mask = masks.pano_mask(image[0])[None]
    log = window_fn(encode_control(image), mask, None)
    return Result(hdr=decode_output(log[:1]), mask=mask)
