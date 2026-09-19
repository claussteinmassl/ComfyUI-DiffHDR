"""Model-agnostic DiffHDR orchestration around an injected window function."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from . import color, frames, masks, timing, windows

# window_fn(control_log [F,H,W,3], mask [F,H,W], reference_log [1,H,W,3] | None) -> log frames [F,H,W,3]
WindowFn = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


@dataclass
class Result:
    """Pipeline output: linear HDR ``[F,H,W,3]`` and the mask used ``[F,H,W]``."""

    hdr: torch.Tensor
    mask: torch.Tensor


def encode_control(images: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] -> log-encoded control frames.

    The transfer functions are applied block by block. Applied to a whole clip at once
    they would keep half a dozen full ``[F,H,W,3]`` temporaries alive, which dominates
    host memory on long videos; the result is exactly the same either way.
    """
    out = torch.empty(images.shape, dtype=torch.float32, device=images.device)
    step = frames.block_size(images.shape[-3], images.shape[-2])
    for start in range(0, images.shape[0], step):
        block = images[start:start + step]
        out[start:start + step] = color.lin_to_log(color.srgb_to_linear(block.float().clamp(0, 1)))
    return out


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
    """Returns ``(images, mask)`` without ever modifying the caller's tensors.

    Nothing is clamped here: every consumer of ``images`` clamps to [0,1] itself
    (:func:`encode_control`, :func:`masks.video_masks`), so a clamp at this point would
    only add a full-clip copy without changing any result.
    """
    if user_mask is not None:
        mask = frames.pad_frames((user_mask.float() > 0.5).float(), images.shape[0])[: images.shape[0]]
        return images, mask
    vm = masks.video_masks(images, use_over=mask_overexposed, use_under=mask_underexposed)
    if mask_underexposed:
        images = masks.paint_underexposed(images, vm.under)
    return images, vm.combined


def _store(out: torch.Tensor, filled: int, chunk: torch.Tensor) -> int:
    """Copies a finished blend chunk into the output buffer; returns the frames written."""
    length = chunk.shape[0]
    if length:
        out[filled:filled + length] = chunk
    return length


def run_video(images: torch.Tensor, window_fn: WindowFn, *, user_mask: Optional[torch.Tensor] = None,
              mask_overexposed: bool = True, mask_underexposed: bool = False,
              reference: Optional[torch.Tensor] = None, reference_ev: float = 5.0,
              window_size: int = 33, window_stride: int = 16, use_prev_window_reference: bool = False,
              fit_to: Optional[tuple[int, int]] = None,
              on_window: Optional[Callable[[int, int], None]] = None,
              timer: Optional[timing.StageTimer] = None) -> Result:
    """Runs image, short-video or long-video reconstruction.

    The input tensors are treated as read-only (ComfyUI caches node outputs, so a node's
    IMAGE input must survive untouched). Host memory is kept to the control frames, the
    mask and the output buffer: the resized clip is released as soon as the control
    frames exist, and finished windows are written straight into the output.

    Args:
        images: ``[F,H,W,3]`` sRGB, at processing size unless ``fit_to`` is given.
        window_fn: Generates log frames for one window.
        user_mask: Optional ``[F,H,W]`` mask overriding detection.
        mask_overexposed: Detect over-exposed regions.
        mask_underexposed: Detect under-exposed regions and paint them to 0.5.
        reference: Optional ``[1,H,W,3]`` sRGB reference image.
        reference_ev: EV boost for the reference.
        window_size: Frames per window (``4k+1``).
        window_stride: Stride between windows.
        use_prev_window_reference: Feed the previous window's output as reference.
        fit_to: Optional ``(height, width)`` to centre-crop and resize the clip to here,
            so the caller never holds a second full-size copy of it.
        on_window: Progress callback ``(index, count)``.
        timer: Optional :class:`timing.StageTimer` collecting per-stage seconds.

    Returns:
        Result with linear HDR and the mask used.
    """
    total = images.shape[0]
    if fit_to is not None:
        with timing.stage(timer, "resize"):
            images = frames.fit(images, fit_to[0], fit_to[1], crop=True)
    with timing.stage(timer, "masks"):
        images, mask = _prepare(images, user_mask, mask_overexposed, mask_underexposed)
    with timing.stage(timer, "control"):
        control = encode_control(images)
        ref_log = prepare_reference(reference, mask[0], reference_ev) if reference is not None else None
    del images          # everything downstream reads the control frames, not the sRGB clip

    if total == 1:
        n = frames.image_mode_num_frames(control.shape[1], control.shape[2])
        with timing.stage(timer, "window"):
            log = window_fn(control.expand(n, -1, -1, -1).contiguous(), mask.expand(n, -1, -1).contiguous(), ref_log)
        return Result(hdr=decode_output(log[:1]), mask=mask)

    size = frames.snap_4n1(window_size)
    plan = windows.plan_windows(total, size, window_stride)
    blender = windows.StreamingBlender(size, window_stride)
    hdr = torch.empty((total, *control.shape[1:]), dtype=torch.float32)
    filled = 0
    for index, win in enumerate(plan):
        if on_window is not None:
            on_window(index, len(plan))
        length = win.end - win.start
        with timing.stage(timer, "window"):
            # The per-window slices stay views of ``control`` / ``mask``; only the last,
            # short window is materialised by ``pad_frames``.
            log = window_fn(frames.pad_frames(control[win.start:win.end], size),
                            frames.pad_frames(mask[win.start:win.end], size), ref_log)[:length].float().clamp_min(0)
        if use_prev_window_reference:
            ref_log = log[min(window_stride, length - 1)][None].clone()
        with timing.stage(timer, "blend"):
            filled += _store(hdr, filled, blender.add(color.log_to_lin(log), win))
    with timing.stage(timer, "blend"):
        filled += _store(hdr, filled, blender.flush())
    return Result(hdr=hdr[:filled], mask=mask)


def run_pano(image: torch.Tensor, window_fn: WindowFn, *, user_mask: Optional[torch.Tensor] = None,
             timer: Optional[timing.StageTimer] = None) -> Result:
    """Runs panorama reconstruction on one equirectangular frame ``[1,H,W,3]``."""
    image = image[:1].float().clamp(0, 1)
    with timing.stage(timer, "masks"):
        if user_mask is not None:
            mask = (user_mask[:1].float() > 0.5).float()
        else:
            mask = masks.pano_mask(image[0])[None]
    with timing.stage(timer, "control"):
        control = encode_control(image)
    with timing.stage(timer, "window"):
        log = window_fn(control, mask, None)
    return Result(hdr=decode_output(log[:1]), mask=mask)
