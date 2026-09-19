"""Frame geometry helpers: target size, resizing, padding, frame counts."""

import math

import torch
import torch.nn.functional as F

RESIZE_MODES = ("crop_to_720p", "native", "custom")
TRAINED_TOKENS = 9 * 45 * 80  # latent frames x (720/16) x (1280/16)

# Frames per processing block, derived from this pixel budget (32 frames at 720p).
# Element-wise stages run block by block so their temporaries stay bounded instead of
# growing with the clip length; the results are unaffected.
BLOCK_PIXELS = 32 * 720 * 1280


def block_size(height: int, width: int) -> int:
    """Frames per processing block for the given frame size."""
    return max(1, BLOCK_PIXELS // max(1, height * width))


def _round16(v: int) -> int:
    return max(16, int(round(v / 16)) * 16)


def resolve_size(height: int, width: int, mode: str, custom_height: int = 720, custom_width: int = 1280) -> tuple[int, int]:
    """Resolves the processing size ``(H, W)``, both multiples of 16.

    Args:
        height: Input height.
        width: Input width.
        mode: ``crop_to_720p`` | ``native`` | ``custom``.
        custom_height: Used for ``custom``.
        custom_width: Used for ``custom``.

    Raises:
        ValueError: On an unknown mode.
    """
    if mode == "crop_to_720p":
        return (1280, 720) if height > width else (720, 1280)
    if mode == "native":
        return max(16, height // 16 * 16), max(16, width // 16 * 16)
    if mode == "custom":
        return _round16(custom_height), _round16(custom_width)
    raise ValueError(f"Unknown resize_mode: {mode}")


def _center_crop_to_aspect(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    ih, iw = x.shape[-2:]
    if ih / iw < height / width:
        cw = max(1, int(ih / height * width))
        left = (iw - cw) // 2
        return x[..., :, left:left + cw]
    ch = max(1, int(iw / width * height))
    top = (ih - ch) // 2
    return x[..., top:top + ch, :]


def fit(images: torch.Tensor, height: int, width: int, crop: bool = True) -> torch.Tensor:
    """Resizes ``[F,H,W,3]`` images to ``height x width`` on the CPU.

    Args:
        images: Input images in [0,1].
        height: Target height.
        width: Target width.
        crop: Centre-crop to the target aspect first (True) or stretch (False).
    """
    if images.shape[1] == height and images.shape[2] == width:
        return images
    x = images.detach().cpu().float().movedim(-1, 1)
    if crop:
        x = _center_crop_to_aspect(x, height, width)
    # ``interpolate`` already returns a tensor this function owns, so clamp it in place
    # instead of allocating a second full-size copy.
    x = F.interpolate(x, size=(height, width), mode="bicubic", antialias=True, align_corners=False)
    return x.clamp_(0, 1).movedim(1, -1)


def fit_mask(mask: torch.Tensor, height: int, width: int, crop: bool = True) -> torch.Tensor:
    """Resizes a ``[F,H,W]`` mask with nearest sampling and binarises at 0.5."""
    x = mask.detach().cpu().float()[:, None]
    if x.shape[-2:] != (height, width):
        if crop:
            x = _center_crop_to_aspect(x, height, width)
        x = F.interpolate(x, size=(height, width), mode="nearest")
    return (x[:, 0] > 0.5).float()


def snap_4n1(n: int) -> int:
    """Smallest ``4k+1 >= n``."""
    return (max(1, n) - 1 + 3) // 4 * 4 + 1


def image_mode_num_frames(height: int, width: int) -> int:
    """Frame count that best matches the trained token budget for single images."""
    spatial = (height // 16) * (width // 16)
    if spatial == 0:
        return 33
    temporal = max(1, math.ceil(TRAINED_TOKENS / spatial))
    return (temporal - 1) * 4 + 1


def pad_frames(x: torch.Tensor, length: int) -> torch.Tensor:
    """Pads dim 0 to ``length`` by repeating the last entry."""
    if x.shape[0] >= length:
        return x
    reps = [length - x.shape[0]] + [1] * (x.dim() - 1)
    return torch.cat([x, x[-1:].repeat(*reps)], dim=0)
