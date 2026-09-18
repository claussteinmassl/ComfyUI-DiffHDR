"""VACE conditioning tensors (same layout as ComfyUI's WanVaceToVideo)."""

from dataclasses import dataclass
from typing import Optional

import torch

VAE_STRIDE = 8


@dataclass
class VaceCond:
    """VACE conditioning for one window."""

    frames: torch.Tensor          # [1,32,T,h,w]
    mask: torch.Tensor            # [1,64,T,h,w]
    trim: int                     # reference latent frames to drop after sampling
    latent_shape: tuple           # (1,16,T,h,w)


def build(vae, control: torch.Tensor, mask: torch.Tensor, reference: Optional[torch.Tensor] = None) -> VaceCond:
    """Encodes control frames, mask and optional reference.

    Args:
        vae: ComfyUI VAE (Wan 2.1).
        control: ``[F,H,W,3]`` log-encoded frames in [0,1]; H, W multiples of 16, F = 4k+1.
        mask: ``[F,H,W]`` binary mask, 1 = regenerate.
        reference: Optional ``[1,H,W,3]`` log-encoded reference image.

    Returns:
        VaceCond: The VACE frames/mask tensors, the reference trim length and
        the latent shape to sample.
    """
    import comfy.latent_formats

    length, height, width, _ = control.shape
    latent_length = (length - 1) // 4 + 1
    m = mask.float()[..., None]
    centered = control.float() - 0.5
    inactive = vae.encode(centered * (1 - m) + 0.5)
    reactive = vae.encode(centered * m + 0.5)
    frames = torch.cat((inactive, reactive), dim=1)

    hm, wm = height // VAE_STRIDE, width // VAE_STRIDE
    vmask = mask.float().view(length, hm, VAE_STRIDE, wm, VAE_STRIDE).permute(2, 4, 0, 1, 3)
    vmask = vmask.reshape(VAE_STRIDE * VAE_STRIDE, length, hm, wm)
    vmask = torch.nn.functional.interpolate(vmask.unsqueeze(0), size=(latent_length, hm, wm), mode="nearest-exact").squeeze(0)

    trim = 0
    if reference is not None:
        ref = vae.encode(reference[:1, :, :, :3].float())
        ref = torch.cat([ref, comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref))], dim=1)
        frames = torch.cat((ref, frames), dim=2)
        vmask = torch.cat((torch.zeros_like(vmask[:, : ref.shape[2]]), vmask), dim=1)
        trim = ref.shape[2]
        latent_length += trim

    return VaceCond(frames=frames, mask=vmask.unsqueeze(0), trim=trim, latent_shape=(1, 16, latent_length, hm, wm))


def apply(conditioning, vc: VaceCond, strength: float = 1.0):
    """Returns a copy of ``conditioning`` carrying the VACE tensors.

    Args:
        conditioning: ComfyUI CONDITIONING list.
        vc: The VACE conditioning built by :func:`build`.
        strength: VACE strength applied to every conditioning entry.

    Returns:
        list: The conditioning list with ``vace_frames``/``vace_mask``/``vace_strength`` set.
    """
    import node_helpers
    return node_helpers.conditioning_set_values(
        conditioning, {"vace_frames": [vc.frames], "vace_mask": [vc.mask], "vace_strength": [strength]}, append=True)
