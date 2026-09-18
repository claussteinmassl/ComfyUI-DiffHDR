"""VAE helpers: fp32 working copy and decoding to frames."""

import weakref

import torch

VAE_PRECISIONS = ("fp32", "as_loaded")
_fp32_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def get_vae(vae, precision: str):
    """Returns the VAE to use for DiffHDR.

    Args:
        vae: ComfyUI VAE supplied by the user.
        precision: ``fp32`` builds (and caches) a float32 copy; ``as_loaded`` returns ``vae`` unchanged.

    Returns:
        The VAE object to run with.
    """
    if precision == "as_loaded" or getattr(vae, "vae_dtype", None) == torch.float32:
        return vae
    if vae not in _fp32_cache:
        import comfy.sd
        sd = {k: v.to(torch.float32) for k, v in vae.get_sd().items()}
        copy = comfy.sd.VAE(sd=sd, dtype=torch.float32)
        copy.throw_exception_if_invalid()
        _fp32_cache[vae] = copy
    return _fp32_cache[vae]


def check_wan_vae(vae) -> None:
    """Raises ValueError unless ``vae`` is a 16-channel Wan 2.1 VAE.

    Args:
        vae: ComfyUI VAE to validate.

    Raises:
        ValueError: If the VAE is not a 16-channel Wan 2.1 VAE.
    """
    if getattr(vae, "latent_channels", None) != 16:
        raise ValueError("DiffHDR requires the Wan 2.1 VAE (wan_2.1_vae.safetensors).")


def to_frames(decoded: torch.Tensor) -> torch.Tensor:
    """Flattens a ``[B,F,H,W,C]`` video decode to ``[F,H,W,C]``.

    Args:
        decoded: Decoded pixels, either ``[B,F,H,W,C]`` or already ``[F,H,W,C]``.

    Returns:
        torch.Tensor: The frames as ``[F,H,W,C]``.
    """
    if decoded.dim() == 5:
        decoded = decoded.reshape(-1, *decoded.shape[-3:])
    return decoded


def decode(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decodes a Wan latent ``[1,16,T,h,w]`` to log frames ``[F,H,W,3]`` (float32).

    Args:
        vae: ComfyUI VAE (Wan 2.1).
        latent: The latent samples ``[1,16,T,h,w]``.

    Returns:
        torch.Tensor: Decoded frames ``[F,H,W,3]`` as float32.
    """
    return to_frames(vae.decode(latent)).float()
