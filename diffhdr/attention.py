"""Attention backend selection with a PyTorch SDPA fallback."""

import logging

ATTENTION_MODES = ("auto", "flash_attn", "sage", "sdpa")
log = logging.getLogger("DiffHDR")

# ComfyUI's ``wrap_attn`` sets this keyword before it dispatches to an
# ``optimized_attention_override``; a wrapped attention function that still sees it calls its
# own implementation instead of the override, which is what keeps forwarding from recursing.
_INSIDE_WRAPPER = "_inside_attn_wrapper"


def _is_gpu(device) -> bool:
    """Reports whether ``device`` is a discrete accelerator (not CPU or MPS).

    Args:
        device: A ``torch.device`` or device string.

    Returns:
        bool: True for CUDA/ROCm/XPU-style devices, False for ``cpu`` and ``mps``.
    """
    return getattr(device, "type", str(device)) not in ("cpu", "mps")


def _available(name: str, device) -> bool:
    """Reports whether an optional attention backend can run on ``device``.

    Args:
        name: ``flash_attn`` or ``sage``.
        device: The torch device sampling will run on.

    Returns:
        bool: True if the package is importable and the device is a GPU.
    """
    import comfy.ldm.modules.attention as ca
    if not _is_gpu(device):
        return False
    if name == "flash_attn":
        return bool(ca.FLASH_ATTENTION_IS_AVAILABLE)
    return bool(ca.SAGE_ATTENTION_IS_AVAILABLE)


def _override(target):
    """Wraps an attention function as an ``optimized_attention_override`` callable.

    Args:
        target: The attention function to call instead of ComfyUI's default.

    Returns:
        Callable: ``override(func, *args, **kwargs)`` that ignores ``func`` and calls ``target``.
    """
    def override(func, *args, **kwargs):
        return target(*args, **{**kwargs, _INSIDE_WRAPPER: True})
    return override


def resolve(mode: str, device):
    """Returns an ``optimized_attention_override`` callable, or None for ComfyUI's default.

    ``auto`` prefers SageAttention, then flash-attn, then ComfyUI's configured default.
    Explicit ``flash_attn``/``sage`` fall back to PyTorch SDPA when unavailable. ComfyUI's
    flash/sage wrappers additionally fall back to SDPA on runtime errors.

    Args:
        mode: One of :data:`ATTENTION_MODES`.
        device: The torch device sampling will run on.

    Returns:
        Callable | None: The override callable, or None to keep ComfyUI's default.

    Raises:
        ValueError: On an unknown mode.
    """
    import comfy.ldm.modules.attention as ca
    if mode not in ATTENTION_MODES:
        raise ValueError(f"Unknown attention mode: {mode}")
    funcs = {"flash_attn": ca.attention_flash, "sage": ca.attention_sage}
    if mode == "sdpa":
        return _override(ca.attention_pytorch)
    if mode == "auto":
        for name in ("sage", "flash_attn"):
            if _available(name, device):
                log.info("DiffHDR attention: %s", name)
                return _override(funcs[name])
        return None
    if _available(mode, device):
        return _override(funcs[mode])
    log.warning("DiffHDR attention '%s' is not available on this system, falling back to PyTorch SDPA.", mode)
    return _override(ca.attention_pytorch)
