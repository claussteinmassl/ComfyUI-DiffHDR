"""Builds the ComfyUI-backed window function used by the pipeline."""

from typing import Callable, Optional

import torch

from . import sampling, timing, vace
from . import vae as dvae
from .pipeline import WindowFn


def make_window_fn(model, vae, positive, negative, steps: int, seed: int,
                   settings: sampling.SamplingSettings,
                   on_step: Optional[Callable[[], None]] = None,
                   timer: Optional[timing.StageTimer] = None) -> WindowFn:
    """Creates ``window_fn(control_log, mask, reference_log) -> log frames``.

    Args:
        model: Prepared ModelPatcher (see :func:`sampling.prepare_model`).
        vae: VAE to use (see :func:`vae.get_vae`).
        positive: Positive conditioning.
        negative: Negative conditioning.
        steps: Sampling steps.
        seed: Noise seed (same for every window, as in the reference implementation).
        settings: Sampler, scheduler and shift to sample with (see :func:`sampling.resolve_settings`).
            The shift is already patched into ``model``; sampler and scheduler are used here.
        on_step: Called once per sampling step.
        timer: Optional :class:`timing.StageTimer` collecting ``encode``/``sample``/``decode``.

    Returns:
        WindowFn: The window function to hand to :mod:`pipeline`.
    """

    def callback(step, x0, x, total_steps):
        if on_step is not None:
            on_step()

    def window_fn(control: torch.Tensor, mask: torch.Tensor, reference: Optional[torch.Tensor]) -> torch.Tensor:
        import comfy.model_management
        comfy.model_management.throw_exception_if_processing_interrupted()
        with timing.stage(timer, "encode"):
            vc = vace.build(vae, control, mask, reference)
        with timing.stage(timer, "sample"):
            latent = sampling.sample(model, vace.apply(positive, vc), vace.apply(negative, vc),
                                     vc.latent_shape, steps, seed, settings.sampler, settings.scheduler,
                                     callback=callback)
        if vc.trim:
            latent = latent[:, :, vc.trim:]
        with timing.stage(timer, "decode"):
            return dvae.decode(vae, latent)[: control.shape[0]]

    return window_fn
