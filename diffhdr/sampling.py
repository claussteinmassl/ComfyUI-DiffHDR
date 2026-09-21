"""Model preparation and sampling for one DiffHDR window.

The sampler, scheduler and shift live here as a small preset table so that the nodes, the
docs and the tests all read the same values. ``original`` reproduces the reference
implementation; ``fast`` is the tuned default measured on Wan2.1-VACE-14B.
"""

from dataclasses import dataclass

import torch

from . import attention, lora

CFG = 1.0


@dataclass(frozen=True)
class SamplingSettings:
    """The three sampling parameters a DiffHDR preset fixes.

    Attributes:
        sampler: ComfyUI sampler name.
        scheduler: ComfyUI scheduler name.
        shift: Flow-matching shift (as set by ``ModelSamplingSD3``).
    """

    sampler: str
    scheduler: str
    shift: float


PRESETS = {
    "fast": SamplingSettings("res_multistep", "simple", 8.0),
    "original": SamplingSettings("euler", "simple", 5.0),
}
DEFAULT_PRESET = "fast"
PRESET_NAMES = [*PRESETS, "custom"]

# Curated options. ``beta`` (crushes highlights) and ``deis`` (invents glow) measured worse
# than doing nothing and are deliberately not offered.
SAMPLERS = ["res_multistep", "dpmpp_2m", "euler"]
SCHEDULERS = ["simple", "normal", "sgm_uniform"]


def resolve_settings(preset: str, sampler: str, scheduler: str, shift: float) -> SamplingSettings:
    """Returns the settings a node should sample with.

    Args:
        preset: ``fast``, ``original`` or ``custom``.
        sampler: Sampler widget value, honoured only for ``custom``.
        scheduler: Scheduler widget value, honoured only for ``custom``.
        shift: Shift widget value, honoured only for ``custom``.

    Returns:
        SamplingSettings: The preset's values, or the widget values for ``custom``.

    Raises:
        ValueError: If ``preset`` is not one of :data:`PRESET_NAMES`.
    """
    if preset in PRESETS:
        return PRESETS[preset]
    if preset != "custom":
        raise ValueError(f"unknown preset {preset!r}; expected one of {PRESET_NAMES}")
    return SamplingSettings(sampler, scheduler, float(shift))


def check_vace_model(model) -> None:
    """Raises ValueError unless ``model`` is a Wan2.1-VACE-14B ModelPatcher.

    Args:
        model: The MODEL supplied by the user.

    Raises:
        ValueError: If the diffusion model has no 8 VACE blocks or a hidden size other than 5120.
    """
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(dm, "vace_blocks", None)
    if blocks is None or len(blocks) != 8 or getattr(dm, "dim", None) != 5120:
        raise ValueError(
            "DiffHDR requires Wan2.1-VACE-14B as MODEL (e.g. wan2.1_vace_14B_fp16.safetensors from "
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged, or a GGUF/fp8 variant). "
            "The 1.3B VACE model and non-VACE Wan models are not compatible with the DiffHDR LoRA."
        )


def set_shift(model, shift: float):
    """Patches flow-matching shift in place on a cloned ModelPatcher (same as ModelSamplingSD3).

    Args:
        model: A ModelPatcher clone; the object patch is applied in place.
        shift: The flow-matching shift.

    Returns:
        The same ModelPatcher, for chaining.
    """
    import comfy.model_sampling

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    original = model.get_model_object("model_sampling")
    ms = ModelSamplingAdvanced(model.model.model_config)
    ms.set_parameters(shift=shift, multiplier=1000)
    if hasattr(original, "noise_scale"):
        ms.set_noise_scale(original.noise_scale)
    model.add_object_patch("model_sampling", ms)
    return model


def prepare_model(model, variant: str, attention_mode: str = "auto",
                  shift: float = PRESETS[DEFAULT_PRESET].shift, apply_diffhdr_lora: bool = True):
    """Returns a patched clone: DiffHDR LoRA, flow-matching shift, optional attention override.

    Args:
        model: The Wan2.1-VACE-14B MODEL supplied by the user; it is never modified.
        variant: LoRA variant, ``standard`` or ``pano``.
        attention_mode: One of :data:`attention.ATTENTION_MODES`.
        shift: Flow-matching shift to patch in (see :data:`PRESETS`).
        apply_diffhdr_lora: Set False to skip the LoRA (the model already carries it).

    Returns:
        The patched ModelPatcher clone.
    """
    import comfy.model_management
    check_vace_model(model)
    patched = lora.apply_lora(model, variant) if apply_diffhdr_lora else model.clone()
    set_shift(patched, shift)
    override = attention.resolve(attention_mode, comfy.model_management.get_torch_device())
    if override is not None:
        # Rebuild the dict instead of writing into it, so a shallow clone can never leak
        # the override back into the user's original model.
        patched.model_options["transformer_options"] = dict(
            patched.model_options.get("transformer_options", {}), optimized_attention_override=override)
    return patched


def sample(model, positive, negative, latent_shape, steps: int, seed: int,
           sampler: str, scheduler: str, callback=None) -> torch.Tensor:
    """Samples one window from pure noise.

    Args:
        model: The prepared ModelPatcher (see :func:`prepare_model`).
        positive: Positive conditioning with VACE tensors applied.
        negative: Negative conditioning with VACE tensors applied.
        latent_shape: The latent shape ``(1,16,T,h,w)`` to sample.
        steps: Number of sampling steps.
        seed: Noise seed.
        sampler: ComfyUI sampler name (see :data:`SAMPLERS`).
        scheduler: ComfyUI scheduler name (see :data:`SCHEDULERS`).
        callback: Optional ``callback(step, x0, x, total_steps)`` progress hook.

    Returns:
        torch.Tensor: The sampled latent ``[1,16,T,h,w]``.
    """
    import comfy.model_management
    import comfy.sample
    latent = torch.zeros(latent_shape, device=comfy.model_management.intermediate_device())
    noise = comfy.sample.prepare_noise(latent, seed)
    return comfy.sample.sample(model, noise, steps, CFG, sampler, scheduler, positive, negative, latent,
                               denoise=1.0, callback=callback, disable_pbar=True, seed=seed)
