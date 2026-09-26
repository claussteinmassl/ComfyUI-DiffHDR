"""Shared input definitions for the DiffHDR nodes."""

from comfy_api.latest import io

from .. import attention, masks, sampling
from .. import vae as dvae

CATEGORY = "DiffHDR"


def model_inputs() -> list:
    return [
        io.Model.Input("model", tooltip="Wan2.1-VACE-14B diffusion model (bf16, fp8 or GGUF). The DiffHDR LoRA is downloaded and applied automatically."),
        io.Vae.Input("vae", tooltip="Wan 2.1 VAE. Used in float32 by default to avoid banding in the log-encoded output."),
    ]


def clip_input():
    return io.Clip.Input("clip", optional=True, tooltip="Optional umT5-xxl text encoder. If not connected, the bundled DiffHDR embeddings are used and the prompt is ignored.")


PRESET_TOOLTIP = (
    "fast = res_multistep / simple / shift 8 (recommended, measured on Wan2.1-VACE-14B: matches the "
    "50-step reference at 10-20 steps, 4-7x faster). original = euler / simple / shift 5 (the reference "
    "implementation's sampler). custom = use the sampler / scheduler / shift widgets below."
)


def preset_input():
    """The preset dropdown, the first input of both all-in-one nodes."""
    return io.Combo.Input("preset", options=list(sampling.PRESET_NAMES), default=sampling.DEFAULT_PRESET,
                          tooltip=PRESET_TOOLTIP)


def sampler_inputs(default_seed: int) -> list:
    return [
        io.Int.Input("steps", default=20, min=1, max=200, tooltip="Sampling steps. 20 is the tuned default; 10 is enough with the fast preset; 50 = reference-implementation default."),
        io.Int.Input("seed", default=default_seed, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, tooltip="Noise seed. Long videos use the same seed for every window."),
    ]


def sampling_inputs() -> list:
    """The three advanced sampling widgets, honoured only when ``preset`` is ``custom``."""
    fast = sampling.PRESETS[sampling.DEFAULT_PRESET]
    return [
        io.Combo.Input("sampler", options=list(sampling.SAMPLERS), default=fast.sampler, tooltip="Sampler, used when preset = custom. res_multistep and dpmpp_2m are equivalent and reach the 50-step reference in far fewer steps than euler."),
        io.Combo.Input("scheduler", options=list(sampling.SCHEDULERS), default=fast.scheduler, tooltip="Scheduler, used when preset = custom. Only simple was measured; beta is deliberately not offered because it crushes highlights."),
        io.Float.Input("shift", default=fast.shift, min=1.0, max=12.0, step=0.5, tooltip="Flow-matching shift, used when preset = custom. 8 measured best with every sampler; 5 is the reference implementation's value."),
    ]


def system_inputs() -> list:
    return [
        io.Combo.Input("attention", options=list(attention.ATTENTION_MODES), default="auto", tooltip="auto: SageAttention, then flash-attn if installed, else ComfyUI's default. Unavailable backends fall back to PyTorch SDPA. SageAttention is quantised attention: choose sdpa or flash_attn for bit-reproducible results."),
        io.Combo.Input("vae_precision", options=list(dvae.VAE_PRECISIONS), default="fp32", tooltip="fp32 is recommended. as_loaded saves memory but can cause banding in highlights."),
    ]


OVER_THRESHOLD_TOOLTIP = (
    "Brightness (sRGB luma, 0-1) above which highlights are regenerated. 0.95 = DiffHDR default: only "
    "(nearly) clipped areas. Lower it, e.g. to 0.85, to also rebuild bright highlights that still hold "
    "some detail; the model sees that detail and extends it. Too low and correctly exposed areas get "
    "reinvented. Check the mask output while tuning."
)
UNDER_THRESHOLD_TOOLTIP = (
    "Level (sRGB, 0-1) below which shadows are regenerated, used when mask_underexposed is on. "
    "0.01 = DiffHDR default: only crushed blacks. Raise it, e.g. to 0.05, to also rebuild dark shadows. "
    "Only pixels below 0.01 are painted grey; darker detail above that stays visible to the model. "
    "Check the mask output while tuning."
)


def over_threshold_input():
    """The over-exposure threshold widget, appended after all older widgets."""
    return io.Float.Input("overexposed_threshold", default=masks.OVER_THR, min=0.5, max=1.0, step=0.01,
                          tooltip=OVER_THRESHOLD_TOOLTIP)


def threshold_inputs() -> list:
    """Over- and under-exposure threshold widgets, appended after all older widgets.

    ComfyUI stores widget values by position, so new widgets go last: saved workflows
    then keep loading with their old values and get the defaults for these.
    """
    return [
        over_threshold_input(),
        io.Float.Input("underexposed_threshold", default=masks.UNDER_THR, min=0.0, max=0.5, step=0.01,
                       tooltip=UNDER_THRESHOLD_TOOLTIP),
    ]


def hdr_outputs() -> list:
    return [
        io.Image.Output(display_name="hdr", tooltip="Linear scene-referred HDR (Rec.709 primaries, float32, values above 1.0). Do not route through 8-bit save nodes."),
        io.Mask.Output(display_name="mask", tooltip="Mask of regenerated regions."),
    ]
