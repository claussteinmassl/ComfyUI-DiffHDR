"""Shared input definitions for the DiffHDR nodes."""

from comfy_api.latest import io

from .. import attention, vae as dvae

CATEGORY = "DiffHDR"


def model_inputs() -> list:
    return [
        io.Model.Input("model", tooltip="Wan2.1-VACE-14B diffusion model (bf16, fp8 or GGUF). The DiffHDR LoRA is downloaded and applied automatically."),
        io.Vae.Input("vae", tooltip="Wan 2.1 VAE. Used in float32 by default to avoid banding in the log-encoded output."),
    ]


def clip_input():
    return io.Clip.Input("clip", optional=True, tooltip="Optional umT5-xxl text encoder. If not connected, the bundled DiffHDR embeddings are used and the prompt is ignored.")


def sampler_inputs(default_seed: int) -> list:
    return [
        io.Int.Input("steps", default=50, min=1, max=200, tooltip="Sampling steps. 50 matches the reference; around 10 is reported to be comparable."),
        io.Int.Input("seed", default=default_seed, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, tooltip="Noise seed. Long videos use the same seed for every window."),
    ]


def system_inputs() -> list:
    return [
        io.Combo.Input("attention", options=list(attention.ATTENTION_MODES), default="auto", tooltip="auto: flash-attn, then SageAttention if installed, else ComfyUI's default. Unavailable backends fall back to PyTorch SDPA."),
        io.Combo.Input("vae_precision", options=list(dvae.VAE_PRECISIONS), default="fp32", tooltip="fp32 is recommended. as_loaded saves memory but can cause banding in highlights."),
    ]


def hdr_outputs() -> list:
    return [
        io.Image.Output(display_name="hdr", tooltip="Linear scene-referred HDR (Rec.709 primaries, float32, values above 1.0). Do not route through 8-bit save nodes."),
        io.Mask.Output(display_name="mask", tooltip="Mask of regenerated regions."),
    ]
