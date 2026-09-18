"""Applies the DiffHDR LoRA to a Wan2.1-VACE-14B model."""

from comfy_api.latest import io

from .. import lora, sampling
from . import common


class DiffHDRApplyLora(io.ComfyNode):
    """Downloads (first use) and applies the DiffHDR LoRA."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffHDRApplyLora",
            display_name="DiffHDR Apply LoRA",
            category=common.CATEGORY,
            description="For modular graphs: patches the DiffHDR LoRA into a Wan2.1-VACE-14B model. Use with ModelSamplingSD3 shift 5, euler/simple, cfg 1.",
            inputs=[
                io.Model.Input("model", tooltip="Wan2.1-VACE-14B diffusion model."),
                io.Combo.Input("variant", options=list(lora.LORA_FILES), default="standard", tooltip="standard: images and videos. pano: equirectangular HDRIs."),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05, tooltip="LoRA strength. 1.0 matches the reference."),
            ],
            outputs=[io.Model.Output(display_name="model", tooltip="Model with the DiffHDR LoRA applied.")],
        )

    @classmethod
    def execute(cls, model, variant, strength) -> io.NodeOutput:
        sampling.check_vace_model(model)
        return io.NodeOutput(lora.apply_lora(model, variant, strength))
