"""Converts VAE-decoded DiffHDR log frames to linear HDR."""

from comfy_api.latest import io

from .. import pipeline
from . import common


class DiffHDRPostprocess(io.ComfyNode):
    """Log-encoded frames -> linear HDR."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffHDRPostprocess",
            display_name="DiffHDR Postprocess",
            category=common.CATEGORY,
            description="Decodes the DiffHDR log curve to linear scene-referred HDR. Use after VAE Decode in modular graphs.",
            inputs=[io.Image.Input("images", tooltip="VAE-decoded, log-encoded frames.")],
            outputs=[io.Image.Output(display_name="hdr", tooltip="Linear HDR (Rec.709 primaries).")],
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        return io.NodeOutput(pipeline.decode_output(images[..., :3]))
