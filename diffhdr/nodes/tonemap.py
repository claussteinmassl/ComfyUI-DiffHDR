"""Tonemaps linear HDR for preview."""

from comfy_api.latest import io

from .. import color
from . import common


class DiffHDRTonemap(io.ComfyNode):
    """Linear HDR -> display sRGB."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffHDRTonemap",
            display_name="DiffHDR Tonemap Preview",
            category=common.CATEGORY,
            description="Converts linear HDR to a displayable sRGB image for preview or LDR export.",
            inputs=[
                io.Image.Input("hdr", tooltip="Linear HDR image."),
                io.Float.Input("exposure", default=0.0, min=-16.0, max=16.0, step=0.25, tooltip="Exposure offset in stops."),
                io.Combo.Input("operator", options=["reinhard", "clip"], default="reinhard", tooltip="reinhard compresses highlights; clip shows the LDR range at the chosen exposure."),
            ],
            outputs=[io.Image.Output(display_name="image", tooltip="sRGB image in [0,1].")],
        )

    @classmethod
    def execute(cls, hdr, exposure, operator) -> io.NodeOutput:
        return io.NodeOutput(color.tonemap(hdr[..., :3].float(), exposure, operator))
