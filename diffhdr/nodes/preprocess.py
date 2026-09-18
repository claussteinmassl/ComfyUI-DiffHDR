"""Builds DiffHDR control frames and masks for native VACE graphs."""

from comfy_api.latest import io

from .. import masks, pipeline
from . import common


class DiffHDRPreprocess(io.ComfyNode):
    """LDR frames -> log-encoded control video + regeneration mask."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffHDRPreprocess",
            display_name="DiffHDR Preprocess",
            category=common.CATEGORY,
            description="Log-encodes LDR frames and detects clipped regions. Connect to WanVaceToVideo (control_video, control_masks).",
            inputs=[
                io.Image.Input("images", tooltip="sRGB LDR frames at the final processing size (multiples of 16)."),
                io.Combo.Input("variant", options=["video", "pano"], default="video", tooltip="Mask detector: video/image (temporally stabilised) or pano."),
                io.Boolean.Input("mask_overexposed", default=True, tooltip="Detect over-exposed regions (video variant)."),
                io.Boolean.Input("mask_underexposed", default=False, tooltip="Detect under-exposed regions and paint them mid-grey (video variant)."),
            ],
            outputs=[
                io.Image.Output(display_name="control_video", tooltip="Log-encoded frames in [0,1]. Keep as float; do not route through 8-bit nodes."),
                io.Mask.Output(display_name="control_masks", tooltip="1 = regenerate."),
            ],
        )

    @classmethod
    def execute(cls, images, variant, mask_overexposed, mask_underexposed) -> io.NodeOutput:
        images = images[..., :3].float().clamp(0, 1)
        if variant == "pano":
            mask = masks.pano_mask(images[0])[None].expand(images.shape[0], -1, -1).contiguous()
        else:
            vm = masks.video_masks(images, use_over=mask_overexposed, use_under=mask_underexposed)
            if mask_underexposed:
                images = masks.paint_underexposed(images, vm.under)
            mask = vm.combined
        return io.NodeOutput(pipeline.encode_control(images), mask)
