"""ComfyUI V3 node registration for DiffHDR."""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .apply_lora import DiffHDRApplyLora
from .hdri import DiffHDRPano
from .postprocess import DiffHDRPostprocess
from .preprocess import DiffHDRPreprocess
from .save_exr import DiffHDRSaveEXR
from .tonemap import DiffHDRTonemap
from .video import DiffHDRVideo


class DiffHDRExtension(ComfyExtension):
    """Registers all DiffHDR nodes."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DiffHDRVideo, DiffHDRPano, DiffHDRApplyLora, DiffHDRPreprocess,
                DiffHDRPostprocess, DiffHDRTonemap, DiffHDRSaveEXR]


async def comfy_entrypoint() -> DiffHDRExtension:
    """ComfyUI entry point."""
    return DiffHDRExtension()
