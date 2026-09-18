"""ComfyUI V3 node registration for DiffHDR."""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class DiffHDRExtension(ComfyExtension):
    """Registers all DiffHDR nodes."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return []


async def comfy_entrypoint() -> DiffHDRExtension:
    """ComfyUI entry point."""
    return DiffHDRExtension()
