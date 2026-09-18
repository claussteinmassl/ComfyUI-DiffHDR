"""ComfyUI-DiffHDR: DiffHDR LDR-to-HDR reconstruction nodes for ComfyUI."""

import importlib.util

if importlib.util.find_spec("comfy_api") is None:
    comfy_entrypoint = None
else:
    from diffhdr.nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
