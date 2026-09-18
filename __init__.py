"""ComfyUI-DiffHDR: DiffHDR LDR-to-HDR reconstruction nodes for ComfyUI."""

import importlib.util

if importlib.util.find_spec("comfy_api") is None:
    comfy_entrypoint = None
else:
    try:
        from .diffhdr.nodes import comfy_entrypoint
    except ImportError as e:
        # This happens when imported without proper package context (e.g., pytest)
        if "attempted relative import with no known parent package" in str(e):
            comfy_entrypoint = None
        else:
            # Real import error - re-raise it
            raise

__all__ = ["comfy_entrypoint"]
