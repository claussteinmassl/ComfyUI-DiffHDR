"""ComfyUI-DiffHDR: DiffHDR LDR-to-HDR reconstruction nodes for ComfyUI."""

import importlib.util

# ``__package__`` is empty when this file is executed as a plain top-level module
# (a script, or a spec without ``submodule_search_locations``); the relative import
# below is impossible then. ComfyUI always loads it as a package, so any ImportError
# raised there is a genuine one and must propagate.
if not __package__ or importlib.util.find_spec("comfy_api") is None:
    comfy_entrypoint = None
else:
    from .diffhdr.nodes import comfy_entrypoint

# ``ComfyExtension`` has no hook for web assets, so the legacy module attribute is used:
# ComfyUI reads it before it dispatches to the V3 entry point.
WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
