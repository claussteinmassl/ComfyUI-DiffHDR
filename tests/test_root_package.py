"""Test root package loading as ComfyUI does it."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_root_package_without_package_context():
    """Loaded as a plain top-level module (no package context) the entrypoint is None.

    The root ``__init__.py`` must decide this structurally (``__package__`` is empty),
    not by string-matching CPython's relative-import error message.
    """
    spec = importlib.util.spec_from_file_location(
        "diffhdr_root_no_package", REPO_ROOT / "__init__.py", submodule_search_locations=None)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        assert not module.__package__
        assert module.comfy_entrypoint is None
    finally:
        sys.modules.pop("diffhdr_root_no_package", None)


@pytest.mark.requires_comfy
def test_root_package_comfy_import():
    """Test that root __init__.py imports correctly when loaded as ComfyUI does.

    ComfyUI imports custom node packages using importlib with
    submodule_search_locations set to the repo root. This test verifies
    the root __init__.py can be imported in that context.
    """
    repo_root_str = str(REPO_ROOT)

    # Save original state
    original_path = sys.path.copy()
    diffhdr_keys = [k for k in sys.modules.keys() if k.startswith("diffhdr") or k.startswith("ComfyUI_DiffHDR")]
    removed_modules = {k: sys.modules.pop(k) for k in diffhdr_keys}

    try:
        # Remove repo root from sys.path to simulate ComfyUI import context
        # (ComfyUI doesn't add the custom node directory to sys.path)
        sys.path = [p for p in sys.path if p != repo_root_str]

        # Import the root __init__.py as ComfyUI would using importlib
        spec = importlib.util.spec_from_file_location(
            "ComfyUI_DiffHDR_under_test",
            REPO_ROOT / "__init__.py",
            submodule_search_locations=[str(REPO_ROOT)],
        )
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        sys.modules["ComfyUI_DiffHDR_under_test"] = module
        spec.loader.exec_module(module)

        # Assert that comfy_entrypoint exists
        assert hasattr(module, "comfy_entrypoint"), "Root module should have comfy_entrypoint"
        assert module.comfy_entrypoint is not None, "comfy_entrypoint should not be None when comfy_api is available"

        # Assert it's a coroutine function
        assert asyncio.iscoroutinefunction(module.comfy_entrypoint), \
            "comfy_entrypoint should be an async coroutine function"
    finally:
        # Restore sys.path and modules
        sys.path = original_path
        sys.modules.pop("ComfyUI_DiffHDR_under_test", None)
        for k, v in removed_modules.items():
            sys.modules[k] = v
