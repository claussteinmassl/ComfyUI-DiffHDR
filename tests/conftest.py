"""Shared pytest configuration."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = REPO_ROOT / ".dev" / "reference" / "DiffHDR"

_comfy_path = os.environ.get("COMFYUI_PATH")
if _comfy_path and Path(_comfy_path).is_dir() and _comfy_path not in sys.path:
    sys.path.insert(0, _comfy_path)


def _has_comfy() -> bool:
    return importlib.util.find_spec("comfy") is not None


def pytest_collection_modifyitems(config, items):
    if _has_comfy():
        return
    skip = pytest.mark.skip(reason="ComfyUI not importable (set COMFYUI_PATH)")
    for item in items:
        if "requires_comfy" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def reference_color_utils():
    """The reference implementation's colour utils, if the local clone exists."""
    path = REFERENCE_ROOT / "utils" / "color_utils.py"
    if not path.is_file():
        pytest.skip("reference clone not available")
    spec = importlib.util.spec_from_file_location("ref_color_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
