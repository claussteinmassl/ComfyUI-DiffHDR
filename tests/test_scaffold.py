import pytest
import torch

import diffhdr
from tests import conftest


def test_version():
    assert diffhdr.__version__ == "0.1.0"


@pytest.mark.parametrize("value, expected", [
    (None, False), ("", False), ("0", False), ("false", False), ("False", False),
    ("1", True), ("true", True), ("yes", True),
])
def test_require_comfy_env_flag(monkeypatch, value, expected):
    """``DIFFHDR_REQUIRE_COMFY`` turns the ComfyUI skip into a hard error in CI."""
    monkeypatch.delenv("DIFFHDR_REQUIRE_COMFY", raising=False)
    if value is not None:
        monkeypatch.setenv("DIFFHDR_REQUIRE_COMFY", value)
    assert conftest.require_comfy() is expected


def test_require_comfy_without_comfy_is_a_usage_error(monkeypatch):
    monkeypatch.setenv("DIFFHDR_REQUIRE_COMFY", "1")
    monkeypatch.setattr(conftest, "_has_comfy", lambda: False)
    with pytest.raises(pytest.UsageError, match="DIFFHDR_REQUIRE_COMFY"):
        conftest.pytest_configure(None)


def test_accelerator_probe_survives_a_backend_that_raises(monkeypatch):
    def boom():
        raise RuntimeError("no driver")

    assert isinstance(conftest._accelerator_available(), bool)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", boom)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert conftest._accelerator_available() is False


@pytest.mark.parametrize("has_comfy, accelerator, expected", [
    (True, False, True),        # CPU-only CI runner: ComfyUI must be told to use the CPU
    (True, True, False),        # a real GPU box keeps ComfyUI's own device choice
    (False, False, False),      # no ComfyUI at all, nothing to configure
])
def test_should_force_cpu(monkeypatch, has_comfy, accelerator, expected):
    monkeypatch.setattr(conftest, "_has_comfy", lambda: has_comfy)
    monkeypatch.setattr(conftest, "_accelerator_available", lambda: accelerator)
    assert conftest._should_force_cpu() is expected


def test_reference_skip_reason_names_the_clone_command():
    assert "git clone https://github.com/Eyeline-Labs/DiffHDR .dev/reference/DiffHDR" in conftest.REFERENCE_HINT
