"""Tests for attention backend selection."""

import logging

import pytest
import torch

pytestmark = pytest.mark.requires_comfy


def _fake_backends(monkeypatch, available):
    """Installs identifiable attention backends and a fixed availability rule.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        available: The backend names ``_available`` should report as usable.
    """
    import comfy.ldm.modules.attention as ca

    from diffhdr import attention

    monkeypatch.setattr(ca, "attention_flash", lambda *a, **k: "flash_attn")
    monkeypatch.setattr(ca, "attention_sage", lambda *a, **k: "sage")
    monkeypatch.setattr(attention, "_available", lambda name, device: name in available)


def _picked(override):
    """Returns the name of the backend an override dispatches to, or None."""
    return None if override is None else override(None)


def test_auto_prefers_sage_over_flash_attn(monkeypatch, caplog):
    from diffhdr import attention
    _fake_backends(monkeypatch, {"flash_attn", "sage"})
    with caplog.at_level(logging.INFO, logger="DiffHDR"):
        override = attention.resolve("auto", torch.device("cuda"))
    assert _picked(override) == "sage"
    assert "DiffHDR attention: sage" in caplog.text


def test_auto_uses_flash_attn_when_sage_is_unavailable(monkeypatch):
    from diffhdr import attention
    _fake_backends(monkeypatch, {"flash_attn"})
    assert _picked(attention.resolve("auto", torch.device("cuda"))) == "flash_attn"


def test_auto_without_a_backend_is_comfyui_default(monkeypatch):
    from diffhdr import attention
    _fake_backends(monkeypatch, set())
    assert attention.resolve("auto", torch.device("cuda")) is None


def test_auto_on_cpu_ignores_importable_backends(monkeypatch):
    import comfy.ldm.modules.attention as ca

    from diffhdr import attention

    monkeypatch.setattr(ca, "FLASH_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(ca, "SAGE_ATTENTION_IS_AVAILABLE", True)
    assert attention.resolve("auto", torch.device("cpu")) is None


def test_auto_on_cpu_is_default():
    from diffhdr import attention
    assert attention.resolve("auto", torch.device("cpu")) is None


def test_sdpa_override_forwards():
    from diffhdr import attention
    override = attention.resolve("sdpa", torch.device("cpu"))
    q = torch.rand(1, 4, 16)
    out = override(lambda *a, **k: (_ for _ in ()).throw(AssertionError("default must not be called")), q, q, q, 2)
    assert out.shape == (1, 4, 16)


def test_unavailable_backend_falls_back(caplog):
    from diffhdr import attention
    override = attention.resolve("flash_attn", torch.device("cpu"))
    assert override is not None and "falling back" in caplog.text.lower()


def test_unknown():
    from diffhdr import attention
    with pytest.raises(ValueError):
        attention.resolve("bogus", torch.device("cpu"))


def test_modes_are_the_public_list():
    from diffhdr import attention
    assert attention.ATTENTION_MODES == ("auto", "flash_attn", "sage", "sdpa")


def test_override_called_through_wrap_attn_runs_once():
    """ComfyUI's wrap_attn must not re-enter the override when it calls the backend."""
    import comfy.ldm.modules.attention as ca

    from diffhdr import attention

    sdpa = attention.resolve("sdpa", torch.device("cpu"))
    calls = []

    def counting(func, *args, **kwargs):
        calls.append(1)
        assert len(calls) < 5, "attention override recursed"
        return sdpa(func, *args, **kwargs)

    q = torch.rand(1, 4, 16)
    out = ca.attention_pytorch(q, q, q, 2, transformer_options={"optimized_attention_override": counting})
    assert out.shape == (1, 4, 16)
    assert len(calls) == 1


def test_override_terminates_with_itself_in_transformer_options():
    """Forwarding must not dispatch back into the override even if it stays installed."""
    from diffhdr import attention

    override = attention.resolve("sdpa", torch.device("cpu"))
    q = torch.rand(1, 4, 16)
    out = override(None, q, q, q, 2, transformer_options={"optimized_attention_override": override})
    assert out.shape == (1, 4, 16)
