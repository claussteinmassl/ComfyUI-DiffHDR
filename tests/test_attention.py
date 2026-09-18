"""Tests for attention backend selection."""

import pytest
import torch

pytestmark = pytest.mark.requires_comfy


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
