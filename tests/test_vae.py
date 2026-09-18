import pytest
import torch

pytestmark = pytest.mark.requires_comfy


def test_as_loaded_is_identity():
    from diffhdr import vae as dvae
    sentinel = object()
    assert dvae.get_vae(sentinel, "as_loaded") is sentinel


def test_to_frames():
    from diffhdr import vae as dvae
    assert dvae.to_frames(torch.zeros(1, 5, 8, 8, 3)).shape == (5, 8, 8, 3)
    assert dvae.to_frames(torch.zeros(5, 8, 8, 3)).shape == (5, 8, 8, 3)
