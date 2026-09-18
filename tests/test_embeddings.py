import pytest
import torch
from safetensors.torch import save_file

from diffhdr import embeddings


def test_load_bundled_pads(tmp_path, monkeypatch):
    save_file({"cond": torch.ones(1, 3, 4096, dtype=torch.bfloat16)}, str(tmp_path / "diffhdr_empty.safetensors"))
    monkeypatch.setattr(embeddings, "ASSET_DIR", tmp_path)
    embeddings._cache.clear()
    cond = embeddings.load_bundled("empty")
    assert cond.shape == (1, 512, 4096) and cond.dtype == torch.float32
    assert cond[0, :3].eq(1).all() and cond[0, 3:].eq(0).all()


def test_missing_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "ASSET_DIR", tmp_path)
    embeddings._cache.clear()
    with pytest.raises(RuntimeError, match="CLIP"):
        embeddings.load_bundled("pano")


def test_conditioning_without_clip(tmp_path, monkeypatch):
    save_file({"cond": torch.ones(1, 2, 4096)}, str(tmp_path / "diffhdr_empty.safetensors"))
    monkeypatch.setattr(embeddings, "ASSET_DIR", tmp_path)
    embeddings._cache.clear()
    pos, neg = embeddings.get_conditioning(None, "", "standard")
    assert pos[0][0].shape == (1, 512, 4096) and neg[0][0].shape == (1, 512, 4096)
