import json
from pathlib import Path

import pytest
import torch

from diffhdr import lora

KEYS = json.loads((Path(__file__).parent / "fixtures" / "lora_keys.json").read_text())


def test_fixture_shape():
    assert len(KEYS) == 160
    assert all(k.startswith("vace_blocks.") and ".default.weight" in k for k in KEYS)


def test_rename_keys():
    sd = {k: torch.zeros(1) for k in KEYS}
    renamed = lora.rename_keys(sd)
    assert len(renamed) == 160 and all(k.startswith("diffusion_model.vace_blocks.") for k in renamed)
    assert lora.rename_keys(renamed).keys() == renamed.keys()


def test_ensure_lora_existing(tmp_path, monkeypatch):
    target = tmp_path / "DiffHDR" / lora.LORA_FILES["standard"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    monkeypatch.setattr(lora, "_candidate_paths", lambda filename: [target])
    monkeypatch.setattr(lora, "LORA_SIZE", 1)
    assert lora.ensure_lora("standard") == target


def test_ensure_lora_download_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(lora, "_candidate_paths", lambda filename: [tmp_path / "DiffHDR" / filename])

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(lora, "_download", boom)
    with pytest.raises(RuntimeError, match="huggingface.co/Eyeline-Labs/DiffHDR"):
        lora.ensure_lora("pano")


def test_unknown_variant():
    with pytest.raises(ValueError):
        lora.ensure_lora("bogus")


@pytest.mark.requires_comfy
def test_comfy_parses_all_patches():
    import comfy.lora
    sd = lora.rename_keys({k: torch.zeros(*shape) for k, shape in KEYS.items()})
    modules = {k.rsplit(".lora_", 1)[0] for k in sd}
    key_map = {m: m + ".weight" for m in modules}
    loaded = comfy.lora.load_lora(sd, key_map)
    assert len(loaded) == lora.EXPECTED_PATCHES == 80
