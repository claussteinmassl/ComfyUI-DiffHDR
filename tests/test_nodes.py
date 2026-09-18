import asyncio

import pytest
import torch

pytestmark = pytest.mark.requires_comfy

EXPECTED = {"DiffHDRVideo", "DiffHDRPano", "DiffHDRApplyLora", "DiffHDRPreprocess",
            "DiffHDRPostprocess", "DiffHDRTonemap", "DiffHDRSaveEXR"}


def _nodes():
    from diffhdr.nodes import comfy_entrypoint
    ext = asyncio.run(comfy_entrypoint())
    return asyncio.run(ext.get_node_list())


def test_registry_and_tooltips():
    nodes = _nodes()
    assert {n.define_schema().node_id for n in nodes} == EXPECTED
    for n in nodes:
        schema = n.define_schema()
        assert schema.category == "DiffHDR"
        for inp in schema.inputs:
            assert inp.tooltip, f"{schema.node_id}.{inp.id} has no tooltip"


def _out(result):
    return result.args if hasattr(result, "args") else result


def test_preprocess_postprocess_roundtrip():
    from diffhdr import color
    from diffhdr.nodes.preprocess import DiffHDRPreprocess
    from diffhdr.nodes.postprocess import DiffHDRPostprocess
    img = torch.rand(3, 32, 32, 3) * 0.8
    img[:, 4:20, 4:20] = 1.0
    control, mask = _out(DiffHDRPreprocess.execute(img, "video", True, False))
    assert control.shape == img.shape and mask.shape == (3, 32, 32) and mask.max() == 1
    (hdr,) = _out(DiffHDRPostprocess.execute(control))
    assert torch.allclose(hdr, color.srgb_to_linear(img), atol=1e-4)


def test_tonemap():
    from diffhdr.nodes.tonemap import DiffHDRTonemap
    (ldr,) = _out(DiffHDRTonemap.execute(torch.rand(1, 8, 8, 3) * 100, 0.0, "reinhard"))
    assert ldr.max() <= 1 and ldr.min() >= 0


def test_save_exr(tmp_path, monkeypatch):
    import folder_paths
    from diffhdr.nodes.save_exr import DiffHDRSaveEXR
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    DiffHDRSaveEXR.execute(torch.rand(3, 16, 16, 3), "DiffHDR/hdr", "exr", "half", "dwab", 100.0, "acescg", 1001, 0.0)
    files = sorted(p.name for p in tmp_path.rglob("*.exr"))
    assert files == ["frame_1001.exr", "frame_1002.exr", "frame_1003.exr"]
    DiffHDRSaveEXR.execute(torch.rand(3, 16, 16, 3), "DiffHDR/hdr", "exr", "half", "zip", 45.0, "linear_rec709", 1, 0.0)
    assert len({p.parent for p in tmp_path.rglob("*.exr")}) == 2   # second sequence got its own folder
    DiffHDRSaveEXR.execute(torch.rand(1, 16, 16, 3), "DiffHDR/hdr", "hdr", "half", "zip", 45.0, "linear_rec709", 1, 0.0)
    assert len(list(tmp_path.rglob("*.hdr"))) == 1
