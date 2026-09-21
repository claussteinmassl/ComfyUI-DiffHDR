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


ALL_IN_ONE = ("DiffHDRVideo", "DiffHDRPano")


def _schemas():
    return {n.define_schema().node_id: n.define_schema() for n in _nodes()}


def _inputs(schema):
    return {inp.id: inp for inp in schema.inputs}


@pytest.mark.parametrize("node_id", ALL_IN_ONE)
def test_preset_is_the_first_input(node_id):
    schema = _schemas()[node_id]
    assert schema.inputs[0].id == "preset"
    assert schema.inputs[1].id == "model"


@pytest.mark.parametrize("node_id", ALL_IN_ONE)
def test_sampling_widgets_sit_right_after_the_seed(node_id):
    ids = [inp.id for inp in _schemas()[node_id].inputs]
    seed = ids.index("seed")
    assert ids[seed + 1:seed + 4] == ["sampler", "scheduler", "shift"]


@pytest.mark.parametrize("node_id", ALL_IN_ONE)
def test_tuned_defaults(node_id):
    from diffhdr import sampling
    inputs = _inputs(_schemas()[node_id])
    assert inputs["preset"].default == "fast"
    assert inputs["steps"].default == 20
    assert inputs["sampler"].default == sampling.PRESETS["fast"].sampler
    assert inputs["scheduler"].default == sampling.PRESETS["fast"].scheduler
    assert inputs["shift"].default == sampling.PRESETS["fast"].shift
    assert (inputs["shift"].min, inputs["shift"].max, inputs["shift"].step) == (1.0, 12.0, 0.5)


@pytest.mark.parametrize("node_id", ALL_IN_ONE)
def test_curated_combo_options(node_id):
    from diffhdr import sampling
    inputs = _inputs(_schemas()[node_id])
    assert inputs["preset"].options == ["fast", "original", "custom"]
    assert inputs["sampler"].options == sampling.SAMPLERS
    assert inputs["scheduler"].options == sampling.SCHEDULERS
    assert "beta" not in inputs["scheduler"].options and "deis" not in inputs["sampler"].options


@pytest.mark.parametrize("node_id", ALL_IN_ONE)
def test_new_inputs_have_tooltips(node_id):
    inputs = _inputs(_schemas()[node_id])
    for name in ("preset", "sampler", "scheduler", "shift", "steps"):
        assert inputs[name].tooltip, f"{node_id}.{name} has no tooltip"
    for name in ("sampler", "scheduler", "shift"):
        assert "custom" in inputs[name].tooltip


def _capture(monkeypatch):
    """Runs an all-in-one node with stubbed sampling and returns what it resolved."""
    from diffhdr import backend, pipeline, sampling
    seen = {}

    def fake_prepare(model, variant, attention_mode="auto", shift=None, apply_diffhdr_lora=True):
        seen["shift"] = shift
        return model

    def fake_window_fn(model, vae, positive, negative, steps, seed, settings, on_step=None, timer=None):
        seen["settings"] = settings
        seen["steps"] = steps
        return lambda control, mask, reference: torch.zeros_like(control)

    monkeypatch.setattr(sampling, "prepare_model", fake_prepare)
    monkeypatch.setattr(backend, "make_window_fn", fake_window_fn)
    monkeypatch.setattr(pipeline, "run_video", lambda *a, **k: pipeline.Result(torch.zeros(1, 16, 16, 3), torch.zeros(1, 16, 16)))
    monkeypatch.setattr(pipeline, "run_pano", lambda *a, **k: pipeline.Result(torch.zeros(1, 16, 16, 3), torch.zeros(1, 16, 16)))
    return seen


def _run_video(preset, sampler, scheduler, shift):
    from diffhdr import vae as dvae
    from diffhdr.nodes.video import DiffHDRVideo
    return DiffHDRVideo.execute(
        preset=preset, model=object(), vae=object(), images=torch.zeros(1, 16, 16, 3), prompt="",
        reference_ev=5.0, resize_mode="native", width=16, height=16, steps=20, seed=10,
        sampler=sampler, scheduler=scheduler, shift=shift,
        mask_overexposed=True, mask_underexposed=False, window_size=33, window_stride=16,
        use_prev_window_reference=False, attention="sdpa", vae_precision="fp32") and dvae


@pytest.mark.parametrize("preset,expected", [("fast", ("res_multistep", "simple", 8.0)),
                                             ("original", ("euler", "simple", 5.0)),
                                             ("custom", ("dpmpp_2m", "sgm_uniform", 6.5))])
def test_video_node_resolves_the_preset(monkeypatch, preset, expected):
    from diffhdr import embeddings, sampling
    from diffhdr import vae as dvae
    seen = _capture(monkeypatch)
    monkeypatch.setattr(dvae, "check_wan_vae", lambda vae: None)
    monkeypatch.setattr(dvae, "get_vae", lambda vae, precision: vae)
    monkeypatch.setattr(embeddings, "get_conditioning", lambda clip, prompt, variant: ([], []))

    _run_video(preset, "dpmpp_2m", "sgm_uniform", 6.5)

    assert seen["settings"] == sampling.SamplingSettings(*expected)
    assert seen["shift"] == expected[2]


def test_pano_node_resolves_the_preset(monkeypatch):
    from diffhdr import embeddings, sampling
    from diffhdr import vae as dvae
    from diffhdr.nodes.hdri import DiffHDRPano
    seen = _capture(monkeypatch)
    monkeypatch.setattr(dvae, "check_wan_vae", lambda vae: None)
    monkeypatch.setattr(dvae, "get_vae", lambda vae, precision: vae)
    monkeypatch.setattr(embeddings, "get_conditioning", lambda clip, prompt, variant: ([], []))

    DiffHDRPano.execute(preset="custom", model=object(), vae=object(), image=torch.zeros(1, 32, 64, 3),
                        prompt="", width=64, height=32, steps=20, seed=42,
                        sampler="euler", scheduler="normal", shift=3.0,
                        attention="sdpa", vae_precision="fp32")

    assert seen["settings"] == sampling.SamplingSettings("euler", "normal", 3.0)
    assert seen["shift"] == 3.0
