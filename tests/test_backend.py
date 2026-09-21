"""Tests for model preparation, sampling guards and the window function factory."""

import pytest
import torch

pytestmark = pytest.mark.requires_comfy


class _DM:
    dim = 5120
    vace_blocks = [0] * 8


def _fake_patcher(model_options=None):
    import comfy.utils

    class M:
        diffusion_model = _DM()

    class MP:
        def __init__(self, options):
            self.model = M()
            self.model_options = options

        def clone(self):
            return MP(comfy.utils.deepcopy_list_dict(self.model_options))

    return MP(model_options if model_options is not None else {})


def test_check_vace_model_rejects():
    from diffhdr import sampling

    class DM:
        dim = 1536
        vace_blocks = [0] * 8

    class M:
        diffusion_model = DM()

    class MP:
        model = M()

    with pytest.raises(ValueError, match="Wan2.1-VACE-14B"):
        sampling.check_vace_model(MP())
    DM.dim = 5120
    sampling.check_vace_model(MP())


def test_check_vace_model_rejects_without_vace_blocks():
    from diffhdr import sampling

    class DM:
        dim = 5120

    class M:
        diffusion_model = DM()

    class MP:
        model = M()

    with pytest.raises(ValueError, match="Wan2.1-VACE-14B"):
        sampling.check_vace_model(MP())


def test_set_shift_patches_shift_and_keeps_noise_scale():
    import comfy.model_sampling

    from diffhdr import sampling

    class Config:
        sampling_settings = {"shift": 1.0, "noise_scale": 1.5}

    class M:
        model_config = Config()

    class MP:
        def __init__(self):
            self.model = M()
            self.object_patches = {}

        def get_model_object(self, name):
            return self.object_patches.get(name) or comfy.model_sampling.ModelSamplingDiscreteFlow(self.model.model_config)

        def add_object_patch(self, name, obj):
            self.object_patches[name] = obj

    mp = MP()
    sampling.set_shift(mp, sampling.PRESETS["original"].shift)
    ms = mp.object_patches["model_sampling"]
    assert ms.shift == 5.0 and ms.multiplier == 1000 and ms.noise_scale == 1.5
    assert isinstance(ms, comfy.model_sampling.CONST)


def test_prepare_model_does_not_mutate_the_users_model(monkeypatch):
    from diffhdr import sampling

    original = _fake_patcher({"transformer_options": {"existing": 1}})
    monkeypatch.setattr(sampling.lora, "apply_lora", lambda model, variant: model.clone())
    monkeypatch.setattr(sampling, "set_shift", lambda model, shift: model)
    monkeypatch.setattr(sampling.attention, "resolve", lambda mode, device: "OVERRIDE")

    patched = sampling.prepare_model(original, "standard", "sdpa")

    assert patched.model_options["transformer_options"]["optimized_attention_override"] == "OVERRIDE"
    assert patched.model_options["transformer_options"]["existing"] == 1
    assert original.model_options == {"transformer_options": {"existing": 1}}


def test_prepare_model_keeps_default_when_override_is_none(monkeypatch):
    from diffhdr import sampling

    original = _fake_patcher()
    monkeypatch.setattr(sampling.lora, "apply_lora", lambda model, variant: model.clone())
    monkeypatch.setattr(sampling, "set_shift", lambda model, shift: model)
    monkeypatch.setattr(sampling.attention, "resolve", lambda mode, device: None)

    patched = sampling.prepare_model(original, "standard", "auto")

    assert "optimized_attention_override" not in patched.model_options.get("transformer_options", {})


def test_window_fn_trims_reference(monkeypatch):
    from diffhdr import backend, sampling, vace

    seen = {}

    def fake_build(vae, control, mask, reference):
        t = (control.shape[0] - 1) // 4 + 1
        trim = 0 if reference is None else 1
        return vace.VaceCond(frames=torch.zeros(1), mask=torch.zeros(1), trim=trim, latent_shape=(1, 16, t + trim, 2, 2))

    def fake_sample(model, positive, negative, latent_shape, steps, seed, sampler, scheduler, callback=None):
        return torch.arange(latent_shape[2]).float().view(1, 1, -1, 1, 1).expand(latent_shape).clone()

    def fake_decode(vae, latent):
        seen["latent"] = latent
        return torch.zeros((latent.shape[2] - 1) * 4 + 1, 16, 16, 3)

    monkeypatch.setattr(backend.vace, "build", fake_build)
    monkeypatch.setattr(backend.vace, "apply", lambda cond, vc: cond)
    monkeypatch.setattr(backend.sampling, "sample", fake_sample)
    monkeypatch.setattr(backend.dvae, "decode", fake_decode)
    fn = backend.make_window_fn(model=None, vae=None, positive=[], negative=[], steps=2, seed=1,
                                settings=sampling.PRESETS["fast"])
    out = fn(torch.zeros(9, 16, 16, 3), torch.zeros(9, 16, 16), torch.zeros(1, 16, 16, 3))
    assert out.shape == (9, 16, 16, 3)
    assert seen["latent"][0, 0, :, 0, 0].tolist() == [1.0, 2.0, 3.0]   # reference latent (index 0) removed


def test_window_fn_without_reference_keeps_all_latents(monkeypatch):
    from diffhdr import backend, sampling, vace

    seen = {}

    def fake_build(vae, control, mask, reference):
        assert reference is None
        t = (control.shape[0] - 1) // 4 + 1
        return vace.VaceCond(frames=torch.zeros(1), mask=torch.zeros(1), trim=0, latent_shape=(1, 16, t, 2, 2))

    def fake_sample(model, positive, negative, latent_shape, steps, seed, sampler, scheduler, callback=None):
        if callback is not None:
            for i in range(steps):
                callback(i, None, None, steps)
        return torch.arange(latent_shape[2]).float().view(1, 1, -1, 1, 1).expand(latent_shape).clone()

    def fake_decode(vae, latent):
        seen["latent"] = latent
        return torch.zeros((latent.shape[2] - 1) * 4 + 1, 16, 16, 3)

    monkeypatch.setattr(backend.vace, "build", fake_build)
    monkeypatch.setattr(backend.vace, "apply", lambda cond, vc: cond)
    monkeypatch.setattr(backend.sampling, "sample", fake_sample)
    monkeypatch.setattr(backend.dvae, "decode", fake_decode)

    steps_seen = []
    fn = backend.make_window_fn(model=None, vae=None, positive=[], negative=[], steps=3, seed=1,
                                settings=sampling.PRESETS["fast"], on_step=lambda: steps_seen.append(1))
    out = fn(torch.zeros(5, 16, 16, 3), torch.zeros(5, 16, 16), None)
    assert out.shape == (5, 16, 16, 3)
    assert seen["latent"][0, 0, :, 0, 0].tolist() == [0.0, 1.0]
    assert len(steps_seen) == 3


def test_window_fn_is_a_pipeline_window_fn(monkeypatch):
    """The factory result must be usable where the pipeline expects a WindowFn."""
    from diffhdr import backend, sampling, vace

    def fake_build(vae, control, mask, reference):
        t = (control.shape[0] - 1) // 4 + 1
        return vace.VaceCond(frames=torch.zeros(1), mask=torch.zeros(1), trim=0, latent_shape=(1, 16, t, 2, 2))

    monkeypatch.setattr(backend.vace, "build", fake_build)
    monkeypatch.setattr(backend.vace, "apply", lambda cond, vc: cond)
    monkeypatch.setattr(backend.sampling, "sample",
                        lambda *a, **k: torch.zeros(a[3] if len(a) > 3 else k["latent_shape"]))
    monkeypatch.setattr(backend.dvae, "decode",
                        lambda vae, latent: torch.zeros((latent.shape[2] - 1) * 4 + 1, 16, 16, 3))

    from diffhdr import pipeline

    fn = backend.make_window_fn(model=None, vae=None, positive=[], negative=[], steps=1, seed=0,
                                settings=sampling.PRESETS["fast"])
    result = pipeline.run_video(torch.zeros(5, 16, 16, 3), fn, window_size=33, window_stride=16)
    assert result.hdr.shape == (5, 16, 16, 3)


def test_window_fn_forwards_the_sampling_settings(monkeypatch):
    """The window function must sample with the settings it was built with."""
    from diffhdr import backend, sampling, vace

    seen = {}

    def fake_build(vae, control, mask, reference):
        return vace.VaceCond(frames=torch.zeros(1), mask=torch.zeros(1), trim=0, latent_shape=(1, 16, 2, 2, 2))

    def fake_sample(model, positive, negative, latent_shape, steps, seed, sampler, scheduler, callback=None):
        seen.update(sampler=sampler, scheduler=scheduler, steps=steps, seed=seed)
        return torch.zeros(latent_shape)

    monkeypatch.setattr(backend.vace, "build", fake_build)
    monkeypatch.setattr(backend.vace, "apply", lambda cond, vc: cond)
    monkeypatch.setattr(backend.sampling, "sample", fake_sample)
    monkeypatch.setattr(backend.dvae, "decode", lambda vae, latent: torch.zeros(5, 16, 16, 3))

    settings = sampling.SamplingSettings("dpmpp_2m", "sgm_uniform", 7.5)
    fn = backend.make_window_fn(model=None, vae=None, positive=[], negative=[], steps=4, seed=3,
                                settings=settings)
    fn(torch.zeros(5, 16, 16, 3), torch.zeros(5, 16, 16), None)

    assert seen == {"sampler": "dpmpp_2m", "scheduler": "sgm_uniform", "steps": 4, "seed": 3}
