"""Tests for the sampling presets and the parameterised sampler call."""

import pytest

pytestmark = pytest.mark.requires_comfy


def test_presets_hold_the_measured_values():
    from diffhdr import sampling

    assert sampling.PRESETS["fast"] == sampling.SamplingSettings("res_multistep", "simple", 8.0)
    assert sampling.PRESETS["original"] == sampling.SamplingSettings("euler", "simple", 5.0)
    assert sampling.DEFAULT_PRESET == "fast"
    assert sampling.PRESET_NAMES == ["fast", "original", "custom"]


def test_sampling_settings_is_frozen():
    from diffhdr import sampling

    settings = sampling.PRESETS["fast"]
    with pytest.raises(Exception):
        settings.sampler = "euler"


def test_resolve_settings_ignores_the_widgets_for_a_named_preset():
    from diffhdr import sampling

    assert sampling.resolve_settings("fast", "euler", "normal", 2.0) == sampling.PRESETS["fast"]
    assert sampling.resolve_settings("original", "dpmpp_2m", "sgm_uniform", 12.0) == sampling.PRESETS["original"]


def test_resolve_settings_uses_the_widgets_for_custom():
    from diffhdr import sampling

    assert sampling.resolve_settings("custom", "dpmpp_2m", "sgm_uniform", 6.5) == \
        sampling.SamplingSettings("dpmpp_2m", "sgm_uniform", 6.5)


def test_resolve_settings_rejects_an_unknown_preset():
    from diffhdr import sampling

    with pytest.raises(ValueError, match="preset"):
        sampling.resolve_settings("turbo", "euler", "simple", 5.0)


def test_curated_options_exclude_the_harmful_settings():
    """``beta`` and ``deis`` were measured to be worse than doing nothing."""
    from diffhdr import sampling

    assert sampling.SAMPLERS == ["res_multistep", "dpmpp_2m", "euler"]
    assert sampling.SCHEDULERS == ["simple", "normal", "sgm_uniform"]
    for preset in sampling.PRESETS.values():
        assert preset.sampler in sampling.SAMPLERS
        assert preset.scheduler in sampling.SCHEDULERS


def test_sample_forwards_the_requested_sampler_and_scheduler(monkeypatch):
    import comfy.sample
    import torch

    from diffhdr import sampling

    seen = {}

    def fake_sample(model, noise, steps, cfg, sampler, scheduler, positive, negative, latent, **kwargs):
        seen.update(steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, seed=kwargs.get("seed"))
        return torch.zeros_like(latent)

    monkeypatch.setattr(comfy.sample, "prepare_noise", lambda latent, seed: torch.zeros_like(latent))
    monkeypatch.setattr(comfy.sample, "sample", fake_sample)

    out = sampling.sample(None, [], [], (1, 16, 3, 2, 2), steps=7, seed=11,
                          sampler="dpmpp_2m", scheduler="sgm_uniform")

    assert out.shape == (1, 16, 3, 2, 2)
    assert seen == {"steps": 7, "cfg": sampling.CFG, "sampler": "dpmpp_2m",
                    "scheduler": "sgm_uniform", "seed": 11}


def test_set_shift_applies_the_requested_shift():
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
    sampling.set_shift(mp, 8.0)
    assert mp.object_patches["model_sampling"].shift == 8.0


def test_prepare_model_applies_the_requested_shift(monkeypatch):
    from diffhdr import sampling

    class DM:
        dim = 5120
        vace_blocks = [0] * 8

    class M:
        diffusion_model = DM()

    class MP:
        model = M()
        model_options = {}

        def clone(self):
            return self

    seen = {}
    monkeypatch.setattr(sampling.lora, "apply_lora", lambda model, variant: model.clone())
    monkeypatch.setattr(sampling, "set_shift", lambda model, shift: seen.setdefault("shift", shift))
    monkeypatch.setattr(sampling.attention, "resolve", lambda mode, device: None)

    sampling.prepare_model(MP(), "standard", "sdpa", shift=8.0)
    assert seen["shift"] == 8.0
