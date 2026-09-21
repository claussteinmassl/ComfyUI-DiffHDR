"""Tests for the stage timer and its wiring into pipeline, backend and nodes."""

import logging
from pathlib import Path

import pytest
import torch

from diffhdr import timing


class Clock:
    """Deterministic monotonic clock."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_stage_timer_accumulates_repeated_stages():
    clock = Clock()
    timer = timing.StageTimer(clock=clock)
    with timer.stage("sample"):
        clock.advance(2.0)
    with timer.stage("decode"):
        clock.advance(0.5)
    with timer.stage("sample"):
        clock.advance(3.0)
    assert timer.summary() == {"sample": 5.0, "decode": 0.5}


def test_stage_timer_summary_keeps_first_use_order():
    clock = Clock()
    timer = timing.StageTimer(clock=clock)
    for name in ("masks", "control", "window", "blend"):
        with timer.stage(name):
            clock.advance(1.0)
    with timer.stage("masks"):
        clock.advance(1.0)
    assert list(timer.summary()) == ["masks", "control", "window", "blend"]


def test_stage_timer_records_time_even_when_the_body_raises():
    clock = Clock()
    timer = timing.StageTimer(clock=clock)
    with pytest.raises(RuntimeError):
        with timer.stage("sample"):
            clock.advance(4.0)
            raise RuntimeError("boom")
    assert timer.summary() == {"sample": 4.0}


def test_stage_timer_format():
    clock = Clock()
    timer = timing.StageTimer(clock=clock)
    for name, dt in (("prepare", 0.8), ("masks", 41.2), ("encode", 11.0),
                     ("sample", 123.4), ("decode", 9.7), ("blend", 0.4)):
        with timer.stage(name):
            clock.advance(dt)
    assert timer.format() == ("prepare=0.8s masks=41.2s encode=11.0s sample=123.4s "
                              "decode=9.7s blend=0.4s total=186.5s")


def test_stage_timer_total_is_wall_clock_since_construction():
    clock = Clock()
    timer = timing.StageTimer(clock=clock)
    clock.advance(5.0)          # untimed gap
    with timer.stage("sample"):
        clock.advance(1.0)
    assert timer.elapsed() == 6.0
    assert timer.format() == "sample=1.0s total=6.0s"


def test_stage_helper_is_a_noop_without_a_timer():
    with timing.stage(None, "masks"):
        pass


def test_stage_helper_delegates_to_the_timer():
    clock = Clock()
    timer = timing.StageTimer(clock=clock)
    with timing.stage(timer, "masks"):
        clock.advance(2.0)
    assert timer.summary() == {"masks": 2.0}


def test_no_cuda_synchronisation_in_the_timing_module():
    source = Path(timing.__file__).read_text(encoding="utf-8")
    assert "torch" not in source and "cuda" not in source


@pytest.mark.parametrize("count,expected", [(1, "video, 1 frame, 1 window(s), 10 steps"),
                                            (33, "video, 33 frames, 1 window(s), 10 steps")])
def test_node_context(count, expected):
    assert timing.node_context("video", count, 1, 10) == expected


def test_log_timing_emits_one_info_line_on_the_diffhdr_logger(caplog):
    clock = Clock()
    timer = timing.StageTimer(clock=clock)
    with timer.stage("sample"):
        clock.advance(1.0)
    with caplog.at_level(logging.INFO, logger="DiffHDR"):
        timing.log_timing("video, 33 frames, 1 window(s), 10 steps", timer)
    records = [r for r in caplog.records if r.name == "DiffHDR"]
    assert len(records) == 1 and records[0].levelno == logging.INFO
    assert records[0].getMessage() == ("DiffHDR timing [video, 33 frames, 1 window(s), 10 steps]: "
                                       "sample=1.0s total=1.0s")


# --- pipeline wiring -------------------------------------------------------

def _clip(f, h=16, w=32):
    x = torch.rand(f, h, w, 3, generator=torch.Generator().manual_seed(f)) * 0.8
    x[:, 2:8, 2:12] = 1.0
    return x


def _identity_window_fn(control, mask, reference):
    return control


def test_run_video_fills_the_expected_stage_keys_for_a_long_clip():
    from diffhdr import pipeline

    timer = timing.StageTimer()
    pipeline.run_video(_clip(65), _identity_window_fn, timer=timer)
    assert list(timer.summary()) == ["masks", "control", "window", "blend"]
    assert all(v >= 0.0 for v in timer.summary().values())


def test_run_video_fills_the_expected_stage_keys_in_image_mode(monkeypatch):
    from diffhdr import pipeline

    monkeypatch.setattr(pipeline.frames, "image_mode_num_frames", lambda h, w: 5)
    timer = timing.StageTimer()
    pipeline.run_video(_clip(1), _identity_window_fn, timer=timer)
    assert list(timer.summary()) == ["masks", "control", "window"]


def test_run_pano_fills_the_expected_stage_keys():
    from diffhdr import pipeline

    timer = timing.StageTimer()
    pipeline.run_pano(_clip(1, 32, 64), _identity_window_fn, timer=timer)
    assert list(timer.summary()) == ["masks", "control", "window"]


def test_pipeline_without_a_timer_is_unchanged():
    from diffhdr import pipeline

    clip = _clip(65)
    a = pipeline.run_video(clip, _identity_window_fn)
    b = pipeline.run_video(clip, _identity_window_fn, timer=timing.StageTimer())
    assert torch.equal(a.hdr, b.hdr) and torch.equal(a.mask, b.mask)


# --- backend wiring --------------------------------------------------------

@pytest.mark.requires_comfy
def test_make_window_fn_times_encode_sample_decode(monkeypatch):
    from diffhdr import backend, sampling, vace

    def fake_build(vae, control, mask, reference):
        t = (control.shape[0] - 1) // 4 + 1
        return vace.VaceCond(frames=torch.zeros(1), mask=torch.zeros(1), trim=0,
                             latent_shape=(1, 16, t, 2, 2))

    monkeypatch.setattr(backend.vace, "build", fake_build)
    monkeypatch.setattr(backend.vace, "apply", lambda cond, vc: cond)
    monkeypatch.setattr(backend.sampling, "sample", lambda *a, **k: torch.zeros(a[3]))
    monkeypatch.setattr(backend.dvae, "decode",
                        lambda vae, latent: torch.zeros((latent.shape[2] - 1) * 4 + 1, 16, 16, 3))

    timer = timing.StageTimer()
    fn = backend.make_window_fn(model=None, vae=None, positive=[], negative=[], steps=1, seed=0,
                                settings=sampling.PRESETS["fast"], timer=timer)
    fn(torch.zeros(5, 16, 16, 3), torch.zeros(5, 16, 16), None)
    assert list(timer.summary()) == ["encode", "sample", "decode"]


# --- node wiring -----------------------------------------------------------

def _patch_node_deps(monkeypatch, module):
    """Replaces every model-dependent call of an all-in-one node with a stub."""
    monkeypatch.setattr(module.dvae, "check_wan_vae", lambda vae: None)
    monkeypatch.setattr(module.dvae, "get_vae", lambda vae, precision: vae)
    monkeypatch.setattr(module.embeddings, "get_conditioning", lambda clip, prompt, variant: ([], []))
    monkeypatch.setattr(module.sampling, "prepare_model", lambda model, variant, attention, shift: model)
    monkeypatch.setattr(module.backend, "make_window_fn",
                        lambda *a, **k: (lambda control, mask, reference: control))
    monkeypatch.setattr(module.pipeline.frames, "image_mode_num_frames", lambda h, w: 5)


@pytest.mark.requires_comfy
def test_video_node_logs_one_timing_line(monkeypatch, caplog):
    from diffhdr.nodes import video

    _patch_node_deps(monkeypatch, video)
    with caplog.at_level(logging.INFO, logger="DiffHDR"):
        video.DiffHDRVideo.execute(preset="fast", model=None, vae=None, images=_clip(5), prompt="",
                                   reference_ev=5.0, resize_mode="native", width=32, height=16,
                                   steps=10, seed=0, sampler="res_multistep", scheduler="simple", shift=8.0,
                                   mask_overexposed=True, mask_underexposed=False,
                                   window_size=33, window_stride=16, use_prev_window_reference=False,
                                   attention="sdpa", vae_precision="fp32")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("DiffHDR timing")]
    assert len(lines) == 1
    assert lines[0].startswith("DiffHDR timing [video, 5 frames, 1 window(s), 10 steps]: ")
    for key in ("prepare=", "conditioning=", "model_patch=", "masks=", "control=", "window=", "total="):
        assert key in lines[0]


@pytest.mark.requires_comfy
def test_video_node_reports_image_mode_and_window_count(monkeypatch, caplog):
    from diffhdr.nodes import video

    _patch_node_deps(monkeypatch, video)
    with caplog.at_level(logging.INFO, logger="DiffHDR"):
        video.DiffHDRVideo.execute(preset="fast", model=None, vae=None, images=_clip(1), prompt="",
                                   reference_ev=5.0, resize_mode="native", width=32, height=16,
                                   steps=10, seed=0, sampler="res_multistep", scheduler="simple", shift=8.0,
                                   mask_overexposed=True, mask_underexposed=False,
                                   window_size=33, window_stride=16, use_prev_window_reference=False,
                                   attention="sdpa", vae_precision="fp32")
        video.DiffHDRVideo.execute(preset="fast", model=None, vae=None, images=_clip(65), prompt="",
                                   reference_ev=5.0, resize_mode="native", width=32, height=16,
                                   steps=10, seed=0, sampler="res_multistep", scheduler="simple", shift=8.0,
                                   mask_overexposed=True, mask_underexposed=False,
                                   window_size=33, window_stride=16, use_prev_window_reference=False,
                                   attention="sdpa", vae_precision="fp32")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("DiffHDR timing")]
    assert lines[0].startswith("DiffHDR timing [image, 1 frame, 1 window(s), 10 steps]: ")
    assert lines[1].startswith("DiffHDR timing [video, 65 frames, 3 window(s), 10 steps]: ")
    assert "blend=" in lines[1]


@pytest.mark.requires_comfy
def test_pano_node_logs_one_timing_line(monkeypatch, caplog):
    from diffhdr.nodes import hdri

    _patch_node_deps(monkeypatch, hdri)
    with caplog.at_level(logging.INFO, logger="DiffHDR"):
        hdri.DiffHDRPano.execute(preset="fast", model=None, vae=None, image=_clip(1, 32, 64), prompt="",
                                 width=64, height=32, steps=10, seed=0, sampler="res_multistep",
                                 scheduler="simple", shift=8.0, attention="sdpa", vae_precision="fp32")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("DiffHDR timing")]
    assert len(lines) == 1
    assert lines[0].startswith("DiffHDR timing [pano, 1 frame, 1 window(s), 10 steps]: ")
    for key in ("prepare=", "conditioning=", "model_patch=", "masks=", "control=", "window=", "total="):
        assert key in lines[0]


@pytest.mark.requires_comfy
def test_save_exr_logs_one_timing_line(tmp_path, monkeypatch, caplog):
    import folder_paths

    from diffhdr.nodes.save_exr import DiffHDRSaveEXR

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    with caplog.at_level(logging.INFO, logger="DiffHDR"):
        DiffHDRSaveEXR.execute(torch.rand(3, 16, 16, 3), "DiffHDR/hdr", "exr", "half", "zip",
                               45.0, "linear_rec709", 1, 0.0)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("DiffHDR timing")]
    assert len(lines) == 1
    assert lines[0].startswith("DiffHDR timing [save_exr, 3 frames, exr/half/zip]: ")
    assert "write=" in lines[0] and "preview=" in lines[0] and "total=" in lines[0]
