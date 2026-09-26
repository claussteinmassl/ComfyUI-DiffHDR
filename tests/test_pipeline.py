import pytest
import torch

from diffhdr import color, pipeline


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, control, mask, reference):
        self.calls.append((control.shape[0], None if reference is None else reference.clone()))
        return control


def _clip(f, h=16, w=32):
    x = torch.rand(f, h, w, 3, generator=torch.Generator().manual_seed(f)) * 0.8
    x[:, 2:8, 2:12] = 1.0
    return x


@pytest.mark.parametrize("f", [20, 33, 65])
def test_video_identity(f):
    rec = Recorder()
    clip = _clip(f)
    out = pipeline.run_video(clip, rec, mask_overexposed=True)
    assert out.hdr.shape == clip.shape and out.mask.shape == clip.shape[:3]
    assert torch.allclose(out.hdr, color.srgb_to_linear(clip), atol=1e-4)
    assert all(n == 33 for n, _ in rec.calls)
    assert len(rec.calls) == (1 if f <= 33 else 3)


def test_image_mode(monkeypatch):
    rec = Recorder()
    clip = _clip(1)
    monkeypatch.setattr(pipeline.frames, "image_mode_num_frames", lambda h, w: 5)
    out = pipeline.run_video(clip, rec)
    assert out.hdr.shape == clip.shape
    assert rec.calls[0][0] == 5
    assert torch.allclose(out.hdr, color.srgb_to_linear(clip), atol=1e-4)


def test_prev_window_reference():
    rec = Recorder()
    clip = _clip(65)
    user_ref = torch.rand(1, 16, 32, 3)
    pipeline.run_video(clip, rec, reference=user_ref, reference_ev=2.0, use_prev_window_reference=True)
    assert rec.calls[0][1] is not None
    expected = color.lin_to_log(color.srgb_to_linear(clip[16:17]))
    assert torch.allclose(rec.calls[1][1], expected, atol=1e-5)


def test_user_mask_and_under_paint():
    rec = Recorder()
    clip = _clip(5)
    # Region sized to survive masks.stabilize's 9x9 blur + open/close (see task-6-report.md deviation note).
    clip[:, 6:16, 16:32] = 0.0
    user = torch.zeros(5, 16, 32)
    user[:, :4, :4] = 1
    out = pipeline.run_video(clip, rec, user_mask=user)
    assert torch.equal(out.mask, user)
    out2 = pipeline.run_video(clip, Recorder(), mask_overexposed=True, mask_underexposed=True)
    assert out2.mask[:, 9:13, 21:27].min() == 1.0


def test_prepare_reference():
    ref = torch.full((1, 4, 4, 3), 0.5)
    mask0 = torch.zeros(4, 4)
    mask0[:2] = 1
    out = pipeline.prepare_reference(ref, mask0, ev=1.0)
    inside = color.lin_to_log(color.srgb_to_linear(torch.tensor(0.5)) * 2.0)
    outside = color.lin_to_log(torch.tensor(2.0))
    assert out[0, 0, 0, 0].item() == pytest.approx(inside.item(), abs=1e-6)
    assert out[0, 3, 0, 0].item() == pytest.approx(outside.item(), abs=1e-6)


def test_pano():
    rec = Recorder()
    img = _clip(1, 32, 64)
    out = pipeline.run_pano(img, rec)
    assert rec.calls[0][0] == 1 and out.hdr.shape == img.shape and out.mask.shape == (1, 32, 64)


@pytest.mark.parametrize("kwargs", [
    {"mask_overexposed": True},
    {"mask_overexposed": True, "mask_underexposed": True},
    {"user_mask": torch.ones(40, 16, 32)},
])
def test_run_video_does_not_touch_its_inputs(kwargs):
    """ComfyUI caches node outputs, so the input IMAGE/MASK must come back unchanged."""
    clip = _clip(40)
    clip[:, 10:14, 20:30] = 0.0
    reference = torch.rand(1, 16, 32, 3, generator=torch.Generator().manual_seed(1))
    before, ref_before = clip.clone(), reference.clone()
    user_mask = kwargs.get("user_mask")
    mask_before = None if user_mask is None else user_mask.clone()
    pipeline.run_video(clip, Recorder(), reference=reference, **kwargs)
    assert torch.equal(clip, before)
    assert torch.equal(reference, ref_before)
    if user_mask is not None:
        assert torch.equal(user_mask, mask_before)


def test_run_video_survives_out_of_range_inputs():
    """Values outside [0,1] must still be clamped exactly as before."""
    clip = _clip(8)
    wild = clip * 1.5 - 0.25
    out = pipeline.run_video(wild, Recorder(), mask_overexposed=True)
    expected = pipeline.run_video(wild.clamp(0, 1), Recorder(), mask_overexposed=True)
    assert torch.equal(out.hdr, expected.hdr) and torch.equal(out.mask, expected.mask)


def test_run_pano_does_not_touch_its_input():
    img = _clip(1, 32, 64)
    before = img.clone()
    pipeline.run_pano(img, Recorder())
    assert torch.equal(img, before)


def test_run_video_passes_the_thresholds_on():
    """A raised under threshold regenerates near-black shadows without painting them grey."""
    seen = {}

    def window_fn(control, mask, reference):
        seen["control"], seen["mask"] = control.clone(), mask.clone()
        return control

    clip = torch.full((5, 48, 64, 3), 0.4)
    clip[:, 4:24, 4:28] = 0.04          # dark, not crushed
    clip[:, 24:44, 36:60] = 0.85        # bright, not clipped
    pipeline.run_video(clip, window_fn, mask_overexposed=True, mask_underexposed=True,
                       over_threshold=0.8, under_threshold=0.06)
    assert seen["mask"][:, 8:20, 8:24].min() == 1.0
    assert seen["mask"][:, 28:40, 40:56].min() == 1.0
    expected = pipeline.encode_control(clip[:1, 8:20, 8:24])
    assert torch.equal(seen["control"][:1, 8:20, 8:24], expected)     # detail not painted


def test_run_pano_passes_the_threshold_on():
    img = torch.full((1, 32, 64, 3), 0.4)
    img[:, 4:12, 8:24] = 0.85
    assert pipeline.run_pano(img, Recorder()).mask[0, 6:10, 10:22].max() == 0.0
    assert pipeline.run_pano(img, Recorder(), over_threshold=0.8).mask[0, 6:10, 10:22].min() == 1.0
