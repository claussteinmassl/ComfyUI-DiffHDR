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
