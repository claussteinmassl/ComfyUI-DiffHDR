import pytest
import torch

from diffhdr import frames


def test_resolve_size():
    assert frames.resolve_size(1080, 1920, "crop_to_720p") == (720, 1280)
    assert frames.resolve_size(1920, 1080, "crop_to_720p") == (1280, 720)
    assert frames.resolve_size(1080, 1920, "native") == (1072, 1920)
    assert frames.resolve_size(500, 500, "custom", 490, 1001) == (496, 1008)
    assert frames.resolve_size(10, 10, "native") == (16, 16)
    with pytest.raises(ValueError):
        frames.resolve_size(10, 10, "bogus")


def test_fit_crop():
    img = torch.rand(2, 100, 300, 3)
    out = frames.fit(img, 64, 96, crop=True)
    assert out.shape == (2, 64, 96, 3) and out.min() >= 0 and out.max() <= 1


def test_fit_stretch_identity():
    img = torch.rand(1, 32, 64, 3)
    assert torch.equal(frames.fit(img, 32, 64, crop=False), img)


def test_fit_mask():
    m = torch.zeros(3, 40, 40)
    m[:, 10:30, 10:30] = 1
    out = frames.fit_mask(m, 32, 48)
    assert out.shape == (3, 32, 48) and set(out.unique().tolist()) <= {0.0, 1.0}


def test_num_frames():
    assert frames.image_mode_num_frames(720, 1280) == 33
    assert frames.image_mode_num_frames(1072, 1920) == 17
    for h, w in [(480, 832), (2048, 2048), (16, 16)]:
        assert (frames.image_mode_num_frames(h, w) - 1) % 4 == 0


def test_pad_frames():
    x = torch.arange(3).float().view(3, 1, 1, 1).expand(3, 2, 2, 3)
    out = frames.pad_frames(x, 5)
    assert out.shape[0] == 5 and torch.equal(out[4], x[2])
    assert frames.pad_frames(x, 3) is x


def test_snap_window():
    assert frames.snap_4n1(33) == 33 and frames.snap_4n1(34) == 37 and frames.snap_4n1(1) == 1
