import torch

from diffhdr import masks


def _clip(frames=5, h=48, w=64):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(frames, h, w, 3, generator=g) * 0.6 + 0.1
    x[:, 8:28, 10:40, :] = 1.0          # clipped highlight
    x[:, 34:46, 44:60, :] = 0.0         # crushed shadow
    return x


def test_shapes_and_binary():
    out = masks.video_masks(_clip(), use_over=True, use_under=False)
    assert out.combined.shape == (5, 48, 64)
    assert set(out.combined.unique().tolist()) <= {0.0, 1.0}
    assert out.combined[:, 12:24, 14:36].min() == 1.0   # highlight core detected
    assert out.combined[:, 36:44, 46:58].max() == 0.0   # shadows ignored without use_under


def test_under_union_and_paint():
    clip = _clip()
    out = masks.video_masks(clip, use_over=True, use_under=True)
    assert out.combined[:, 38:42, 48:56].min() == 1.0
    painted = masks.paint_underexposed(clip, out.under)
    assert torch.all(painted[:, 38:42, 48:56, :] == 0.5)
    assert torch.equal(painted[:, 0:4, 0:4], clip[:, 0:4, 0:4])


def test_requires_one_flag():
    import pytest
    with pytest.raises(ValueError):
        masks.video_masks(_clip(), use_over=False, use_under=False)


def test_matches_reference(reference_color_utils):
    ref = reference_color_utils
    clip = _clip()
    ours = masks.video_masks(clip, use_over=True, use_under=True)
    prev_o = prev_u = None
    for i in range(clip.shape[0]):
        chw = clip[i].permute(2, 0, 1)
        o, u = ref.exposure_masks_from_srgb(chw, over_luma_thr_srgb=0.95, under_luma_thr_srgb=0.01,
                                            softness=0.02, clip_eps_srgb=0.02, use_local_contrast_gate=False)
        o, prev_o = ref.stabilize_soft_mask(o, prev_o, alpha=0.7, k_smooth=9, k_open_close=9)
        u, prev_u = ref.stabilize_soft_mask(u, prev_u, alpha=0.7, k_smooth=9, k_open_close=9)
        assert torch.equal(ours.over[i], (o > 0.2).float())
        assert torch.equal(ours.under[i], (u > 0.2).float())


def test_pano_mask():
    img = torch.full((32, 64, 3), 0.4)
    img[4:12, 8:24] = 1.0
    m = masks.pano_mask(img)
    assert m.shape == (32, 64)
    assert m[6:10, 10:22].min() == 1.0 and m[20:, :].max() == 0.0
