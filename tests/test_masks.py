import pytest
import torch
import torch.nn.functional as F

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


def test_unused_mask_is_skipped_without_changing_the_used_one():
    """The over mask must be bit-identical whether or not the under mask is asked for."""
    clip = _clip()
    both = masks.video_masks(clip, use_over=True, use_under=True)
    over_only = masks.video_masks(clip, use_over=True, use_under=False)
    under_only = masks.video_masks(clip, use_over=False, use_under=True)
    assert torch.equal(over_only.over, both.over)
    assert torch.equal(over_only.combined, both.over)
    assert torch.equal(under_only.under, both.under)
    assert float(over_only.under.abs().sum()) == 0.0    # not computed
    assert float(under_only.over.abs().sum()) == 0.0


@pytest.mark.parametrize("k", [3, 9])
def test_separable_max_pool_is_bit_identical_to_the_dense_one(k):
    x = torch.rand(4, 1, 37, 53, generator=torch.Generator().manual_seed(k))
    assert torch.equal(masks._max_pool(x, k), F.max_pool2d(x, k, 1, k // 2))


def test_pooling_calls_do_not_grow_with_the_frame_count(monkeypatch):
    """Performance guard: the mask stages are batched, so pooling is O(1) in frames."""
    calls = {"max": 0, "avg": 0}
    real_max, real_avg = F.max_pool2d, F.avg_pool2d
    monkeypatch.setattr(masks.F, "max_pool2d",
                        lambda *a, **k: (calls.__setitem__("max", calls["max"] + 1), real_max(*a, **k))[1])
    monkeypatch.setattr(masks.F, "avg_pool2d",
                        lambda *a, **k: (calls.__setitem__("avg", calls["avg"] + 1), real_avg(*a, **k))[1])
    masks.video_masks(_clip(frames=1), use_over=True, use_under=False)
    one = dict(calls)
    calls.update(max=0, avg=0)
    masks.video_masks(_clip(frames=16), use_over=True, use_under=False)
    assert calls == one, f"pooling calls grew with the frame count: {one} -> {calls}"


def test_per_frame_helpers_still_match_the_batched_path():
    """The documented single-frame API must agree with the batched implementation."""
    clip = _clip(frames=3)
    batched = masks.video_masks(clip, use_over=True, use_under=True)
    prev_o = prev_u = None
    for i in range(clip.shape[0]):
        o, u = masks.soft_exposure_masks(clip[i])
        o, prev_o = masks.stabilize(o, prev_o)
        u, prev_u = masks.stabilize(u, prev_u)
        assert torch.equal(batched.over[i], (o > masks.BIN_THR).float())
        assert torch.equal(batched.under[i], (u > masks.BIN_THR).float())


def test_pano_mask():
    img = torch.full((32, 64, 3), 0.4)
    img[4:12, 8:24] = 1.0
    m = masks.pano_mask(img)
    assert m.shape == (32, 64)
    assert m[6:10, 10:22].min() == 1.0 and m[20:, :].max() == 0.0


def test_video_masks_does_not_touch_its_input():
    clip = _clip()
    before = clip.clone()
    masks.video_masks(clip, use_over=True, use_under=True)
    assert torch.equal(clip, before)
    masks.video_masks(clip, use_over=True, use_under=False)
    assert torch.equal(clip, before)


def test_stabilize_sequence_reuses_the_soft_mask_buffer():
    """Memory guard: EMA, morphology and binarisation all run in the input buffer."""
    soft = torch.rand(6, 24, 32, generator=torch.Generator().manual_seed(3))
    out = masks._stabilize_sequence(soft)
    assert out is soft
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_unrequested_mask_costs_no_memory():
    """Memory guard: the mask that was not asked for is a zero-stride view of one element."""
    out = masks.video_masks(_clip(), use_over=True, use_under=False)
    assert out.under.shape == (5, 48, 64) and float(out.under.abs().sum()) == 0.0
    assert out.under.numel() * out.under.element_size() > out.under.untyped_storage().nbytes()
    assert out.combined is out.over          # no separate copy when only one mask is used


def _patches(value_a, value_b, frames=3, h=48, w=64):
    """Mid-grey clip with two uniform 20x24 patches at sRGB ``value_a`` and ``value_b``."""
    x = torch.full((frames, h, w, 3), 0.4)
    x[:, 4:24, 4:28] = value_a
    x[:, 24:44, 36:60] = value_b
    return x


def test_default_thresholds_are_the_reference_values():
    clip = _clip()
    default = masks.video_masks(clip, use_over=True, use_under=True)
    explicit = masks.video_masks(clip, use_over=True, use_under=True,
                                 over_threshold=masks.OVER_THR, under_threshold=masks.UNDER_THR)
    assert torch.equal(default.over, explicit.over) and torch.equal(default.under, explicit.under)
    assert masks.pano_mask(clip[0]).equal(masks.pano_mask(clip[0], over_thr=masks.OVER_THR))


def test_lower_over_threshold_catches_nearly_clipped_highlights():
    clip = _patches(0.85, 1.0)
    default = masks.video_masks(clip, use_over=True, use_under=False)
    lowered = masks.video_masks(clip, use_over=True, use_under=False, over_threshold=0.8)
    assert default.over[:, 8:20, 8:24].max() == 0.0          # 0.85 is not clipped by default
    assert lowered.over[:, 8:20, 8:24].min() == 1.0
    assert torch.all(lowered.over >= default.over)            # a lower threshold only adds


def test_over_threshold_moves_the_channel_clip_term_too():
    """At 1.0 only truly clipped pixels count; the fixed 0.98 channel test must follow."""
    clip = _patches(0.985, 1.0)
    default = masks.video_masks(clip, use_over=True, use_under=False)
    strict = masks.video_masks(clip, use_over=True, use_under=False, over_threshold=1.0)
    assert default.over[:, 8:20, 8:24].min() == 1.0
    assert strict.over[:, 8:20, 8:24].max() == 0.0
    assert strict.over[:, 28:40, 40:56].min() == 1.0          # pure white stays masked


def test_higher_under_threshold_catches_nearly_crushed_shadows():
    clip = _patches(0.04, 0.0)
    default = masks.video_masks(clip, use_over=False, use_under=True)
    raised = masks.video_masks(clip, use_over=False, use_under=True, under_threshold=0.06)
    assert default.under[:, 8:20, 8:24].max() == 0.0
    assert raised.under[:, 8:20, 8:24].min() == 1.0
    assert torch.all(raised.under >= default.under)


def test_only_truly_crushed_pixels_are_painted():
    """The paint mask stays at the reference threshold so shadow detail reaches the model."""
    clip = _patches(0.04, 0.0)
    default = masks.video_masks(clip, use_over=False, use_under=True)
    raised = masks.video_masks(clip, use_over=False, use_under=True, under_threshold=0.06)
    assert default.paint is default.under
    assert torch.equal(raised.paint, default.under)
    lowered = masks.video_masks(clip, use_over=False, use_under=True, under_threshold=0.0)
    assert torch.all(lowered.paint <= lowered.under)           # never paint what is kept
    painted = masks.paint_underexposed(clip, raised.paint)
    assert torch.equal(painted[:, 8:20, 8:24], clip[:, 8:20, 8:24])
    assert torch.all(painted[:, 28:40, 40:56] == 0.5)


def test_paint_is_empty_without_under_detection():
    out = masks.video_masks(_clip(), use_over=True, use_under=False)
    assert float(out.paint.abs().sum()) == 0.0


def test_pano_threshold():
    img = torch.full((32, 64, 3), 0.4)
    img[4:12, 8:24] = 0.85
    img[20:28, 40:56] = 0.985
    assert masks.pano_mask(img)[6:10, 10:22].max() == 0.0
    assert masks.pano_mask(img, over_thr=0.8)[6:10, 10:22].min() == 1.0
    assert masks.pano_mask(img)[22:26, 42:54].min() == 1.0
    assert masks.pano_mask(img, over_thr=1.0)[22:26, 42:54].max() == 0.0
