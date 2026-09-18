import pytest
import torch

pytestmark = pytest.mark.requires_comfy


class FakeVAE:
    """Deterministic stand-in: [F,H,W,3] -> [1,16,T,H/8,W/8]."""

    def encode(self, pixels):
        f, h, w, _ = pixels.shape
        t = (f - 1) // 4 + 1
        x = pixels.movedim(-1, 1).mean(dim=1, keepdim=True)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (h // 8, w // 8))[:: 4][:t]
        return x.permute(1, 0, 2, 3)[None].repeat(1, 16, 1, 1, 1) * torch.arange(1, 17).view(1, 16, 1, 1, 1)


def _inputs(f=9, h=32, w=48):
    g = torch.Generator().manual_seed(3)
    control = torch.rand(f, h, w, 3, generator=g)
    mask = (torch.rand(f, h, w, generator=g) > 0.5).float()
    return control, mask


@pytest.mark.parametrize("with_ref", [False, True])
def test_matches_comfy_node(with_ref):
    from comfy_extras.nodes_wan import WanVaceToVideo
    from diffhdr import vace
    control, mask = _inputs()
    ref = torch.rand(1, 32, 48, 3) if with_ref else None
    vc = vace.build(FakeVAE(), control, mask, ref)
    pos = [[torch.zeros(1, 4, 8), {}]]
    out = WanVaceToVideo.execute(pos, pos, FakeVAE(), 48, 32, 9, 1, 1.0, control_video=control,
                                 control_masks=mask, reference_image=ref)
    cond = out.args[0][0][1] if hasattr(out, "args") else out[0][0][1]
    assert torch.allclose(vc.frames, cond["vace_frames"][0])
    assert torch.allclose(vc.mask, cond["vace_mask"][0])
    assert vc.trim == (1 if with_ref else 0)
    assert vc.latent_shape == (1, 16, 3 + vc.trim, 4, 6)


def test_apply_sets_values():
    from diffhdr import vace
    control, mask = _inputs()
    vc = vace.build(FakeVAE(), control, mask, None)
    cond = vace.apply([[torch.zeros(1, 4, 8), {}]], vc)
    assert cond[0][1]["vace_strength"] == [1.0] and cond[0][1]["vace_frames"][0] is vc.frames
