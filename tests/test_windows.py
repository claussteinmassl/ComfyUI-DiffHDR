import pytest
import torch

from diffhdr import windows


def test_plan_basic():
    w = windows.plan_windows(65, 33, 16)
    assert [(x.start, x.end) for x in w] == [(0, 33), (16, 49), (32, 65)]
    assert w[0].first and w[-1].last and not w[1].first and not w[1].last


def test_plan_short_and_tail():
    assert [(x.start, x.end) for x in windows.plan_windows(20, 33, 16)] == [(0, 20)]
    w = windows.plan_windows(34, 33, 16)
    assert [(x.start, x.end) for x in w] == [(0, 33), (16, 34)]


def test_plan_validation():
    with pytest.raises(ValueError):
        windows.plan_windows(10, 33, 33)


def test_ramp():
    r = windows.ramp_weights(33, first=False, last=True, border=17)
    assert r[0] == pytest.approx(0.5 / 17) and r[16] == pytest.approx(16.5 / 17) and r[17:].eq(1).all()
    assert windows.ramp_weights(5, True, True, 17).eq(1).all()


@pytest.mark.parametrize("total,size,stride", [(65, 33, 16), (34, 33, 16), (100, 33, 16), (50, 17, 8), (33, 33, 16)])
def test_streaming_equals_dense(total, size, stride):
    g = torch.Generator().manual_seed(1)
    plan = windows.plan_windows(total, size, stride)
    chunks = [torch.rand(w.end - w.start, 2, 2, 3, generator=g) for w in plan]
    dense_v = torch.zeros(total, 2, 2, 3)
    dense_w = torch.zeros(total)
    for w, c in zip(plan, chunks):
        r = windows.ramp_weights(c.shape[0], w.first, w.last, size - stride)
        dense_v[w.start:w.end] += c * r.view(-1, 1, 1, 1)
        dense_w[w.start:w.end] += r
    dense = dense_v / dense_w.view(-1, 1, 1, 1)

    blender = windows.StreamingBlender(size, stride)
    out = [blender.add(c, w) for w, c in zip(plan, chunks)] + [blender.flush()]
    out = torch.cat([o for o in out if o.shape[0] > 0])
    assert out.shape[0] == total
    assert torch.allclose(out, dense, atol=1e-6)


def test_constant():
    plan = windows.plan_windows(70, 33, 16)
    blender = windows.StreamingBlender(33, 16)
    out = [blender.add(torch.full((w.end - w.start, 1, 1, 3), 2.5), w) for w in plan] + [blender.flush()]
    assert torch.allclose(torch.cat(out), torch.full((70, 1, 1, 3), 2.5))
