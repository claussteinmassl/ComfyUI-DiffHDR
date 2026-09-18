import numpy as np
import pytest
import torch

from diffhdr import color


def test_srgb_roundtrip():
    x = torch.linspace(0, 1, 101)
    assert torch.allclose(color.linear_to_srgb(color.srgb_to_linear(x)), x, atol=1e-5)


def test_srgb_known_value():
    assert color.srgb_to_linear(torch.tensor(0.5)).item() == pytest.approx(0.21404114, abs=1e-6)


def test_log_endpoints():
    assert color.lin_to_log(torch.tensor(0.0)).item() == 0.0
    assert color.lin_to_log(torch.tensor(65536.0)).item() == pytest.approx(1.0, abs=1e-6)
    assert color.lin_to_log(torch.tensor(1e9)).item() == pytest.approx(1.0, abs=1e-6)  # clamped


def test_log_roundtrip():
    x = torch.logspace(-4, 3, 200)
    back = color.log_to_lin(color.lin_to_log(x))
    assert torch.allclose(back, x, rtol=1e-4, atol=1e-6)


def test_log_matches_reference(reference_color_utils):
    x = torch.rand(4, 8, 8, 3) * 100
    assert torch.allclose(color.lin_to_log(x), reference_color_utils.Lin_to_Log(x), atol=1e-6)
    y = torch.rand(4, 8, 8, 3)
    assert torch.allclose(color.log_to_lin(y), reference_color_utils.Log_to_Lin(y), rtol=1e-5, atol=1e-6)


def test_luma():
    assert color.luma709(torch.ones(2, 2, 3)).allclose(torch.ones(2, 2))


def test_primaries_matrices():
    acescg = np.array([[0.613097, 0.339523, 0.047379],
                       [0.070194, 0.916354, 0.013452],
                       [0.020616, 0.109570, 0.869815]])
    ap0 = np.array([[0.439633, 0.382989, 0.177378],
                    [0.089776, 0.813439, 0.096784],
                    [0.017541, 0.111547, 0.870912]])
    assert np.allclose(color.primaries_matrix("acescg"), acescg, atol=1e-4)
    assert np.allclose(color.primaries_matrix("aces2065_1"), ap0, atol=1e-4)
    assert np.allclose(color.primaries_matrix("linear_rec709"), np.eye(3))
    for name in color.COLORSPACES:
        assert np.allclose(color.primaries_matrix(name).sum(axis=1), 1.0, atol=1e-5)


def test_convert_primaries_keeps_white_and_shape():
    img = torch.full((2, 4, 4, 3), 3.0)
    out = color.convert_primaries(img, "acescg")
    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-4)


def test_tonemap_range():
    hdr = torch.rand(1, 8, 8, 3) * 500
    for op in ("reinhard", "clip"):
        out = color.tonemap(hdr, exposure=1.0, operator=op)
        assert out.min() >= 0 and out.max() <= 1
