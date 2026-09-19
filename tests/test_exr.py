import numpy as np
import OpenEXR
import pytest
import torch

from diffhdr import color, exr


def _img(h=64, w=96, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy((rng.random((h, w, 3)) ** 4 * 50).astype(np.float32))


def _read(path):
    # Do not use `with OpenEXR.File(...) as f:` here: closing the file via
    # the context manager's __exit__ clears the header dict in place on this
    # binding version, so it would come back empty by the time this function
    # returns. Reading without the context manager keeps it populated.
    f = OpenEXR.File(str(path))
    return f.header(), f.channels()["RGB"].pixels


def test_float_zip_roundtrip(tmp_path):
    img = _img()
    p = tmp_path / "a.exr"
    exr.write_exr(p, img, bit_depth="float", compression="zip")
    header, px = _read(p)
    assert px.dtype == np.float32 and np.array_equal(px, img.numpy())
    assert header["chromaticities"] == pytest.approx(color.CHROMATICITIES["linear_rec709"], abs=1e-6)


def test_half(tmp_path):
    img = _img()
    p = tmp_path / "h.exr"
    exr.write_exr(p, img, bit_depth="half", compression="piz")
    _, px = _read(p)
    assert px.dtype == np.float16
    assert np.allclose(px.astype(np.float32), img.numpy(), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("comp", exr.COMPRESSIONS)
def test_all_compressions(tmp_path, comp):
    p = tmp_path / f"{comp}.exr"
    exr.write_exr(p, _img(), bit_depth="half", compression=comp)
    _, px = _read(p)
    assert px.shape == (64, 96, 3)


def test_dwa_level(tmp_path):
    img = _img(256, 256)
    lo, hi = tmp_path / "lo.exr", tmp_path / "hi.exr"
    exr.write_exr(lo, img, compression="dwab", dwa_level=45.0)
    exr.write_exr(hi, img, compression="dwab", dwa_level=400.0)
    assert hi.stat().st_size < lo.stat().st_size
    assert _read(hi)[0]["dwaCompressionLevel"] == pytest.approx(400.0)


def test_colorspaces(tmp_path):
    img = _img()
    p = tmp_path / "cg.exr"
    exr.write_exr(p, img, bit_depth="float", compression="zip", colorspace="acescg")
    header, px = _read(p)
    assert header["chromaticities"] == pytest.approx(color.CHROMATICITIES["acescg"], abs=1e-5)
    assert np.allclose(px, color.convert_primaries(img, "acescg").numpy(), atol=1e-5)
    p2 = tmp_path / "ap0.exr"
    exr.write_exr(p2, img, colorspace="aces2065_1")
    assert _read(p2)[0]["acesImageContainerFlag"] == 1


def test_hdr_roundtrip(tmp_path):
    img = torch.from_numpy(np.geomspace(0.01, 1000, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3))
    p = tmp_path / "x.hdr"
    exr.write_hdr(p, img)
    back = exr.read_hdr(p)
    assert np.allclose(back, img.numpy(), rtol=0.01)


def test_sequence_paths(tmp_path):
    paths = exr.sequence_paths(tmp_path, "shot", 7, 3, start_frame=1001, ext="exr")
    assert [p.name for p in paths] == ["frame_1001.exr", "frame_1002.exr", "frame_1003.exr"]
    assert paths[0].parent.name == "shot_00007"
    single = exr.sequence_paths(tmp_path, "shot", 7, 1, start_frame=1, ext="hdr")
    assert single == [tmp_path / "shot_00007.hdr"]


def test_sequence_paths_widen_when_crossing_9999(tmp_path):
    """One padding width for the whole sequence keeps lexical order across 9999."""
    paths = exr.sequence_paths(tmp_path, "shot", 1, 4, start_frame=9998, ext="exr")
    assert [p.name for p in paths] == ["frame_09998.exr", "frame_09999.exr",
                                       "frame_10000.exr", "frame_10001.exr"]
    assert [p.name for p in paths] == sorted(p.name for p in paths)
    wide = exr.sequence_paths(tmp_path, "shot", 1, 2, start_frame=1000000, ext="exr")
    assert [p.name for p in wide] == ["frame_1000000.exr", "frame_1000001.exr"]
