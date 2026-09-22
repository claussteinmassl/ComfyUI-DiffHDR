"""OpenEXR and Radiance HDR writers."""

from pathlib import Path

import numpy as np
import torch

from .color import CHROMATICITIES, convert_primaries

COMPRESSIONS = ("none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab")
BIT_DEPTHS = ("half", "float")


def _openexr():
    try:
        import OpenEXR
    except ImportError as e:  # pragma: no cover
        raise ImportError("The OpenEXR package is required: pip install 'OpenEXR>=3.3'") from e
    return OpenEXR


def write_exr(path, image: torch.Tensor, bit_depth: str = "half", compression: str = "zip",
              dwa_level: float = 45.0, colorspace: str = "linear_rec709") -> None:
    """Writes a linear Rec.709 image as a scanline RGB EXR.

    Args:
        path: Destination file.
        image: ``[H,W,3]`` linear Rec.709 tensor.
        bit_depth: ``half`` or ``float``.
        compression: One of :data:`COMPRESSIONS`.
        dwa_level: ``dwaCompressionLevel`` header value, used for ``dwaa``/``dwab``.
        colorspace: Output primaries; pixels are converted and chromaticities written.

    Raises:
        ValueError: On unknown ``bit_depth`` / ``compression`` / ``colorspace``.
    """
    if bit_depth not in BIT_DEPTHS:
        raise ValueError(f"Unknown bit_depth: {bit_depth}")
    if compression not in COMPRESSIONS:
        raise ValueError(f"Unknown compression: {compression}")
    if colorspace not in CHROMATICITIES:
        raise ValueError(f"Unknown colorspace: {colorspace}")
    OpenEXR = _openexr()
    comp = {
        "none": OpenEXR.NO_COMPRESSION, "rle": OpenEXR.RLE_COMPRESSION, "zips": OpenEXR.ZIPS_COMPRESSION,
        "zip": OpenEXR.ZIP_COMPRESSION, "piz": OpenEXR.PIZ_COMPRESSION, "pxr24": OpenEXR.PXR24_COMPRESSION,
        "b44": OpenEXR.B44_COMPRESSION, "b44a": OpenEXR.B44A_COMPRESSION,
        "dwaa": OpenEXR.DWAA_COMPRESSION, "dwab": OpenEXR.DWAB_COMPRESSION,
    }[compression]
    pixels = convert_primaries(image.detach().cpu().float(), colorspace).numpy()
    if bit_depth == "half":
        pixels = np.clip(pixels, -65504.0, 65504.0).astype(np.float16)
    header = {
        "compression": comp,
        "type": OpenEXR.scanlineimage,
        "chromaticities": tuple(float(v) for v in CHROMATICITIES[colorspace]),
    }
    if compression in ("dwaa", "dwab"):
        header["dwaCompressionLevel"] = float(dwa_level)
    if colorspace == "aces2065_1":
        header["acesImageContainerFlag"] = 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OpenEXR.File(header, {"RGB": np.ascontiguousarray(pixels)}).write(str(path))


def write_hdr(path, image: torch.Tensor) -> None:
    """Writes a linear Rec.709 image as an uncompressed Radiance RGBE ``.hdr`` file."""
    rgb = np.clip(image.detach().cpu().float().numpy(), 0.0, None)
    h, w, _ = rgb.shape
    brightest = rgb.max(axis=-1)
    mantissa, exponent = np.frexp(brightest)
    scale = np.where(brightest < 1e-32, 0.0, mantissa * 256.0 / np.maximum(brightest, 1e-32))
    rgbe = np.zeros((h, w, 4), dtype=np.uint8)
    rgbe[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(brightest < 1e-32, 0, exponent + 128).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {h} +X {w}\n".encode("ascii"))
        f.write(rgbe.tobytes())


def sequence_paths(folder, prefix: str, counter: int, count: int, start_frame: int, ext: str) -> list[Path]:
    """Output paths: one file ``<prefix>_<counter>.<ext>`` or a folder of zero-padded frames.

    Frame numbers use a single padding width for the whole sequence -- at least four
    digits, more when the last frame needs them -- so the files stay in lexical order
    even when the sequence crosses 9999.
    """
    folder = Path(folder)
    stem = f"{prefix}_{counter:05d}"
    if count == 1:
        return [folder / f"{stem}.{ext}"]
    pad = max(4, len(str(start_frame + count - 1)))
    return [folder / stem / f"frame_{start_frame + i:0{pad}d}.{ext}" for i in range(count)]
