"""Saves linear HDR images as OpenEXR or Radiance HDR."""

from pathlib import Path

import folder_paths
from comfy_api.latest import io, ui

from .. import color, exr, timing
from . import common


class DiffHDRSaveEXR(io.ComfyNode):
    """EXR / HDR writer with colourspace and compression control."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffHDRSaveEXR",
            display_name="DiffHDR Save EXR",
            category=common.CATEGORY,
            description="Writes linear HDR as EXR (single file or frame sequence) or Radiance .hdr.",
            is_output_node=True,
            inputs=[
                io.Image.Input("images", tooltip="Linear Rec.709 HDR image or frame batch."),
                io.String.Input("filename_prefix", default="DiffHDR/hdr", tooltip="Prefix inside the ComfyUI output folder. Batches are written to <prefix>_<counter>/frame_####.exr."),
                io.Combo.Input("format", options=["exr", "hdr"], default="exr", tooltip="exr: OpenEXR. hdr: Radiance RGBE (always linear Rec.709; bit depth, compression and colorspace are ignored)."),
                io.Combo.Input("bit_depth", options=list(exr.BIT_DEPTHS), default="half", tooltip="half (16-bit float) or float (32-bit)."),
                io.Combo.Input("compression", options=list(exr.COMPRESSIONS), default="zip", tooltip="EXR compression. dwaa/dwab are lossy; zip/piz are lossless."),
                io.Float.Input("dwa_compression_level", default=45.0, min=0.0, max=500.0, step=1.0, tooltip="DWAA/DWAB compression level (dwaCompressionLevel). Higher = smaller and lossier. 45 is the OpenEXR default."),
                io.Combo.Input("colorspace", options=list(color.COLORSPACES), default="linear_rec709", tooltip="Output primaries. Pixels are converted and chromaticities are written to the header."),
                io.Int.Input("start_frame", default=1, min=0, max=9999999, tooltip="Frame number of the first file in a sequence. Numbers are padded to at least four digits, widened for the whole sequence when the last frame needs more."),
                io.Float.Input("preview_exposure", default=0.0, min=-16.0, max=16.0, step=0.25, tooltip="Exposure in stops for the tonemapped UI preview."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, images, filename_prefix, format, bit_depth, compression, dwa_compression_level,
                colorspace, start_frame, preview_exposure) -> io.NodeOutput:
        images = images[..., :3].float()
        count, height, width = images.shape[:3]
        folder, name, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        while count > 1 and (Path(folder) / f"{name}_{counter:05d}").exists():
            counter += 1  # get_save_image_path only counts files, not sequence folders
        paths = exr.sequence_paths(Path(folder), name, counter, count, start_frame, format)
        timer = timing.StageTimer()
        with timer.stage("write"):
            for image, path in zip(images, paths):
                if format == "hdr":
                    exr.write_hdr(path, image)
                else:
                    exr.write_exr(path, image, bit_depth=bit_depth, compression=compression,
                                  dwa_level=dwa_compression_level, colorspace=colorspace)
        with timer.stage("preview"):
            picks = sorted({0, count // 2, count - 1})
            preview = color.tonemap(images[picks], preview_exposure, "reinhard")
            output = io.NodeOutput(ui=ui.PreviewImage(preview, cls=cls))
        flavour = "hdr" if format == "hdr" else f"exr/{bit_depth}/{compression}"
        plural = "" if count == 1 else "s"
        timing.log_timing(f"save_exr, {count} frame{plural}, {flavour}", timer)
        return output
