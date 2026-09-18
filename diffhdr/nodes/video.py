"""DiffHDR all-in-one node for images, videos and long videos."""

import comfy.utils
from comfy_api.latest import io

from .. import backend, embeddings, frames, pipeline, sampling, timing, windows
from .. import vae as dvae
from . import common


class DiffHDRVideo(io.ComfyNode):
    """LDR to HDR reconstruction for a single image or a frame sequence."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffHDRVideo",
            display_name="DiffHDR (Image / Video)",
            category=common.CATEGORY,
            description="Reconstructs HDR from LDR images or videos with DiffHDR. One frame = image mode, up to window_size frames = one window, more = sliding windows with blending.",
            inputs=[
                *common.model_inputs(),
                io.Image.Input("images", tooltip="sRGB LDR image or frame batch."),
                common.clip_input(),
                io.String.Input("prompt", default="", multiline=True, tooltip="Optional prompt (DiffHDR was trained with an empty prompt). Only used when a CLIP is connected."),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask of regions to regenerate (white = regenerate). Overrides automatic detection."),
                io.Image.Input("reference_image", optional=True, tooltip="Optional sRGB reference guiding the content of over-exposed regions."),
                io.Float.Input("reference_ev", default=5.0, min=-10.0, max=10.0, step=0.1, tooltip="Exposure boost in stops applied to the reference image."),
                io.Combo.Input("resize_mode", options=list(frames.RESIZE_MODES), default="crop_to_720p", tooltip="crop_to_720p: centre-crop and resize to 1280x720 (training resolution; 720x1280 for portrait). native: keep size, floored to multiples of 16. custom: use width/height."),
                io.Int.Input("width", default=1280, min=16, max=8192, step=16, tooltip="Width for resize_mode=custom."),
                io.Int.Input("height", default=720, min=16, max=8192, step=16, tooltip="Height for resize_mode=custom."),
                *common.sampler_inputs(default_seed=10),
                io.Boolean.Input("mask_overexposed", default=True, tooltip="Detect and regenerate over-exposed (clipped) regions."),
                io.Boolean.Input("mask_underexposed", default=False, tooltip="Also detect and regenerate under-exposed (crushed) regions."),
                io.Int.Input("window_size", default=33, min=5, max=129, step=4, tooltip="Frames per window (4n+1). 33 is the training length."),
                io.Int.Input("window_stride", default=16, min=1, max=128, tooltip="Frames between window starts for long videos. Must be smaller than window_size."),
                io.Boolean.Input("use_prev_window_reference", default=False, tooltip="Long videos: use the previous window's output as reference for temporal consistency."),
                *common.system_inputs(),
            ],
            outputs=common.hdr_outputs(),
        )

    @classmethod
    def execute(cls, model, vae, images, prompt, reference_ev, resize_mode, width, height, steps, seed,
                mask_overexposed, mask_underexposed, window_size, window_stride, use_prev_window_reference,
                attention, vae_precision, clip=None, mask=None, reference_image=None) -> io.NodeOutput:
        timer = timing.StageTimer()
        with timer.stage("prepare"):
            dvae.check_wan_vae(vae)
            images = images[..., :3]
            h, w = frames.resolve_size(images.shape[1], images.shape[2], resize_mode, height, width)
            images = frames.fit(images, h, w, crop=True)
            if mask is not None:
                mask = frames.fit_mask(mask if mask.dim() == 3 else mask[None], h, w, crop=True)
            if reference_image is not None:
                reference_image = frames.fit(reference_image[:1, ..., :3], h, w, crop=True)

        total = images.shape[0]
        size = frames.snap_4n1(window_size)
        n_windows = 1 if total <= size else len(windows.plan_windows(total, size, window_stride))
        pbar = comfy.utils.ProgressBar(n_windows * steps)

        with timer.stage("conditioning"):
            positive, negative = embeddings.get_conditioning(clip, prompt, "standard")
        with timer.stage("model_patch"):
            patched = sampling.prepare_model(model, "standard", attention)
        window_fn = backend.make_window_fn(patched, dvae.get_vae(vae, vae_precision), positive, negative,
                                           steps, seed, on_step=lambda: pbar.update(1), timer=timer)
        result = pipeline.run_video(images, window_fn, user_mask=mask, mask_overexposed=mask_overexposed,
                                    mask_underexposed=mask_underexposed, reference=reference_image,
                                    reference_ev=reference_ev, window_size=size, window_stride=window_stride,
                                    use_prev_window_reference=use_prev_window_reference, timer=timer)
        timing.log_timing(timing.node_context("image" if total == 1 else "video", total, n_windows, steps), timer)
        return io.NodeOutput(result.hdr, result.mask)
