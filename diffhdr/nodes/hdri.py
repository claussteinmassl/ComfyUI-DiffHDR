"""DiffHDR all-in-one node for equirectangular HDRI panoramas."""

import comfy.utils
from comfy_api.latest import io

from .. import backend, embeddings, frames, pipeline, sampling, timing
from .. import vae as dvae
from . import common


class DiffHDRPano(io.ComfyNode):
    """LDR panorama to HDRI reconstruction."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffHDRPano",
            display_name="DiffHDR HDRI (Panorama)",
            category=common.CATEGORY,
            description="Reconstructs an HDR environment map from a clipped LDR equirectangular panorama using the DiffHDR panorama LoRA.",
            inputs=[
                *common.model_inputs(),
                io.Image.Input("image", tooltip="sRGB equirectangular panorama (2:1). Only the first image of a batch is used."),
                common.clip_input(),
                io.String.Input("prompt", default=embeddings.PANO_PROMPT, multiline=True, tooltip="Prompt used in training. Only used when a CLIP is connected."),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask of regions to regenerate (white = regenerate). Overrides automatic detection."),
                io.Int.Input("width", default=2048, min=16, max=8192, step=16, tooltip="Processing width. The panorama is stretched, not cropped."),
                io.Int.Input("height", default=1024, min=16, max=8192, step=16, tooltip="Processing height."),
                *common.sampler_inputs(default_seed=42),
                *common.system_inputs(),
            ],
            outputs=common.hdr_outputs(),
        )

    @classmethod
    def execute(cls, model, vae, image, prompt, width, height, steps, seed, attention, vae_precision,
                clip=None, mask=None) -> io.NodeOutput:
        timer = timing.StageTimer()
        with timer.stage("prepare"):
            dvae.check_wan_vae(vae)
            h, w = frames.resolve_size(image.shape[1], image.shape[2], "custom", height, width)
            image = frames.fit(image[:1, ..., :3], h, w, crop=False)
            if mask is not None:
                mask = frames.fit_mask(mask if mask.dim() == 3 else mask[None], h, w, crop=False)
        pbar = comfy.utils.ProgressBar(steps)
        with timer.stage("conditioning"):
            positive, negative = embeddings.get_conditioning(clip, prompt, "pano")
        with timer.stage("model_patch"):
            patched = sampling.prepare_model(model, "pano", attention)
        window_fn = backend.make_window_fn(patched, dvae.get_vae(vae, vae_precision), positive, negative,
                                           steps, seed, on_step=lambda: pbar.update(1), timer=timer)
        result = pipeline.run_pano(image, window_fn, user_mask=mask, timer=timer)
        timing.log_timing(timing.node_context("pano", 1, 1, steps), timer)
        return io.NodeOutput(result.hdr, result.mask)
