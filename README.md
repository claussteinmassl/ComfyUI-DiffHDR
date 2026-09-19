# ComfyUI-DiffHDR

LDR-to-HDR reconstruction for images, videos and equirectangular HDRI panoramas, built on ComfyUI's
native Wan2.1-VACE-14B objects. This is a from-scratch ComfyUI port of
[DiffHDR](https://github.com/Eyeline-Labs/DiffHDR) (Eyeline Labs) — it reimplements DiffHDR's mask
detection, log-color encoding and long-video blending on top of ComfyUI's own `MODEL`/`VAE`/`CLIP`
objects, VACE conditioning and sampler, instead of vendoring DiffSynth-Studio.

## Status

Validated end-to-end against the reference implementation on an NVIDIA A100 80GB PCIe, using
byte-identical pre-sized inputs and the reference default of 50 sampling steps on both sides.

| Case | Mask IoU vs reference mask code | Log-space PSNR outside the mask | Reference's own seed-to-seed PSNR |
|---|---|---|---|
| Single image (1280×720) | 1.000000 | **57.8 dB** | 47.0 dB |
| Video, 33 frames (1280×720) | 1.000000 | **59.1 dB** | 42.1 dB |
| Long video, **all 99 demo frames, 6 blended windows** | 1.000000 | **57.5 dB** | — |
| Long video, 65 frames, 3 blended windows | 1.000000 | **53.4 dB** | 33.2 dB |
| HDRI panorama (2048×1024) | 1.000000 | **51.1 dB** | — |

The 99-frame run covers the full demo sequence, so the last window is short (19 real frames) and is
padded to the trained length by repeating the last frame, exactly as the reference does. Inside the
reconstructed region the two agree to within 5 % on p99.9 luminance (25.4 vs 24.2) and 0.3 % on the
mean log value (0.4255 vs 0.4244), and the largest frame-to-frame step in masked mean log at a
window boundary is 0.0035 against a median of 0.0014 over a 0.377–0.450 range — i.e. no visible
seam, including at the padded last window.

Wherever a reference seed-to-seed floor could be measured, the output is *closer to the reference*
than two runs of the reference itself (with different seeds) are to each other, and the exposure
masks are bit-identical to the reference algorithm in every case. Inside the reconstructed (masked) region the robust highlight statistics agree closely
— e.g. for the 33-frame video p99.9 luminance 3.82 vs 3.72 and mean log value 0.4004 vs 0.3995 —
while the single brightest pixel is not a meaningful comparison, because the reference's own two
seeds differ on it by a factor of ~4 (p99.9 of 3.72 vs 13.65 between reference seeds 10 and 11).

Also verified on real weights: the DiffHDR LoRA downloads from a clean install and applies exactly
80 patches for both variants; the float32 VAE working copy really is float32 in all 194 parameters
(the checkpoint otherwise loads as bfloat16); and the sampler schedule matches the reference's
`linspace(1, 0, N+1)[:-1]` with shift 5 to within 6e-8 at 50, 20 and 10 steps.

## What it is

DiffHDR turns clipped, 8-bit LDR footage into linear, scene-referred HDR by treating LDR-to-HDR
conversion as a generative radiance-inpainting problem inside the latent space of a video diffusion
model (Wan2.1-VACE-14B). It works in a log-gamma color space and uses spatio-temporal priors from
the video model to synthesize plausible detail in over- and under-exposed regions while recovering
continuous radiance in the correctly-exposed ones. This pack exposes that as native ComfyUI nodes:
seven `DiffHDR*` nodes that plug into stock `UNETLoader` / `VAELoader` / `CLIPLoader` /
`WanVaceToVideo` / `KSampler` objects — no VACE fork, no bundled inference engine.

Three ways to use it:

- **Image**: a single clipped LDR frame in, one linear HDR frame out.
- **Video**: an LDR frame sequence in; long clips are processed as overlapping, blended sliding
  windows.
- **HDRI (panorama)**: a clipped equirectangular LDR panorama in, a full HDR environment map out,
  using a separate LoRA trained for panoramas.

## Requirements & model downloads

Wan2.1-VACE-14B and the Wan 2.1 VAE are the same checkpoints used by any native ComfyUI VACE
workflow; the DiffHDR LoRA is downloaded automatically on first use.

| Model | File | Source | Folder |
|---|---|---|---|
| Wan2.1 VACE 14B | `wan2.1_vace_14B_fp16.safetensors` | [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models) (`split_files/diffusion_models/`) | `models/diffusion_models` |
| Wan2.1 VACE 14B GGUF (low memory / Apple Silicon) | e.g. `Wan2.1_14B_VACE-Q4_K_M.gguf`, `Q5_K_M`, `Q8_0` | [`QuantStack/Wan2.1_14B_VACE-GGUF`](https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF) — requires the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom node (`UnetLoaderGGUF` in place of `UNETLoader`) | `models/diffusion_models` |
| Wan 2.1 VAE | `wan_2.1_vae.safetensors` | same Comfy-Org repo ([`split_files/vae/`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae)) | `models/vae` |
| umT5-xxl (**optional** — only for custom prompts, see below) | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | same Comfy-Org repo ([`split_files/text_encoders/`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders)) | `models/text_encoders` |
| DiffHDR LoRAs | `DiffHDR.safetensors` (image/video), `DiffHDR_Pano.safetensors` (HDRI) | [`ZhengmingYu/DiffHDR`](https://huggingface.co/ZhengmingYu/DiffHDR) — downloaded automatically | `models/loras/DiffHDR` |

All filenames above were verified against the current file listing of each Hugging Face repository,
and every one of them was downloaded and run end-to-end during GPU validation.

**About the CLIP input — optional**: DiffHDR was trained with an empty prompt for images/video and
a fixed prompt for panoramas, so both all-in-one nodes ship with exactly those embeddings baked in
(`assets/embeds/`) and run with **no CLIP model loaded at all**. Leave `clip` unconnected and you
save the ~11 GB umT5-xxl download and its load time; the `prompt` widget is then ignored.

The bundled embeddings were produced with ComfyUI's own umT5-xxl encoder (`scripts/make_embeds.py`)
and verified on the GPU: cosine similarity against the DiffSynth prompter output used by the
reference implementation is 0.999997 (empty prompt) and 0.99982 (panorama prompt) — closer than the
reference's own bfloat16 run is to its float32 run (0.99996 / 0.99955). A 33-frame video
reconstructed with the bundled embeddings matches the same run with a live `CLIPLoader` at
**85.5 dB** log-space PSNR. Connect a umT5-xxl CLIP (`CLIPLoader`, type `wan`) only if you want to
experiment with your own prompts.

## Installation

1. Clone this repository into `ComfyUI/custom_nodes/ComfyUI-DiffHDR`:
   `git clone https://github.com/cs-agentic/ComfyUI-DiffHDR.git ComfyUI/custom_nodes/ComfyUI-DiffHDR`
2. `pip install -r requirements.txt` inside your ComfyUI Python environment (adds `OpenEXR` and
   `huggingface_hub`; `torch` and `safetensors` are assumed to already be provided by ComfyUI).
3. Download the Wan2.1-VACE-14B model and Wan 2.1 VAE from the table above into the usual ComfyUI
   model folders. The DiffHDR LoRA is fetched automatically the first time a DiffHDR node runs.
4. Restart ComfyUI. The nodes register under the **DiffHDR** category.

## Nodes

### DiffHDR (Image / Video) — `DiffHDRVideo`

All-in-one reconstruction for a single image or a frame sequence. One frame runs in image mode; up
to `window_size` frames run as a single window; more frames run as overlapping, blended sliding
windows.

- **Inputs**: `model` (Wan2.1-VACE-14B), `vae` (Wan 2.1 VAE), `images` (LDR image or batch),
  `clip` (optional umT5-xxl; see above), `prompt` (only used with `clip` connected), `mask`
  (optional, overrides automatic detection), `reference_image` (optional, guides content of
  over-exposed regions), `reference_ev` (exposure boost applied to the reference, default 5 stops),
  `resize_mode` (`crop_to_720p` / `native` / `custom`), `width`, `height` (for `custom`), `steps`
  (default 50), `seed`, `mask_overexposed`, `mask_underexposed`, `window_size` (frames per window,
  4n+1, default 33 — the training length), `window_stride` (frames between window starts, default
  16), `use_prev_window_reference` (temporal consistency across windows for long videos),
  `attention`, `vae_precision`.
- **Outputs**: `hdr` (linear scene-referred HDR image/batch), `mask` (regenerated regions).

### DiffHDR HDRI (Panorama) — `DiffHDRPano`

Reconstructs an HDR environment map from a single clipped, equirectangular LDR panorama using the
DiffHDR panorama LoRA.

- **Inputs**: `model`, `vae`, `image` (equirectangular, 2:1; only the first frame of a batch is
  used), `clip` (optional), `prompt` (defaults to the training prompt, only used with `clip`
  connected), `mask` (optional), `width` (default 2048), `height` (default 1024 — the panorama is
  stretched to this size, not cropped), `steps` (default 50), `seed`, `attention`, `vae_precision`.
- **Outputs**: `hdr`, `mask`.

### DiffHDR Apply LoRA — `DiffHDRApplyLora`

For modular graphs: downloads (on first use) and patches the DiffHDR LoRA into a Wan2.1-VACE-14B
model. Follow with `ModelSamplingSD3` (shift 5), and sample with `euler` / `simple`, `cfg` 1.

- **Inputs**: `model`, `variant` (`standard`: image/video, `pano`: HDRI), `strength` (default 1.0).
- **Outputs**: `model`.

### DiffHDR Preprocess — `DiffHDRPreprocess`

Builds the log-encoded control video and regeneration mask for a native `WanVaceToVideo` graph.

- **Inputs**: `images` (sRGB LDR frames, sized to multiples of 16), `variant` (`video` / `pano`
  mask detector), `mask_overexposed`, `mask_underexposed`.
- **Outputs**: `control_video` (log-encoded frames, feed to `WanVaceToVideo.control_video`),
  `control_masks` (feed to `WanVaceToVideo.control_masks`).

### DiffHDR Postprocess — `DiffHDRPostprocess`

Decodes the DiffHDR log curve to linear HDR after `VAE Decode` in a modular graph.

- **Inputs**: `images` (VAE-decoded, log-encoded frames).
- **Outputs**: `hdr` (linear HDR).

### DiffHDR Tonemap Preview — `DiffHDRTonemap`

Converts linear HDR to a displayable sRGB image, for `PreviewImage` or an LDR export.

- **Inputs**: `hdr`, `exposure` (stops, default 0), `operator` (`reinhard` compresses highlights,
  `clip` shows the LDR range at the chosen exposure as-is).
- **Outputs**: `image` (sRGB, `[0,1]`).

### DiffHDR Save EXR — `DiffHDRSaveEXR`

Writes linear HDR as OpenEXR (single file or frame sequence) or Radiance `.hdr`.

- **Inputs**: `images` (linear Rec.709 HDR image or batch), `filename_prefix` (default
  `DiffHDR/hdr`), `format` (`exr` / `hdr`), `bit_depth` (`half` / `float`), `compression` (`none`,
  `rle`, `zips`, `zip`, `piz`, `pxr24`, `b44`, `b44a`, `dwaa`, `dwab`), `dwa_compression_level`
  (default 45, only used by `dwaa`/`dwab`), `colorspace` (`linear_rec709` / `acescg` /
  `aces2065_1`), `start_frame` (first frame number of a sequence), `preview_exposure` (exposure for
  the tonemapped UI preview only).
- **Outputs**: none (output node; also emits a tonemapped preview in the UI).

## Workflows

Four example workflows are in `workflows/` (ComfyUI UI-format JSON, load with **Open** or
drag-and-drop). All of them reference `wan2.1_vace_14B_fp16.safetensors`,
`wan_2.1_vae.safetensors` and `umt5_xxl_fp8_e4m3fn_scaled.safetensors` — set the loader widgets to
match whatever you actually have installed if you used the GGUF or a different precision.

- **`diffhdr_image.json`** — the simplest graph: `UNETLoader` + `VAELoader` + `CLIPLoader` (type
  `wan`) + `LoadImage` into `DiffHDR (Image / Video)`, then `DiffHDR Save EXR` and
  `DiffHDR Tonemap Preview` → `PreviewImage`.
- **`diffhdr_video.json`** — the same all-in-one node fed by `LoadVideo` + `GetVideoComponents`
  (core ComfyUI video nodes, no VideoHelperSuite dependency) instead of `LoadImage`.
- **`diffhdr_hdri.json`** — `LoadImage` (equirectangular panorama) into
  `DiffHDR HDRI (Panorama)`, then save/tonemap/preview.
- **`diffhdr_modular.json`** — the fully manual native-VACE graph:
  `DiffHDR Apply LoRA` → `ModelSamplingSD3` (shift 5); `LoadVideo` + `GetVideoComponents` →
  `DiffHDR Preprocess` → `WanVaceToVideo` (conditioned by `CLIPTextEncode`) → `KSampler`
  (`euler` / `simple`, `cfg` 1) → `TrimVideoLatent` → `VAEDecode` → `DiffHDR Postprocess` → save /
  tonemap / preview. Use this as a starting point when you need to combine DiffHDR with other VACE
  controls or additional LoRAs, since every step is a separate node instead of one bundled node.

## Performance

Every DiffHDR node logs one INFO line per execution with its own stage breakdown, so you can see
where your time goes on your own hardware instead of guessing:

```
DiffHDR timing [video, 33 frames, 1 window(s), 10 steps]: prepare=0.0s conditioning=1.0s
model_patch=0.0s masks=6.2s control=1.9s encode=7.6s sample=118.2s decode=5.5s window=131.5s
blend=1.0s total=141.8s
DiffHDR timing [save_exr, 33 frames, exr/float/zip]: write=7.2s preview=0.5s total=7.7s
```

`total` is wall clock for the whole node, so it is not the sum of the stages: `window` already
contains `encode`/`sample`/`decode`, and the difference between `total` and the stages is work no
stage covers.

### Where the time goes

Measured on an **NVIDIA A100 80GB PCIe** (CUDA 12.8, PyTorch 2.8, ComfyUI 0.36.0), bf16
`wan2.1_vace_14B_fp16.safetensors` with the float32 VAE, PyTorch SDPA attention, umT5 connected,
1280×720 (panorama 2048×1024), **warm** (model already resident), **10 sampling steps**. Seconds.

| Stage | Image | Video, 33 frames | Panorama |
|---|---|---|---|
| `prepare` (resize / fit) | 0.0 | 0.0 | 0.0 |
| `conditioning` (text embeddings) | 0.8 | 1.0 | 0.8 |
| `model_patch` (LoRA + shift + attention) | 0.0 | 0.0 | 0.0 |
| `masks` (exposure detection + stabilisation) | 0.4 | 6.2 | 0.1 |
| `control` (log encoding, reference prep) | 0.1 | 1.9 | 0.2 |
| `encode` (VACE tensors, 2 VAE encodes) | 8.0 | 7.6 | 0.8 |
| **`sample`** | **118.2** | **118.2** | **20.1** |
| `decode` (VAE) | 5.5 | 5.5 | 0.3 |
| `blend` (window overlap) | — | 1.0 | — |
| **node total** | **133.3** | **141.8** | **22.4** |
| Save EXR (float32 / ZIP, write + preview) | 0.4 | 7.7 | 0.8 |
| graph overhead (image loading, queue) | 6.4 | 6.1 | 1.9 |
| **end-to-end** | **140.1** | **155.6** | **25.1** |

Three things follow directly from these measurements:

* **Sampling is linear in steps and is the whole story.** From 10 and 50 steps on the same job:
  `(588.1 − 118.2) / 40` = **11.7 s per step** at 720p × 33 frames, with an intercept of 0.7 s.
* **The per-job fixed cost is small**: node total minus `sample` is **24 s** for a 33-frame video
  and **15 s** for an image. (Earlier versions of this README quoted "roughly 300 s of fixed cost";
  that figure was *derived* from a line fit on a CPU-starved host, and it is wrong — the number
  above is measured stage by stage.)
* **Image mode is not cheaper because it samples less.** It replicates the single frame to the
  trained 33-frame window, so it runs the *same* VAE and sampling work: `window` is 131.9 s
  (image) against 131.5 s (video). The whole difference is per-frame CPU work, mostly mask
  detection over 33 frames instead of 1.

### Totals

Same host and settings; warm. A cold first run after a ComfyUI restart adds a **170 s** model load
(measured 350.9 s cold vs 180.7 s warm for the same job).

| Mode | Steps | Output frames | Warm wall time | s / output frame | Peak VRAM |
|---|---|---|---|---|---|
| Image (720p) | 10 | 1 | 140 s | 140 | 62.9 GB |
| Video, 33 frames (720p) | 10 | 33 | 156 s | 4.7 | 63.4 GB |
| Video, 33 frames (720p) | 50 | 33 | 636 s | 19.3 | 62.9 GB |
| Long video, 99 frames, 6 windows | 20 | 99 | 1559 s (260 s / window) | 15.7 | 63.4 GB |
| Long video, 99 frames, 6 windows | 50 | 99 | 3706 s (618 s / window) | 37.4 | 63.4 GB |
| Long video, 99 frames, `use_prev_window_reference` | 20 | 99 | 1754 s | 17.7 | 63.4 GB |
| HDRI panorama 2048×1024 | 10 | 1 | 25 s | 25 | 48.7 GB |

The panorama is far cheaper than the video modes because it is a single latent frame: 2048×1024 is
256×128 latent pixels against 9 latent frames × 80×45 for a 720p clip.

### The same GPU can be twice as slow

These jobs are GPU-bound only during `sample`. Everything else is CPU work, and two rented
"A100 80GB PCIe" hosts measured **2.4× apart** end-to-end on the identical 10-step 33-frame job
with identical code (181 s vs 438 s) even though the sampling component — the only GPU-bound part
— differed by about 5 % (118 s measured here against the ~124 s implied by the other host's
step-count fit). The cause is CPU contention, and one setting dominates it: PyTorch sizes its
thread pool from the cores the container *advertises*, not from the cores the instance is
*allocated*. On a pod that showed 128 cores to `torch` but was allocated 16 vCPU, capping the pool
nearly halved the mask stage:

| `torch.get_num_threads()` | mask stage, 33 frames @720p |
|---|---|
| 64 (the default there) | 6.4 s |
| 16 | 3.5 s |

If your `DiffHDR timing` line shows a large `masks` or `control` value relative to `sample`, set
`OMP_NUM_THREADS` to the number of cores you actually have before starting ComfyUI.

### Quantised base models and attention backends

Measured in an earlier session on a different A100 80GB PCIe host (the one at the slow end of the
range above), 33-frame video at 20 steps. The wall times are only comparable **to each other**; the
VRAM figures and the PSNRs are host-independent.

| Base model | Warm wall time | Peak VRAM | Log-space PSNR vs bf16 |
|---|---|---|---|
| bf16 `wan2.1_vace_14B_fp16.safetensors` | 575 s | 62.2 GB | — |
| `weight_dtype = fp8_e4m3fn` (same file) | 684 s | **47.2 GB** | 41.2 dB |
| GGUF `Wan2.1_14B_VACE-Q4_K_M.gguf` | 605 s | **41.0 GB** | 40.7 dB |

Attention backends produce the same result: `auto` and `flash_attn` are both within **85.6 dB** of
`sdpa` (`auto` selects flash-attn when it is importable). flash-attn was the fastest at 530 s for
the 20-step video versus 575 s for SDPA on that host.

The reference implementation's own timings, on that same earlier host and the same inputs, were
1185 s (image), 994 s (33-frame video) and 592 s (panorama); on the faster host its 99-frame long
video took 4538 s against our 3706 s. Each reference invocation is a fresh process that loads the
14B model from scratch (~270–400 s of the total), so they are not directly comparable to the warm
numbers above.

### How many steps?

- **50 steps** reproduces the published setting and is what all parity numbers above were measured at.
- **20 steps** is usually the best speed/quality trade-off — 40 % faster — but how much highlight
  energy it gives up depends on the shot. On the 33-frame test clip the reconstructed highlights
  stay within ~4 % of the 50-step result (masked p99.9 luminance 3.98 vs 3.82) at 47 dB masked
  PSNR; on the much brighter 99-frame long-video sequence 20 steps reconstructs **~20 % less**
  (masked p99.9 19.4 vs the reference's 24.2, against 25.4 at 50 steps), while staying at 58.8 dB
  outside the mask. Check the highlights on your own footage before committing to 20.
- **10 steps** is fine for previews and for finding a seed, but it does *not* reconstruct the same
  amount of highlight energy: on the test image the masked p99.9 luminance drops from 8.87 to 5.61
  (−37 %) and the peak from 15.0 to 6.4. Use it to iterate, then re-run the keeper at 20 or 50.
- `use_prev_window_reference` on long videos costs about 13 % more time (1754 s vs 1559 s at 20
  steps over 99 frames) and visibly tightens temporal consistency: the per-frame masked mean log
  value varies over 0.425–0.454 with it against 0.384–0.452 without.

## Platform notes

- **CUDA**: `attention=auto` prefers flash-attn, then SageAttention, then ComfyUI's own default
  attention backend, depending on what is importable in your environment; unavailable/incompatible
  backends fall back to PyTorch SDPA automatically.
- **ROCm**: treated like any other discrete accelerator; flash-attn/SageAttention availability
  follows whatever ComfyUI itself detects for the device, with the same SDPA fallback.
- **MPS (Apple Silicon)**: flash-attn and SageAttention are GPU-only backends in ComfyUI and are
  not selected on `mps`, so `attention=auto` (or any explicit choice) transparently uses PyTorch
  SDPA. For VRAM-constrained Apple Silicon machines, use a GGUF-quantized Wan2.1-VACE-14B checkpoint
  via the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom node instead of the full
  fp16 safetensors.
- **CPU**: supported for testing (this is how the automated test suite and CI run) but far too slow
  for real inference on a 14B video diffusion model.
- **Windows**: the code path has no Windows-specific branches, but it has not been run on a Windows
  GPU yet — see Limitations.

## HDR/EXR notes

- The `hdr` output of every DiffHDR node is **linear, scene-referred** Rec.709 data with values
  that legitimately exceed `1.0` in reconstructed highlights. Never route it through an 8-bit save
  or preview node directly — use `DiffHDR Tonemap Preview` first, or write it out with
  `DiffHDR Save EXR`.
- `DiffHDR Save EXR` can target OpenEXR (`half`/16-bit or `float`/32-bit, with `none` / `rle` /
  `zips` / `zip` / `piz` / `pxr24` / `b44` / `b44a` / lossy `dwaa` / `dwab` compression) or Radiance
  `.hdr` (RGBE). The Radiance path is always linear Rec.709 and ignores the bit depth, compression
  and colorspace widgets.
- `colorspace` converts pixels from linear Rec.709 to the chosen primaries (`linear_rec709`,
  `acescg`, `aces2065_1`) and writes the matching chromaticities into the EXR header, so downstream
  tools read the file correctly regardless of which primaries you picked.
- A batch of frames is written as `<filename_prefix>_<counter>/frame_####.exr`; a single image is
  written directly under `filename_prefix`. `start_frame` sets the first frame number in a
  sequence. Frame numbers are padded to at least four digits, widened for the whole sequence
  when the last frame needs more, so the files always stay in lexical order.

## Limitations

- DiffHDR was trained at 720p (1280×720 or 720×1280) on 33-frame windows; `resize_mode=crop_to_720p`
  and the default `window_size=33` match that training distribution most closely. Other sizes and
  window lengths work but are extrapolating beyond the training distribution.
- `window_size` must be `4n+1`; `window_stride` must be smaller than `window_size`.
- Long clips cost host RAM in proportion to their length. At 1280×720 a frame needs about
  **37 MB of system RAM** while the node runs — roughly 26 MB of it allocated by DiffHDR
  itself (log-encoded control frames, mask, HDR output buffer) plus the ~11 MB input IMAGE
  that ComfyUI keeps cached. The figure is derived from the tensor sizes
  (`1280·720·3·4` bytes per float32 RGB frame, `1280·720·4` per mask frame), not measured
  on a specific machine. A 500-frame 720p clip therefore needs roughly 18 GB of free RAM;
  split longer shots or process them in parts.
- The published numbers were measured on a single A100 80GB PCIe; other GPUs will differ.
- The Windows GPU path is untested.

## Credits & License

This project is licensed under **Apache-2.0** (see `LICENSE`).

It is an independent ComfyUI integration of:

- **[DiffHDR](https://github.com/Eyeline-Labs/DiffHDR)** by Eyeline Labs (Apache-2.0) — mask
  detection, the log encoding curve and the long-video blending scheme are ported from this
  project. Paper: *DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models*,
  [arXiv:2604.06161](https://arxiv.org/abs/2604.06161), project page
  [eyeline-labs.github.io/DiffHDR](https://eyeline-labs.github.io/DiffHDR/).
- **[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)** (Apache-2.0), which DiffHDR
  itself builds on.
- **[Wan 2.1](https://github.com/Wan-Video/Wan2.1)** (Apache-2.0), the underlying video diffusion
  model, used here through ComfyUI's native Wan2.1-VACE-14B support.
- The **[DiffHDR LoRA weights](https://huggingface.co/ZhengmingYu/DiffHDR)** (Apache-2.0) are
  downloaded at runtime and are not part of this repository.

See `NOTICE` for the full attribution.
