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
| Long video, 65 frames, 3 blended windows | 1.000000 | **53.4 dB** | 33.2 dB |
| HDRI panorama (2048×1024) | 1.000000 | **51.1 dB** | — |

In every case the output is *closer to the reference* than two runs of the reference itself (with
different seeds) are to each other, and the exposure masks are bit-identical to the reference
algorithm. Inside the reconstructed (masked) region the robust highlight statistics agree closely
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
reference's own bfloat16 GPU run is to its float32 run (0.99987 / 0.99953). A 33-frame video
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

Measured on an **NVIDIA A100 80GB PCIe** (CUDA 12.8, PyTorch 2.8, ComfyUI 0.36.0), bf16
`wan2.1_vace_14B_fp16.safetensors` with the float32 VAE, PyTorch SDPA attention, at 1280×720.
"Warm" means the model was already resident in the ComfyUI process; the first run after a restart
additionally pays a **153 s** model load (measured: 591 s cold vs 438 s warm for the same job).

| Mode | Steps | Output frames | Warm wall time | s / output frame | Peak VRAM |
|---|---|---|---|---|---|
| Image (720p) | 50 | 1 | 636 s | 636 | 62.9 GB |
| Image (720p) | 10 | 1 | 150 s | 150 | 62.9 GB |
| Video, 33 frames (720p) | 50 | 33 | 935 s | 28.3 | 63.5 GB |
| Video, 33 frames (720p) | 20 | 33 | 575 s | 17.4 | 62.2 GB |
| Video, 33 frames (720p) | 10 | 33 | 438 s | 13.3 | 64.2 GB |
| Long video, 65 frames, 3 windows | 50 | 65 | 2701 s (900 s / window) | 41.6 | 64.6 GB |
| HDRI panorama 2048×1024 | 50 | 1 | 115 s | 115 | 49.0 GB |

Quantised base models, 33-frame video at 20 steps:

| Base model | Warm wall time | Peak VRAM | Log-space PSNR vs bf16 |
|---|---|---|---|
| bf16 `wan2.1_vace_14B_fp16.safetensors` | 575 s | 62.2 GB | — |
| `weight_dtype = fp8_e4m3fn` (same file) | 684 s | **47.2 GB** | 41.2 dB |
| GGUF `Wan2.1_14B_VACE-Q4_K_M.gguf` | 605 s | **41.0 GB** | 40.7 dB |

Attention backends produce the same result: `auto` and `flash_attn` are both within **85.6 dB** of
`sdpa` (`auto` selects flash-attn when it is importable). flash-attn was the fastest at 530 s for
the 20-step video versus 575 s for SDPA.

Two notes on where the time goes. Image mode is much cheaper per run than a 33-frame video at the
same step count (150 s vs 438 s at 10 steps) even though both sample the same ~32 400 tokens,
because the per-frame exposure-mask stabilisation runs once instead of 33 times. And the sampler
itself costs about 12 s/step at 720p×33 frames, so a 33-frame job has roughly 300 s of fixed cost
(masking, two VAE encodes and one decode, all in float32) on top of `12 s × steps`.

The reference implementation's own timings on the same GPU and inputs were 1185 s (image), 994 s
(33-frame video), 3579 s (65-frame long video) and 592 s (panorama) — but each of those is a fresh
process that loads the 14B model from scratch every time (~270-400 s of the total), so they are not
directly comparable to the warm numbers above.

### How many steps?

- **50 steps** reproduces the published setting and is what the parity numbers above were measured at.
- **20 steps** is the best speed/quality trade-off: 40 % faster, and the reconstructed highlights
  stay within ~4 % of the 50-step result (masked p99.9 luminance 3.98 vs 3.82) at 47 dB masked PSNR.
  It is still closer to the 50-step reference than two reference seeds are to each other.
- **10 steps** is fine for previews and for finding a seed, but it does *not* reconstruct the same
  amount of highlight energy: on the test image the masked p99.9 luminance drops from 8.87 to 5.61
  (−37 %) and the peak from 15.0 to 6.4. Use it to iterate, then re-run the keeper at 20 or 50.

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
  sequence.

## Limitations

- DiffHDR was trained at 720p (1280×720 or 720×1280) on 33-frame windows; `resize_mode=crop_to_720p`
  and the default `window_size=33` match that training distribution most closely. Other sizes and
  window lengths work but are extrapolating beyond the training distribution.
- `window_size` must be `4n+1`; `window_stride` must be smaller than `window_size`.
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
