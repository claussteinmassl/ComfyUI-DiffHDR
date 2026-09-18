# ComfyUI-DiffHDR

LDR-to-HDR reconstruction for images, videos and equirectangular HDRI panoramas, built on ComfyUI's
native Wan2.1-VACE-14B objects. This is a from-scratch ComfyUI port of
[DiffHDR](https://github.com/Eyeline-Labs/DiffHDR) (Eyeline Labs) — it reimplements DiffHDR's mask
detection, log-color encoding and long-video blending on top of ComfyUI's own `MODEL`/`VAE`/`CLIP`
objects, VACE conditioning and sampler, instead of vendoring DiffSynth-Studio.

## Status

This node pack has been unit-tested (mask detection, log-curve math, LoRA patching, EXR I/O, node
schemas, ...) against oracle values from the reference implementation, and the ComfyUI-facing code
has been smoke-tested against a local, model-free ComfyUI instance. **GPU validation against the
reference implementation is in progress** — no end-to-end runs against real weights, no
performance/VRAM numbers and no parity claims are made yet.

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
| umT5-xxl (required for now, see below) | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | same Comfy-Org repo ([`split_files/text_encoders/`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders)) | `models/text_encoders` |
| DiffHDR LoRAs | `DiffHDR.safetensors` (image/video), `DiffHDR_Pano.safetensors` (HDRI) | [`ZhengmingYu/DiffHDR`](https://huggingface.co/ZhengmingYu/DiffHDR) — downloaded automatically | `models/loras/DiffHDR` |

All filenames above were verified against the current file listing of each Hugging Face repository.

**About the CLIP input**: DiffHDR was trained with an empty prompt for images/video and a fixed
prompt for panoramas, so the `DiffHDR (Image / Video)` and `DiffHDR HDRI (Panorama)` nodes are
meant to run with those exact embeddings bundled into the pack and no CLIP model loaded at all.
Those bundled embeddings have not shipped yet — **until they do, connect a umT5-xxl CLIP input**
(`CLIPLoader`, type `wan`) to these nodes; leaving `clip` unconnected will fail with a clear error
telling you to do so. This note will be updated once the bundled embeddings ship.

## Installation

1. Clone or copy this repository into `ComfyUI/custom_nodes/ComfyUI-DiffHDR`.
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
- No performance, VRAM or timing numbers are published here — see Status.
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
