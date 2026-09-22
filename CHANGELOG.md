# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org/).

## 0.3.1 — 2026-09-22

### Fixed

- The DiffHDR LoRAs are downloaded from
  [`Eyeline-Labs/DiffHDR`](https://huggingface.co/Eyeline-Labs/DiffHDR); the previous Hugging Face
  location no longer answers. Same file names, checksums verified against the new repository, same
  160 LoRA tensors on the eight VACE blocks.

## 0.3.0 — 2026-09-21

### Changed

- **`use_prev_window_reference` is on by default.** Every sliding window of a long video is
  sampled on its own; without a reference each window invents its own version of the clipped
  content and the blend visibly cross-fades between them. Feeding each window the previous
  window's output frame was measured to raise the agreement of neighbouring windows from 28.7 to
  34.4 dB and to cut the drift along the clip from −15.8 % to −4.6 %, at 2 % more time. The
  tooltip explains it. Two alternatives (a fixed first-window reference, and joint denoising with
  ComfyUI's context windows) were measured and rejected — see the README.
- New README hero image from the first frame of the demo dance sequence.

## 0.2.0 — 2026-09-21

### Added

- **Sampler presets on both all-in-one nodes.** A `preset` dropdown is now the first input:
  `fast` (`res_multistep` / `simple` / shift 8, the default), `original` (`euler` / `simple` /
  shift 5, the reference implementation's setting) and `custom`.
- **`sampler`, `scheduler` and `shift` widgets** below `seed`, honoured when `preset` is
  `custom`. The option lists are curated — the `beta` scheduler and the `deis` sampler measured
  worse than doing nothing and are not offered.
- **Preset-sync frontend extension** (`web/diffhdr_presets.js`): picking a preset fills the three
  widgets, editing one of them switches the preset to `custom`. Python resolves the preset again
  on execution, so the nodes behave identically without the extension.
- README section *Samplers, turbo LoRAs and SageAttention 3 (measured)* with the results of 323
  GPU runs on an RTX PRO 6000 Blackwell, and images throughout the README.
- **The sliding-window long-video path is measured too**: all 99 demo frames through the
  all-in-one node in six blended windows. `fast` at 20 steps reaches 46.5 dB against the 50-step
  `original` reference and beats `original` at 6 steps (38.6 vs 35.7 dB); no preset or step count
  changes the window seams. Two 50-step runs that differ only in the noise seed are 25.1 dB
  apart, so every measured setting is inside the reference's own distribution — the `fast`
  default stands for long videos as well.
- `workflows/experimental/`: two graphs that put a Wan 2.1 turbo LoRA (FastWan rank 64, AccVid
  rank 32) in front of the all-in-one node at `original` / 6 steps and export MP4s at 0 EV and
  −4 EV. They are a creative look, not an HDR reconstruction — measured at 21.8 dB / +212 %
  highlight energy and 26.8 dB / +27 % against the 50-step reference — and are labelled as such.

### Changed

- **Tuned defaults**: `steps` 50 → 20, `shift` 5 → 8, sampler `euler` → `res_multistep`. Select
  the `original` preset to reproduce the reference implementation.
- `attention=auto` now prefers SageAttention, then flash-attn, then ComfyUI's default (2026-09-19).
- Example workflows re-exported with the new defaults; the modular graph's `KSampler` and
  `ModelSamplingSD3` values match the `fast` preset. The three all-in-one graphs no longer
  contain a text encoder — the bundled umT5 embeddings make it optional.
- Long-video reconstruction uses noticeably less host memory, and the ComfyUI-dependent tests run
  in a dedicated CI job that falls back to the CPU on machines without an accelerator.

## 0.1.0 — 2026-09-19

Initial release: seven V3 nodes (`DiffHDRVideo`, `DiffHDRPano`, `DiffHDRApplyLora`,
`DiffHDRPreprocess`, `DiffHDRPostprocess`, `DiffHDRTonemap`, `DiffHDRSaveEXR`) porting DiffHDR
onto ComfyUI's native Wan2.1-VACE-14B objects, validated against the reference implementation on
a GPU (51–59 dB parity for image, video, long video and panorama).
