# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org/).

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
- README section *Samplers, turbo LoRAs and SageAttention 3 (measured)* with the results of 313
  GPU runs on an RTX PRO 6000 Blackwell, and images throughout the README.

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
