# Experimental workflows

These graphs are a **creative look, not an HDR reconstruction.** They put a Wan 2.1
step-distillation ("turbo") LoRA in front of `DiffHDR (Image / Video)` and sample the
`original` preset at 6 steps. Turbo LoRAs were measured to make DiffHDR *less* faithful —
they shift the value of the log-encoded output, so the reconstructed highlights blow out —
and they do not make sampling faster per step. Use them when you like the result, not when
you need the dynamic range back. See the README section *"Samplers, turbo LoRAs and
SageAttention 3 (measured)"* for the numbers.

| file | LoRA | source | licence |
|---|---|---|---|
| `diffhdr_video_turbo_fastwan.json` | `FastWan_T2V_14B_480p_lora_rank_64_bf16.safetensors` | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) (`FastWan/`) | upstream FastVideo release is Apache-2.0 |
| `diffhdr_video_turbo_accvid.json` | `Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors` | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) | no licence tag upstream — check before commercial use |

**The LoRA files are not downloaded automatically.** Download the file named above and put
it into `ComfyUI/models/loras/`, then pick it in the `LoraLoaderModelOnly` node.

Each graph writes a lossless EXR sequence and, through `DiffHDR Tonemap Preview` →
`Create Video` → `Save Video`, two MP4s: one at 0 EV with the `reinhard` operator and one at
−4 EV with the `clip` operator, so the reconstructed range is visible side by side.
