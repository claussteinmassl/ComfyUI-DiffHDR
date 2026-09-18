"""Generates the bundled DiffHDR umT5 text embeddings with ComfyUI's text encoder.

The DiffHDR nodes work without a CLIP input by loading these pre-computed
embeddings from ``assets/embeds``. Run this once inside a ComfyUI installation
that has ``umt5_xxl_fp16.safetensors`` available.

Usage (from the ComfyUI root):
    python custom_nodes/ComfyUI-DiffHDR/scripts/make_embeds.py models/text_encoders/umt5_xxl_fp16.safetensors
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path.cwd()))
import comfy.sd  # noqa: E402

PROMPTS = {"empty": "", "pano": "Restore the full dynamic range of this clipped HDRI panorama."}
OUT = Path(__file__).resolve().parent.parent / "assets" / "embeds"


def kept_length(cond: torch.Tensor) -> int:
    """Returns the number of leading tokens before the zero padding.

    Args:
        cond: Conditioning tensor ``[1,T,4096]``.

    Returns:
        int: The token count to keep; the full length when padding is not zeroed.
    """
    nonzero = cond[0].abs().sum(dim=-1) > 0
    if not bool(nonzero.any()):
        return 1
    length = int(nonzero.nonzero().max().item()) + 1
    if bool(nonzero[length:].any()):
        return int(cond.shape[1])
    return length


def main() -> None:
    """Encodes both DiffHDR prompts and writes them to ``assets/embeds``."""
    clip = comfy.sd.load_clip(ckpt_paths=[sys.argv[1]], clip_type=comfy.sd.CLIPType.WAN)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, prompt in PROMPTS.items():
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0].float().cpu()
        assert cond.shape[-1] == 4096, cond.shape
        length = kept_length(cond)
        save_file({"cond": cond[:, :length].to(torch.bfloat16).contiguous()}, str(OUT / f"diffhdr_{name}.safetensors"))
        print(name, tuple(cond.shape), "kept tokens:", length)


if __name__ == "__main__":
    main()
