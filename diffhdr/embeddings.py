"""Text conditioning: bundled umT5 embeddings or live CLIP encoding."""

from pathlib import Path

import torch

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "embeds"
PANO_PROMPT = "Restore the full dynamic range of this clipped HDRI panorama."
DEFAULT_PROMPTS = {"standard": "", "pano": PANO_PROMPT}
_BUNDLED = {"standard": "empty", "pano": "pano"}
SEQ_LEN = 512
_cache: dict[str, torch.Tensor] = {}


def load_bundled(name: str) -> torch.Tensor:
    """Loads ``assets/embeds/diffhdr_<name>.safetensors`` as ``[1,512,4096]`` float32.

    Args:
        name: Asset base name (``empty`` or ``pano``).

    Returns:
        torch.Tensor: The cached ``[1,512,4096]`` float32 embedding.

    Raises:
        RuntimeError: If the asset is missing.
    """
    if name not in _cache:
        from safetensors.torch import load_file
        path = ASSET_DIR / f"diffhdr_{name}.safetensors"
        if not path.is_file():
            raise RuntimeError(f"Bundled DiffHDR text embedding not found ({path}). Connect a CLIP (umT5-xxl) input instead.")
        cond = load_file(str(path))["cond"].float()
        if cond.shape[1] < SEQ_LEN:
            cond = torch.nn.functional.pad(cond, (0, 0, 0, SEQ_LEN - cond.shape[1]))
        _cache[name] = cond
    return _cache[name]


def get_conditioning(clip, prompt: str, variant: str):
    """Returns ``(positive, negative)`` conditioning lists.

    Args:
        clip: ComfyUI CLIP (umT5-xxl) or None to use bundled embeddings.
        prompt: Prompt text; only honoured when ``clip`` is given.
        variant: ``standard`` or ``pano``; selects the bundled positive embedding.

    Returns:
        tuple: The positive and negative CONDITIONING lists.
    """
    if clip is None:
        import logging
        if prompt.strip() not in ("", DEFAULT_PROMPTS[variant]):
            logging.getLogger("DiffHDR").warning("Custom prompt ignored: connect a CLIP input to encode prompts.")
        return [[load_bundled(_BUNDLED[variant]), {}]], [[load_bundled("empty"), {}]]
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    return positive, negative
