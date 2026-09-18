"""DiffHDR LoRA handling: locate or download, rename keys, patch the model."""

import logging
from pathlib import Path

HF_REPO = "ZhengmingYu/DiffHDR"
LORA_FILES = {"standard": "DiffHDR.safetensors", "pano": "DiffHDR_Pano.safetensors"}
LORA_SIZE = 61_360_760
EXPECTED_PATCHES = 80
SUBFOLDER = "DiffHDR"

_PREFIX = "diffusion_model."
_sd_cache: dict[str, dict] = {}
log = logging.getLogger("DiffHDR")


def rename_keys(sd: dict) -> dict:
    """Prefixes DiffSynth-style keys with ``diffusion_model.`` (idempotent)."""
    return {k if k.startswith(_PREFIX) else _PREFIX + k: v for k, v in sd.items()}


def _candidate_paths(filename: str) -> list[Path]:
    """All ``<loras dir>/DiffHDR/<filename>`` locations; the first is the download target."""
    import folder_paths
    return [Path(d) / SUBFOLDER / filename for d in folder_paths.get_folder_paths("loras")]


def _download(filename: str, target_dir: Path) -> None:
    from huggingface_hub import hf_hub_download
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=str(target_dir))


def ensure_lora(variant: str) -> Path:
    """Returns the local LoRA path, downloading it on first use.

    Args:
        variant: ``standard`` (image/video) or ``pano`` (HDRI).

    Raises:
        ValueError: On an unknown variant.
        RuntimeError: If the file cannot be downloaded or has the wrong size.
    """
    if variant not in LORA_FILES:
        raise ValueError(f"Unknown DiffHDR LoRA variant: {variant}")
    filename = LORA_FILES[variant]
    candidates = _candidate_paths(filename)
    for path in candidates:
        if path.is_file() and path.stat().st_size == LORA_SIZE:
            return path
    target = candidates[0]
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
    try:
        log.info("Downloading %s to %s", url, target)
        _download(filename, target.parent)
    except Exception as e:
        raise RuntimeError(f"Could not download the DiffHDR LoRA. Download {url} manually and place it at {target}. ({e})") from e
    if not target.is_file() or target.stat().st_size != LORA_SIZE:
        raise RuntimeError(f"DiffHDR LoRA at {target} is missing or incomplete. Delete it and download {url} manually.")
    return target


def load_lora_sd(variant: str) -> dict:
    """Loads and caches the key-renamed LoRA state dict."""
    if variant not in _sd_cache:
        import comfy.utils
        _sd_cache[variant] = rename_keys(comfy.utils.load_torch_file(str(ensure_lora(variant)), safe_load=True))
    return _sd_cache[variant]


def apply_lora(model, variant: str, strength: float = 1.0):
    """Returns a clone of ``model`` with the DiffHDR LoRA patched in.

    Raises:
        ValueError: If the model is not Wan2.1-VACE-14B (fewer than 80 patches match).
    """
    import comfy.lora
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(load_lora_sd(variant), key_map)
    patched = model.clone()
    applied = patched.add_patches(loaded, strength)
    if len(applied) < EXPECTED_PATCHES:
        raise ValueError(
            f"DiffHDR LoRA matched {len(applied)}/{EXPECTED_PATCHES} weights. Connect a Wan2.1-VACE-14B model "
            "(e.g. wan2.1_vace_14B_fp16.safetensors from Comfy-Org/Wan_2.1_ComfyUI_repackaged)."
        )
    return patched
