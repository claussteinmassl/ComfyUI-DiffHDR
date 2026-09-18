"""Sliding-window planning and streaming overlap blending for long videos."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Window:
    """A temporal window ``[start, end)`` over real (unpadded) frames."""

    start: int
    end: int
    first: bool
    last: bool


def plan_windows(total: int, size: int, stride: int) -> list[Window]:
    """Plans overlapping windows.

    Args:
        total: Number of frames in the clip.
        size: Window size in frames.
        stride: Step between window starts, ``1 <= stride < size``.

    Returns:
        Windows in order. A window is skipped when its predecessor already reached the clip end.

    Raises:
        ValueError: If ``stride`` is not in ``[1, size)``.
    """
    if not 1 <= stride < size:
        raise ValueError(f"window_stride must be in [1, window_size); got {stride} / {size}")
    out = []
    for t in range(0, total, stride):
        if t - stride >= 0 and t - stride + size >= total:
            continue
        end = min(t + size, total)
        out.append(Window(start=t, end=end, first=t == 0, last=end == total))
    return out


def ramp_weights(length: int, first: bool, last: bool, border: int) -> torch.Tensor:
    """Blend weights with linear ramps of ``border`` frames on non-boundary ends."""
    x = torch.ones(length, dtype=torch.float32)
    border = min(border, length)
    if border <= 0:
        return x
    ramp = (torch.arange(border, dtype=torch.float32) + 0.5) / border
    if not first:
        x[:border] = ramp
    if not last:
        x[-border:] = ramp.flip(0)
    return x


class StreamingBlender:
    """Accumulates window outputs and emits frames once no later window can touch them.

    Args:
        size: Window size.
        stride: Window stride.
    """

    def __init__(self, size: int, stride: int):
        self.size = size
        self.stride = stride
        self._values: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None

    def add(self, chunk: torch.Tensor, window: Window) -> torch.Tensor:
        """Adds a window result ``[L,H,W,3]`` (linear) and returns the frames that are final."""
        chunk = chunk.detach().cpu().float()
        length = chunk.shape[0]
        w = ramp_weights(length, window.first, window.last, self.size - self.stride)
        weighted = chunk * w.view(-1, 1, 1, 1)
        if self._values is None:
            self._values, self._weights = weighted, w
        else:
            overlap = min(self._values.shape[0], length)
            self._values = torch.cat([self._values[:overlap] + weighted[:overlap], weighted[overlap:]])
            self._weights = torch.cat([self._weights[:overlap] + w[:overlap], w[overlap:]])
        count = min(self.stride, self._values.shape[0])
        return self._emit(count)

    def flush(self) -> torch.Tensor:
        """Returns all remaining frames."""
        if self._values is None:
            return torch.zeros(0)
        return self._emit(self._values.shape[0])

    def _emit(self, count: int) -> torch.Tensor:
        out = self._values[:count] / self._weights[:count].clamp_min(1e-8).view(-1, 1, 1, 1)
        self._values, self._weights = self._values[count:], self._weights[count:]
        return out
