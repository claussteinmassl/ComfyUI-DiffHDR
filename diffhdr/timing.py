"""Wall-clock stage timing for the DiffHDR nodes.

The timer is deliberately free of device synchronisation: stage boundaries are
placed where the data has already landed on the CPU or on the intermediate
device anyway, so no extra stalls are introduced by measuring.

Stages may nest (``window`` contains ``encode``/``sample``/``decode``), in which
case the reported seconds overlap. ``total`` is always the wall clock since the
timer was created, not the sum of the stages, so the difference between the two
is the work that no stage covers.
"""

import logging
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Callable, Iterator, Optional

LOGGER_NAME = "DiffHDR"


class StageTimer:
    """Accumulates wall-clock seconds per named stage.

    Args:
        clock: Monotonic clock returning seconds. Injectable for tests.
    """

    def __init__(self, clock: Callable[[], float] = perf_counter):
        self._clock = clock
        self._start = clock()
        self._stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Times the enclosed block and adds it to the stage ``name``."""
        started = self._clock()
        try:
            yield
        finally:
            self._stages[name] = self._stages.get(name, 0.0) + (self._clock() - started)

    def elapsed(self) -> float:
        """Seconds since the timer was created."""
        return self._clock() - self._start

    def summary(self) -> dict[str, float]:
        """Seconds per stage, in first-use order."""
        return dict(self._stages)

    def format(self) -> str:
        """One log-friendly line, e.g. ``masks=41.2s sample=123.4s total=186.5s``."""
        parts = [f"{name}={seconds:.1f}s" for name, seconds in self._stages.items()]
        parts.append(f"total={self.elapsed():.1f}s")
        return " ".join(parts)


def stage(timer: Optional[StageTimer], name: str):
    """Times ``name`` on ``timer``; a no-op context manager when ``timer`` is None."""
    return nullcontext() if timer is None else timer.stage(name)


def node_context(mode: str, frame_count: int, window_count: int, steps: int) -> str:
    """Builds the bracketed context of a node timing line.

    Args:
        mode: ``image``, ``video`` or ``pano``.
        frame_count: Number of input frames.
        window_count: Number of sampling windows.
        steps: Sampling steps per window.
    """
    plural = "" if frame_count == 1 else "s"
    return f"{mode}, {frame_count} frame{plural}, {window_count} window(s), {steps} steps"


def log_timing(context: str, timer: StageTimer) -> None:
    """Logs one INFO line ``DiffHDR timing [<context>]: <stages>`` on the DiffHDR logger."""
    logging.getLogger(LOGGER_NAME).info("DiffHDR timing [%s]: %s", context, timer.format())
