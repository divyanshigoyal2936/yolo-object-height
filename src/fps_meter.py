"""Real-time and average FPS measurement."""
import time
from collections import deque

import numpy as np


class FPSMeter:
    """Call update() once per processed frame.

    realtime_fps : 1 / mean frame time over the last `window` frames.
    average_fps  : frames / total time, ignoring the first `warmup_frames`
                   (model warm-up and camera auto-exposure distort them).
    Frame time is the full loop time (capture + inference + drawing + display).
    """

    def __init__(self, window: int = 30, warmup_frames: int = 10):
        self.warmup_frames = warmup_frames
        self._recent = deque(maxlen=window)
        self._recent_latency = deque(maxlen=window)
        self._frame_times = []
        self._latencies = []
        self._last = None
        self._seen = 0

    def update(self, latency_ms: float = None) -> None:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            self._recent.append(dt)
            self._seen += 1
            if self._seen > self.warmup_frames:
                self._frame_times.append(dt)
                if latency_ms is not None:
                    self._latencies.append(latency_ms)
        if latency_ms is not None:
            self._recent_latency.append(latency_ms)
        self._last = now

    @property
    def realtime_fps(self) -> float:
        total = sum(self._recent)
        return len(self._recent) / total if total > 0 else 0.0

    @property
    def average_fps(self) -> float:
        total = sum(self._frame_times)
        if total > 0:
            return len(self._frame_times) / total
        return self.realtime_fps  # still warming up

    @property
    def latency_ms(self) -> float:
        return float(np.mean(self._recent_latency)) if self._recent_latency else 0.0

    def summary(self) -> dict:
        if not self._frame_times:
            return {"frames_measured": 0}
        ft = np.array(self._frame_times)
        out = {
            "frames_measured": int(len(ft)),
            "avg_fps": round(float(len(ft) / ft.sum()), 2),
            "median_fps": round(float(1.0 / np.median(ft)), 2),
            "min_fps": round(float(1.0 / ft.max()), 2),
            "p95_frame_time_ms": round(float(np.percentile(ft, 95) * 1000), 2),
        }
        if self._latencies:
            out["avg_detection_latency_ms"] = round(float(np.mean(self._latencies)), 2)
        return out
