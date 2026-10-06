"""Optional CSV logging of every detection (useful for later analysis)."""
import csv
from datetime import datetime
from pathlib import Path


class CSVLogger:
    HEADER = ["frame", "timestamp", "object", "instance", "track_id",
              "confidence", "x1", "y1", "x2", "y2", "floor_gap_cm"]

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def log(self, frame_idx: int, detections) -> None:
        ts = datetime.now().isoformat(timespec="milliseconds")
        for d in detections:
            self._writer.writerow([frame_idx, ts, d.name, d.index, d.track_id,
                                   f"{d.confidence:.4f}", *d.bbox, d.floor_gap_cm])

    def close(self) -> None:
        self._file.close()
