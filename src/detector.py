"""YOLO segmentation wrapper with multi-instance support."""
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    contour: Optional[np.ndarray]        # (N, 2) int32 polygon, or None
    index: int = 0                        # 1..N, left-to-right in this frame
    track_id: Optional[int] = None        # persistent ID when tracking is on
    floor_gap_cm: float = 0.0

    @property
    def label_id(self) -> int:
        return self.track_id if self.track_id is not None else self.index


class ObjectDetector:

    def __init__(self, model_path: str, conf: float = 0.35, iou: float = 0.5,
                 imgsz: int = 640, device: Optional[str] = None,
                 max_det: int = 100, track: bool = False):
        if not Path(model_path).exists():
            fallback = Path(model_path).name
            print(f"[warn] {model_path} not found, letting Ultralytics fetch '{fallback}'.")
            model_path = fallback
        self.model = YOLO(model_path)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device, self.max_det, self.track = device, max_det, track
        self.names: Dict[int, str] = self.model.names
        self._name_to_id = {n.lower(): i for i, n in self.names.items()}
        self.class_ids: Optional[List[int]] = None

    # ---- target handling -------------------------------------------------
    def resolve_targets(self, targets: Sequence[str]):
        """Split user input into (valid class names, {invalid: suggestions})."""
        valid, invalid = [], {}
        for t in targets:
            t = t.lower().strip()
            if t in self._name_to_id:
                valid.append(t)
            else:
                invalid[t] = get_close_matches(t, self._name_to_id.keys(), n=3)
        return valid, invalid

    def set_targets(self, names: Optional[Sequence[str]]) -> None:
        """Restrict inference to these classes (None = all classes)."""
        self.class_ids = None if names is None else [self._name_to_id[n] for n in names]

    # ---- inference -------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        kwargs = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                      max_det=self.max_det, classes=self.class_ids, verbose=False)
        if self.device is not None:
            kwargs["device"] = self.device

        if self.track:
            result = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
        else:
            result = self.model.predict(frame, **kwargs)[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = (result.boxes.id.cpu().numpy().astype(int)
                     if result.boxes.id is not None else [None] * len(xyxy))
        polygons = result.masks.xy if result.masks is not None else [None] * len(xyxy)

        detections = []
        for box, conf, cid, tid, poly in zip(xyxy, confs, cls_ids, track_ids, polygons):
            contour = None
            if poly is not None and len(poly) >= 3:
                contour = np.round(poly).astype(np.int32)
            detections.append(Detection(
                name=self.names[int(cid)].lower(),
                confidence=float(conf),
                bbox=tuple(int(v) for v in box),
                contour=contour,
                track_id=None if tid is None else int(tid),
            ))

        detections.sort(key=lambda d: d.bbox[0])      # stable left-to-right numbering
        for i, d in enumerate(detections, start=1):
            d.index = i
        return detections
