"""Configuration loading: built-in defaults <- config.yaml <- calibration.yaml."""
import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "model": {"path": "models/yolo11n-seg.pt", "imgsz": 640, "conf": 0.35,
              "iou": 0.5, "device": None, "max_det": 100},
    "camera": {"width": 640, "height": 480},
    "height": {"floor_y": 430, "pixel_to_cm": 0.25},
    "display": {"show_masks": True, "mask_alpha": 0.35, "draw_floor_links": True},
    "fps": {"window": 30, "warmup_frames": 10},
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str = "config.yaml") -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    path = Path(path)
    for candidate in (path, path.with_name("calibration.yaml")):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                _merge(cfg, yaml.safe_load(f))
    return cfg
