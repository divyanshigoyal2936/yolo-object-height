"""Reproducible FPS / latency benchmark (no GUI, no drawing).

  python scripts/benchmark.py --source 0 --frames 300 --classes apple
  python scripts/benchmark.py --source demo.mp4 --device cpu
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import __version__

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detector import ObjectDetector  # noqa: E402


def open_capture(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)   # reliable on Windows
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(source))
        return cap
    return cv2.VideoCapture(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolo11n-seg.pt")
    ap.add_argument("--source", default="0")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None)
    ap.add_argument("--classes", nargs="*", help="class names to detect (default: all)")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    detector = ObjectDetector(args.model, conf=args.conf, imgsz=args.imgsz, device=args.device)
    if args.classes:
        valid, _ = detector.resolve_targets(args.classes)
        detector.set_targets(valid)

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    is_live = args.source.isdigit()

    detect_ms, total_ms, resolution = [], [], None
    processed, failed = 0, 0
    while processed < args.warmup + args.frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            failed += 1
            if is_live and failed < 30:
                continue
            break
        failed = 0
        resolution = frame.shape[1], frame.shape[0]
        t1 = time.perf_counter()
        detector.detect(frame)
        t2 = time.perf_counter()
        if processed >= args.warmup:
            detect_ms.append((t2 - t1) * 1000)
            total_ms.append((t2 - t0) * 1000)
        processed += 1
    cap.release()

    if not total_ms:
        raise SystemExit("Not enough frames for a benchmark.")

    d, t = np.array(detect_ms), np.array(total_ms)
    report = {
        "frames": len(t), "warmup_frames": args.warmup, "input_resolution": resolution,
        "imgsz": args.imgsz, "model": args.model,
        "device": args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__, "ultralytics": __version__,
        "avg_fps": round(float(len(t) / (t.sum() / 1000)), 2),
        "median_fps": round(float(1000 / np.median(t)), 2),
        "p95_latency_ms": round(float(np.percentile(t, 95)), 2),
        "avg_detect_ms": round(float(d.mean()), 2),
        "detect_only_fps": round(float(1000 / d.mean()), 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
