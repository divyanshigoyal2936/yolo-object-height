"""Real-time multi-instance object detection, counting and height-from-floor estimation.

Examples
  python main.py                                   # GUI asks for object names
  python main.py --objects apple,cup               # skip the GUI
  python main.py --objects person --track          # persistent IDs (ByteTrack)
  python main.py --objects apple --source demo.mp4 --save-video outputs/out.mp4 --log-csv outputs/log.csv
"""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import time

import cv2

from src import visualizer
from src.config import load_config
from src.detector import ObjectDetector
from src.fps_meter import FPSMeter
from src.height_estimator import HeightEstimator
from src.logger import CSVLogger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--objects", help="comma-separated COCO class names (skips the GUI)")
    p.add_argument("--source", default="0", help="webcam index or video path (default 0)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", help="override model path")
    p.add_argument("--conf", type=float, help="override confidence threshold")
    p.add_argument("--track", action="store_true", help="enable ByteTrack persistent IDs")
    p.add_argument("--save-video", help="write annotated video to this path")
    p.add_argument("--log-csv", help="write every detection to this CSV")
    return p.parse_args()


def get_targets(args):
    if args.objects:
        raw = args.objects.split(",")
    else:
        from src.gui import InputGUI          # imported lazily: not needed with --objects
        raw = InputGUI().get_objects() or []
    return list(dict.fromkeys(o.strip().lower() for o in raw if o.strip()))


def open_source(source, cam_cfg):
    """Open a webcam (DirectShow first, more reliable than MSMF on Windows) or a video file."""
    if str(source).isdigit():
        index = int(source)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        return cap
    return cv2.VideoCapture(source)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["path"] = args.model
    if args.conf is not None:
        cfg["model"]["conf"] = args.conf

    targets = get_targets(args)
    if not targets:
        print("No object selected.")
        return

    m = cfg["model"]
    detector = ObjectDetector(m["path"], m["conf"], m["iou"], m["imgsz"],
                              m["device"], m["max_det"], track=args.track)

    targets, invalid = detector.resolve_targets(targets)
    for name, suggestions in invalid.items():
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        print(f"[warn] '{name}' is not a class of this model.{hint}")
    if not targets:
        print("No valid object names. Exiting.")
        return
    detector.set_targets(targets)

    estimator = HeightEstimator(cfg["height"]["floor_y"], cfg["height"]["pixel_to_cm"])
    meter = FPSMeter(cfg["fps"]["window"], cfg["fps"]["warmup_frames"])
    disp = cfg["display"]
    show_masks = disp["show_masks"]

    cap = open_source(args.source, cfg["camera"])
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        return

    logger = CSVLogger(args.log_csv) if args.log_csv else None
    writer = None
    Path("outputs").mkdir(exist_ok=True)
    frame_idx = 0
    failed_reads = 0
    is_live = str(args.source).isdigit()

    print("System started. Press Q to quit, S for screenshot, M to toggle masks.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if is_live and failed_reads < 30:
                    continue                      # webcam hiccup: try again
                break
            failed_reads = 0

            frame_idx += 1
            h, w = frame.shape[:2]
            if estimator.floor_y >= h:
                print(f"[warn] floor_y={estimator.floor_y} is outside the {h}px frame; clamping.")
                estimator.floor_y = h - 1

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            latency_ms = (time.perf_counter() - t0) * 1000

            for d in detections:
                d.floor_gap_cm = estimator.calculate_height(d.bbox)
            counts = Counter(d.name for d in detections)

            meter.update(latency_ms)

            visualizer.draw_floor_line(frame, estimator.floor_y)
            visualizer.draw_detections(frame, detections, estimator.floor_y, show_masks,
                                       disp["mask_alpha"], disp["draw_floor_links"])
            visualizer.draw_hud(frame, targets, counts, meter)

            if logger:
                logger.log(frame_idx, detections)
            if args.save_video:
                if writer is None:
                    src_fps = cap.get(cv2.CAP_PROP_FPS)
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps if src_fps and src_fps > 1 else 30, (w, h))
                writer.write(frame)

            cv2.imshow("Object Detection + Height Estimation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_masks = not show_masks
            if key == ord("s"):
                path = f"outputs/screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(path, frame)
                print("Saved", path)
    finally:
        cap.release()
        if writer:
            writer.release()
        if logger:
            logger.close()
        cv2.destroyAllWindows()

    print("\nFPS summary:", meter.summary())


if __name__ == "__main__":
    main()
