"""Evaluate a YOLO segmentation model on an annotated dataset (box + mask metrics).

All numbers come straight from Ultralytics' validator (COCO-style mAP).
Nothing is estimated or hard-coded.

  python scripts/evaluate.py --model models/yolo11n-seg.pt --data datasets/apples/data.yaml --split test
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO, __version__


def extract(metric, names):
    """Overall + per-class numbers from an Ultralytics Metric object."""
    overall = {
        "precision": float(metric.mp),       # at the confidence that maximises F1
        "recall": float(metric.mr),          # at the confidence that maximises F1
        "mAP50": float(metric.map50),
        "mAP50-95": float(metric.map),
    }
    per_class = {}
    for i, class_id in enumerate(metric.ap_class_index):
        p, r, ap50, ap = metric.class_result(i)
        per_class[names[int(class_id)]] = {
            "precision": float(p), "recall": float(r),
            "mAP50": float(ap50), "mAP50-95": float(ap)}
    return overall, per_class


def markdown_table(title, overall, per_class):
    lines = [f"### {title}", "",
             "| Class | Precision | Recall | mAP@50 | mAP@50:95 |", "|---|---|---|---|---|",
             f"| **all** | {overall['precision']:.3f} | {overall['recall']:.3f} | "
             f"{overall['mAP50']:.3f} | {overall['mAP50-95']:.3f} |"]
    for name, v in per_class.items():
        lines.append(f"| {name} | {v['precision']:.3f} | {v['recall']:.3f} | "
                     f"{v['mAP50']:.3f} | {v['mAP50-95']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolo11n-seg.pt")
    ap.add_argument("--data", required=True, help="dataset data.yaml (YOLO segmentation format)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="keep low for mAP (standard practice)")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    ap.add_argument("--device", default=None)
    ap.add_argument("--classes", nargs="*", help="evaluate only these class names")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    kwargs = dict(data=args.data, split=args.split, imgsz=args.imgsz, conf=args.conf,
                  iou=args.iou, plots=True, project=str(out), name="val", exist_ok=True)
    if args.device:
        kwargs["device"] = args.device
    if args.classes:
        lookup = {n.lower(): i for i, n in model.names.items()}
        kwargs["classes"] = [lookup[c.lower()] for c in args.classes]

    metrics = model.val(**kwargs)

    names = metrics.names
    box, box_pc = extract(metrics.box, names)
    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "ultralytics_version": __version__,
        "model": args.model, "data": args.data, "split": args.split,
        "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
        "box": {"overall": box, "per_class": box_pc},
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in metrics.speed.items()},
    }
    md = [f"## Evaluation: `{args.model}` on `{args.data}` ({args.split} split)\n",
          markdown_table("Bounding boxes", box, box_pc)]

    if hasattr(metrics, "seg"):                      # segmentation models only
        seg, seg_pc = extract(metrics.seg, names)
        report["mask"] = {"overall": seg, "per_class": seg_pc}
        md.append(markdown_table("Instance masks", seg, seg_pc))

    md.append(f"Speed (ms/image): {report['speed_ms_per_image']}\n")

    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"Saved: {out / 'metrics.json'}, {out / 'metrics.md'} (plus PR/F1 curves and confusion matrix in {out / 'val'})")


if __name__ == "__main__":
    main()
