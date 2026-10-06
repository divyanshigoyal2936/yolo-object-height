# Real-Time Multi-Instance Object Detection, Counting & Height-from-Floor Estimation

Python · OpenCV · Ultralytics YOLO11-seg

A webcam/video pipeline that detects every instance of the object classes you ask for
(e.g. 5 apples in a basket), draws a mask + box + confidence per instance, counts them,
estimates each object's distance above a calibrated floor line, and reports FPS.
A separate script evaluates the model with mAP@50 and mAP@50:95 on an annotated dataset.

## Features
- Multi-instance detection and per-class counting (separate mask, box, ID and confidence per instance)
- Optional persistent IDs with ByteTrack (`--track`)
- Real-time FPS, average FPS (warm-up excluded) and detection latency overlay
- Distance-from-floor estimation (original feature) with interactive calibration
- Class-name validation with suggestions ("aple" -> "apple")
- Optional CSV logging and annotated-video export
- Reproducible benchmark (`scripts/benchmark.py`) and evaluation (`scripts/evaluate.py`)

## Structure
```
.
├── main.py               # run the application
├── config.yaml           # model / camera / calibration / display settings
├── requirements.txt
├── src/
│   ├── detector.py       # YOLO wrapper, Detection dataclass
│   ├── height_estimator.py
│   ├── fps_meter.py
│   ├── visualizer.py
│   ├── logger.py
│   ├── gui.py            # tkinter input window
│   └── config.py
├── scripts/
│   ├── evaluate.py       # precision / recall / mAP@50 / mAP@50:95
│   ├── benchmark.py      # FPS / latency measurement
│   └── calibrate.py      # floor line + cm-per-pixel
├── models/               # put yolo11n-seg.pt here (git-ignored)
├── datasets/             # annotated datasets (git-ignored)
├── results/              # evaluation / benchmark outputs
└── outputs/              # screenshots, videos, CSV logs
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```
Place `yolo11n-seg.pt` in `models/` (or let Ultralytics download it on first run).
Linux only: `sudo apt install python3-tk` for the GUI.

## Run
```bash
python main.py                                    # GUI asks for object names
python main.py --objects apple                    # no GUI
python main.py --objects apple,orange --track
python main.py --objects apple --source demo.mp4 --save-video outputs/out.mp4 --log-csv outputs/log.csv
```
Keys: `Q` quit, `S` screenshot, `M` toggle masks.
Object names must be COCO class names for the default model (`apple`, `cup`, `cell phone`, `person` ...).

## Calibrate height estimation
```bash
python scripts/calibrate.py
```
Freeze a frame, click the floor line, click both ends of a known length (e.g. ruler on the floor),
type its length in cm. Values go to `calibration.yaml`. Keep the camera fixed afterwards.
The estimate is approximate: it assumes a fixed, level camera and objects at about the same depth as the reference.

## Benchmark FPS
```bash
python scripts/benchmark.py --source 0 --frames 300 --classes apple
python scripts/benchmark.py --source demo.mp4 --device cpu
```
Writes `results/benchmark.json` (hardware, resolution, avg/median FPS, p95 latency).

## Evaluate (mAP)
Requires a labelled dataset in YOLO segmentation format with a `data.yaml`.

```bash
python scripts/evaluate.py --model models/yolo11n-seg.pt --data datasets/<name>/data.yaml --split test
```
Writes `results/eval/metrics.json`, `metrics.md`, PR/F1 curves and the confusion matrix.

Choose the dataset carefully:
1. **Fine-tune on your own data (recommended for a resume).** Annotate ~150-300 images of your target
   objects (Roboflow / CVAT, export "YOLO segmentation"), split train/val/test, then
   `yolo segment train data=datasets/<name>/data.yaml model=yolo11n-seg.pt epochs=50 imgsz=640`
   and evaluate `runs/segment/train/weights/best.pt` on the **test** split.
   Also report the pretrained model on the same test set for a before/after comparison
   (class IDs must match, see 3).
2. **COCO val2017 subset.** Pretrained weights were not trained on val2017, so this is a fair test of the
   pretrained model. Do not use coco128-seg / coco8-seg for reporting: they overlap the training data.
3. **Class-ID warning.** A COCO-pretrained model uses COCO IDs (apple = 47). A custom single-class dataset
   uses ID 0, so evaluating the pretrained model on it directly gives wrong numbers. Either fine-tune, or
   write the dataset labels with COCO IDs.

## Reporting results (resume / report)
- Always state: model, dataset name + number of test images, split, image size, conf/IoU used, hardware.
- **mAP@50 / mAP@50:95**: report both, for boxes and masks (`metrics.md` has both). mAP@50:95 is
  the mean AP over IoU 0.50 to 0.95 (step 0.05); mAP@50 is at IoU 0.50 only.
- **Precision / recall**: Ultralytics reports them at the confidence threshold that maximises F1, not at
  your demo threshold of 0.35. Say so.
- **Evaluation threshold**: mAP is computed with a very low conf (0.001); the demo uses 0.35.
  These are different settings, so don't mix their numbers.
- **FPS**: report the `benchmark.py` result (average and median over >= 300 frames after warm-up) plus the hardware.
  Say whether it is detection-only or end-to-end (with drawing/display). Webcam FPS can be capped by the camera.
- Never report training/validation-set numbers as test results. Don't round up.

Example wording (replace placeholders with YOUR measured values):
> Fine-tuned YOLO11n-seg on a custom N-image apple dataset; on a held-out test set of M images,
> achieved mAP@50 = X and mAP@50:95 = Y (masks), precision P / recall R. Runs at Z FPS
> (avg, 640 px, <hardware>) with real-time multi-instance counting.

## Limitations
- Height/distance is a 2D pixel-to-cm approximation, not true 3D depth.
- Heavily overlapping objects may merge or be missed; tune `model.iou` and `model.conf` in `config.yaml`.

## Troubleshooting
- **Windows webcam error `MSMF ... can't grab frame`**: close other apps using the camera. The code already opens
  webcams with the DirectShow backend and retries dropped frames; try `--source 1` if you have several cameras.
- **Wrong cm values**: run `python scripts/calibrate.py` once with the camera in its final position.
