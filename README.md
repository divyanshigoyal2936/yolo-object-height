# 🎯 YOLO Object Height

**Real-time object detection, counting & height-from-floor estimation**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green?logo=opencv&logoColor=white)
![YOLO11](https://img.shields.io/badge/YOLO11--seg-Ultralytics-orange)

Show an apple (or 5 of them) to your webcam. The system finds **every** instance, outlines it,
counts it, and tells you roughly **how far above the floor** it is.

---

## 📸 Demo
<!-- Add a screenshot or GIF: ![demo](docs/demo.png) -->

---

## ✨ What it does

| | Feature | Details |
|---|---|---|
| 🍎 | **Multi-instance detection** | Separate mask, box, ID and confidence for each object |
| 🔢 | **Counting** | Live count per selected class |
| 📏 | **Height from floor** | Pixel gap to a calibrated floor line, converted to cm |
| ⚡ | **FPS monitor** | Real-time FPS, average FPS and detection latency |
| 🧭 | **Tracking (optional)** | Stable IDs with ByteTrack (`--track`) |
| 📝 | **Logging (optional)** | CSV log of every detection, annotated video export |
| ✅ | **Smart input** | Typos are caught: "aple" → *Did you mean apple?* |
| 📊 | **Evaluation** | mAP@50, mAP@50:95, precision, recall |

---

## 🚀 Quick start

```bash
# 1️⃣ Install
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2️⃣ Run
python main.py --objects apple
```

> 💡 Weights: put `yolo11n-seg.pt` in `models/` (downloaded automatically on first run if missing).
> 🐧 Linux only: `sudo apt install python3-tk` for the GUI.

---

## ▶️ Usage

| Goal | Command |
|---|---|
| Use the input window | `python main.py` |
| One object | `python main.py --objects apple` |
| Several objects | `python main.py --objects apple,cup,person` |
| Names with spaces | `python main.py --objects "cell phone",laptop` |
| Fewer false detections | `python main.py --objects apple --conf 0.5` |
| Persistent IDs | `python main.py --objects person --track` |
| Video file + save output | `python main.py --objects apple --source demo.mp4 --save-video outputs/out.mp4` |
| Log detections | `python main.py --objects apple --log-csv outputs/log.csv` |

⌨️ **Keys:** `Q` quit · `S` screenshot · `M` toggle masks

🏷️ Object names are COCO classes (`apple`, `cup`, `bottle`, `cell phone`, `person`, ...).

---

## 📐 Calibrate the height estimate

Do this once, with the camera in its final fixed position.

```bash
python scripts/calibrate.py
```

1. Press **SPACE** to freeze a frame
2. Click the **floor line**
3. Click both ends of a **known length** (e.g. a ruler on the floor)
4. Type the real length in cm

Saved to `calibration.yaml` automatically.

> ⚠️ The result is an **approximation**: it assumes a fixed, level camera and objects at about the same depth as the reference.

---

## ⏱️ Benchmark FPS

```bash
python scripts/benchmark.py --source 0 --frames 300 --classes apple
```

Saves `results/benchmark.json` with your hardware, resolution, average/median FPS and p95 latency.

---

## 📊 Evaluate accuracy (mAP)

You need a labelled dataset in **YOLO segmentation format** (with a `data.yaml`).

```bash
# Optional: fine-tune on your own images
yolo segment train data=datasets/<name>/data.yaml model=yolo11n-seg.pt epochs=50 imgsz=640

# Evaluate on the TEST split
python scripts/evaluate.py --model runs/segment/train/weights/best.pt --data datasets/<name>/data.yaml --split test
```

Outputs `results/eval/metrics.json`, `metrics.md`, PR/F1 curves and the confusion matrix.

| ⚠️ Watch out | Why |
|---|---|
| Class-ID mismatch | COCO uses apple = 47, a custom dataset uses 0. Evaluating a COCO model on it directly gives wrong numbers. |
| Don't use `coco8-seg` / `coco128-seg` | They overlap the pretrained model's training data. |
| Report the **test** split | Never report training or validation numbers as test results. |

---

## 📈 Results

> Fill in **only values you measured**.

| Metric | Value |
|---|---|
| 💻 Hardware | `<CPU / GPU>` |
| 🖼️ Input resolution | `<e.g. 640x480>` |
| ⚡ Average FPS (300 frames) | `<X>` |
| 🗂️ Test set | `<dataset name, M images>` |
| 🎯 mAP@50 (mask) | `<X>` |
| 🎯 mAP@50:95 (mask) | `<X>` |
| ✅ Precision / Recall | `<P> / <R>` |

**📝 How to read the metrics**
- **mAP@50**: average precision at IoU 0.50
- **mAP@50:95**: average over IoU 0.50 to 0.95 (step 0.05), stricter
- **Precision / Recall**: measured at the confidence that maximises F1, not at the demo threshold (0.35)
- **mAP** is computed at conf 0.001; the demo uses 0.35. Don't mix the two.
- **FPS**: `avg_fps` is end-to-end (capture + detection); `detect_only_fps` is detection alone

---

## 🗂️ Project structure

```
📦 yolo-object-height
├── 📄 main.py                 ← run the app
├── ⚙️ config.yaml             ← settings
├── 📄 requirements.txt
├── 📁 src/
│   ├── detector.py            ← YOLO wrapper
│   ├── height_estimator.py    ← pixel → cm distance
│   ├── fps_meter.py           ← FPS / latency
│   ├── visualizer.py          ← drawing
│   ├── logger.py              ← CSV log
│   ├── gui.py                 ← input window
│   └── config.py
├── 📁 scripts/
│   ├── calibrate.py           ← floor line + cm per pixel
│   ├── benchmark.py           ← FPS measurement
│   └── evaluate.py            ← mAP / precision / recall
├── 📁 models/                 ← weights (git-ignored)
├── 📁 datasets/               ← your data (git-ignored)
├── 📁 results/                ← benchmark & eval outputs
└── 📁 outputs/                ← screenshots, videos, logs
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| 📷 `MSMF ... can't grab frame` (Windows) | Close other apps using the camera. The code already uses DirectShow and retries dropped frames. Try `--source 1` if you have several cameras. |
| 📏 cm values look wrong | Run `python scripts/calibrate.py` |
| ❓ "not a class of this model" | Use a COCO class name; check the suggestion printed |

---

## ⚠️ Limitations
- Height is a **2D pixel-to-cm approximation**, not true 3D depth.
- Heavily overlapping objects may merge or be missed. Tune `model.iou` and `model.conf` in `config.yaml`.
- The default model knows only the **80 COCO classes**. Other objects need a fine-tuned model.

---

## 🧰 Tech stack
Python · OpenCV · Ultralytics YOLO11 (instance segmentation) · PyTorch · NumPy · Tkinter · YAML
