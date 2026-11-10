# Real-Time Multi-Instance Object Detection & Height Estimation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A computer vision pipeline built with Python, OpenCV, and Ultralytics YOLO11-seg. It detects and segments multiple instances of target object classes, counts them in real time, and calculates each object's vertical distance from a calibrated floor baseline. Includes both an interactive Streamlit web application and an OpenCV live-feed pipeline with ByteTrack support.

---

## Features

- **Multi-Instance Segmentation & Counting:** Generates instance masks, bounding boxes, class labels, and per-instance confidence scores.
- **Height-from-Floor Estimation:** Approximates real-world distance ($cm$) above a calibrated ground line using pixel-to-metric scaling.
- **Interactive Web App (`app.py`):** Browser-based UI powered by Streamlit for image upload, custom class filtering, and dynamic calibration adjustments.
- **Real-Time Webcam Pipeline (`main.py`):** Desktop application with HUD overlay displaying live FPS, warmup-excluded average FPS, and detection latency.
- **Persistent Object Tracking:** Optional ByteTrack integration (`--track`) to preserve IDs across frames.
- **Evaluation & Benchmarking:** Built-in evaluation scripts for computing mAP@50, mAP@50:95, precision-recall curves, and latency benchmarks.

---

## Project Structure

Project Structure

yolo-object-height/ ├── app.py # Streamlit web application (cloud/browser deployment) ├── main.py # Core CLI & webcam processing pipeline ├── config.yaml # Camera, model, calibration, and HUD configuration ├── requirements.txt # Project dependencies ├── .streamlit/ │ └── config.toml # Streamlit UI theme and server settings ├── src/ │ ├── detector.py # YOLO11-seg inference wrapper & dataclass │ ├── height_estimator.py # Pixel-to-centimeter vertical projection logic │ ├── visualizer.py # OpenCV annotation routines (masks, boxes, HUD) │ ├── fps_meter.py # Frame rate and latency tracking │ ├── logger.py # CSV detection logger │ ├── gui.py # Tkinter class selection dialog │ └── config.py # YAML configuration loader ├── scripts/ │ ├── calibrate.py # Interactive 2-click floor line & scale calibration │ ├── benchmark.py # Hardware FPS and p95 latency benchmark │ └── evaluate.py # mAP@50 and mAP@50:95 evaluation on annotated datasets ├── models/ # Model weight directory (git-ignored) └── outputs/ # Captured frames, video exports, and CSV logs

---
## Installation
### 1. Clone the repository

git clone https://github.com/divyanshigoyal2936/yolo-object-height.git
cd yolo-object-height
2. Set up a virtual environment

# Windows
python -m venv .venv
.venv\Scripts\activate
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies

pip install -r requirements.txt

Model Weights: The default model yolo11n-seg.pt will automatically download from Ultralytics on your first run. Alternatively, you can place it manually inside the models/ directory.

Usage
Option A: Interactive Web App (Streamlit)

Launch the browser UI to test detections and height calculations on images:


streamlit run app.py
Option B: Real-Time Webcam / Video Pipeline (CLI)

Run the desktop pipeline with your default webcam:


# Opens Tkinter prompt to select target objects
python main.py
# Specify target classes directly (comma-separated COCO classes)
python main.py --objects person,bottle,chair
# Enable ByteTrack persistent tracking
python main.py --objects person --track
# Process a video file and save the annotated output
python main.py --objects cup,laptop --source sample.mp4 --save-video outputs/annotated.mp4 --log-csv outputs/log.csv

Keyboard Shortcuts during live stream:

Q or ESC: Quit pipeline
S: Save high-resolution screenshot to outputs/
M: Toggle segmentation mask overlay on/off
Floor Calibration

Height estimation requires setting the baseline floor position and pixel-to-centimeter scale:


python scripts/calibrate.py
A reference frame will open.
Click on the floor level to define floor_y.
Click two reference points of known distance (e.g., a 30 cm ruler placed vertically or on the floor) and input the length.
Settings are automatically saved to calibration.yaml and loaded during runtime.
Benchmarking & Evaluation
Benchmark Runtime Performance

Measures median FPS, average FPS, and p95 latency across a fixed frame count:


python scripts/benchmark.py --source 0 --frames 300 --classes person

Results are exported to results/benchmark.json.

Model Evaluation (mAP)

Evaluate segmentation and detection performance against a YOLO-format annotated test set:


python scripts/evaluate.py --model models/yolo11n-seg.pt --data datasets/custom/data.yaml --split test

Outputs validation metrics to results/eval/metrics.json along with PR and F1 curves.

Limitations
Geometric Approximation: Ground height calculation uses a 2D planar projection. For accurate metric readings, the camera angle must remain fixed and objects should be situated at a similar focal plane as the calibration reference.
Occlusion: Dense clusters of overlapping items may cause merged segmentation polygons. Adjust iou and conf thresholds in config.yaml to optimize for specific scenes.
License

This project is licensed under the MIT License.











# 🎯 YOLO Object Height

**Real-time object detection, counting & height-from-floor estimation**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green?logo=opencv&logoColor=white)
![YOLO11](https://img.shields.io/badge/YOLO11--seg-Ultralytics-orange)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)


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
├── app.py                 streamlit web application
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
