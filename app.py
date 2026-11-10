import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter

from src.config import load_config
from src.detector import ObjectDetector
from src.height_estimator import HeightEstimator
from src import visualizer

st.set_page_config(page_title="YOLO Object Detection & Height Estimation", layout="wide")

st.title("🎯 Real-Time Object Detection & Height Estimation")
st.write("Upload an image to detect objects, show segmentation masks, and calculate their height from the floor line.")

@st.cache_resource
def get_detector(model_path="yolo11n-seg.pt", conf=0.25):
    return ObjectDetector(model_path, conf=conf, iou=0.45, imgsz=640, device="cpu", max_det=100)

cfg = load_config("config.yaml")

# Sidebar settings
st.sidebar.header("Configuration")
target_input = st.sidebar.text_input("Target Classes (comma separated)", value="person, bottle, cup, chair, apple")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, float(cfg["model"]["conf"]), 0.05)
floor_y = st.sidebar.slider("Floor Line Y (pixels)", 100, 1000, int(cfg["height"]["floor_y"]), 10)
pixel_to_cm = st.sidebar.number_input("Pixel to cm scale", value=float(cfg["height"]["pixel_to_cm"]))
show_masks = st.sidebar.checkbox("Show Segmentation Masks", value=True)

# Parse targets
targets = [t.strip().lower() for t in target_input.split(",") if t.strip()]

detector = get_detector(cfg["model"]["path"], conf_thresh)
targets, invalid = detector.resolve_targets(targets)
if invalid:
    st.sidebar.warning(f"Ignored unknown classes: {', '.join(invalid.keys())}")
detector.set_targets(targets)

estimator = HeightEstimator(floor_y, pixel_to_cm)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    img = Image.open(uploaded_file).convert("RGB")
    frame = np.array(img)
    # Convert RGB to BGR for OpenCV processing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]
    if estimator.floor_y >= h:
        estimator.floor_y = h - 1

    # Run Detection
    detections = detector.detect(frame_bgr)
    for d in detections:
        d.floor_gap_cm = estimator.calculate_height(d.bbox)
    counts = Counter(d.name for d in detections)

    # Visualize results
    visualizer.draw_floor_line(frame_bgr, estimator.floor_y)
    visualizer.draw_detections(frame_bgr, detections, estimator.floor_y, show_masks,
                               cfg["display"]["mask_alpha"], cfg["display"]["draw_floor_links"])

    # Convert back to RGB for Streamlit display
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(result_rgb, caption="Processed Image", use_container_width=True)
    with col2:
        st.subheader("📊 Detection Summary")
        if counts:
            for name, count in counts.items():
                st.write(f"- **{name.capitalize()}**: {count}")
            st.divider()
            st.subheader("📏 Estimated Height from Floor")
            for idx, d in enumerate(detections, 1):
                st.write(f"**#{idx} {d.name}**: {d.floor_gap_cm:.1f} cm")
        else:
            st.info("No target objects detected.")