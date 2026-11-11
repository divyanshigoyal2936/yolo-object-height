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
st.write("Open your camera, specify the object class you want to detect, and measure its height from the floor line.")

@st.cache_resource
def get_detector(model_path="yolo11n-seg.pt", conf=0.25):
    return ObjectDetector(model_path, conf=conf, iou=0.45, imgsz=640, device="cpu", max_det=100)

cfg = load_config("config.yaml")

# Sidebar configuration
st.sidebar.header("⚙️ Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, float(cfg["model"]["conf"]), 0.05)
floor_y = st.sidebar.slider("Floor Line Y (pixels)", 100, 1000, int(cfg["height"]["floor_y"]), 10)
pixel_to_cm = st.sidebar.number_input("Pixel to cm scale", value=float(cfg["height"]["pixel_to_cm"]), step=0.01)
show_masks = st.sidebar.checkbox("Show Segmentation Masks", value=True)

# Main area inputs: User types the object to detect
col_input, col_hint = st.columns([2, 1])
with col_input:
    target_input = st.text_input(
        "🔍 Enter object(s) you want to detect (comma-separated):",
        value="cup, person, bottle",
        help="Type any COCO class name (e.g., person, bottle, cup, chair, laptop, cell phone, apple)"
    )
with col_hint:
    st.caption("Common COCO classes: person, bottle, cup, chair, laptop, cell phone, apple, book, dog, cat")

# Choose input method (Webcam or File Upload)
mode = st.radio("Choose input mode:", ["📷 Use Live Webcam", "📁 Upload Image"], horizontal=True)

image_source = None
if mode == "📷 Use Live Webcam":
    # Opens your computer or phone webcam directly in the browser
    image_source = st.camera_input("Open Camera and take a snapshot:")
else:
    image_source = st.file_uploader("Upload an image file:", type=["jpg", "jpeg", "png"])

if image_source is not None:
    # Parse target classes
    targets = [t.strip().lower() for t in target_input.split(",") if t.strip()]
    if not targets:
        st.warning("Please type at least one object name above to detect.")
    else:
        detector = get_detector(cfg["model"]["path"], conf_thresh)
        targets, invalid = detector.resolve_targets(targets)
        if invalid:
            st.warning(f"Ignored unknown class names: {', '.join(invalid.keys())}")
        
        if not targets:
            st.error("No valid object names found to detect. Please check your spelling.")
        else:
            detector.set_targets(targets)
            estimator = HeightEstimator(floor_y, pixel_to_cm)

            # Read image data
            img = Image.open(image_source).convert("RGB")
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            h, w = frame_bgr.shape[:2]
            if estimator.floor_y >= h:
                estimator.floor_y = h - 1

            # Run detection
            with st.spinner("Detecting objects and calculating height..."):
                detections = detector.detect(frame_bgr)
                for d in detections:
                    d.floor_gap_cm = estimator.calculate_height(d.bbox)
                counts = Counter(d.name for d in detections)

                # Annotate image
                visualizer.draw_floor_line(frame_bgr, estimator.floor_y)
                visualizer.draw_detections(frame_bgr, detections, estimator.floor_y, show_masks,
                                           cfg["display"]["mask_alpha"], cfg["display"]["draw_floor_links"])

                result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Display output
            res_col, stats_col = st.columns([2, 1])
            with res_col:
                st.image(result_rgb, caption="Processed Camera Frame", use_container_width=True)
            with stats_col:
                st.subheader("📊 Results")
                if counts:
                    st.success(f"Detected {len(detections)} instance(s) of '{', '.join(targets)}'")
                    for name, count in counts.items():
                        st.write(f"- **{name.capitalize()}**: {count} detected")
                    
                    st.divider()
                    st.subheader("📏 Height from Floor")
                    for idx, d in enumerate(detections, 1):
                        st.write(f"**#{idx} {d.name}**: {d.floor_gap_cm:.1f} cm above floor")
                else:
                    st.info(f"No '{', '.join(targets)}' detected in the frame. Try adjusting camera angle, distance, or lighting.")